"""Module: G_P -> G_R lift (conditional instantiation) | Paper section: §3.1 | Status: wip"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import structlog
from pydantic import BaseModel

from ssp.graph.g_percept import PerceptualGraph
from ssp.graph.g_reason import ReasoningGraph
from ssp.ontology.relations import L1SuppressionRelation
from ssp.ontology.risk_events import (
    EntityFilter,
    InstantiationCondition,
    RiskEventTemplate,
    RiskEventType,
)
from ssp.ontology.schema import Edge, FactorNode, Node
from ssp.ontology.template_registry import TemplateRegistry

logger = structlog.get_logger()


def _matches_entity_filter(node: Node, ef: EntityFilter) -> bool:
    """Check if a node matches an EntityFilter (type + subtypes + required_attributes).

    ADR-026: a node carries a SET of subtypes; it matches the filter's accepted
    subtypes iff the two sets intersect (a ceramic mug with
    subtypes={fragile_object, container} matches BOTH a fragile filter and a
    container filter, so it instantiates both fragile_breakage and spill_damage).
    """
    if node.type != ef.entity_type:
        return False
    if ef.subtypes and not (set(node.subtypes) & set(ef.subtypes)):
        return False
    for attr_name, attr_value in ef.required_attributes.items():
        node_val = getattr(node.attributes, attr_name, None)
        if node_val is None or str(node_val) != attr_value:
            return False
    return True


CondStatus = Literal["met", "unmet", "unresolved"]


def _evaluate_instantiation_condition(
    cond: InstantiationCondition,
    hazard: Node,
    target: Node,
    g_p: PerceptualGraph,
    relation_blind: bool = False,
) -> tuple[CondStatus, str]:
    """Evaluate a single instantiation condition. Returns (status, evidence).

    Three-state (ADR-029):
      - "met":        the condition is satisfied.
      - "unmet":      the condition is DEFINITELY false (identity/spatial
                      mismatch, or an evidence attribute with a known value not in
                      the accepted set) -> not-applicable.
      - "unresolved": an EVIDENCE condition is undecidable (attribute unknown /
                      missing; contents_observation=unknown) -> uncertain candidate.

    Scope note (ADR-029): identity gates (`entity_type_in`, incl. subtype-set
    membership) and spatial `relation_exists` use a predicate-local closed-world
    assumption -- a mismatch/absence is "unmet", never "unresolved". Their own
    epistemic completeness (e.g. "is the subtype classifier complete?") is
    deferred to a later ADR. Only `entity_attr_in` and observation-bound
    `relation_exists` (contents) participate in the "unresolved" state this round.

    relation_blind (E3 / ADR-020): when True, `relation_exists` conditions are
    skipped (treated as vacuously met). Default False = unchanged.
    """
    op = cond.op
    args = cond.args

    if relation_blind and op == "relation_exists":
        return "met", "relation_blind: relation_exists skipped"

    if op == "entity_type_in":
        field = str(args.get("field", ""))
        values = args.get("values", [])
        if not isinstance(values, list):
            values = [str(values)]

        # Identity gate: predicate-local closed-world -> met / unmet only.
        if field == "hazard.subtype":
            # ADR-026: match if the hazard's subtype set intersects `values`.
            passed = bool(set(hazard.subtypes) & set(values))
            evidence = f"hazard.subtypes={hazard.subtypes} in {values}"
        elif field == "hazard.type":
            passed = str(hazard.type) in values
            evidence = f"hazard.type={hazard.type} in {values}"
        elif field == "target.subtype":
            passed = bool(set(target.subtypes) & set(values))
            evidence = f"target.subtypes={target.subtypes} in {values}"
        elif field == "target.type":
            passed = str(target.type) in values
            evidence = f"target.type={target.type} in {values}"
        else:
            return "unmet", f"unknown field: {field}"
        return ("met" if passed else "unmet"), evidence

    if op == "entity_attr_in":
        entity_role = str(args.get("entity", ""))
        attr_name = str(args.get("attr", ""))
        values = args.get("values", [])
        if not isinstance(values, list):
            values = [str(values)]

        node = hazard if entity_role == "hazard" else target
        node_val = getattr(node.attributes, attr_name, None)
        if node_val is None and attr_name == "vulnerability":
            node_val = node.vulnerability

        # Evidence gate: three-state (ADR-029).
        #   value in set          -> met
        #   value known, not in   -> unmet (definite non-applicability)
        #   value unknown / None   -> unresolved (undecidable -> uncertain candidate)
        if node_val is None:
            return "unresolved", f"{entity_role}.{attr_name} missing (unresolved)"
        if str(node_val) == "unknown":
            return "unresolved", f"{entity_role}.{attr_name}=unknown (unresolved)"
        if str(node_val) in values:
            return "met", f"{entity_role}.{attr_name}={node_val} in {values}"
        return "unmet", f"{entity_role}.{attr_name}={node_val} not in {values}"

    if op == "relation_exists":
        relations_raw = args.get("relation", [])
        if not isinstance(relations_raw, list):
            relations_raw = [str(relations_raw)]
        src_role = str(args.get("src", "hazard"))
        dst_role = str(args.get("dst", "target"))
        dst_type = args.get("dst_type")  # ADR-028: optional third-party target
        observation = args.get("observation")  # ADR-029: explicit completeness binding

        src_id = hazard.id if src_role == "hazard" else target.id

        # ADR-028/029: when `dst_type` is given, match "does the src-role entity
        # have this relation to ANY node of type dst_type?" (e.g. container
        # contains SOME substance -- positive evidence of contents).
        if dst_type is not None:
            dst_type_str = str(dst_type)
            found_relation: str | None = None
            for edge in g_p.edges:
                rel_str = str(edge.relation)
                if rel_str not in relations_raw or edge.src != src_id:
                    continue
                dst_node = g_p.nodes.get(edge.dst)
                if dst_node is not None and str(dst_node.type) == dst_type_str:
                    found_relation = rel_str
                    break
            if found_relation is not None:
                return "met", f"relation({found_relation}) from {src_id} to a {dst_type_str}"
            # No positive evidence. ADR-029: three-state via an EXPLICIT
            # observation binding (no hard-coded "read hazard.contents"). The
            # binding names which entity's completeness attribute to read.
            if observation is not None and isinstance(observation, dict):
                obs_role = str(observation.get("entity_role", src_role))
                obs_field = str(observation.get("field", ""))
                complete_value = str(observation.get("complete_value", "complete"))
                obs_node = hazard if obs_role == "hazard" else target
                obs_val = getattr(obs_node.attributes, obs_field, None)
                if obs_val is not None and str(obs_val) == complete_value:
                    # observation complete + no edge -> known-absent (e.g. empty)
                    return "unmet", (
                        f"no {relations_raw} to a {dst_type_str}; "
                        f"{obs_role}.{obs_field}={complete_value} (known absent)"
                    )
                # observation incomplete/unknown/missing -> undecidable
                return "unresolved", (
                    f"no {relations_raw} to a {dst_type_str}; "
                    f"{obs_role}.{obs_field}={obs_val} (contents unresolved)"
                )
            # No observation binding: predicate-local closed-world (unmet).
            return "unmet", f"no {relations_raw} from {src_id} to a {dst_type_str}"

        dst_id = target.id if dst_role == "target" else hazard.id

        # Spatial relation. Special case: if src and dst resolve to the same
        # entity, check if it participates in any edge with the relation.
        same_entity = src_id == dst_id
        found_relation = None

        for edge in g_p.edges:
            rel_str = str(edge.relation)
            if rel_str not in relations_raw:
                continue
            if same_entity:
                if edge.src == src_id or edge.dst == src_id:
                    found_relation = rel_str
                    break
            else:
                if edge.src == src_id and edge.dst == dst_id:
                    found_relation = rel_str
                    break

        # Spatial relation: predicate-local closed-world -> met / unmet only.
        if found_relation is not None:
            return "met", f"relation({found_relation}) between {src_id} and {dst_id}"
        return "unmet", f"no {relations_raw} between {src_id} and {dst_id}"

    return "unmet", f"unknown op: {op}"


def evaluate_intrinsic_conditions(
    template: RiskEventTemplate,
    factor: FactorNode,
    g_p: PerceptualGraph,
) -> tuple[Literal["met", "unmet", "unresolved"], str]:
    """Evaluate a template's intrinsic_activation_conditions against a factor
    (ADR-025 + ADR-027). These are scene-intrinsic physical preconditions (e.g.
    stability in {unstable,tipping}) that gate the action-free view only -- lift
    already instantiated the factor, so an action can still activate it regardless.

    Returns (status, reason) with a three-state status the scene-intrinsic router
    (ADR-027) consumes directly instead of matching on the reason string:
      - "met":        all preconditions satisfied, or the template has no gate
                      -> the factor is an active intrinsic risk (activated/suppressed).
      - "unmet":      a precondition is DEFINITELY false (e.g. stability=stable);
                      we KNOW the trigger is absent -> latent hazard.
      - "unresolved": a precondition is UNDECIDABLE (unknown value or missing
                      attribute); we cannot confirm or rule out -> uncertain.
    A definite "not a risk now" is never conflated with "evidence insufficient".
    """
    conds = template.intrinsic_activation_conditions
    if not conds:
        return "met", ""

    hazard = g_p.nodes[factor.hazard_id]
    target = g_p.nodes[factor.target_id]
    # Reuse the shared three-state evaluator (ADR-029): same status + reason
    # taxonomy as instantiation, evaluated here at the gate layer. Precedence
    # unmet > unresolved > met: a definite non-trigger (-> latent) dominates an
    # undecidable one (-> uncertain).
    first_unresolved: str | None = None
    for cond in conds:
        status, evidence = _evaluate_instantiation_condition(cond, hazard, target, g_p)
        if status == "unmet":
            return "unmet", f"intrinsic precondition unmet: {evidence}"
        if status == "unresolved" and first_unresolved is None:
            first_unresolved = evidence
    if first_unresolved is not None:
        return "unresolved", f"intrinsic precondition unresolved: {first_unresolved}"
    return "met", ""


def _has_grounding_edge(
    g_p: PerceptualGraph,
    suppressor_id: str,
    relation: L1SuppressionRelation,
    factor_node: FactorNode,
) -> bool:
    """Per ADR-013: a candidate suppressor only counts if g_p contains an
    explicit L1 edge of the same relation type binding it to the factor's
    hazard (for isolated_by/neutralized_by) or target (for guarded_by).

    Edge directions used in v0_3 / v0_3_aug / sandbox:
      - isolated_by    : src=suppressor, dst=hazard
      - neutralized_by : src=suppressor, dst=hazard
      - guarded_by     : src=suppressor, dst=target
    """
    if relation == L1SuppressionRelation.GUARDED_BY:
        expected_dst = factor_node.target_id
    else:
        expected_dst = factor_node.hazard_id
    for edge in g_p.edges:
        if (edge.src == suppressor_id
                and edge.dst == expected_dst
                and edge.relation == relation):
            return True
    return False


def _find_suppressors(
    g_p: PerceptualGraph,
    factor_node: FactorNode,
    template: RiskEventTemplate,
) -> list[tuple[str, L1SuppressionRelation, str, float, list[RiskEventType] | None]]:
    """Find L1-grounded suppressor entities for a factor node.

    Per ADR-013: a suppressor is included iff
      (1) it satisfies the template's suppressor_filter (entity-side condition);
      (2) g_p contains an explicit L1 edge of the same relation type binding
          the candidate to the factor's hazard (isolated_by / neutralized_by)
          or target (guarded_by).

    Returns list of
    (suppressor_id, relation, mitigation_mode, residual_multiplier, suppression_dims).
    Per ADR-015: residual_multiplier is the fraction of risk that remains;
    mitigation_mode is "hard" or "soft". suppression_dims=None means suppress
    only the factor node's own re_type dimension.
    """
    suppressors: list[tuple[str, L1SuppressionRelation, str, float, list[RiskEventType] | None]] = []
    for mitigation in template.mitigation_relations:
        for node in g_p.nodes.values():
            if node.id in (factor_node.hazard_id, factor_node.target_id):
                continue
            if not _matches_entity_filter(node, mitigation.suppressor_filter):
                continue
            if not _has_grounding_edge(g_p, node.id, mitigation.relation, factor_node):
                continue
            suppressors.append((
                node.id, mitigation.relation, mitigation.mitigation_mode,
                mitigation.residual_multiplier, mitigation.suppression_dims,
            ))
    return suppressors


def _is_self_referential(template: RiskEventTemplate) -> bool:
    """A template is self-referential when its risk accumulates on the hazard
    itself, with no independent vulnerable target (the falling object IS the
    object at risk; the breaking cup IS the fragile object at risk).

    Signal (ADR-024): every propagation edge is a hazard self-loop
    (from_role == to_role == "hazard"). Such templates must instantiate only
    the diagonal target == hazard, never a hazard x target Cartesian product
    (the cross-pairs are semantically redundant duplicates -- their propagation
    and activation rules both bind target back to hazard anyway).

    Two-entity templates return False and keep full instantiation. Note that
    fall_injury's loop is on `target` (from_role == to_role == "target", hazard
    is a distinct furniture support), so it is NOT self-referential here.
    """
    edges = template.propagation_edges
    return bool(edges) and all(
        e.from_role == "hazard" and e.to_role == "hazard" for e in edges
    )


class UncertainFactorCandidate(BaseModel):
    """A factor whose identity/spatial conditions matched but at least one
    EVIDENCE condition is undecidable (ADR-029). It is NEVER built into G_R and
    NEVER propagated -- kept physically separate from the reasoning graph so a
    candidate can never pollute the propagation fixed point (a structural
    guarantee, not a filtering discipline). The parser routes it to an
    instantiation-stage `UncertainRiskEvent`."""

    template_id: str
    re_type: RiskEventType
    hazard_id: str
    target_id: str
    unresolved_conditions: list[str]  # which evidence conditions were undecidable
    resolved_evidence: list[str]  # conditions already met (provenance)
    constraint_template_ids: list[str]

    @property
    def factor_id(self) -> str:
        """Synthetic id mirroring the resolved-factor id scheme (never in G_R)."""
        return f"UC_{self.template_id}_{self.hazard_id}_{self.target_id}"


@dataclass
class LiftResult:
    """Result of the canonical epistemic lift (ADR-029).

    resolved_graph is the ONLY thing that enters propagation. unresolved_candidates
    are held separately -- lift never discards them (No Silent Failure).
    """

    resolved_graph: ReasoningGraph
    unresolved_candidates: list[UncertainFactorCandidate] = field(default_factory=list)


def lift(
    g_p: PerceptualGraph,
    registry: TemplateRegistry,
    relation_blind: bool = False,
) -> LiftResult:
    """Canonical epistemic lift: G_P -> (G_R, unresolved candidates).

    For each template and each candidate (hazard, target) pair, evaluates the
    instantiation_conditions with three-state semantics (ADR-029) and routes by
    precedence **unmet > unresolved > met**:
      - any condition unmet  -> not-applicable, no factor, no candidate;
      - else any unresolved  -> an UncertainFactorCandidate (NOT built into G_R,
                                NOT propagated -- No Silent Failure: never dropped);
      - all met              -> a resolved FactorNode + propagation/suppression
                                edges in G_R (unchanged from prior behavior).

    Never discards unresolved candidates. Graph-only callers must use
    `lift_resolved_only()` (explicitly lossy) so information loss is auditable.

    Args:
        g_p: Perceptual scene graph with L0 edges.
        registry: Loaded template registry.
        relation_blind: E3 ablation (ADR-020). When True, `relation_exists`
            instantiation conditions are treated as vacuously met so lift degrades
            to object-co-occurrence. Default False = unchanged.

    Returns:
        LiftResult(resolved_graph, unresolved_candidates).
    """
    nodes = list(g_p.nodes.values())
    edges: list[Edge] = []
    factor_nodes: list[FactorNode] = []
    candidates: list[UncertainFactorCandidate] = []
    factor_counter = 0

    for template in registry.all_templates():
        hazard_candidates = [n for n in nodes if _matches_entity_filter(n, template.hazard_source)]
        target_candidates: list[Node] = []
        for tf in template.vulnerable_targets:
            for n in nodes:
                if _matches_entity_filter(n, tf) and n not in target_candidates:
                    target_candidates.append(n)

        # ADR-024: self-referential templates (object_fall, fragile_breakage)
        # accumulate risk on the hazard itself -- the vulnerable target IS the
        # hazard. Instantiate only the diagonal target == hazard; skip the
        # spurious hazard x target cross-pairs that otherwise blow up N^2.
        self_referential = _is_self_referential(template)

        for hazard in hazard_candidates:
            targets_for_hazard = [hazard] if self_referential else target_candidates
            for target_node in targets_for_hazard:
                # ADR-029 three-state aggregation (unmet > unresolved > met).
                any_unmet = False
                unresolved_conds: list[str] = []
                evidence_list: list[str] = []

                for cond in template.instantiation_conditions:
                    status, evidence = _evaluate_instantiation_condition(
                        cond, hazard, target_node, g_p, relation_blind
                    )
                    if status == "unmet":
                        any_unmet = True
                        break
                    if status == "unresolved":
                        unresolved_conds.append(evidence)
                    else:
                        evidence_list.append(evidence)

                if any_unmet:
                    continue  # not-applicable: no factor, no candidate

                if unresolved_conds:
                    # Undecidable evidence -> uncertain candidate, kept OUT of G_R.
                    candidates.append(UncertainFactorCandidate(
                        template_id=template.id,
                        re_type=template.name,
                        hazard_id=hazard.id,
                        target_id=target_node.id,
                        unresolved_conditions=unresolved_conds,
                        resolved_evidence=evidence_list,
                        constraint_template_ids=list(template.constraint_template_ids),
                    ))
                    continue

                # All met -> resolved factor node.
                factor_counter += 1
                re_type_idx = list(RiskEventType).index(template.name)
                severity = [0.0] * len(RiskEventType)
                likelihood = [0.0] * len(RiskEventType)
                severity[re_type_idx] = template.severity_default
                likelihood[re_type_idx] = template.likelihood_prior

                fn = FactorNode(
                    id=f"FN_{template.id}_{hazard.id}_{target_node.id}",
                    re_type=template.name,
                    template_id=template.id,
                    hazard_id=hazard.id,
                    target_id=target_node.id,
                    severity=severity,
                    likelihood=likelihood,
                    confidence=min(hazard.confidence, target_node.confidence),
                    instantiation_evidence=evidence_list,
                    constraint_template_ids=list(template.constraint_template_ids),
                )
                factor_nodes.append(fn)

                # Create propagation edges (entity -> factor node)
                for prop_spec in template.propagation_edges:
                    src_id = hazard.id if prop_spec.from_role == "hazard" else target_node.id
                    edges.append(Edge(
                        src=src_id,
                        dst=fn.id,
                        relation=prop_spec.relation,
                        sign="+",
                        weight=prop_spec.base_weight,
                        confidence=min(hazard.confidence, target_node.confidence),
                    ))

                # Find and attach suppressors
                suppressors = _find_suppressors(g_p, fn, template)
                for sup_id, sup_relation, sup_mode, residual_mult, sup_dims in suppressors:
                    edges.append(Edge(
                        src=sup_id,
                        dst=fn.id,
                        relation=sup_relation,
                        sign="-",
                        weight=residual_mult,
                        confidence=g_p.nodes[sup_id].confidence,
                        suppression_dims=sup_dims,
                        mitigation_mode=sup_mode,
                    ))

    logger.info(
        "lift_complete",
        num_factor_nodes=len(factor_nodes),
        num_unresolved_candidates=len(candidates),
        num_edges=len(edges),
        templates_evaluated=len(registry.all_templates()),
    )

    resolved_graph = ReasoningGraph(
        nodes=nodes,
        edges=edges,
        factor_nodes=factor_nodes,
    )
    return LiftResult(resolved_graph=resolved_graph, unresolved_candidates=candidates)


def lift_resolved_only(
    g_p: PerceptualGraph,
    registry: TemplateRegistry,
    relation_blind: bool = False,
) -> ReasoningGraph:
    """Explicitly lossy adapter for graph-only callers (ADR-029).

    Returns just the resolved G_R, DISCARDING unresolved candidates. Named so the
    information loss is auditable; when candidates are dropped a warning records
    the count (No Silent Failure). Prefer `lift()` unless you genuinely only need
    the graph.
    """
    result = lift(g_p, registry, relation_blind)
    if result.unresolved_candidates:
        logger.warning(
            "lift_resolved_only_discards_candidates",
            num_discarded=len(result.unresolved_candidates),
            re_types=sorted({c.re_type.value for c in result.unresolved_candidates}),
        )
    return result.resolved_graph
