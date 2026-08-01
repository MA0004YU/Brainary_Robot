"""Module: G_P -> G_R lift (conditional instantiation) | Paper section: §3.1 | Status: wip"""

from __future__ import annotations

import structlog

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
    """Check if a node matches an EntityFilter (type + subtypes + required_attributes)."""
    if node.type != ef.entity_type:
        return False
    if ef.subtypes and (node.subtype is None or node.subtype not in ef.subtypes):
        return False
    for attr_name, attr_value in ef.required_attributes.items():
        node_val = getattr(node.attributes, attr_name, None)
        if node_val is None or str(node_val) != attr_value:
            return False
    return True


def _evaluate_instantiation_condition(
    cond: InstantiationCondition,
    hazard: Node,
    target: Node,
    g_p: PerceptualGraph,
    relation_blind: bool = False,
) -> tuple[bool, str]:
    """Evaluate a single instantiation condition. Returns (passed, evidence_string).

    relation_blind (E3 / ADR-020): when True, `relation_exists` conditions are
    skipped (treated as vacuously passed) so the lift becomes an object-co-
    occurrence baseline that ignores L0 relations. Default False = unchanged.
    """
    op = cond.op
    args = cond.args

    if relation_blind and op == "relation_exists":
        return True, "relation_blind: relation_exists skipped"

    if op == "entity_type_in":
        field = str(args.get("field", ""))
        values = args.get("values", [])
        if not isinstance(values, list):
            values = [str(values)]

        if field == "hazard.subtype":
            passed = hazard.subtype is not None and hazard.subtype in values
            evidence = f"hazard.subtype={hazard.subtype} in {values}"
        elif field == "hazard.type":
            passed = str(hazard.type) in values
            evidence = f"hazard.type={hazard.type} in {values}"
        elif field == "target.subtype":
            passed = target.subtype is not None and target.subtype in values
            evidence = f"target.subtype={target.subtype} in {values}"
        elif field == "target.type":
            passed = str(target.type) in values
            evidence = f"target.type={target.type} in {values}"
        else:
            passed = False
            evidence = f"unknown field: {field}"
        return passed, evidence

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
        passed = node_val is not None and str(node_val) in values
        evidence = f"{entity_role}.{attr_name}={node_val} in {values}"
        return passed, evidence

    if op == "relation_exists":
        relations_raw = args.get("relation", [])
        if not isinstance(relations_raw, list):
            relations_raw = [str(relations_raw)]
        src_role = str(args.get("src", "hazard"))
        dst_role = str(args.get("dst", "target"))

        src_id = hazard.id if src_role == "hazard" else target.id
        dst_id = target.id if dst_role == "target" else hazard.id

        # Special case: if src and dst resolve to same entity, check if entity
        # participates in any edge with the specified relation (either direction)
        same_entity = src_id == dst_id
        found_relation: str | None = None

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

        passed = found_relation is not None
        evidence = f"relation({found_relation or 'none'}) between {src_id} and {dst_id}"
        return passed, evidence

    return False, f"unknown op: {op}"


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


def lift(
    g_p: PerceptualGraph,
    registry: TemplateRegistry,
    relation_blind: bool = False,
) -> ReasoningGraph:
    """Lift G_P to G_R by conditionally instantiating factor nodes from RE templates.

    For each template, evaluates instantiation_conditions against candidate
    (hazard, target) pairs. Only creates factor nodes where ALL conditions pass.
    Also creates propagation edges (hazard -> factor, factor -> target) and
    suppression edges (suppressor -> factor).

    Args:
        g_p: Perceptual scene graph with L0 edges.
        registry: Loaded template registry.
        relation_blind: E3 ablation (ADR-020). When True, `relation_exists`
            instantiation conditions are skipped so lift degrades to object-
            co-occurrence (ignores L0 relations). Default False = unchanged.

    Returns:
        ReasoningGraph with entity nodes, factor nodes, and L1 edges.
    """
    nodes = list(g_p.nodes.values())
    edges: list[Edge] = []
    factor_nodes: list[FactorNode] = []
    factor_counter = 0

    for template in registry.all_templates():
        hazard_candidates = [n for n in nodes if _matches_entity_filter(n, template.hazard_source)]
        target_candidates: list[Node] = []
        for tf in template.vulnerable_targets:
            for n in nodes:
                if _matches_entity_filter(n, tf) and n not in target_candidates:
                    target_candidates.append(n)

        for hazard in hazard_candidates:
            for target_node in target_candidates:
                all_passed = True
                evidence_list: list[str] = []

                for cond in template.instantiation_conditions:
                    passed, evidence = _evaluate_instantiation_condition(
                        cond, hazard, target_node, g_p, relation_blind
                    )
                    if passed:
                        evidence_list.append(evidence)
                    else:
                        all_passed = False
                        break

                if not all_passed:
                    continue

                # Create factor node
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
        num_edges=len(edges),
        templates_evaluated=len(registry.all_templates()),
    )

    return ReasoningGraph(
        nodes=nodes,
        edges=edges,
        factor_nodes=factor_nodes,
    )
