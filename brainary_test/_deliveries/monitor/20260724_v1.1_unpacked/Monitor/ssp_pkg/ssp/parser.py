"""Module: Top-level SSP pipeline | Paper section: §5 | Status: wip"""

from __future__ import annotations

from pathlib import Path

import structlog

from ssp.graph.g_percept import PerceptualGraph
from ssp.graph.g_reason import ReasoningGraph
from ssp.graph.lift import (
    LiftResult,
    UncertainFactorCandidate,
    evaluate_intrinsic_conditions,
    lift,
)
from ssp.ontology.relations import L1SuppressionRelation
from ssp.ontology.schema import CandidateAction, Edge, FactorNode, RiskVector
from ssp.ontology.template_registry import TemplateRegistry
from ssp.propagation.fixed_point import PropagationResult
from ssp.propagation.operator import propagate
from ssp.propagation.params import PropagationParams
from ssp.query.activate import ActivationResult, _try_activate_factor, activate_by_action
from ssp.query.result import (
    ActivatedRiskEvent,
    InactiveRiskEvent,
    LatentRiskEvent,
    MitigationApplied,
    ParserMeta,
    QueryResult,
    SuppressedRiskEvent,
    UncertainRiskEvent,
)

# ADR-015: SUPPRESSION_THRESHOLD removed. Classification routes a factor into
# the `suppressed` bucket iff an effective HARD mitigation edge is present
# (mode-as-gate); residual risk r* is reported but does not decide membership.
# epsilon_risk is a sanity guard (warn when a hard-mitigated factor still has
# materially nonzero residual), NOT a tunable threshold.

logger = structlog.get_logger()


class SceneSafetyParser:
    """Top-level facade: G_P -> lift -> G_R -> propagate -> activate -> QueryResult.

    Concrete risk-identification interface: definitively identifies which risk
    events are activated by an action, their severity/likelihood, and suppression
    status. Diagnosis, not prescription -- it never makes accept/reject decisions
    (that boundary is the downstream L3 gate's job), but its diagnostic output is a
    concrete identification, not a vague description.
    """

    def __init__(
        self,
        templates_dir: Path,
        propagation_params: PropagationParams | None = None,
    ) -> None:
        self._registry = TemplateRegistry(templates_dir)
        self._registry.load_all()
        self._params = propagation_params or PropagationParams()

    @property
    def registry(self) -> TemplateRegistry:
        return self._registry

    # --- Ablation seams (P8 / ADR-020) -------------------------------------
    # Default implementations reproduce the original inline pipeline byte-for-
    # byte; the full parser's behaviour is unchanged. Ablation subclasses
    # override one hook each (E2: _post_lift, E3: _lift, E4: the action branch).
    def _lift(self, g_p: PerceptualGraph) -> LiftResult:
        """G_P -> LiftResult (resolved_graph + unresolved candidates). Override to
        ablate relation-aware instantiation (E3)."""
        return lift(g_p, self._registry)

    def _post_lift(self, g_r: ReasoningGraph) -> ReasoningGraph:
        """Hook between lift and propagate. Override to strip edges (E2)."""
        return g_r

    def _propagate(self, g_r: ReasoningGraph) -> PropagationResult:
        """Run the bounded propagation operator over G_R."""
        return propagate(g_r, self._params)

    def _activate(
        self, g_r: ReasoningGraph, g_p: PerceptualGraph, action: CandidateAction,
    ) -> ActivationResult:
        """Action activation over factor nodes. Override to ablate relation-aware
        activation (E3, relation_blind) without duplicating the build method."""
        return activate_by_action(g_r, g_p, action, self._registry)

    def query_risk(
        self,
        g_p: PerceptualGraph,
        action: CandidateAction | None = None,
    ) -> QueryResult:
        """Run the full SSP pipeline.

        Args:
            g_p: Perceptual scene graph (L0 edges).
            action: Optional candidate action. If None, returns scene-intrinsic mode.

        Returns:
            QueryResult with three-way classification of risk events.
        """
        g_p.validate()
        lift_result = self._lift(g_p)
        candidates = lift_result.unresolved_candidates
        g_r = self._post_lift(lift_result.resolved_graph)
        prop_result = self._propagate(g_r)

        if action is None:
            return self._build_scene_intrinsic_result(g_r, g_p, prop_result, candidates)
        return self._build_action_conditioned_result(
            g_r, g_p, prop_result, action, candidates
        )

    def _candidate_to_uncertain(
        self, cand: UncertainFactorCandidate,
    ) -> UncertainRiskEvent:
        """Convert an instantiation-stage candidate (ADR-029) to an uncertain
        event. risk=None: the candidate was never built into G_R nor propagated,
        so there is no r* -- None, not an empty vector (which would read as zero)."""
        return UncertainRiskEvent(
            factor_id=cand.factor_id,
            re_type=cand.re_type,
            hazard_id=cand.hazard_id,
            target_id=cand.target_id,
            uncertainty_stage="instantiation",
            risk=None,
            reason="; ".join(cand.unresolved_conditions),
            constraint_template_ids=cand.constraint_template_ids,
            unresolved_conditions=cand.unresolved_conditions,
        )

    def _candidate_matches_action(
        self, cand: UncertainFactorCandidate, action: CandidateAction,
        g_p: PerceptualGraph,
    ) -> bool:
        """True iff `action` would activate this candidate's template (ADR-029).

        A candidate is not in G_R, so we probe with a transient FactorNode using
        the SAME activation logic as resolved factors. Action-irrelevant
        candidates stay not-applicable -- an action must not surface every unknown
        risk in the scene, only the ones it would touch."""
        probe = FactorNode(
            id=cand.factor_id,
            re_type=cand.re_type,
            template_id=cand.template_id,
            hazard_id=cand.hazard_id,
            target_id=cand.target_id,
        )
        template = self._registry.get(cand.re_type)
        return _try_activate_factor(probe, action, template, g_p) is not None

    def _build_scene_intrinsic_result(
        self,
        g_r: ReasoningGraph,
        g_p: PerceptualGraph,
        prop_result: PropagationResult,
        candidates: list[UncertainFactorCandidate],
    ) -> QueryResult:
        """Build result for scene-intrinsic mode (no action specified).

        ADR-027 "latent-hazard map" B semantics: the action-free view is a map of
        every structurally-instantiable hazard, routed by its intrinsic precondition
        status (the FIRST gate, per ADR-025). Four buckets, no `inactive`:
          - met        -> activated / suppressed (classify_factor; an active risk)
          - unmet      -> latent   (trigger DEFINITELY absent now, e.g. stability=
                          stable; evidence for pre-emptive constraints)
          - unresolved -> uncertain (gating state unknown/missing; raise the typed
                          u_semantic signal instead of fabricating or dropping)
        Templates with no intrinsic gate return "met" (unchanged from ADR-025).
        This preserves ADR-015's invariant that `suppressed` presupposes an active
        risk -- a non-active precondition never becomes "suppressed".
        """
        activated: list[ActivatedRiskEvent] = []
        suppressed: list[SuppressedRiskEvent] = []
        latent: list[LatentRiskEvent] = []
        uncertain: list[UncertainRiskEvent] = []

        for fn_id, factor in g_r.factor_nodes.items():
            risk = prop_result.final_risks.get(fn_id, RiskVector())
            template = self._registry.get(factor.re_type)
            status, reason = evaluate_intrinsic_conditions(template, factor, g_p)

            if status == "unmet":
                latent.append(LatentRiskEvent(
                    factor_id=fn_id,
                    re_type=factor.re_type,
                    hazard_id=factor.hazard_id,
                    target_id=factor.target_id,
                    risk=risk,
                    reason=reason,
                    constraint_template_ids=factor.constraint_template_ids,
                ))
                continue
            if status == "unresolved":
                # Raise the typed semantic-uncertainty signal (#38) without
                # mutating the shared propagation vector.
                uncertain_risk = risk.model_copy(deep=True)
                uncertain_risk.uncertainty.semantic = max(
                    uncertain_risk.uncertainty.semantic,
                    self._params.u_semantic_unresolved,
                )
                uncertain.append(UncertainRiskEvent(
                    factor_id=fn_id,
                    re_type=factor.re_type,
                    hazard_id=factor.hazard_id,
                    target_id=factor.target_id,
                    uncertainty_stage="intrinsic_activation",
                    risk=uncertain_risk,
                    reason=reason,
                    constraint_template_ids=factor.constraint_template_ids,
                ))
                continue

            event = self._classify_factor(
                g_r, fn_id, factor, risk,
                activation_strength=1.0,
                activation_evidence=["scene_intrinsic"],
            )
            if isinstance(event, SuppressedRiskEvent):
                suppressed.append(event)
            else:
                activated.append(event)

        # ADR-029: instantiation-stage candidates. In scene_intrinsic (the
        # latent-hazard map) ALL candidates are surfaced -- an undecidable
        # evidence precondition is a scene fact independent of any action.
        for cand in candidates:
            uncertain.append(self._candidate_to_uncertain(cand))

        return QueryResult(
            scope="scene_intrinsic",
            activated_risk_events=activated,
            inactive_risk_events=[],
            suppressed_events=suppressed,
            latent_risk_events=latent,
            uncertain_risk_events=uncertain,
            all_factor_risks=dict(prop_result.final_risks),
            propagation_result=prop_result,
            parser_meta=ParserMeta(
                parse_mode="scene_intrinsic",
                num_entities=len(g_r.nodes),
                num_factor_nodes=len(g_r.factor_nodes),
                num_activated=len(activated),
                num_inactive=0,
                num_suppressed=len(suppressed),
                num_latent=len(latent),
                num_uncertain=len(uncertain),
                propagation_iterations=prop_result.iterations,
                converged=prop_result.converged,
                theory_guaranteed=prop_result.theory_guaranteed,
                warnings=prop_result.warnings,
            ),
        )

    def _build_action_conditioned_result(
        self,
        g_r: ReasoningGraph,
        g_p: PerceptualGraph,
        prop_result: PropagationResult,
        action: CandidateAction,
        candidates: list[UncertainFactorCandidate],
    ) -> QueryResult:
        """Build result for action-conditioned mode."""
        activation_result = self._activate(g_r, g_p, action)

        activated: list[ActivatedRiskEvent] = []
        inactive: list[InactiveRiskEvent] = []
        suppressed: list[SuppressedRiskEvent] = []
        uncertain: list[UncertainRiskEvent] = []

        for fa in activation_result.activated:
            factor = g_r.factor_nodes[fa.factor_id]
            risk = prop_result.final_risks.get(fa.factor_id, RiskVector())
            event = self._classify_factor(
                g_r, fa.factor_id, factor, risk,
                activation_strength=fa.activation_strength,
                activation_evidence=fa.evidence,
            )
            if isinstance(event, SuppressedRiskEvent):
                suppressed.append(event)
            else:
                activated.append(event)

        for fn_id in activation_result.inactive_factor_ids:
            factor = g_r.factor_nodes[fn_id]
            risk = prop_result.final_risks.get(fn_id, RiskVector())
            inactive.append(InactiveRiskEvent(
                factor_id=fn_id,
                re_type=factor.re_type,
                hazard_id=factor.hazard_id,
                target_id=factor.target_id,
                risk=risk,
                reason=f"not activated by action {action.id} (type={action.type})",
            ))

        # ADR-029: only candidates whose template's activation rule matches THIS
        # action become uncertain -- action-irrelevant candidates stay
        # not-applicable (an action must not surface every unknown scene risk).
        for cand in candidates:
            if self._candidate_matches_action(cand, action, g_p):
                uncertain.append(self._candidate_to_uncertain(cand))

        return QueryResult(
            scope="action_conditioned",
            activated_risk_events=activated,
            inactive_risk_events=inactive,
            suppressed_events=suppressed,
            uncertain_risk_events=uncertain,
            all_factor_risks=dict(prop_result.final_risks),
            propagation_result=prop_result,
            parser_meta=ParserMeta(
                parse_mode="action_conditioned",
                num_entities=len(g_r.nodes),
                num_factor_nodes=len(g_r.factor_nodes),
                num_activated=len(activated),
                num_inactive=len(inactive),
                num_suppressed=len(suppressed),
                num_uncertain=len(uncertain),
                propagation_iterations=prop_result.iterations,
                converged=prop_result.converged,
                theory_guaranteed=prop_result.theory_guaranteed,
                warnings=prop_result.warnings,
            ),
        )

    def _hard_mitigation_edges(self, g_r: ReasoningGraph, fn_id: str) -> list[Edge]:
        """Incoming hard-mitigation suppression edges for a factor node (ADR-015)."""
        return [
            edge
            for edge in g_r.edges
            if edge.dst == fn_id
            and isinstance(edge.relation, L1SuppressionRelation)
            and edge.mitigation_mode == "hard"
        ]

    def _soft_mitigation_edges(self, g_r: ReasoningGraph, fn_id: str) -> list[Edge]:
        """Incoming soft-mitigation suppression edges for a factor node (ADR-015)."""
        return [
            edge
            for edge in g_r.edges
            if edge.dst == fn_id
            and isinstance(edge.relation, L1SuppressionRelation)
            and edge.mitigation_mode == "soft"
        ]

    @staticmethod
    def _peak_severity(risk: RiskVector) -> float:
        """Max severity across all risk dimensions (0.0 if empty)."""
        return max(risk.severity.values(), default=0.0)

    def _classify_factor(
        self,
        g_r: ReasoningGraph,
        fn_id: str,
        factor: FactorNode,
        risk: RiskVector,
        activation_strength: float,
        activation_evidence: list[str],
    ) -> ActivatedRiskEvent | SuppressedRiskEvent:
        """Route a factor into activated/suppressed using mitigation mode + r*.

        Per ADR-015 (four-layer separation, mode-as-gate / decision M):
          suppressed = an effective HARD mitigation edge is present (the scene
                       asserts a fact that removes the hazard). Membership is
                       decided by mitigation_mode, NOT by a residual threshold.
          activated  = otherwise; residual_risk carries r* (already discounted
                       by any soft mitigation), and mitigation_applied records
                       a soft discount when present.
        Soft mitigation never moves a factor into suppressed.

        epsilon_risk is a SANITY guard, not a gate: if a factor claims hard
        mitigation yet its residual severity is materially above zero, a warning
        is logged so the offending template surfaces (e.g. FRAG neutralized_by
        surface with residual 0.6). It does not change membership.
        """
        hard_edges = self._hard_mitigation_edges(g_r, fn_id)
        if hard_edges:
            residual_peak = self._peak_severity(risk)
            if residual_peak >= self._params.epsilon_risk:
                logger.warning(
                    "hard_mitigation_residual_nonzero",
                    factor_id=fn_id,
                    re_type=factor.re_type.value,
                    residual_peak=residual_peak,
                    suppressors=[e.src for e in hard_edges],
                )
            return SuppressedRiskEvent(
                factor_id=fn_id,
                re_type=factor.re_type,
                hazard_id=factor.hazard_id,
                target_id=factor.target_id,
                suppressor_ids=[e.src for e in hard_edges],
                hard_mitigation_evidence=[
                    f"{e.relation.value}({e.src}->{fn_id}) residual_multiplier={e.weight}"
                    for e in hard_edges
                ],
                residual_risk=risk,
            )

        soft_edges = self._soft_mitigation_edges(g_r, fn_id)
        mitigation_applied: MitigationApplied | None = None
        if soft_edges:
            mitigation_applied = MitigationApplied(
                suppressor_ids=[e.src for e in soft_edges],
                mode="soft",
                residual_multiplier=min(e.weight for e in soft_edges),
            )

        return ActivatedRiskEvent(
            factor_id=fn_id,
            re_type=factor.re_type,
            hazard_id=factor.hazard_id,
            target_id=factor.target_id,
            risk=risk,
            residual_risk=risk,
            confidence=factor.confidence,
            activation_strength=activation_strength,
            activation_evidence=activation_evidence,
            constraint_template_ids=factor.constraint_template_ids,
            mitigation_applied=mitigation_applied,
        )

