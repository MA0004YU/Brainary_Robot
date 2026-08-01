"""Module: QueryResult strong-typed output models | Paper section: §5 | Status: wip"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

from ssp.ontology.risk_events import RiskEventType
from ssp.ontology.schema import RiskVector
from ssp.propagation.fixed_point import PropagationResult


class MitigationApplied(BaseModel):
    """Record of a soft mitigation that discounted (but did not eliminate) risk.

    Per ADR-015: soft mitigation only changes residual_risk; it never moves the
    factor out of `activated`. This documents which suppressor(s) applied the
    discount and the residual multiplier used.
    """

    suppressor_ids: list[str]
    mode: Literal["soft"]
    residual_multiplier: float


class ActivatedRiskEvent(BaseModel):
    """A risk event activated by the current action and NOT hard-suppressed.

    Per ADR-015: residual_risk is the propagation operator's output r* (with the
    suppression gate already applied). If a soft mitigation discounted the risk,
    `mitigation_applied` records it but the event stays activated.
    """

    factor_id: str
    re_type: RiskEventType
    hazard_id: str
    target_id: str
    risk: RiskVector
    residual_risk: RiskVector
    confidence: float
    activation_strength: float
    activation_evidence: list[str]
    constraint_template_ids: list[str]
    mitigation_applied: MitigationApplied | None = None
    baseline_risk: RiskVector | None = None


class InactiveRiskEvent(BaseModel):
    """A risk event that exists but is not activated by the current action.

    Action-conditioned mode only (the "not-applicable" bucket: this action does
    not touch this factor). In scene_intrinsic mode the action-free view uses
    the latent / uncertain buckets instead (ADR-027)."""

    factor_id: str
    re_type: RiskEventType
    hazard_id: str
    target_id: str
    risk: RiskVector
    reason: str


class LatentRiskEvent(BaseModel):
    """A latent hazard: its affordance is structurally present but the intrinsic
    physical trigger is DEFINITELY not met now (ADR-027, scene_intrinsic only).

    Not an active risk (marking it activated would be the D3 over-conservatism
    ADR-025 removed), but positive evidence for pre-emptive constraint
    generation: a plausible action (pick / bump / pour) would activate it, so it
    carries `constraint_template_ids`. Example: a stable ceramic mug of water is
    a latent object_fall / fragile_breakage / spill_damage, not an active one.
    `reason` states which precondition is unmet (e.g. stability=stable)."""

    factor_id: str
    re_type: RiskEventType
    hazard_id: str
    target_id: str
    risk: RiskVector
    reason: str
    constraint_template_ids: list[str]


class UncertainRiskEvent(BaseModel):
    """A risk SSP can neither confirm nor rule out. Two provenance stages
    (ADR-027 + ADR-029), disambiguated by `uncertainty_stage`:

    - "intrinsic_activation" (ADR-027, scene_intrinsic only): the factor WAS
      instantiated and propagated, but its scene-intrinsic gate precondition
      (e.g. stability) is undecidable. `risk` carries the propagated vector with
      `uncertainty.semantic` raised.
    - "instantiation" (ADR-029, both modes): a template's identity/spatial
      conditions matched but an evidence condition (e.g. energy, containment,
      contents_observation) is undecidable, so the factor was NEVER built into
      G_R and NEVER propagated. `risk` is None (NOT an empty vector, which would
      read as "computed to zero"). `unresolved_conditions` lists why.

    Either way SSP raises a typed uncertainty signal instead of fabricating
    activation (over-conservatism) or silently dropping (silent false negative,
    No Silent Failure). `constraint_template_ids` lets L3 act conservatively or
    re-observe."""

    factor_id: str
    re_type: RiskEventType
    hazard_id: str
    target_id: str
    uncertainty_stage: Literal["instantiation", "intrinsic_activation"]
    risk: RiskVector | None
    reason: str
    constraint_template_ids: list[str]
    unresolved_conditions: list[str] = []

    @model_validator(mode="after")
    def _check_stage_risk_consistency(self) -> UncertainRiskEvent:
        """instantiation -> risk is None (never propagated);
        intrinsic_activation -> risk is not None (propagated, u_semantic raised)."""
        if self.uncertainty_stage == "instantiation" and self.risk is not None:
            raise ValueError(
                "instantiation-stage uncertain event must have risk=None "
                "(the factor was never propagated)"
            )
        if self.uncertainty_stage == "intrinsic_activation" and self.risk is None:
            raise ValueError(
                "intrinsic_activation-stage uncertain event must carry a risk vector"
            )
        return self

    @property
    def risk_computed(self) -> bool:
        """True iff the risk vector was actually propagated (intrinsic_activation)."""
        return self.uncertainty_stage == "intrinsic_activation"


class SuppressedRiskEvent(BaseModel):
    """A risk event hard-eliminated by an effective hard mitigation.

    Per ADR-015: only HARD mitigation (isolated_by / neutralized_by, a fact that
    removes the hazard) routes a factor here, confirmed by residual_risk ~ 0.
    Soft mitigation never lands in this bucket.
    """

    factor_id: str
    re_type: RiskEventType
    hazard_id: str
    target_id: str
    suppressor_ids: list[str]
    hard_mitigation_evidence: list[str]
    residual_risk: RiskVector


class ParserMeta(BaseModel):
    """Metadata about the parsing run."""

    parse_mode: Literal["scene_intrinsic", "action_conditioned"]
    num_entities: int
    num_factor_nodes: int
    num_activated: int
    num_inactive: int
    num_suppressed: int
    num_latent: int = 0  # ADR-027: scene_intrinsic latent hazards
    num_uncertain: int = 0  # ADR-027: scene_intrinsic undecidable-precondition hazards
    propagation_iterations: int
    converged: bool
    theory_guaranteed: bool
    warnings: list[str] = []


class QueryResult(BaseModel):
    """Three-way classified query result: concrete identification of activated /
    inactive / suppressed risk events (diagnosis, not prescription -- no accept/reject)."""

    scope: Literal["action_conditioned", "scene_intrinsic"]
    activated_risk_events: list[ActivatedRiskEvent]
    inactive_risk_events: list[InactiveRiskEvent]
    suppressed_events: list[SuppressedRiskEvent]
    # ADR-027: scene_intrinsic-only buckets (empty in action_conditioned mode).
    latent_risk_events: list[LatentRiskEvent] = []
    uncertain_risk_events: list[UncertainRiskEvent] = []
    all_factor_risks: dict[str, RiskVector]
    propagation_result: PropagationResult
    parser_meta: ParserMeta
