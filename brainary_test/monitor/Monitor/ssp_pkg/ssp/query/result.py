"""Module: QueryResult strong-typed output models | Paper section: §5 | Status: wip"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

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
    """A risk event that exists but is not activated by the current action."""

    factor_id: str
    re_type: RiskEventType
    hazard_id: str
    target_id: str
    risk: RiskVector
    reason: str


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
    all_factor_risks: dict[str, RiskVector]
    propagation_result: PropagationResult
    parser_meta: ParserMeta
