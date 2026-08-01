"""Module: SSP-Bench v0.1 schema | Paper section: §4.3 | Status: wip"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ssp.ontology.entities import EntityType, VictimClass, Vulnerability
from ssp.ontology.relations import L0Relation, L1SuppressionRelation
from ssp.ontology.risk_events import RiskEventType
from ssp.ontology.schema import CandidateAction

# --- Robot state ---


class RobotState(BaseModel):
    """Robot state at the time of scene capture. All fields optional for graph-first data."""

    model_config = ConfigDict(extra="forbid")

    position: tuple[float, float, float] | None = None
    holding: list[str] = []
    velocity: float | None = None
    orientation: float | None = None


# --- Scene description ---


class BenchEntity(BaseModel):
    """An entity in a benchmark sample scene."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: EntityType
    subtype: str | None = None
    attributes: dict[str, str | float | bool] = {}
    vulnerability: Vulnerability | None = None
    # NOTE (ADR-016): `intrinsic_hazard` removed from the input contract. It is a
    # risk *conclusion* (lift output), not a perception input — real robot
    # perception yields subtype/attributes, never a pre-labeled hazard verdict.
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    source: str = "manual"


class BenchRelation(BaseModel):
    """A relation in a benchmark sample scene.

    Allowed relation types:
    - L0Relation: observable scene relations, direct parser input (G_P layer).
    - L1SuppressionRelation: used ONLY in mitigated variants to express annotated
      mitigation mechanisms (isolated_by, guarded_by, neutralized_by). These are
      benchmark GT annotations, NOT perception frontend outputs.

    L1PropagationRelation is NEVER allowed here — propagation edges are inferred
    by the parser's lift + template instantiation logic.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    type: L0Relation | L1SuppressionRelation
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    params: dict[str, float] | None = None


# --- Ground truth ---


class SceneRiskEventGT(BaseModel):
    """Ground truth for a scene-intrinsic risk event instance."""

    model_config = ConfigDict(extra="forbid")

    re_type: RiskEventType
    template_id: str
    # ADR-023 (v0.3): victim decoupled from the RE mechanism. victim_class is the
    # vulnerable-target class; victim_entity_id binds to a concrete G_P node (set at
    # S4 once an instantiated scene graph exists; None at the S2 label-only stage).
    victim_class: VictimClass | None = None
    victim_entity_id: str | None = None


class ActionActivationGT(BaseModel):
    """Ground truth for a risk event activated by a specific action."""

    model_config = ConfigDict(extra="forbid")

    re_type: RiskEventType
    template_id: str
    # ADR-023 (v0.3): see SceneRiskEventGT — victim is an orthogonal dimension.
    victim_class: VictimClass | None = None
    victim_entity_id: str | None = None


class SuppressionGT(BaseModel):
    """Ground truth for an active suppression mechanism (structural/binary, no threshold)."""

    model_config = ConfigDict(extra="forbid")

    re_type: RiskEventType
    template_id: str
    suppressor_entity: str
    suppression_type: L1SuppressionRelation


class GroundTruth(BaseModel):
    """Ground truth labels for a benchmark sample.

    Primary GT (SSP parser evaluation):
    - scene_risk_events: scene-intrinsic baseline risk events
    - per_action_activation: which RE templates are activated by which actions (diagnosis)
    - suppression_active: which suppression mechanisms are effective (E2 ablation)

    Auxiliary GT (downstream / optional):
    - per_action_safety: L3 gate labels (prescription), NOT used for SSP correctness
    """

    model_config = ConfigDict(extra="forbid")

    scene_risk_events: list[SceneRiskEventGT]
    per_action_activation: dict[str, list[ActionActivationGT]]
    suppression_active: list[SuppressionGT] = []
    per_action_safety: (
        dict[str, Literal["safe", "unsafe", "conditional"]] | None
    ) = None


# --- Annotation metadata ---


class Annotation(BaseModel):
    """Annotation metadata for provenance and quality tracking."""

    model_config = ConfigDict(extra="allow")

    author: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    dual_annotated: bool = False
    created_at: datetime
    # Optional provenance fields populated during P5c LLM-assisted merging
    # and human review (kept as opaque dicts/lists — see ADR-009/010/011).
    merge_source: dict | None = None
    human_review_log: list[dict] = []


# --- Top-level sample ---


class BenchSample(BaseModel):
    """A complete SSP-Bench sample (one scene, one variant of a pair).

    Paired design: each pair_id has exactly two samples (risky + mitigated)
    that differ only in mitigation relations and corresponding GT.
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    pair_id: str
    variant: Literal["risky", "mitigated"]
    source: Literal["graph_first", "ai2thor", "behavior", "golden", "composite"]
    entities: list[BenchEntity]
    relations: list[BenchRelation]
    robot_state: RobotState | None = None
    candidate_actions: list[CandidateAction]
    ground_truth: GroundTruth
    annotation: Annotation


# --- Paired consistency validation ---


def validate_pair(risky: BenchSample, mitigated: BenchSample) -> list[str]:
    """Validate paired consistency between risky and mitigated samples.

    Rules:
    1. pair_id must match
    2. risky.variant == "risky", mitigated.variant == "mitigated"
    3. risky sample must NOT contain L1SuppressionRelation edges
    4. mitigated sample MUST contain at least one suppression edge
    5. risky entity ids must be a subset of mitigated entity ids
    6. suppression_active GT must reference entities present in mitigated scene
    7. suppression_active GT must align with suppression edges in relations

    Returns
    -------
    List of error strings. Empty list = valid pair.
    """
    errors: list[str] = []

    if risky.pair_id != mitigated.pair_id:
        errors.append(
            f"pair_id mismatch: risky={risky.pair_id}, mitigated={mitigated.pair_id}"
        )

    if risky.variant != "risky":
        errors.append(f"risky sample has variant={risky.variant!r}, expected 'risky'")
    if mitigated.variant != "mitigated":
        errors.append(
            f"mitigated sample has variant={mitigated.variant!r}, expected 'mitigated'"
        )

    risky_suppression = [
        r for r in risky.relations if isinstance(r.type, L1SuppressionRelation)
    ]
    if risky_suppression:
        errors.append(
            f"risky sample contains {len(risky_suppression)} suppression edge(s)"
        )

    mit_suppression = [
        r for r in mitigated.relations if isinstance(r.type, L1SuppressionRelation)
    ]
    if not mit_suppression:
        errors.append("mitigated sample contains no suppression edges")

    risky_ids = {e.id for e in risky.entities}
    mitigated_ids = {e.id for e in mitigated.entities}
    missing = risky_ids - mitigated_ids
    if missing:
        errors.append(
            f"risky entities missing from mitigated: {sorted(missing)}"
        )

    for sup in mitigated.ground_truth.suppression_active:
        if sup.suppressor_entity not in mitigated_ids:
            errors.append(
                f"suppression GT references non-existent entity: {sup.suppressor_entity}"
            )
        sup_edges = [
            r for r in mitigated.relations
            if r.type == sup.suppression_type
            and (r.source == sup.suppressor_entity or r.target == sup.suppressor_entity)
        ]
        if not sup_edges:
            errors.append(
                f"suppression GT ({sup.suppressor_entity}, {sup.suppression_type}) "
                f"has no matching edge in relations"
            )

    return errors
