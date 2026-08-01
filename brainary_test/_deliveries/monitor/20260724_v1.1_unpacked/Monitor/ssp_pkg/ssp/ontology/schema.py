"""Module: Pydantic schema definitions for ontology types | Paper section: §2 | Status: frozen"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from ssp.ontology.entities import EntityType, Vulnerability
from ssp.ontology.relations import (
    L0Relation,
    L1PropagationRelation,
    L1SuppressionRelation,
)
from ssp.ontology.risk_events import RiskEventType

# --- State attribute enums (§2.3) ---


class PoseState(StrEnum):
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    BENDING = "bending"
    REACHING = "reaching"
    FALLING = "falling"
    UNKNOWN = "unknown"


class StabilityState(StrEnum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    TIPPING = "tipping"
    UNKNOWN = "unknown"


class OrientationState(StrEnum):
    UPRIGHT = "upright"
    TILTED = "tilted"
    INVERTED = "inverted"
    UNKNOWN = "unknown"


class EnergyState(StrEnum):
    HOT = "hot"
    COLD = "cold"
    ELECTRIFIED = "electrified"
    RADIANT = "radiant"  # v0.3 (ADR-023): UV / welding-arc / radiant emitter (BURN radiant)
    NONE = "none"
    UNKNOWN = "unknown"


class ContainmentState(StrEnum):
    OPEN = "open"
    SEALED = "sealed"
    LOCKED = "locked"
    UNKNOWN = "unknown"


class MotionState(StrEnum):
    STATIC = "static"
    MOVING_SLOW = "moving_slow"
    MOVING_FAST = "moving_fast"
    UNKNOWN = "unknown"


class AttentionState(StrEnum):
    AWARE = "aware"
    UNAWARE = "unaware"
    DISTRACTED = "distracted"
    UNKNOWN = "unknown"


class ContentsObservation(StrEnum):
    """Completeness of container-contents observation (ADR-029, v0.5).

    Distinguishes "known empty" from "unknown contents" for a container -- the
    absence of a `contains` edge is otherwise ambiguous. Only `complete` licenses
    the predicate-local closed-world inference "no contains edge => known empty".

    - COMPLETE: perception has inspected and fully represented this container's
      contents (so no contains edge => genuinely empty => spill precondition unmet).
    - UNKNOWN: contents not (fully) observed (no contains edge => unresolved =>
      spill becomes an uncertain candidate, not a silent drop).

    Default is None (absent), treated as UNKNOWN -- NEVER defaulted to COMPLETE,
    which would just rename the old empty/unknown ambiguity.
    """

    COMPLETE = "complete"
    UNKNOWN = "unknown"


class UncertaintyTag(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    PREDICTED = "predicted"
    ASSUMED = "assumed"


# --- Composite models ---


class StateSchema(BaseModel):
    """State attributes for a node (§2.3). All optional."""

    pose: PoseState | None = None
    stability: StabilityState | None = None
    orientation: OrientationState | None = None
    energy: EnergyState | None = None
    containment: ContainmentState | None = None
    motion: MotionState | None = None
    attention: AttentionState | None = None
    # ADR-029: container-contents observation completeness (spill "known empty"
    # vs "unknown"). Default None == UNKNOWN; never defaulted to COMPLETE.
    contents_observation: ContentsObservation | None = None


class UncertaintyVector(BaseModel):
    """Typed uncertainty (§3.4): four independent dimensions."""

    percept: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    relation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    semantic: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    ood: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class RiskVector(BaseModel):
    """Vectorized risk representation (§2.7)."""

    severity: dict[RiskEventType, float] = {}
    likelihood: dict[RiskEventType, float] = {}
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    uncertainty: UncertaintyVector = UncertaintyVector()


# --- Core graph elements ---


class Node(BaseModel):
    """A typed node in any SSP graph layer.

    ADR-026 (v0.4): a physical object carries a SET of hazard-relevant subtypes
    (`subtypes`), not a single one. `PhysicalObjectSubtype` was always a
    hazard-affordance taxonomy (sharp / fragile / container / electronic / ...),
    and real objects bear several at once -- a ceramic mug of water is BOTH
    `fragile_object` and `container`, so it must be able to instantiate both
    `fragile_breakage` and `spill_damage`. Cardinality was the only bug; no new
    subtype or RE type is introduced.

    Backward compat: a legacy single `subtype` (str) in JSON / kwargs is coerced
    into `subtypes=[subtype]` by a pre-validator, so the 2678 frozen v0.1 files
    and existing tests load unchanged. The read-only `.subtype` property returns
    the first subtype (deprecated; template matching uses the full `subtypes` set).
    """

    id: str
    type: EntityType
    subtypes: list[str] = []
    attributes: StateSchema = StateSchema()
    vulnerability: Vulnerability | None = None
    # NOTE (ADR-016): `intrinsic_hazard` removed. Risk-event existence is derived
    # by lift from type+attributes, never read off a pre-labeled node field.
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    uncertainty: UncertaintyTag = UncertaintyTag.OBSERVED
    source: str = "manual"

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_subtype(cls, data: object) -> object:
        """Accept the legacy single `subtype` field (ADR-026 back-compat).

        If `subtype` is present and `subtypes` is not, normalize:
        None -> [], "x" -> ["x"]. If both are given, `subtypes` wins and the
        legacy key is dropped. No-op for inputs that are not dicts (e.g. an
        already-built Node passed through model_validate).
        """
        if not isinstance(data, dict) or "subtype" not in data:
            return data
        legacy = data.pop("subtype")
        if not data.get("subtypes"):
            data["subtypes"] = [] if legacy is None else [legacy]
        return data

    @property
    def subtype(self) -> str | None:
        """Deprecated (ADR-026): first subtype, for legacy read sites only.

        Template matching must use the full `subtypes` set, not this property --
        a multi-subtype node would otherwise silently match on its first subtype
        only. Kept to let un-migrated read sites keep working during transition.
        """
        return self.subtypes[0] if self.subtypes else None


class Edge(BaseModel):
    """A typed directed edge in any SSP graph layer."""

    src: str
    dst: str
    relation: L0Relation | L1PropagationRelation | L1SuppressionRelation
    sign: Literal["+", "-"]
    weight: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    params: dict[str, float] = {}
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    uncertainty: UncertaintyTag = UncertaintyTag.OBSERVED
    suppression_dims: list[RiskEventType] | None = None
    # Only meaningful when sign="-". None = suppress factor node's own re_type dim.
    # Explicit list = suppress exactly those RE-type dimensions.
    mitigation_mode: Literal["hard", "soft"] | None = None
    # Only meaningful when sign="-" (ADR-015). For suppression edges, `weight`
    # carries the residual_multiplier (fraction of risk that REMAINS):
    # risk_after = risk_before * weight. mitigation_mode distinguishes a fact
    # that removes the hazard (hard) from a discount (soft); only hard can route
    # a factor into the `suppressed` bucket.


class FactorNode(BaseModel):
    """A risk event instance (factor node) in G_R. Reifies a template-defined hyperedge."""

    id: str
    re_type: RiskEventType
    template_id: str
    hazard_id: str
    target_id: str
    severity: list[float] = []
    likelihood: list[float] = []
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    instantiation_evidence: list[str] = []
    constraint_template_ids: list[str] = []


# --- Action schema (§5) ---

ACTION_TYPES = Literal[
    "pick", "place", "handover", "pour", "push", "pull",
    "move", "navigate", "pass_near", "bump", "open", "close",
]

PARTICIPANT_ROLES = Literal[
    "target", "recipient", "source", "destination",
    "path_object", "tool", "substance", "nearby_human",
]


class ActionParticipant(BaseModel):
    """A participant in a candidate action with a typed role."""

    role: PARTICIPANT_ROLES
    entity_id: str


class ActionKinematics(BaseModel):
    """Kinematic parameters of a candidate action."""

    speed: float | None = None
    force: float | None = None
    distance: float | None = None
    clearance: float | None = None
    trajectory_entities: list[str] = []


class CandidateAction(BaseModel):
    """A candidate action to be evaluated for risk activation."""

    id: str
    type: ACTION_TYPES
    participants: list[ActionParticipant]
    kinematics: ActionKinematics | None = None
    expected_effects: list[str] = []
    params: dict[str, str | float | bool] = {}

    def participant(self, role: str) -> str | None:
        """Get entity_id for a given role, or None if not present."""
        for p in self.participants:
            if p.role == role:
                return p.entity_id
        return None
