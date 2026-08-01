"""Module: Risk event type definitions (14 RE templates, v0.2) | Paper section: §2.6 | Status: frozen"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ssp.ontology.entities import EntityType
from ssp.ontology.relations import L1PropagationRelation, L1SuppressionRelation


class RiskEventType(StrEnum):
    """Risk event types (closed set, 14 types — v0.2 ADR-021, renamed v0.3 ADR-023).

    Split by event_role on the RiskEventTemplate (not by enum):
    - TerminalRiskEvent (10): energy/substance directly harms a vulnerable target.
    - IntermediateRiskOutcome (4): a link on the propagation chain that can
      re-activate downstream terminals (declared via downstream_events).

    v0.1 -> v0.2 changes: CONTAMINATION split into CHEM + BIO;
    FALL_FROM_ELEVATED_SUPPORT split into HUMAN_FALL (terminal) + OBJECT_FALL
    (intermediate); BURN narrowed (thermal harm) and FIRE added (combustion spread);
    new PINCH / ELEC / CHOKE. Both removed values are gone (clean 14, no aliases).

    v0.2 -> v0.3 changes (ADR-023, victim/mechanism decoupling): the two victim-welded
    names are renamed to mechanism names, with victim carried by the orthogonal
    VictimClass dimension (see entities.VictimClass + template.vulnerable_targets):
    COLLISION_WITH_HUMAN -> COLLISION_IMPACT, HUMAN_FALL -> FALL_INJURY. No RE *type*
    is added or removed (still 14); robot is NOT a victim. Clean rename, no aliases.

    v0.3 coverage extension (ADR-023, B3 human review): two more renames/extensions to
    close ontology gaps found during DESPITE adjudication, still 14 types, no new type:
    - CHEMICAL_EXPOSURE -> HAZARDOUS_MATERIAL_EXPOSURE: broadened from chemicals to also
      cover contact biological hazards (pathogens / blood / reagents that harm by
      contact, not ingestion). Food/utensil/eating-surface contamination still maps to
      BIO_FOOD_CONTAMINATION.
    - BURN_INJURY (no rename): broadened from "thermal burn" to "thermal/radiant burn or
      energy injury" to cover UV / welding-arc / radiant eye injury.
    """

    # --- TerminalRiskEvent (10) ---
    COLLISION_IMPACT = "collision_impact"
    CUT_INJURY = "cut_injury"
    PINCH_CRUSH_INJURY = "pinch_crush_injury"
    FALL_INJURY = "fall_injury"
    BURN_INJURY = "burn_injury"
    ELECTRICAL_SHOCK = "electrical_shock"
    HAZARDOUS_MATERIAL_EXPOSURE = "hazardous_material_exposure"
    BIO_FOOD_CONTAMINATION = "bio_food_contamination"
    CHOKING_INGESTION = "choking_ingestion"
    DANGEROUS_OBJECT_TRANSFER = "dangerous_object_transfer"
    # --- IntermediateRiskOutcome (4) ---
    SPILL_DAMAGE = "spill_damage"
    FRAGILE_BREAKAGE = "fragile_breakage"
    OBJECT_FALL = "object_fall"
    FIRE_COMBUSTION = "fire_combustion"


class EntityFilter(BaseModel):
    """Filter to match entities by type and attributes."""

    entity_type: EntityType
    subtypes: list[str] = []
    required_attributes: dict[str, str] = {}


class PropagationEdgeSpec(BaseModel):
    """Specification for a propagation edge in a RE template."""

    from_role: str
    to_role: str
    relation: L1PropagationRelation
    base_weight: Annotated[float, Field(ge=0.0, le=1.0)]


class MitigationSpec(BaseModel):
    """Specification for a suppression/mitigation relation.

    Per ADR-015:
    - mitigation_mode distinguishes a *fact that removes the hazard* (hard, e.g.
      isolated_by / neutralized_by) from a *discount on a hazard that still
      exists* (soft, e.g. guarded_by). Only hard mitigation can route a factor
      into the `suppressed` bucket.
    - residual_multiplier is the fraction of risk that REMAINS:
      risk_after = risk_before * residual_multiplier. 0.0 = fully eliminated,
      1.0 = no effect. Lower = stronger mitigation. Replaces the old
      `suppression_strength`, which conflated "fraction removed" and "residual
      multiplier" in opposite directions across templates.
    """

    relation: L1SuppressionRelation
    suppressor_filter: EntityFilter
    mitigation_mode: Literal["hard", "soft"]
    residual_multiplier: Annotated[float, Field(ge=0.0, le=1.0)]
    suppression_dims: list["RiskEventType"] | None = None
    # None = suppress only the factor node's own re_type dimension (default behavior).
    # Explicit list = suppress exactly those RE-type dimensions, regardless of factor node's re_type.


# --- Instantiation conditions (used by lift to decide whether to create factor node) ---


class InstantiationCondition(BaseModel):
    """Condition for instantiating a factor node during lift.

    ADR-029: `args` may carry a nested `observation` dict (entity_role / field /
    complete_value) that binds a relation_exists contents check to an explicit
    completeness attribute for three-state (met/unmet/unresolved) evaluation.
    """

    op: Literal["entity_type_in", "relation_exists", "entity_attr_in"]
    args: dict[str, str | list[str] | dict[str, str]]


# --- Activation rules (used by query to decide if action activates a factor node) ---


class ActivationCondition(BaseModel):
    """Typed predicate for action activation evaluation."""

    op: Literal["participant_is", "entity_attr_in", "relation_exists", "kinematics_lt"]
    args: dict[str, str | float | list[str]]


class ActivationRule(BaseModel):
    """Defines how an action type activates this RE."""

    action_type: str
    participant_bindings: dict[str, str] = {}
    conditions: list[ActivationCondition] = []
    activation_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


# --- Legacy ActionActivator (kept for backward compat during transition) ---


class ActionActivator(BaseModel):
    """Defines what action types can activate this RE (legacy, use ActivationRule)."""

    action_type: str
    target_role: str
    conditions: dict[str, str] = {}


class RiskEventTemplate(BaseModel):
    """Full risk event template (§4)."""

    id: str
    name: RiskEventType
    hazard_source: EntityFilter
    vulnerable_targets: list[EntityFilter]

    # ADR-021: two-tier event_role. terminal_harm = energy/substance directly harms a
    # vulnerable target; intermediate_outcome = a chain link that can re-activate the
    # RE ids listed in downstream_events. These are DIAGNOSTIC metadata only — SSP does
    # NOT implement chain propagation here (interface only) and does not aggregate
    # downstream risk into a prescription.
    event_role: Literal["terminal_harm", "intermediate_outcome"] = "terminal_harm"
    downstream_events: list[RiskEventType] = []

    instantiation_conditions: list[InstantiationCondition] = []
    activation_conditions: list[str] = []
    # ADR-025: typed physical preconditions evaluated ONLY in scene_intrinsic mode
    # (e.g. stability in {unstable,tipping} for object_fall). Distinct from
    # instantiation_conditions (gate lift, all modes) and activation_rules (gate
    # action activation): a factor whose intrinsic precondition is unmet still
    # instantiates (so an action can activate it) but is routed to `inactive` in
    # the action-free scene-intrinsic view. Empty = no scene-intrinsic gate.
    intrinsic_activation_conditions: list[InstantiationCondition] = []

    propagation_edges: list[PropagationEdgeSpec]
    mitigation_relations: list[MitigationSpec]

    action_activators: list[ActionActivator] = []
    activation_rules: list[ActivationRule] = []

    constraint_template_ids: list[str]
    severity_default: Annotated[float, Field(ge=0.0, le=1.0)]
    likelihood_prior: Annotated[float, Field(ge=0.0, le=1.0)]
