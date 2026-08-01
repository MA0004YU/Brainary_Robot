"""Module: Entity types (closed set) | Paper section: §2.2 | Status: frozen"""

from enum import StrEnum


class EntityType(StrEnum):
    """Top-level entity types (closed set, v0.1)."""

    PHYSICAL_OBJECT = "physical_object"
    HUMAN = "human"
    ANIMAL = "animal"
    ROBOT_COMPONENT = "robot_component"
    SUBSTANCE = "substance"
    ZONE = "zone"
    SURFACE = "surface"


class PhysicalObjectSubtype(StrEnum):
    """Subtypes of PhysicalObject."""

    SHARP = "sharp_object"
    HOT = "hot_object"
    FRAGILE = "fragile_object"
    CONTAINER = "container"
    TOOL = "tool"
    ELECTRONIC = "electronic"
    FOOD_ITEM = "food_item"
    CHEMICAL_CONTAINER = "chemical_container"
    FURNITURE_SUPPORT = "furniture_support"
    OTHER = "other"
    # --- v0.2 additions (ADR-021) ---
    MOVABLE_BARRIER = "movable_barrier"  # door / drawer / clamp / gripper (PINCH source)
    ELECTRICAL_SOURCE = "electrical_source"  # socket / live appliance / bare wire (ELEC)
    FLAMMABLE = "flammable"  # combustible material (FIRE target)
    SMALL_INGESTIBLE = "small_ingestible"  # small part / button battery / pill (CHOKE source)
    # --- v0.3 additions (ADR-023, coverage extension) ---
    BIOHAZARD = "biohazard"  # pathogen / blood / bio reagent — contact bio hazard (HAZMAT)
    RADIANT_SOURCE = "radiant_source"  # UV lamp / welding arc / radiant emitter (BURN radiant)


class Vulnerability(StrEnum):
    """Human vulnerability levels."""

    ADULT_NORMAL = "adult_normal"
    CHILD = "child"
    ELDERLY = "elderly"
    IMPAIRED = "impaired"
    UNKNOWN = "unknown"


class VictimClass(StrEnum):
    """Vulnerable-target classes (closed set, v0.3 — ADR-023).

    Orthogonal to RiskEventType: a mechanism-named RE (cut/burn/fall_injury/...) can
    harm any of these victim classes. Decoupling victim from the RE label is what lets
    `collision_impact` / `fall_injury` serve human OR animal targets.

    - HUMAN / ANIMAL map to EntityType.HUMAN / ANIMAL.
    - PROPERTY = a physical object / surface damaged as a victim (EntityType
      physical_object / surface).
    - ENVIRONMENT = a zone/surface that becomes unsafe (EntityType zone / surface).

    Deliberately EXCLUDES robot_component (robot self-damage is task-failure, not SSP
    scope — see ADR-023; robot appears only as hazard_source) and any compliance/legal
    notion (that is L3/L5 governance, not SSP physical diagnosis).
    """

    HUMAN = "human"
    ANIMAL = "animal"
    PROPERTY = "property"
    ENVIRONMENT = "environment"


class ZoneType(StrEnum):
    """Zone subtypes."""

    WORKSPACE = "workspace"
    PASSAGE = "passage"
    RESTRICTED = "restricted"
    HAZARD_ZONE = "hazard_zone"
