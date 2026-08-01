"""Module: Relation types (L0 observed + L1 inferred) | Paper section: §2.4-2.5 | Status: frozen"""

from enum import StrEnum


class L0Relation(StrEnum):
    """L0 observation relations (directly perceivable, 9 types)."""

    NEAR = "near"
    CONTACT = "contact"
    SUPPORTS = "supports"
    CONTAINS = "contains"
    HOLDS = "holds"
    INSIDE = "inside"
    ORIENTED_TOWARD = "oriented_toward"
    MOVING_TOWARD = "moving_toward"
    REACHABLE = "reachable"


class L1PropagationRelation(StrEnum):
    """L1 propagation relations (inferred, positive edges)."""

    COULD_CONTACT = "could_contact"
    COULD_FALL_FROM = "could_fall_from"
    COULD_SPILL_TO = "could_spill_to"
    COULD_IGNITE = "could_ignite"
    COULD_CONTAMINATE = "could_contaminate"
    COULD_TRANSFER = "could_transfer"
    # --- v0.2 additions (ADR-021) ---
    COULD_CRUSH = "could_crush"  # pinch/crush propagation (PINCH)
    COULD_SHOCK = "could_shock"  # electrical-shock propagation (ELEC)
    COULD_CHOKE = "could_choke"  # ingestion/choking propagation (CHOKE)
    # FALLHUM/FALLOBJ reuse could_fall_from; FIRE reuses could_ignite;
    # BIO reuses could_contaminate (ADR-021 relation-reuse decision).


class L1SuppressionRelation(StrEnum):
    """L1 suppression relations (inferred, negative edges)."""

    GUARDED_BY = "guarded_by"
    ISOLATED_BY = "isolated_by"
    NEUTRALIZED_BY = "neutralized_by"


# Union type for all L1 relations
L1Relation = L1PropagationRelation | L1SuppressionRelation

# All relation types
AnyRelation = L0Relation | L1PropagationRelation | L1SuppressionRelation


PROPAGATION_RELATIONS: frozenset[L1PropagationRelation] = frozenset(
    L1PropagationRelation,
)
SUPPRESSION_RELATIONS: frozenset[L1SuppressionRelation] = frozenset(
    L1SuppressionRelation,
)
