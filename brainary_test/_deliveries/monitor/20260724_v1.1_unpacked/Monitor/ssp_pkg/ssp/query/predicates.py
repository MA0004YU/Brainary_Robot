"""Module: Typed predicate evaluator for activation rules | Paper section: §5 | Status: wip"""

from __future__ import annotations

from ssp.graph.g_percept import PerceptualGraph
from ssp.ontology.risk_events import ActivationCondition
from ssp.ontology.schema import CandidateAction, FactorNode


def evaluate_predicate(
    condition: ActivationCondition,
    action: CandidateAction,
    factor: FactorNode,
    g_p: PerceptualGraph,
    relation_blind: bool = False,
) -> bool:
    """Evaluate a single typed activation predicate.

    Supported ops:
      - participant_is: action participant role matches factor hazard/target
      - entity_attr_in: entity attribute is in a set of values
      - relation_exists: a relation exists in G_P between specified entities
      - kinematics_lt: action kinematic parameter is below threshold

    relation_blind (E3 / ADR-020): when True, `relation_exists` predicates are
    skipped (vacuously True) so activation no longer consumes L0 relation
    information, mirroring the relation-blind lift. Default False = unchanged.
    """
    op = condition.op
    args = condition.args

    if relation_blind and op == "relation_exists":
        return True

    if op == "participant_is":
        role = str(args.get("role", ""))
        matches = str(args.get("matches", ""))
        participant_id = action.participant(role)
        if participant_id is None:
            return False
        if matches == "hazard":
            return participant_id == factor.hazard_id
        if matches == "target":
            return participant_id == factor.target_id
        return participant_id == matches

    if op == "entity_attr_in":
        entity_ref = str(args.get("entity", ""))
        attr_name = str(args.get("attr", ""))
        values = args.get("values", [])
        if not isinstance(values, list):
            values = [str(values)]

        # Resolve entity_id from reference
        if entity_ref == "hazard":
            entity_id = factor.hazard_id
        elif entity_ref == "target":
            entity_id = factor.target_id
        elif entity_ref == "recipient":
            resolved = action.participant("recipient")
            if resolved is None:
                return False
            entity_id = resolved
        else:
            entity_id = entity_ref

        node = g_p.nodes.get(entity_id)
        if node is None:
            return False

        node_val = getattr(node.attributes, attr_name, None)
        if node_val is None and attr_name == "vulnerability":
            node_val = node.vulnerability
        return node_val is not None and str(node_val) in values

    if op == "relation_exists":
        relations_raw = args.get("relation", [])
        if not isinstance(relations_raw, list):
            relations_raw = [str(relations_raw)]
        src_ref = str(args.get("src", "hazard"))
        dst_ref = str(args.get("dst", "target"))

        src_id = _resolve_entity_ref(src_ref, factor, action)
        dst_id = _resolve_entity_ref(dst_ref, factor, action)
        if src_id is None or dst_id is None:
            return False

        for edge in g_p.edges:
            if str(edge.relation) in relations_raw and edge.src == src_id and edge.dst == dst_id:
                return True
        return False

    if op == "kinematics_lt":
        field = str(args.get("field", ""))
        threshold_raw = args.get("threshold", 0.0)
        threshold = float(str(threshold_raw))
        if action.kinematics is None:
            return False
        kin_val = getattr(action.kinematics, field, None)
        if kin_val is None:
            return False
        return float(kin_val) < threshold

    return False


def _resolve_entity_ref(
    ref: str,
    factor: FactorNode,
    action: CandidateAction,
) -> str | None:
    """Resolve an entity reference to an entity_id."""
    if ref == "hazard":
        return factor.hazard_id
    if ref == "target":
        return factor.target_id
    participant_id = action.participant(ref)
    if participant_id is not None:
        return participant_id
    return ref
