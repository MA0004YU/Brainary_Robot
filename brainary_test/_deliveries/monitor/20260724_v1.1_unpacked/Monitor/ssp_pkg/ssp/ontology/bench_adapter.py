"""Module: BenchSample → parser input adapter | Paper section: §4.3 | Status: wip"""

from __future__ import annotations

from ssp.graph.g_percept import PerceptualGraph
from ssp.ontology.bench_schema import BenchSample
from ssp.ontology.relations import L0Relation, L1SuppressionRelation
from ssp.ontology.schema import CandidateAction, Edge, Node, StateSchema


def bench_to_gp(sample: BenchSample) -> tuple[PerceptualGraph, list[CandidateAction]]:
    """Convert a BenchSample to parser input.

    Includes L0 relations and L1 suppression relations.

    Per ADR-013, L1 suppression edges are perception-level evidence of
    mitigation (a lid sitting on a pot, a locked cabinet, an aware guardian
    watching a child) and the lift step needs them to ground a suppressor
    onto a specific factor. They are *not* evaluation labels — those live in
    `ground_truth.suppression_active`.

    L1 propagation relations are still excluded — those are inferred by the
    parser's lift + template instantiation logic.

    Returns
    -------
    (g_p, actions) tuple ready for SceneSafetyParser.query_risk().
    """
    nodes: list[Node] = []
    for entity in sample.entities:
        state = StateSchema()
        for attr_name in ("pose", "stability", "orientation", "energy",
                          "containment", "motion", "attention"):
            if attr_name in entity.attributes:
                setattr(state, attr_name, entity.attributes[attr_name])

        nodes.append(Node(
            id=entity.id,
            type=entity.type,
            subtype=entity.subtype,
            attributes=state,
            vulnerability=entity.vulnerability,
            confidence=entity.confidence,
            source=entity.source,
        ))

    edges: list[Edge] = []
    for rel in sample.relations:
        if not isinstance(rel.type, (L0Relation, L1SuppressionRelation)):
            continue
        edges.append(Edge(
            src=rel.source,
            dst=rel.target,
            relation=rel.type,
            sign="+",
            weight=1.0,
            params=rel.params or {},
            confidence=rel.confidence,
        ))

    g_p = PerceptualGraph(nodes=nodes, edges=edges)
    return g_p, list(sample.candidate_actions)
