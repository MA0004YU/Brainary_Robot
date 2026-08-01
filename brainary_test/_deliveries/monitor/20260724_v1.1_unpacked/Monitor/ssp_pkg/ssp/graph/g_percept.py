"""Module: Perceptual Scene Graph (G_P) | Paper section: §2.1 L0 | Status: wip"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

from ssp.ontology.relations import L0Relation, L1SuppressionRelation
from ssp.ontology.schema import Edge, Node


class OntologyViolationError(Exception):
    """Raised when graph content violates ontology constraints."""


class PerceptualGraph:
    """G_P: Perceptual Scene Graph.

    Allowed edges:
      - L0Relation: physical scene relations (near, supports, reachable, ...)
      - L1SuppressionRelation: observable mitigation evidence (a lid sealing
        a pot, a locked cabinet, an aware guardian). Per ADR-013 these are
        perception-level facts, not GT, and lift uses them to ground a
        suppressor onto a specific factor.

    L1PropagationRelation is NOT allowed — propagation edges are inferred
    by lift + template instantiation.
    """

    def __init__(self, nodes: list[Node], edges: list[Edge]) -> None:
        self._nodes: dict[str, Node] = {n.id: n for n in nodes}
        self._edges: list[Edge] = list(edges)

    @property
    def nodes(self) -> dict[str, Node]:
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        return self._edges

    def validate(self) -> None:
        """Validate that all edges use L0 / L1 suppression relations and
        reference valid nodes."""
        node_ids = set(self._nodes.keys())
        for edge in self._edges:
            if edge.src not in node_ids:
                raise OntologyViolationError(
                    f"Edge src '{edge.src}' not in node set",
                )
            if edge.dst not in node_ids:
                raise OntologyViolationError(
                    f"Edge dst '{edge.dst}' not in node set",
                )
            if not isinstance(edge.relation, (L0Relation, L1SuppressionRelation)):
                raise OntologyViolationError(
                    f"G_P only allows L0 or L1 suppression relations, "
                    f"got '{edge.relation}'",
                )
            if edge.sign != "+":
                raise OntologyViolationError(
                    f"G_P edges must have sign='+', got '{edge.sign}'",
                )

    def to_networkx(self) -> nx.DiGraph[str, dict[str, object]]:
        """Convert to networkx DiGraph for computation."""
        g: nx.DiGraph[str, dict[str, object]] = nx.DiGraph()
        for node in self._nodes.values():
            g.add_node(node.id, **node.model_dump())
        for edge in self._edges:
            g.add_edge(edge.src, edge.dst, **edge.model_dump(exclude={"src", "dst"}))
        return g

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "layer": "G_P",
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self._edges],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> PerceptualGraph:
        """Deserialize from JSON string."""
        parsed: dict[str, Any] = json.loads(data)
        nodes = [Node.model_validate(n) for n in parsed["nodes"]]
        edges = [Edge.model_validate(e) for e in parsed["edges"]]
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerceptualGraph:
        """Construct from a dictionary."""
        nodes = [Node.model_validate(n) for n in data["nodes"]]
        edges = [Edge.model_validate(e) for e in data["edges"]]
        return cls(nodes=nodes, edges=edges)
