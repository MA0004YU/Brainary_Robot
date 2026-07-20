"""Module: Action-Conditioned Risk Graph (G_R^a) | Paper section: §2.1 L2 | Status: wip"""

from __future__ import annotations

import json
from typing import Any

from ssp.graph.g_percept import OntologyViolationError
from ssp.graph.g_reason import ReasoningGraph
from ssp.ontology.schema import Edge, Node


class ActivatedGraph:
    """G_R^(a): subgraph of G_R activated by a candidate action."""

    def __init__(
        self,
        nodes: list[Node],
        edges: list[Edge],
        action_id: str | None = None,
    ) -> None:
        self._nodes: dict[str, Node] = {n.id: n for n in nodes}
        self._edges: list[Edge] = list(edges)
        self.action_id = action_id

    @property
    def nodes(self) -> dict[str, Node]:
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        return self._edges

    def validate(self) -> None:
        """Validate node references."""
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

    @classmethod
    def from_reasoning_graph(
        cls,
        g_r: ReasoningGraph,
        activated_node_ids: set[str],
        action_id: str,
    ) -> ActivatedGraph:
        """Extract activated subgraph from G_R."""
        nodes = [n for n in g_r.nodes.values() if n.id in activated_node_ids]
        edges = [
            e for e in g_r.edges if e.src in activated_node_ids and e.dst in activated_node_ids
        ]
        return cls(nodes=nodes, edges=edges, action_id=action_id)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "layer": "G_R_a",
            "action_id": self.action_id,
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self._edges],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> ActivatedGraph:
        """Deserialize from JSON string."""
        parsed: dict[str, Any] = json.loads(data)
        nodes = [Node.model_validate(n) for n in parsed["nodes"]]
        edges = [Edge.model_validate(e) for e in parsed["edges"]]
        return cls(
            nodes=nodes,
            edges=edges,
            action_id=parsed.get("action_id"),
        )
