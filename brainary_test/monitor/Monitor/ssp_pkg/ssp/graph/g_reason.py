"""Module: Risk Propagation Graph (G_R) | Paper section: §2.1 L1 | Status: wip"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

from ssp.graph.g_percept import OntologyViolationError
from ssp.ontology.relations import (
    L1PropagationRelation,
    L1SuppressionRelation,
)
from ssp.ontology.schema import Edge, FactorNode, Node, RiskVector


class ReasoningGraph:
    """G_R: Risk Propagation Graph with entity nodes and factor nodes.

    Factor nodes represent instantiated risk events (template-constrained).
    Propagation occurs along template-defined paths, not arbitrary diffusion.
    """

    def __init__(
        self,
        nodes: list[Node],
        edges: list[Edge],
        factor_nodes: list[FactorNode] | None = None,
    ) -> None:
        self._nodes: dict[str, Node] = {n.id: n for n in nodes}
        self._edges: list[Edge] = list(edges)
        self._factor_nodes: dict[str, FactorNode] = (
            {fn.id: fn for fn in factor_nodes} if factor_nodes else {}
        )
        self._risk_vectors: dict[str, RiskVector] = {}

    @property
    def nodes(self) -> dict[str, Node]:
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        return self._edges

    @property
    def factor_nodes(self) -> dict[str, FactorNode]:
        return self._factor_nodes

    @property
    def risk_vectors(self) -> dict[str, RiskVector]:
        return self._risk_vectors

    def add_factor_node(self, fn: FactorNode) -> None:
        self._factor_nodes[fn.id] = fn

    def set_risk_vector(self, node_id: str, rv: RiskVector) -> None:
        self._risk_vectors[node_id] = rv

    def all_node_ids(self) -> set[str]:
        """All valid node IDs (entity + factor)."""
        return set(self._nodes.keys()) | set(self._factor_nodes.keys())

    def validate(self) -> None:
        """Validate that all edges use L1 relations and reference valid nodes."""
        valid_ids = self.all_node_ids()
        for edge in self._edges:
            if edge.src not in valid_ids:
                raise OntologyViolationError(
                    f"Edge src '{edge.src}' not in node set",
                )
            if edge.dst not in valid_ids:
                raise OntologyViolationError(
                    f"Edge dst '{edge.dst}' not in node set",
                )
            is_propagation = isinstance(edge.relation, L1PropagationRelation)
            is_suppression = isinstance(edge.relation, L1SuppressionRelation)
            if not (is_propagation or is_suppression):
                raise OntologyViolationError(
                    f"G_R only allows L1 relations, got '{edge.relation}'",
                )
            if is_propagation and edge.sign != "+":
                raise OntologyViolationError(
                    f"Propagation edge must have sign='+', got '{edge.sign}'",
                )
            if is_suppression and edge.sign != "-":
                raise OntologyViolationError(
                    f"Suppression edge must have sign='-', got '{edge.sign}'",
                )
        for fn in self._factor_nodes.values():
            if fn.hazard_id not in self._nodes:
                raise OntologyViolationError(
                    f"FactorNode '{fn.id}' hazard_id '{fn.hazard_id}' not in entity nodes",
                )
            if fn.target_id not in self._nodes:
                raise OntologyViolationError(
                    f"FactorNode '{fn.id}' target_id '{fn.target_id}' not in entity nodes",
                )

    def to_networkx(self) -> nx.DiGraph[str, dict[str, object]]:
        """Convert to networkx DiGraph for computation."""
        g: nx.DiGraph[str, dict[str, object]] = nx.DiGraph()
        for node in self._nodes.values():
            g.add_node(node.id, **node.model_dump(), node_kind="entity")
        for fn in self._factor_nodes.values():
            g.add_node(fn.id, **fn.model_dump(), node_kind="factor")
        for edge in self._edges:
            g.add_edge(edge.src, edge.dst, **edge.model_dump(exclude={"src", "dst"}))
        return g

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "layer": "G_R",
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "factor_nodes": [fn.model_dump(mode="json") for fn in self._factor_nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self._edges],
            "risk_vectors": {
                k: v.model_dump(mode="json") for k, v in self._risk_vectors.items()
            },
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> ReasoningGraph:
        """Deserialize from JSON string."""
        parsed: dict[str, Any] = json.loads(data)
        nodes = [Node.model_validate(n) for n in parsed["nodes"]]
        edges = [Edge.model_validate(e) for e in parsed["edges"]]
        factor_nodes = [
            FactorNode.model_validate(fn) for fn in parsed.get("factor_nodes", [])
        ]
        graph = cls(nodes=nodes, edges=edges, factor_nodes=factor_nodes)
        for node_id, rv_data in parsed.get("risk_vectors", {}).items():
            graph.set_risk_vector(node_id, RiskVector.model_validate(rv_data))
        return graph
