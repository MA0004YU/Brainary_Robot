"""
Long-term memory modules from the architecture figure:
- Conceptual model of the world (entities, dynamics priors)
- Temporal knowledge (events, routines)
- Spatial knowledge (persistent maps / topology)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


NodeId = str


@dataclass
class ConceptualWorldModel:
    """
    Persistent conceptual structure: object entities, affordance tags, and simple rules.
    Represented as an entity table plus optional relation edges for prototyping.
    """

    entities: Dict[NodeId, Dict[str, Any]] = field(default_factory=dict)
    relations: List[Tuple[NodeId, str, NodeId]] = field(default_factory=list)

    def upsert_entity(self, eid: NodeId, record: Dict[str, Any]) -> None:
        base = self.entities.get(eid, {})
        base.update(record)
        self.entities[eid] = base

    def add_relation(self, src: NodeId, rel_type: str, dst: NodeId) -> None:
        self.relations.append((src, rel_type, dst))


@dataclass
class TemporalKnowledgeBase:
    """
    Sequences and histories: keyed episodes, ordered event strings, optional timestamps.
    """

    episodes: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))

    def record_episode_summary(self, summary: Dict[str, Any]) -> None:
        self.episodes.append(summary)


@dataclass
class SpatialKnowledgeGraph:
    """
    Large-scale spatial memory: metric landmarks or topological nodes.
    For sim2real, you can back this with a pose graph SLAM module behind the same API.
    """

    nodes: Dict[NodeId, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Tuple[NodeId, NodeId, Dict[str, Any]]] = field(default_factory=list)

    def add_node(self, nid: NodeId, attrs: Dict[str, Any]) -> None:
        self.nodes[nid] = attrs

    def add_edge(self, a: NodeId, b: NodeId, attrs: Optional[Dict[str, Any]] = None) -> None:
        self.edges.append((a, b, attrs or {}))


@dataclass
class ManipulationLongTermMemory:
    """Bundles the three long-term compartments."""

    conceptual: ConceptualWorldModel = field(default_factory=ConceptualWorldModel)
    temporal: TemporalKnowledgeBase = field(default_factory=TemporalKnowledgeBase)
    spatial: SpatialKnowledgeGraph = field(default_factory=SpatialKnowledgeGraph)
