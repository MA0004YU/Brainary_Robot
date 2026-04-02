"""
Working memory lanes matching the architecture figure:
- Logical / descriptive: symbolic, abstract, semantic
- Spatial: 3D
- Task-oriented schedule (icon in figure): short-horizon goals, timing hints, step index
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from embodiedbench.memory_manip.config import MemorySystemConfig


@dataclass
class SymbolicWorkingMemory:
    """
    Discrete tokens: action IDs, object handles, skill names, or language fragments
    that the policy reasons over at the current decision cycle.
    """

    tokens: Deque[str] = field(default_factory=lambda: deque(maxlen=64))

    def push(self, token: str) -> None:
        self.tokens.append(token)

    def extend(self, items: List[str]) -> None:
        for t in items:
            self.tokens.append(t)

    def snapshot(self) -> List[str]:
        return list(self.tokens)


@dataclass
class AbstractWorkingMemory:
    """
    High-level task representations: subgoals, phase labels, planner hypotheses.
    Stored as short strings or small dicts (your stack can upgrade to embeddings later).
    """

    notes: Deque[str] = field(default_factory=lambda: deque(maxlen=32))

    def push_note(self, note: str) -> None:
        self.notes.append(note)

    def snapshot(self) -> List[str]:
        return list(self.notes)


@dataclass
class SemanticWorkingMemory:
    """
    Meaning-level facts: object categories, colors, affordances, relations holding *now*.
    """

    facts: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=128))

    def add_fact(self, fact: Dict[str, Any]) -> None:
        self.facts.append(fact)

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self.facts)


@dataclass
class Spatial3DWorkingMemory:
    """
    Immediate 3D context: recent gripper poses, keypoint clouds, or voxel summaries.
    Uses numpy for efficient exchange with perception / control.
    """

    poses_xyz: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=32))
    # Optional dense patch-grid features aligned with perception (T,P,D) after pooling.
    scene_embedding: Optional[np.ndarray] = None

    def push_pose(self, xyz: np.ndarray) -> None:
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
        self.poses_xyz.append(arr)

    def set_scene_embedding(self, vec: Optional[np.ndarray]) -> None:
        if vec is None:
            self.scene_embedding = None
        else:
            self.scene_embedding = np.asarray(vec, dtype=np.float32)

    def snapshot_poses(self) -> List[np.ndarray]:
        return [np.copy(p) for p in self.poses_xyz]


@dataclass
class TaskScheduleWorkingMemory:
    """
    Task-clock / milestone buffer (planner icon in the figure): step counters,
    deadlines, flags for which subgoal is active.
    """

    global_step: int = 0
    episode_step: int = 0
    milestones: Deque[str] = field(default_factory=lambda: deque(maxlen=64))

    def tick(self) -> None:
        self.global_step += 1
        self.episode_step += 1

    def reset_episode_clock(self) -> None:
        self.episode_step = 0

    def add_milestone(self, label: str) -> None:
        self.milestones.append(label)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "milestones": list(self.milestones),
        }


@dataclass
class ManipulationWorkingMemory:
    """Single container for all working-memory lanes."""

    symbolic: SymbolicWorkingMemory = field(default_factory=SymbolicWorkingMemory)
    abstract: AbstractWorkingMemory = field(default_factory=AbstractWorkingMemory)
    semantic: SemanticWorkingMemory = field(default_factory=SemanticWorkingMemory)
    spatial_3d: Spatial3DWorkingMemory = field(default_factory=Spatial3DWorkingMemory)
    task_schedule: TaskScheduleWorkingMemory = field(default_factory=TaskScheduleWorkingMemory)

    def clear_volatile(self) -> None:
        """Clear fast buffers at episode boundary; does not reset global task clock."""
        self.symbolic.tokens.clear()
        self.abstract.notes.clear()
        self.semantic.facts.clear()
        self.spatial_3d.poses_xyz.clear()
        self.spatial_3d.scene_embedding = None
        self.task_schedule.milestones.clear()
        self.task_schedule.reset_episode_clock()

    @staticmethod
    def from_config(cfg: "MemorySystemConfig") -> "ManipulationWorkingMemory":
        """Construct queues with capacity taken from MemorySystemConfig."""
        return ManipulationWorkingMemory(
            symbolic=SymbolicWorkingMemory(tokens=deque(maxlen=cfg.symbolic_queue_max)),
            abstract=AbstractWorkingMemory(notes=deque(maxlen=cfg.abstract_queue_max)),
            semantic=SemanticWorkingMemory(facts=deque(maxlen=cfg.semantic_fact_max)),
            spatial_3d=Spatial3DWorkingMemory(poses_xyz=deque(maxlen=cfg.spatial_3d_pose_max)),
            task_schedule=TaskScheduleWorkingMemory(
                milestones=deque(maxlen=cfg.task_event_max)
            ),
        )
