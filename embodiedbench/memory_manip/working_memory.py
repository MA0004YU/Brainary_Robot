"""
Working Memory: volatile state for a single episode.

Cleared at every episode boundary via clear_episode().
Five slots:
  ActiveGoal   - current instruction + GoalReasoner intent structure
  Observation  - latest perception snapshot (features, visible objects, text)
  ActionBuffer - circular buffer of (action, outcome) pairs this episode
  RobotState   - proprioceptive state; all fields None in simulation
  TaskClock    - step counters, budget, and milestone flags
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from embodiedbench.memory_manip.config import MemorySystemConfig


# ---------------------------------------------------------------------------
# ActiveGoal
# ---------------------------------------------------------------------------

@dataclass
class ActiveGoal:
    """Current task instruction and extracted 5-Whys intent structure."""

    instruction: str = ""
    task_variation: str = ""
    # Full dict from GoalReasoner: target_object, deep_intent, reasoning_chain, …
    intent: Dict[str, Any] = field(default_factory=dict)
    # acceptable_alternatives_properties list from GoalReasoner
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def set(
        self,
        instruction: str,
        task_variation: str = "",
        intent: Dict[str, Any] = None,
        alternatives: List[Dict[str, Any]] = None,
    ) -> None:
        self.instruction = instruction
        self.task_variation = task_variation
        self.intent = intent or {}
        self.alternatives = alternatives or []

    def clear(self) -> None:
        self.instruction = ""
        self.task_variation = ""
        self.intent = {}
        self.alternatives = []

    def snapshot(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "task_variation": self.task_variation,
            "intent": self.intent,
            "alternatives": self.alternatives,
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Latest perception snapshot: visual features, visible objects, and text."""

    # Mean-pooled (T,P,D) scene embedding vector, or None before first perception call
    scene_embedding: Optional[np.ndarray] = None
    # Raw (T,P,D) array from the last perception call
    raw_features: Optional[np.ndarray] = None
    # Objects currently visible according to SolutionSpaceAnalyzer
    visible_objects: List[str] = field(default_factory=list)
    # Objects known from memory: {obj_name: {location, requires_open, …}}
    memory_objects: Dict[str, Any] = field(default_factory=dict)
    # Raw environment observation text
    text: str = ""
    # Name of the robot's current location
    current_location: str = ""
    # Map of previously visited locations to their exploration state
    visited_locations: Dict[str, str] = field(default_factory=dict)
    # Object currently held by the gripper, or None
    held_object: Optional[str] = None

    def update_features(self, features: np.ndarray) -> None:
        """Ingest a (T, P, D) array and pool it into scene_embedding."""
        arr = np.asarray(features, dtype=np.float32)
        self.raw_features = arr
        if arr.ndim == 3:
            self.scene_embedding = arr.mean(axis=(0, 1))

    def update_scene(
        self,
        visible_objects: List[str] = None,
        memory_objects: Dict[str, Any] = None,
        text: str = None,
        current_location: str = None,
        visited_locations: Dict[str, str] = None,
        held_object: Optional[str] = None,
    ) -> None:
        if visible_objects is not None:
            self.visible_objects = visible_objects
        if memory_objects is not None:
            self.memory_objects = memory_objects
        if text is not None:
            self.text = text
        if current_location is not None:
            self.current_location = current_location
        if visited_locations is not None:
            self.visited_locations = visited_locations
        if held_object is not None:
            self.held_object = held_object

    def clear(self) -> None:
        self.scene_embedding = None
        self.raw_features = None
        self.visible_objects = []
        self.memory_objects = {}
        self.text = ""
        self.current_location = ""
        self.visited_locations = {}
        self.held_object = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "visible_objects": self.visible_objects,
            "memory_objects": self.memory_objects,
            "text": self.text,
            "current_location": self.current_location,
            "visited_locations": self.visited_locations,
            "held_object": self.held_object,
            "has_scene_embedding": self.scene_embedding is not None,
        }


# ---------------------------------------------------------------------------
# ActionBuffer
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    """A single executed (action, outcome) pair."""
    step: int
    action: str
    action_id: Any
    success: Optional[bool]
    feedback: str = ""
    reasoning: str = ""


@dataclass
class ActionBuffer:
    """Circular buffer of recent actions and their outcomes within the episode."""

    records: Deque[ActionRecord] = field(default_factory=lambda: deque(maxlen=32))

    def record(
        self,
        step: int,
        action: str,
        action_id: Any,
        success: Optional[bool] = None,
        feedback: str = "",
        reasoning: str = "",
    ) -> None:
        self.records.append(ActionRecord(
            step=step,
            action=action,
            action_id=action_id,
            success=success,
            feedback=feedback,
            reasoning=reasoning,
        ))

    def clear(self) -> None:
        self.records.clear()

    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "step": r.step,
                "action": r.action,
                "action_id": r.action_id,
                "success": r.success,
                "feedback": r.feedback,
                "reasoning": r.reasoning,
            }
            for r in self.records
        ]


# ---------------------------------------------------------------------------
# RobotState  (real-robot proprioception; all fields None in simulation)
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """
    Proprioceptive state for real-robot deployments.

    Every field is Optional so simulation runs transparently without changes.
    Real-robot adapters call update() at each control tick to fill these fields.
    """

    # End-effector pose in world frame: [x, y, z, roll, pitch, yaw]
    gripper_pose: Optional[np.ndarray] = None
    # Joint angles in radians, shape (DOF,)
    joint_angles: Optional[np.ndarray] = None
    # Wrist force-torque: [Fx, Fy, Fz, Tx, Ty, Tz]
    force_torque: Optional[np.ndarray] = None
    # Gripper aperture: 0.0 = fully closed, 1.0 = fully open
    gripper_aperture: Optional[float] = None
    # Unix timestamp of the last sensor reading
    timestamp: Optional[float] = None

    def update(
        self,
        gripper_pose: Optional[np.ndarray] = None,
        joint_angles: Optional[np.ndarray] = None,
        force_torque: Optional[np.ndarray] = None,
        gripper_aperture: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        if gripper_pose is not None:
            self.gripper_pose = np.asarray(gripper_pose, dtype=np.float32)
        if joint_angles is not None:
            self.joint_angles = np.asarray(joint_angles, dtype=np.float32)
        if force_torque is not None:
            self.force_torque = np.asarray(force_torque, dtype=np.float32)
        if gripper_aperture is not None:
            self.gripper_aperture = float(gripper_aperture)
        if timestamp is not None:
            self.timestamp = float(timestamp)

    def clear(self) -> None:
        self.gripper_pose = None
        self.joint_angles = None
        self.force_torque = None
        self.gripper_aperture = None
        self.timestamp = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "gripper_pose": self.gripper_pose.tolist() if self.gripper_pose is not None else None,
            "joint_angles": self.joint_angles.tolist() if self.joint_angles is not None else None,
            "force_torque": self.force_torque.tolist() if self.force_torque is not None else None,
            "gripper_aperture": self.gripper_aperture,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# TaskClock
# ---------------------------------------------------------------------------

@dataclass
class TaskClock:
    """Step counters, budget tracking, and milestone flags."""

    global_step: int = 0      # Never reset; accumulates across all episodes
    episode_step: int = 0     # Reset at each episode boundary
    max_steps: int = 50
    milestones: Deque[str] = field(default_factory=lambda: deque(maxlen=64))

    def tick(self) -> None:
        self.global_step += 1
        self.episode_step += 1

    def reset_episode(self) -> None:
        self.episode_step = 0
        self.milestones.clear()

    def add_milestone(self, label: str) -> None:
        self.milestones.append(label)

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.episode_step)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "max_steps": self.max_steps,
            "remaining_steps": self.remaining_steps,
            "milestones": list(self.milestones),
        }


# ---------------------------------------------------------------------------
# WorkingMemory  (top-level container)
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemory:
    """Single container for all five working-memory slots."""

    goal: ActiveGoal = field(default_factory=ActiveGoal)
    observation: Observation = field(default_factory=Observation)
    action_buffer: ActionBuffer = field(default_factory=lambda: ActionBuffer(records=deque(maxlen=32)))
    robot_state: RobotState = field(default_factory=RobotState)
    clock: TaskClock = field(default_factory=TaskClock)

    def clear_episode(self) -> None:
        """Clear all volatile state at episode boundary. global_step is preserved."""
        self.goal.clear()
        self.observation.clear()
        self.action_buffer.clear()
        self.robot_state.clear()
        self.clock.reset_episode()

    @staticmethod
    def from_config(cfg: "MemorySystemConfig") -> "WorkingMemory":
        return WorkingMemory(
            action_buffer=ActionBuffer(records=deque(maxlen=cfg.action_buffer_max)),
            clock=TaskClock(max_steps=cfg.episode_max_steps),
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "goal": self.goal.snapshot(),
            "observation": self.observation.snapshot(),
            "action_buffer": self.action_buffer.snapshot(),
            "robot_state": self.robot_state.snapshot(),
            "clock": self.clock.snapshot(),
        }
