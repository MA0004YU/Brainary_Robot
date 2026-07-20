"""
External I/O ports for the embodied manipulation stack.

Each interface has a Null* stub that keeps the system runnable before the real
module is integrated.  Teams should subclass the abstract base, implement the
single abstract method, then register the instance with the memory system:

    memory.register_perception(MyVisionPerception())
    memory.register_monitor(MySpineMonitor())
    memory.register_simulation(MyIsaacForwardModel())

─────────────────────────────────────────────────────────────────────────────
PerceptionInterface  (感知模块负责同学实现)
─────────────────────────────────────────────────────────────────────────────
Input : raw observation from env.reset() / env.step() — dict containing
        image paths, obs object, or Isaac Sim sensor data
Output: float32 numpy array shaped (T, P, D)
        T = temporal frames, P = patches/tokens per frame, D = channel dim

For the spine-brain pipeline, perception must ALSO produce a scene_state dict
in the format {cube_pose, target_pose, ee_pose, gripper_width, robot_joint_pos}
and call memory.ingest_scene_state(scene_state) directly, or pass it back to
the driver script to feed into the VLM Brain.

Example skeleton:
    class VisionPerception(PerceptionInterface):
        def extract_features(self, observation, *, meta=None):
            imgs = observation.get("image_paths", [])
            scene_state = self._run_detector(imgs)       # your vision pipeline
            # scene_state → VLM Brain input file (handled by driver)
            return np.zeros((1, 64, 512), dtype=np.float32)  # or real features

─────────────────────────────────────────────────────────────────────────────
MonitorInterface  (Monitor / 脊脑对接负责同学实现)
─────────────────────────────────────────────────────────────────────────────
Input : observation dict (from evaluator or spine brain execution result),
        memory_snapshot dict (from memory.snapshot())
Output: (ok: bool, alert: str | None)
        ok=False triggers a milestone log entry in working memory and can be
        used by the driver to re-plan.

For the spine-brain pipeline, wrap the result of condition_evaluator.py:
    class SpineConditionMonitor(MonitorInterface):
        def check(self, observation, memory_snapshot):
            timeout   = observation.get("timeout", False)
            collision = observation.get("collision_detected", False)
            ok = not timeout and not collision
            alert = "timeout" if timeout else ("collision" if collision else None)
            return ok, alert

─────────────────────────────────────────────────────────────────────────────
SimulationInterface  (Isaac Sim负责同学实现，阶段二)
─────────────────────────────────────────────────────────────────────────────
Input : state_summary dict (working memory snapshot), action (any format),
        horizon int
Output: predicted state dict (e.g. predicted ee_pose, object_pose after action)

For Isaac Sim, query the physics engine for a 1-step forward prediction:
    class IsaacForwardModel(SimulationInterface):
        def predict(self, state_summary, action, *, horizon=1):
            return self._isaac_step(state_summary["working"]["robot_state"], action)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PerceptionInterface(ABC):
    """Vision-backed perception. See module docstring for implementation guide."""

    @abstractmethod
    def extract_features(
        self, observation: Any, *, meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Return float32 array shaped (T, P, D)."""


class SimulationInterface(ABC):
    """Forward simulation / mental model rollouts. See module docstring."""

    @abstractmethod
    def predict(
        self,
        state_summary: Dict[str, Any],
        action: Any,
        *,
        horizon: int = 1,
    ) -> Dict[str, Any]:
        """Return predicted state or distribution summary."""


class MonitorInterface(ABC):
    """Runtime monitoring: success predicates and constraint violations. See module docstring."""

    @abstractmethod
    def check(
        self,
        observation: Any,
        memory_snapshot: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Return (ok, optional_alert_message)."""


class NullPerception(PerceptionInterface):
    """Returns zeros (1,1,1) until VisionPerception is registered."""

    def extract_features(
        self, observation: Any, *, meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        return np.zeros((1, 1, 1), dtype=np.float32)


class NullSimulation(SimulationInterface):
    """Echoes the input action as a trivial prediction."""

    def predict(
        self,
        state_summary: Dict[str, Any],
        action: Any,
        *,
        horizon: int = 1,
    ) -> Dict[str, Any]:
        return {"echo_action": action, "horizon": horizon, "note": "null_simulation"}


class NullMonitor(MonitorInterface):
    """Always reports OK until SpineConditionMonitor is registered."""

    def check(
        self,
        observation: Any,
        memory_snapshot: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        return True, None
