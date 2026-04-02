"""
Orchestrates working memory, long-term memory, consolidation (WM -> LTM), and I/O stubs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from embodiedbench.memory_manip.config import MemorySystemConfig
from embodiedbench.memory_manip.interfaces import (
    MonitorInterface,
    NullMonitor,
    NullPerception,
    NullSimulation,
    PerceptionInterface,
    SimulationInterface,
)
from embodiedbench.memory_manip.longterm_memory import ManipulationLongTermMemory
from embodiedbench.memory_manip.working_memory import ManipulationWorkingMemory


class EmbodiedManipulationMemorySystem:
    """
    Top-level memory facade for EB-Manipulation (and future real-robot deployments).

    Data flow (conceptual): perception -> working memory (semantic + 3D lanes),
    planner symbols -> symbolic/abstract lanes, consolidation -> LTM compartments.
    simulation / monitor hooks read/write conceptual LTM and working snapshots.
    """

    def __init__(
        self,
        config: Optional[MemorySystemConfig] = None,
        *,
        perception: Optional[PerceptionInterface] = None,
        simulation: Optional[SimulationInterface] = None,
        monitor: Optional[MonitorInterface] = None,
    ) -> None:
        self.config = config or MemorySystemConfig()
        self.working = ManipulationWorkingMemory.from_config(self.config)
        self.long_term = ManipulationLongTermMemory()

        self.perception = perception or NullPerception()
        self.simulation = simulation or NullSimulation()
        self.monitor = monitor or NullMonitor()

        self._last_instruction: str = ""
        self._last_task_variation: str = ""

    # --- episode lifecycle -------------------------------------------------
    def reset_episode(self) -> None:
        """Call when the gym env resets."""
        self.working.clear_volatile()

    def begin_task(self, instruction: str, task_variation: str) -> None:
        """Seed abstract working memory with language instruction metadata."""
        self._last_instruction = instruction
        self._last_task_variation = task_variation
        self.working.abstract.push_note(f"instruction:{instruction}")
        self.working.abstract.push_note(f"variation:{task_variation}")
        self.working.task_schedule.add_milestone("task_start")

    def end_episode(
        self,
        *,
        success: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Optional consolidation at episode end."""
        summary = {
            "instruction": self._last_instruction,
            "variation": self._last_task_variation,
            "success": success,
            "working_global_step": self.working.task_schedule.global_step,
            **(extra or {}),
        }
        self.long_term.temporal.record_episode_summary(summary)
        if self.config.auto_consolidate_on_episode_end:
            self.consolidate_to_long_term(episode_summary=summary)

    # --- perception hook ---------------------------------------------------
    def on_perception_features(
        self,
        features: np.ndarray,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ingest (T,P,D) tensor: pool into working memory semantic/3D lanes.

        Default policy: mean-pool over T,P -> vector stored in spatial_3d.scene_embedding;
        store a lightweight semantic fact with shape metadata for traceability.
        """
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected perception features of shape (T,P,D), got {arr.shape}")
        pooled = arr.mean(axis=(0, 1))
        self.working.spatial_3d.set_scene_embedding(pooled)
        fact: Dict[str, Any] = {
            "type": "perception_grid",
            "shape": list(arr.shape),
            "meta": meta or {},
        }
        self.working.semantic.add_fact(fact)

    def run_perception(self, observation: Any) -> np.ndarray:
        """Pull features from the perception module and ingest them."""
        feats = self.perception.extract_features(observation)
        self.on_perception_features(feats)
        return feats

    # --- planner / control hooks -------------------------------------------
    def record_planner_reasoning(self, reasoning_text: str) -> None:
        """Attach latest language reasoning to abstract WM."""
        if reasoning_text:
            self.working.abstract.push_note(reasoning_text[:2000])

    def record_discrete_action(self, action_vector: Union[List[float], np.ndarray]) -> None:
        """Log a manipulation step into symbolic + spatial buffers."""
        act = np.asarray(action_vector, dtype=np.float32).reshape(-1)
        sym = ",".join(str(int(x)) if float(x).is_integer() else f"{x:.3f}" for x in act.tolist())
        self.working.symbolic.push(f"act[{sym}]")
        self.working.spatial_3d.push_pose(act[: min(3, act.shape[0])])
        self.working.task_schedule.tick()

    def record_environment_feedback(self, info: Dict[str, Any]) -> None:
        """Fold env info dict into semantic facts (success bits, messages)."""
        slim = {k: info[k] for k in ("task_success", "action_success") if k in info}
        if slim:
            self.working.semantic.add_fact({"type": "env_feedback", **slim})

    # --- simulation / monitor hooks --------------------------------------
    def propose_simulated_outcome(
        self, action: Any, *, horizon: int = 1
    ) -> Dict[str, Any]:
        """Route through conceptual WM + simulation stub."""
        state_summary = {
            "instruction": self._last_instruction,
            "variation": self._last_task_variation,
            "entities": len(self.long_term.conceptual.entities),
        }
        return self.simulation.predict(state_summary, action, horizon=horizon)

    def run_monitor(self, observation: Any) -> bool:
        snap = self.snapshot()
        ok, msg = self.monitor.check(observation, snap)
        if not ok and msg:
            self.working.semantic.add_fact({"type": "monitor_alert", "message": msg})
        return ok

    # --- consolidation -----------------------------------------------------
    def consolidate_to_long_term(self, *, episode_summary: Dict[str, Any]) -> None:
        """
        Project working-memory traces into persistent structures.

        This reference implementation only sketches the mapping; extend with your
        embedding models / graph learners.
        """
        # Conceptual: anchor entity for current task variation
        eid = f"task::{episode_summary.get('variation', 'unknown')}"
        self.long_term.conceptual.upsert_entity(
            eid,
            {
                "instruction": episode_summary.get("instruction"),
                "last_success": episode_summary.get("success"),
            },
        )

        # Spatial: connect last poses as a short chain
        poses = self.working.spatial_3d.snapshot_poses()
        for i, p in enumerate(poses[-8:]):
            nid = f"{eid}::pose::{episode_summary.get('working_global_step', 0)}::{i}"
            self.long_term.spatial.add_node(nid, {"xyz": p.tolist()})
            if i:
                prev = f"{eid}::pose::{episode_summary.get('working_global_step', 0)}::{i-1}"
                self.long_term.spatial.add_edge(prev, nid, {"kind": "trajectory"})

    # --- inspection --------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """JSON-friendly dict for logging, monitor, or debugging."""
        return {
            "working": {
                "symbolic": self.working.symbolic.snapshot(),
                "abstract": self.working.abstract.snapshot(),
                "semantic": self.working.semantic.snapshot(),
                "spatial_poses": [p.tolist() for p in self.working.spatial_3d.snapshot_poses()],
                "scene_embedding": None
                if self.working.spatial_3d.scene_embedding is None
                else self.working.spatial_3d.scene_embedding.tolist(),
                "task_schedule": self.working.task_schedule.snapshot(),
            },
            "long_term": {
                "conceptual_entities": list(self.long_term.conceptual.entities.keys()),
                "temporal_episodes": len(self.long_term.temporal.episodes),
                "spatial_nodes": len(self.long_term.spatial.nodes),
            },
            "rig": self.config.rig_metadata,
        }
