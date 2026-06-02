"""
EmbodiedManipulationMemorySystem: top-level three-layer memory facade.

Layer data flow:
  Perception     -> working.observation  (scene embedding, visible objects)
  GoalReasoner   -> working.goal         (instruction, 5-Whys intent)
  SolutionSpace  -> working.observation  (visible/memory objects, location)
  TaskValidator  -> working.action_buffer (action + outcome per step)
  Real robot     -> working.robot_state  (gripper pose, joints, force/torque)

At episode end:
  working   -> episodic  (full trajectory saved via consolidate_episode)
  episodic  -> semantic  (batch generalization every N episodes)

Planning_agent bridge (read-only):
  persistent_memory.md  -> semantic (user prefs, scene knowledge)
  session_context.md    -> working  (goal intent, visited locations)

Public API is backward-compatible with the previous EmbodiedManipulationMemorySystem:
  reset_episode, begin_task, end_episode,
  on_perception_features, run_perception,
  record_planner_reasoning, record_discrete_action, record_environment_feedback,
  propose_simulated_outcome, run_monitor,
  query_longterm_memory, snapshot.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
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
from embodiedbench.memory_manip.working_memory import WorkingMemory
from embodiedbench.memory_manip.episodic_memory import EpisodicMemory, EpisodeRecord, StepRecord
from embodiedbench.memory_manip.semantic_memory import SemanticMemory
from embodiedbench.memory_manip.longterm_memory import EmbodiedLTMClient
from embodiedbench.memory_manip.bridges import PlanningAgentBridge
from embodiedbench.memory_manip.episode_scorer import ManipulationScorer


class EmbodiedManipulationMemorySystem:
    """
    Three-layer memory system for embodied manipulation.

    Compatible with both EmbodiedBench simulation and real-robot deployments.
    External perception / simulation / monitor modules plug in via the same
    interface stubs as before; this class never imports from them directly.
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
        store = self.config.get_store_dir()
        store.mkdir(parents=True, exist_ok=True)

        # Three memory layers
        self.working = WorkingMemory.from_config(self.config)
        self.episodic = EpisodicMemory(
            store_path=store / "episodes.jsonl",
            max_episodes=self.config.episodic_max_episodes,
            decay_rate=self.config.activation_decay_rate,
            access_boost=self.config.activation_access_boost,
            prune_threshold=self.config.activation_prune_threshold,
            min_activation_retrieval=self.config.activation_min_retrieval,
        )
        self.semantic = SemanticMemory(store_path=store / "semantic_kb.json")
        self._scorer = ManipulationScorer(self.config)

        # Optional remote EmbodiedLTM service
        self._ltm_client = EmbodiedLTMClient(
            base_url=self.config.embodiedltm_base_url,
            timeout_sec=self.config.embodiedltm_timeout_sec,
        )

        # External I/O stubs (unchanged interface)
        self.perception = perception or NullPerception()
        self.simulation = simulation or NullSimulation()
        self.monitor = monitor or NullMonitor()

        # Planning_module bridge (read-only)
        self._bridge = PlanningAgentBridge(self.config.planning_module_dir)

        # Episode-level bookkeeping
        self._episode_id: str = ""
        self._scene_id: str = ""
        self._episode_start_time: float = 0.0
        self._episodes_since_generalize: int = 0
        self._pending_generalization: List[EpisodeRecord] = []

    # -----------------------------------------------------------------------
    # Episode lifecycle
    # -----------------------------------------------------------------------

    def reset_episode(self, scene_id: str = "") -> None:
        """Call when the gym env resets. Clears volatile working memory."""
        self._episode_id = str(uuid.uuid4())[:8]
        self._scene_id = scene_id
        self._episode_start_time = time.time()
        self.working.clear_episode()

    def begin_task(
        self,
        instruction: str,
        task_variation: str = "",
        intent: Dict[str, Any] = None,
    ) -> None:
        """
        Seed working memory with the task instruction.

        intent: optional GoalReasoner output dict (target_object, deep_intent, …).
        """
        self.working.goal.set(
            instruction=instruction,
            task_variation=task_variation,
            intent=intent or {},
            alternatives=(intent or {}).get("acceptable_alternatives_properties", []),
        )
        self.working.clock.add_milestone("task_start")

    def end_episode(
        self,
        *,
        success: bool,
        extra: Optional[Dict[str, Any]] = None,
        blueprint_skills: Optional[List[str]] = None,
    ) -> None:
        """
        Consolidate working memory into episodic memory at episode end.

        Triggers semantic generalization every N episodes.
        Sends a lightweight summary to the EmbodiedLTM remote service.
        """
        duration = time.time() - self._episode_start_time

        record = EpisodeRecord(
            episode_id=self._episode_id,
            task_instruction=self.working.goal.instruction,
            task_variation=self.working.goal.task_variation,
            scene_id=self._scene_id,
            success=success,
            total_steps=self.working.clock.episode_step,
            duration_sec=round(duration, 2),
            intent=self.working.goal.intent,
            extra=extra or {},
            creation_episode=self.episodic._episode_counter,
        )

        # Populate step records from the action buffer
        for ar in self.working.action_buffer.records:
            record.add_step(StepRecord(
                step_idx=ar.step,
                obs_summary=self.working.observation.text[:200],
                visible_objects=list(self.working.observation.visible_objects),
                current_location=self.working.observation.current_location,
                action=ar.action,
                action_id=ar.action_id,
                success=ar.success,
                feedback=ar.feedback,
                reasoning=ar.reasoning,
            ))

        # Attach Blueprint skill sequence from the spine brain (if provided)
        record.blueprint_skills = list(blueprint_skills) if blueprint_skills else []

        # Compute importance score before persisting
        breakdown = self._scorer.compute(record, semantic=self.semantic)
        record.initial_score = breakdown.total

        self.episodic.save_episode(record)
        self._pending_generalization.append(record)
        self._episodes_since_generalize += 1

        # Batch generalization: episodic -> semantic every N episodes
        if self._episodes_since_generalize >= self.config.episodic_generalize_every_n:
            generalized_ids = [r.episode_id for r in self._pending_generalization]
            self.semantic.generalize_from_episodes(self._pending_generalization)
            self.episodic.mark_abstracted(generalized_ids)
            self.episodic.decay_activations()
            self._pending_generalization.clear()
            self._episodes_since_generalize = 0

        # Push lightweight summary to optional remote EmbodiedLTM service
        self._ltm_client.insert_memory(
            query_text=json.dumps(
                {
                    "instruction": self.working.goal.instruction,
                    "episode_id": self._episode_id,
                    "success": success,
                },
                ensure_ascii=False,
            )
        )

    def decay_and_prune(self) -> Dict[str, Any]:
        """
        Force an activation decay pass and prune eligible episodes.

        Returns a summary dict with pruned count and current activation stats.
        Call periodically (e.g., at session end) to keep memory lean.
        """
        self.episodic.decay_activations()
        pruned = self.episodic.prune()
        return {"pruned": pruned, "activation_stats": self.episodic.get_activation_stats()}

    # -----------------------------------------------------------------------
    # Spine brain bridge
    # -----------------------------------------------------------------------

    def record_blueprint_execution(
        self,
        task_instruction: str,
        blueprint_skills: List[str],
        success: bool,
        scene_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the result of a Blueprint execution from the spine brain.

        Call this after franka_state_machine_cerebellum finishes a blueprint.
        Updates TaskSchema with the skill sequence so the VLM Brain can
        reference a known-good ordering on the next attempt.

        Parameters
        ----------
        task_instruction : natural language task string
        blueprint_skills : ordered skill names executed, e.g.
                           ["move_above", "descend", "grasp", "lift", "place", "retreat"]
        success          : whether the full blueprint execution succeeded
        scene_state      : optional scene_state dict from the spine brain
                           (keys: cube_pose, target_pose, ee_pose, gripper_width)
        """
        self.semantic.task_schema.record_blueprint_skills(
            task_instruction, blueprint_skills, success
        )
        self.semantic.save()
        if scene_state:
            self.working.observation.update_scene(
                text=json.dumps(scene_state, ensure_ascii=False)
            )

    def query_vlm_context(
        self,
        task_instruction: str,
    ) -> Dict[str, Any]:
        """Return structured memory context for VLM Brain input.

        Designed to be serialized and written alongside scene_state.json so
        the VLM Brain can make better-informed blueprint decisions.

        Returns
        -------
        dict with keys:
          task_type                  : str
          task_success_rate          : float | None
          recommended_blueprint_skills : list[str]  – from successful history
          object_location_priors     : {obj_name: top_location}
          similar_episodes           : list of lightweight episode summaries
        """
        schema = self.semantic.query_task_schema(task_instruction)
        similar = self.episodic.query_similar(task_instruction, top_k=3)

        object_priors: Dict[str, str] = {}
        for obj_name in ["cube", "target"]:
            kb = self.semantic.query_object(obj_name)
            likely = kb.get("likely_locations", [])
            if likely:
                object_priors[obj_name] = likely[0][0]

        return {
            "task_type": schema.get("task_type"),
            "task_success_rate": schema.get("success_rate"),
            "recommended_blueprint_skills": schema.get("blueprint_skills", []),
            "object_location_priors": object_priors,
            "similar_episodes": [
                {
                    "episode_id": r.episode_id,
                    "task": r.task_instruction,
                    "success": r.success,
                    "blueprint_skills": r.blueprint_skills,
                }
                for r in similar
            ],
        }

    # -----------------------------------------------------------------------
    # Planning_module bridge
    # -----------------------------------------------------------------------

    def sync_from_planning_module(self) -> Dict[str, Any]:
        """
        Pull user preferences and scene knowledge from the Planning_module.

        Tries MD files first (legacy path), then queries the shared EmbodiedLTM
        HTTP service that Planning_module writes to.

        Call once at session start. Never modifies Planning_module files.
        Returns a status dict describing what was synced.
        """
        return self._bridge.sync(self.working, self.semantic, ltm_client=self._ltm_client)

    def sync_from_planning_agent(self) -> Dict[str, Any]:
        """Backward-compatible alias for sync_from_planning_module()."""
        return self.sync_from_planning_module()

    # -----------------------------------------------------------------------
    # Perception hook
    # -----------------------------------------------------------------------

    def on_perception_features(
        self,
        features: np.ndarray,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ingest a (T, P, D) float32 feature array into working memory."""
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected (T, P, D) array, got shape {arr.shape}")
        self.working.observation.update_features(arr)

    def run_perception(self, observation: Any) -> np.ndarray:
        """Pull features from the perception module and ingest into working memory."""
        feats = self.perception.extract_features(observation)
        self.on_perception_features(feats)
        return feats

    # -----------------------------------------------------------------------
    # Working memory update hooks
    # -----------------------------------------------------------------------

    def update_observation(
        self,
        visible_objects: List[str] = None,
        memory_objects: Dict[str, Any] = None,
        text: str = None,
        current_location: str = None,
        visited_locations: Dict[str, str] = None,
        held_object: Optional[str] = None,
    ) -> None:
        """Push SolutionSpaceAnalyzer outputs into working memory observation."""
        self.working.observation.update_scene(
            visible_objects=visible_objects,
            memory_objects=memory_objects,
            text=text,
            current_location=current_location,
            visited_locations=visited_locations,
            held_object=held_object,
        )

    def record_action(
        self,
        action: str,
        action_id: Any,
        success: Optional[bool] = None,
        feedback: str = "",
        reasoning: str = "",
    ) -> None:
        """Record an executed action and its outcome; advance the task clock."""
        self.working.action_buffer.record(
            step=self.working.clock.episode_step,
            action=action,
            action_id=action_id,
            success=success,
            feedback=feedback,
            reasoning=reasoning,
        )
        self.working.clock.tick()

    def record_goal_intent(self, intent: Dict[str, Any]) -> None:
        """Push GoalReasoner output into the working memory goal slot."""
        self.working.goal.intent = intent
        self.working.goal.alternatives = intent.get("acceptable_alternatives_properties", [])

    def update_robot_state(
        self,
        gripper_pose: Optional[np.ndarray] = None,
        joint_angles: Optional[np.ndarray] = None,
        force_torque: Optional[np.ndarray] = None,
        gripper_aperture: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Update proprioceptive state. No-op in simulation (all args None)."""
        self.working.robot_state.update(
            gripper_pose=gripper_pose,
            joint_angles=joint_angles,
            force_torque=force_torque,
            gripper_aperture=gripper_aperture,
            timestamp=timestamp,
        )

    # --- backward-compatible aliases ---------------------------------------

    def record_planner_reasoning(self, reasoning_text: str) -> None:
        """Backward-compatible alias: store reasoning text in the goal slot."""
        if reasoning_text:
            existing = self.working.goal.intent.get("reasoning_text", "")
            self.working.goal.intent["reasoning_text"] = (existing + " " + reasoning_text).strip()

    def record_discrete_action(self, action_vector: Union[List[float], np.ndarray]) -> None:
        """Backward-compatible alias: log a raw action vector as a step."""
        act = np.asarray(action_vector, dtype=np.float32).reshape(-1)
        sym = ",".join(
            str(int(x)) if float(x).is_integer() else f"{x:.3f}" for x in act.tolist()
        )
        self.record_action(action=f"act[{sym}]", action_id=sym)
        # Also push gripper-relevant xyz into robot_state for real-robot readiness
        if act.shape[0] >= 3:
            self.working.robot_state.update(gripper_pose=act[:6] if act.shape[0] >= 6 else None)

    def record_environment_feedback(self, info: Dict[str, Any]) -> None:
        """Backward-compatible alias: fold env info dict into the last action record."""
        task_success = info.get("task_success")
        action_success = info.get("action_success")
        feedback_str = f"task_success={task_success} action_success={action_success}"
        if self.working.action_buffer.records:
            last = self.working.action_buffer.records[-1]
            last.success = bool(action_success) if action_success is not None else last.success
            last.feedback = feedback_str
        self.working.clock.add_milestone(f"env_feedback:{feedback_str[:60]}")

    # -----------------------------------------------------------------------
    # Semantic memory query API
    # -----------------------------------------------------------------------

    def query_object(self, obj: str) -> Dict[str, Any]:
        """Return object affordances and historical locations from semantic memory."""
        return self.semantic.query_object(obj)

    def query_user_preference(self, need: str) -> Dict[str, Any]:
        """Return user preferences for a functional need from semantic memory."""
        return self.semantic.query_user_preference(need)

    def query_task_schema(self, instruction: str) -> Dict[str, Any]:
        """Return historical success rate and common steps for similar tasks."""
        return self.semantic.query_task_schema(instruction)

    # -----------------------------------------------------------------------
    # Episodic memory query API
    # -----------------------------------------------------------------------

    def query_similar_episodes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return lightweight summaries of past episodes similar to query."""
        records = self.episodic.query_similar(query, top_k=top_k)
        return [
            {
                "episode_id": r.episode_id,
                "task_instruction": r.task_instruction,
                "success": r.success,
                "total_steps": r.total_steps,
                "objects_encountered": list(r.objects_encountered.keys()),
            }
            for r in records
        ]

    def get_object_location_history(self, obj: str) -> Dict[str, int]:
        """Return {location: frequency} for an object across all stored episodes."""
        return self.episodic.get_object_location_history(obj)

    # -----------------------------------------------------------------------
    # Simulation / monitor hooks (unchanged interface)
    # -----------------------------------------------------------------------

    def propose_simulated_outcome(
        self, action: Any, *, horizon: int = 1
    ) -> Dict[str, Any]:
        """Route action through the simulation stub."""
        state_summary = self.working.snapshot()
        return self.simulation.predict(state_summary, action, horizon=horizon)

    def run_monitor(self, observation: Any) -> bool:
        """Run the monitor stub; log any alert into the task clock milestones."""
        snap = self.snapshot()
        ok, msg = self.monitor.check(observation, snap)
        if not ok and msg:
            self.working.clock.add_milestone(f"monitor_alert:{msg[:80]}")
        return ok

    # -----------------------------------------------------------------------
    # Remote LTM query (backward-compatible)
    # -----------------------------------------------------------------------

    def query_longterm_memory(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Forward a query to the optional EmbodiedLTM remote service."""
        return self._ltm_client.query_memory(query, mode=mode)

    # -----------------------------------------------------------------------
    # Snapshot / inspection
    # -----------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Full JSON-serializable snapshot for logging, monitor, or debugging."""
        return {
            "episode_id": self._episode_id,
            "scene_id": self._scene_id,
            "working": self.working.snapshot(),
            "episodic": self.episodic.summary(),
            "semantic": self.semantic.summary(),
            "rig": self.config.rig_metadata,
        }
