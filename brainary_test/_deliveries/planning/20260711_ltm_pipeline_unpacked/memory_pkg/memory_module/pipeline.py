"""
PerceptionMemoryPipeline: main orchestrator connecting
    perception_vlm (QwenVLMPerception)
  → three-layer memory (EmbodiedManipulationMemorySystem)
  → planning interface (PlanningMemoryInterface)

This is the single entry point for integrators. The pipeline wires together
all four components and manages the episode lifecycle.

Quick-start:
    from memory_module import PerceptionMemoryPipeline

    # 1. Create pipeline (lazy-loads VLM model)
    pipeline = PerceptionMemoryPipeline.create(store_dir="memory_store/")

    # 2. Session start
    pipeline.session_start()

    # 3. Episode start
    pipeline.begin_episode(scene_id="ep001", task_instruction="pick up the cube")

    # 4. Per-step loop
    results = pipeline.process_perception(["rgb.png"], current_location="table")
    ctx = pipeline.get_planning_context()
    # → pass ctx to your planning module

    pipeline.record_action("grasp", success=True, feedback="gripper closed")

    # 5. Episode end
    pipeline.end_episode(success=True, blueprint_skills=["move_above", "grasp", "lift"])

    # 6. Session end
    pipeline.session_end()
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    from embodiedbench.memory_manip.config import MemorySystemConfig
    _HAS_MEMORY = True
except ImportError:
    EmbodiedManipulationMemorySystem = None  # type: ignore
    MemorySystemConfig = None                # type: ignore
    _HAS_MEMORY = False

try:
    from perception_vlm import QwenVLMPerception
    DEFAULT_VLM_MEMORY_PATH = "perception_vlm_memory.json"
    _HAS_PERCEPTION = True
except ImportError:
    QwenVLMPerception = None  # type: ignore
    DEFAULT_VLM_MEMORY_PATH = "perception_vlm_memory.json"
    _HAS_PERCEPTION = False

from memory_module.perception_adapter import VisionPerceptionAdapter
from memory_module.planning_interface import PlanningMemoryInterface, PlanningContext


class PerceptionMemoryPipeline:
    """
    End-to-end pipeline: QwenVLMPerception → three-layer memory → planning.

    The pipeline is the only object integrators need to instantiate.
    All other memory_module classes are accessible as attributes:
        pipeline.adapter   — VisionPerceptionAdapter
        pipeline.planning  — PlanningMemoryInterface
        pipeline.memory    — EmbodiedManipulationMemorySystem (advanced use only)
    """

    def __init__(
        self,
        perceiver: Any,
        memory: Any,
        *,
        perception_adapter: Optional[Any] = None,
        planning_interface: Optional[Any] = None,
    ) -> None:
        """
        Low-level constructor. Prefer PerceptionMemoryPipeline.create() instead.

        Parameters
        ----------
        perceiver          : QwenVLMPerception instance
        memory             : EmbodiedManipulationMemorySystem instance
        perception_adapter : VisionPerceptionAdapter (auto-created if None)
        planning_interface : PlanningMemoryInterface (auto-created if None)
        """
        self.perceiver = perceiver
        self.memory = memory
        self.adapter: VisionPerceptionAdapter = (
            perception_adapter or VisionPerceptionAdapter(perceiver, memory)
        )
        self.planning: PlanningMemoryInterface = (
            planning_interface or PlanningMemoryInterface(memory)
        )

        # Register adapter so memory.run_perception() works too
        self.memory.register_perception(self.adapter)

        self._current_task: str = ""
        self._current_scene_id: str = ""
        self._step: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        store_dir: Optional[str] = None,
        vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        vlm_memory_path: str = DEFAULT_VLM_MEMORY_PATH,
        use_auxiliary_views: bool = True,
        enable_depth: bool = False,
        embodiedltm_base_url: Optional[str] = None,
        planning_module_dir: Optional[str] = None,
        rig_metadata: Optional[Dict[str, Any]] = None,
        robot_dof: int = 7,
    ) -> "PerceptionMemoryPipeline":
        """
        Create a fully initialized pipeline.

        All arguments are optional. Defaults are suitable for offline testing
        without a GPU (VLM loads on CPU) or remote services.

        Parameters
        ----------
        store_dir              : directory for persistent memory files.
                                 Defaults to embodiedbench/memory_manip/store/.
        vlm_model_id           : HuggingFace model ID for Qwen VLM.
        vlm_memory_path        : JSON path for VLM-level long-term memory.
        use_auxiliary_views    : enable contour/texture auxiliary views (recommended).
        enable_depth           : enable depth-estimation auxiliary view.
        embodiedltm_base_url   : URL of remote EmbodiedLTM service (None=disabled).
        planning_module_dir    : path to Planning_module directory (for MD-file sync).
        rig_metadata           : robot/rig metadata stored with each episode.
                                 e.g., {"robot": "franka", "dof": 7, "env": "isaac_sim"}
        robot_dof              : robot degrees of freedom (used for state arrays).
        """
        if not _HAS_PERCEPTION:
            raise RuntimeError(
                "perception_vlm.py not importable. "
                "Add E:/Brainary_Robot to PYTHONPATH and install its dependencies."
            )
        if not _HAS_MEMORY:
            raise RuntimeError(
                "embodiedbench package not importable. "
                "Add E:/Brainary_Robot to PYTHONPATH."
            )

        perceiver = QwenVLMPerception(
            model_id=vlm_model_id,
            memory_path=vlm_memory_path,
            use_auxiliary_views=use_auxiliary_views,
            enable_depth=enable_depth,
        )

        config = MemorySystemConfig(
            store_dir=store_dir,
            embodiedltm_base_url=embodiedltm_base_url,
            planning_module_dir=planning_module_dir,
            rig_metadata=rig_metadata or {},
            robot_dof=robot_dof,
        )
        memory_sys = EmbodiedManipulationMemorySystem(config=config)

        return cls(perceiver=perceiver, memory=memory_sys)

    @classmethod
    def create_from_existing(
        cls,
        perceiver: Any,
        memory: Any,
    ) -> "PerceptionMemoryPipeline":
        """
        Create pipeline from already-instantiated perceiver and memory objects.

        Use this when you manage the VLM model and memory system yourself,
        e.g., when sharing the model across multiple pipeline instances.
        """
        return cls(perceiver=perceiver, memory=memory)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def session_start(self, sync_planning_module: bool = False) -> Dict[str, Any]:
        """
        Call once at session start (before the first episode).

        Optionally syncs preferences and scene knowledge from the planning
        module's MD files or shared EmbodiedLTM service.

        Parameters
        ----------
        sync_planning_module : if True, read Planning_module persistent_memory.md
                               and session_context.md (if they exist).

        Returns
        -------
        sync status dict, or {} if sync_planning_module=False
        """
        if sync_planning_module:
            return self.memory.sync_from_planning_module()
        return {}

    def session_end(self) -> Dict[str, Any]:
        """
        Call once at session end.

        Runs activation decay and prunes stale episodes from episodic memory.

        Returns
        -------
        {"pruned": int, "activation_stats": {...}}
        """
        return self.memory.decay_and_prune()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(
        self,
        scene_id: str,
        task_instruction: str,
        task_variation: str = "",
        intent: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Begin a new episode. Clears volatile working memory.

        Parameters
        ----------
        scene_id         : unique scene / task-instance identifier
                           (e.g., "task_0042_v2", or Isaac Sim episode UUID)
        task_instruction : natural language instruction
                           (e.g., "pick up the red cube and place it on the target")
        task_variation   : optional variation tag (e.g., "v3", "hard")
        intent           : optional GoalReasoner output dict:
                           {target_object, deep_intent, reasoning_chain,
                            acceptable_alternatives_properties}
        """
        self._current_task = task_instruction
        self._current_scene_id = scene_id
        self._step = 0
        self.memory.reset_episode(scene_id=scene_id)
        self.memory.begin_task(
            instruction=task_instruction,
            task_variation=task_variation,
            intent=intent,
        )

    def end_episode(
        self,
        success: bool,
        blueprint_skills: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        End the current episode and consolidate memory.

        Saves a full EpisodeRecord to episodic memory, triggers semantic
        generalization every N episodes, and pushes a summary to the optional
        remote EmbodiedLTM service.

        Parameters
        ----------
        success          : True if the task was completed successfully
        blueprint_skills : ordered skill names executed by the spine brain
                           e.g. ["move_above", "descend", "grasp", "lift", "place", "retreat"]
        extra            : arbitrary metadata dict stored with the episode
        """
        self.memory.end_episode(
            success=success,
            blueprint_skills=blueprint_skills,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Per-step API
    # ------------------------------------------------------------------

    def process_perception(
        self,
        image_paths: Sequence[str],
        candidate_labels: Optional[List[str]] = None,
        current_location: str = "unknown",
        scene_state: Optional[Dict[str, Any]] = None,
        save_vlm_memory: bool = True,
    ) -> List[Any]:
        """
        Run perception and update working memory. Call this at every control step.

        Flow:
          QwenVLMPerception.recognize_batch()
            → encode_recognition_results() → (T, P, D) features
            → memory.on_perception_features()
            → memory.update_observation(visible_objects, scene_text, ...)
          [optional] memory.ingest_scene_state(scene_state)

        Parameters
        ----------
        image_paths      : RGB image file paths from the current timestep.
                           Typically one image from the robot's wrist/head camera.
        candidate_labels : hint vocabulary for the VLM (e.g., task-relevant objects).
                           Ranked by historical confidence; helps reduce hallucination.
        current_location : semantic robot location (e.g., "table", "shelf", "bin").
        scene_state      : spine brain scene_state dict, if available:
                           {cube_pose, target_pose, ee_pose, gripper_width,
                            robot_joint_pos, step_index}
                           When provided, robot proprioceptive state is also updated.
        save_vlm_memory  : persist VLM-level long-term memory JSON after this batch.

        Returns
        -------
        List[RecognitionResult] — raw VLM outputs, available for downstream inspection.
        """
        self._step += 1

        results = self.adapter.perceive(
            image_paths=list(image_paths),
            candidate_labels=candidate_labels,
            current_location=current_location,
            save_memory=save_vlm_memory,
        )

        if scene_state:
            self.memory.ingest_scene_state(scene_state)

        return results

    def get_planning_context(
        self,
        top_k_episodes: int = 3,
        include_preferences: bool = True,
    ) -> PlanningContext:
        """
        Get the complete memory context bundle for the planning module.

        Call this after process_perception() each step. The returned
        PlanningContext aggregates all three memory layers into one object.

        Parameters
        ----------
        top_k_episodes     : max number of similar past episodes to retrieve
        include_preferences: include semantic user-preference lookups

        Returns
        -------
        PlanningContext with:
          - visible_objects, scene_description, object_attributes (perception)
          - task_success_rate, recommended_skills, common_steps (semantic)
          - similar_episodes (episodic)
          - object_location_priors, user_preferences (semantic)
          - recent_actions (working memory)
        """
        return self.planning.get_planning_context(
            task_instruction=self._current_task,
            top_k_episodes=top_k_episodes,
            include_preferences=include_preferences,
        )

    def record_action(
        self,
        action: str,
        action_id: Any = None,
        success: Optional[bool] = None,
        feedback: str = "",
        reasoning: str = "",
    ) -> None:
        """
        Record an action and its outcome into working memory.

        Call this after each action the planner dispatches, whether from
        a VLM-generated plan, a Blueprint, or a heuristic.

        Parameters
        ----------
        action    : action description (e.g., "grasp", "move_to(table)", "open_gripper")
        action_id : optional numeric or string ID (e.g., discrete action index)
        success   : True/False, or None if outcome is not yet known
        feedback  : raw feedback text from environment or execution engine
        reasoning : planner reasoning text (stored for introspection)
        """
        self.planning.record_action_result(
            action=action,
            action_id=action_id,
            success=success,
            feedback=feedback,
            reasoning=reasoning,
        )

    def update_robot_state(
        self,
        gripper_pose: Optional[np.ndarray] = None,
        joint_angles: Optional[np.ndarray] = None,
        force_torque: Optional[np.ndarray] = None,
        gripper_aperture: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update robot proprioceptive state in working memory.

        For real-robot deployments: call this whenever fresh sensor data arrives.
        Not required in simulation (all fields may remain None).

        Parameters
        ----------
        gripper_pose    : 7D array [x, y, z, qw, qx, qy, qz]
        joint_angles    : (DOF,) joint position array
        force_torque    : 6D array [Fx, Fy, Fz, Tx, Ty, Tz]
        gripper_aperture: scalar in [0.0, 1.0]
        timestamp       : Unix timestamp of sensor reading
        """
        self.memory.update_robot_state(
            gripper_pose=gripper_pose,
            joint_angles=joint_angles,
            force_torque=force_torque,
            gripper_aperture=gripper_aperture,
            timestamp=timestamp,
        )

    def record_blueprint_execution(
        self,
        blueprint_skills: List[str],
        success: bool,
        scene_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a completed spine brain Blueprint execution.

        Updates TaskSchema in semantic memory so future VLM Brain calls can
        reference a known-good skill ordering.

        Call this when franka_state_machine_cerebellum finishes execution.

        Parameters
        ----------
        blueprint_skills : ordered list of skill names that were executed
                           e.g. ["move_above", "descend", "grasp", "lift", "place", "retreat"]
        success          : whether the full blueprint completed successfully
        scene_state      : final scene_state dict from the spine brain (optional)
        """
        self.memory.record_blueprint_execution(
            task_instruction=self._current_task,
            blueprint_skills=blueprint_skills,
            success=success,
            scene_state=scene_state,
        )

    def set_task_constraints(self, constraints: Dict[str, Any]) -> None:
        """
        Inject task-specific constraints into working memory.

        Call this after begin_episode() when the task rules are known.
        The constraints will appear in ctx.constraints on every subsequent
        get_planning_context() call.

        Parameters
        ----------
        constraints : dict, e.g.:
            {
                "category_rules": {
                    "食品类": "篮子一",
                    "杯具类": "篮子二",
                    "工具类": "篮子三",
                },
                "no_category_mixing": True,
                "collision_avoidance": True,
            }
        """
        existing = self.memory.working.goal.intent.get("constraints", {})
        existing.update(constraints)
        self.memory.working.goal.intent["constraints"] = existing

    def record_occlusion(
        self,
        blocked_object: str,
        blocker_object: str,
        blocker_on_collection_list: bool = False,
        blocker_target_basket: str = "",
    ) -> None:
        """
        Record an occlusion detected by the Simulation module.

        Writes into working memory milestones so the next get_planning_context()
        call includes this occlusion in ctx.constraints["occlusions"].

        The planning module (PM) then reads ctx.constraints["occlusions"] to
        decide whether to move the blocker directly to its target basket
        (blocker_on_collection_list=True) or temporarily set it aside.

        Parameters
        ----------
        blocked_object           : the object whose path is blocked
        blocker_object           : the object causing the obstruction
        blocker_on_collection_list : True if the blocker is also in the task's
                                    collection list (→ PM should place it directly)
        blocker_target_basket    : target basket for the blocker (if on list)
        """
        tag = (
            f"occlusion:blocked={blocked_object},"
            f"blocker={blocker_object},"
            f"blocker_on_list={blocker_on_collection_list},"
            f"target={blocker_target_basket}"
        )
        self.memory.working.clock.add_milestone(tag)

    def clear_occlusions(self, blocked_object: str = "") -> None:
        """
        Remove occlusion records from working memory milestones.

        Call this when an occlusion is resolved (the blocker has been moved).
        The next export_planning_input() call will no longer include the
        cleared occlusion in constraints["occlusions"].

        Parameters
        ----------
        blocked_object : if given, only remove occlusions where
                         blocked == this object (targeted clear).
                         if empty string (default), clear ALL occlusion records.

        Examples
        --------
        # 清除所有遮挡记录
        pipeline.clear_occlusions()

        # 只清除"香蕉"的遮挡记录（大食品盒已移走）
        pipeline.clear_occlusions(blocked_object="香蕉")
        """
        milestones = self.memory.working.clock.milestones
        if blocked_object:
            keep = [
                m for m in milestones
                if not (
                    isinstance(m, str)
                    and m.startswith("occlusion:")
                    and f"blocked={blocked_object}" in m
                )
            ]
        else:
            keep = [
                m for m in milestones
                if not (isinstance(m, str) and m.startswith("occlusion:"))
            ]
        milestones.clear()
        milestones.extend(keep)

    def export_planning_input(self, output_path: str) -> Dict[str, Any]:
        """
        Write the three planning-essential fields to a JSON file.

        Call this after process_perception() each step.
        The planning module reads this file as its sole input from memory.

        Output JSON:
            {
                "task_instruction": "...",
                "manipulable_objects": {"香蕉": ["grasp", "place"], ...},
                "available_skills":    ["move_above", "descend", ...],
                "constraints": { "category_rules": {...}, "occlusions": [...] }
            }

        Parameters
        ----------
        output_path : e.g. "run_001/planning_input.json"

        Returns
        -------
        The dict that was written.
        """
        return self.planning.export_planning_input(self._current_task, output_path)

    def export_context_for_vlm(self, output_path: str) -> Dict[str, Any]:
        """
        Export memory context to JSON for the VLM Brain.

        Write this file before invoking brain_pipeline.py or run_vlm_inference.py.
        The VLM Brain reads this file alongside scene_state.json to produce
        an informed Blueprint.

        Parameters
        ----------
        output_path : target path, e.g. "run_001/memory_context.json"

        Returns
        -------
        The context dict that was written.
        """
        return self.memory.export_vlm_context(self._current_task, output_path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Full JSON-serializable snapshot of current memory state (debug)."""
        return self.memory.snapshot()

    def get_last_perception_results(self) -> List[Any]:
        """Return the RecognitionResult list from the most recent process_perception()."""
        return self.adapter.last_results

    @property
    def step(self) -> int:
        """Number of process_perception() calls since begin_episode()."""
        return self._step

    @property
    def current_task(self) -> str:
        """Natural language task instruction for the current episode."""
        return self._current_task

    @property
    def current_scene_id(self) -> str:
        """Scene ID for the current episode."""
        return self._current_scene_id
