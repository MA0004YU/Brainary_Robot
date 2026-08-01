"""
PlanningMemoryInterface: read-only memory query interface for the planning module.

Design goals:
  1. The planning module never imports from embodiedbench.memory_manip directly.
  2. PlanningContext is a plain dataclass — no memory-system coupling.
  3. The interface is safe to call at any time; it never modifies persistent state
     (except via explicit record_action_result() calls).

Typical usage inside the planning module:
    # Receive the interface object from the pipeline (dependency injection)
    ctx: PlanningContext = planning_iface.get_planning_context("pick up the cube")

    # Use ctx fields to build a plan
    print(ctx.to_prompt_text())

    # After each action:
    planning_iface.record_action_result("grasp", success=True, feedback="closed")

    # Export for VLM Brain:
    planning_iface.export_context_for_vlm("pick up the cube", "run_001/memory_context.json")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    _HAS_MEMORY = True
except ImportError:
    EmbodiedManipulationMemorySystem = None  # type: ignore
    _HAS_MEMORY = False


# Spine brain 定义的完整 skillset（来自 VLM_BRAIN_INTERFACE.md）
DEFAULT_AVAILABLE_SKILLS: List[str] = [
    "move_above",
    "descend",
    "grasp",
    "lift",
    "place",
    "retreat",
    "wait",
    "align_orientation",
    "reach",
]

# 物体没有历史 affordance 记录时的默认值
DEFAULT_AFFORDANCES: List[str] = ["grasp", "place"]


# ---------------------------------------------------------------------------
# PlanningContext dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlanningContext:
    """
    Complete memory context bundle for the planning module.

    This dataclass is the single structured artifact that the planning module
    receives from the memory system. It contains everything needed to produce
    a plan without the planner touching memory internals.

    Fields
    ------
    task_instruction     : natural language instruction for the current episode
    task_type            : inferred task category (e.g., "pick_and_place")

    visible_objects      : object names detected by perception this timestep
    scene_description    : compact text summary from VLM perception
    object_attributes    : {obj_name: {color, shape, texture, confidence, ...}}

    memory_objects       : objects known from working memory (may not be visible)
    current_location     : semantic robot location (e.g., "table")
    visited_locations    : {location: status} map of explored locations
    held_object          : object currently in gripper, or None

    task_success_rate    : historical success rate for this task type (0.0–1.0)
    common_steps         : most common action sequence for this task type
    recommended_skills   : skill names from best-performing Blueprint (spine brain)

    object_location_priors : {obj_name: [location, ...]} ordered by frequency
    user_preferences       : {functional_need: {preferred: [...], avoided: [...]}}

    similar_episodes     : lightweight summaries of up to K similar past episodes
    recent_actions       : last ≤10 action records from the current episode

    ── 新增三项 ─────────────────────────────────────────────────────────────

    manipulable_objects  : {obj_name: [affordance, ...]} 当前可见/已知物体的可操作能力
                           e.g. {"香蕉": ["grasp", "place"], "剪刀": ["grasp", "place"]}
                           来源：语义记忆 ObjectKB.affordances（历史积累），
                                 无历史时默认 ["grasp", "place"]

    available_skills     : 当前可调用的完整 skill 列表
                           e.g. ["move_above", "descend", "grasp", "lift", "place", ...]
                           来源：脊脑 Blueprint 接口定义（静态）+
                                 语义记忆 TaskSchema 中历史出现过的 skills（动态补充）

    constraints          : 物理规则与环境约束字典，包含：
                           - category_rules   : {类别: 目标篮子}，任务开始时由 Planner 注入
                           - no_category_mixing : bool，是否禁止跨类别混放
                           - collision_avoidance : bool，是否启用碰撞检测
                           - occlusions       : [{blocked, blocker, blocker_on_list, target}]
                                               Simulation 模块检测到遮挡后写入
                           来源：working memory goal.intent["constraints"]（Planner 注入）
                                 + working memory clock.milestones 中的 occlusion: 记录
                                   （Simulation 通过 pipeline.record_occlusion() 写入）
    """

    task_instruction: str = ""
    task_type: str = ""

    # Perception outputs (current timestep)
    visible_objects: List[str] = field(default_factory=list)
    scene_description: str = ""
    object_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Working memory: objects known from prior steps
    memory_objects: Dict[str, Any] = field(default_factory=dict)

    # Navigation context
    current_location: str = "unknown"
    visited_locations: Dict[str, str] = field(default_factory=dict)
    held_object: Optional[str] = None

    # Semantic memory: task history
    task_success_rate: Optional[float] = None
    common_steps: List[str] = field(default_factory=list)
    recommended_skills: List[str] = field(default_factory=list)

    # Semantic memory: spatial priors
    object_location_priors: Dict[str, List[str]] = field(default_factory=dict)

    # Semantic memory: user preferences
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    # Episodic memory: similar past episodes
    similar_episodes: List[Dict[str, Any]] = field(default_factory=list)

    # Working memory: recent episode actions
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)

    # ── 三个新增字段 ──────────────────────────────────────────────────────

    # 可操作的物体（带 affordance 标注）
    # 来源：语义记忆 ObjectKB，无历史时默认 ["grasp", "place"]
    manipulable_objects: Dict[str, List[str]] = field(default_factory=dict)

    # 当前完整可用 skillset
    # 来源：脊脑接口静态定义 + 语义记忆 TaskSchema 动态补充
    available_skills: List[str] = field(default_factory=list)

    # 物理规则与环境约束
    # 来源：working memory intent["constraints"]（Planner 注入）
    #       + clock milestones 中的 occlusion: 条目（Simulation 写入）
    constraints: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str) -> None:
        """Atomically save context to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(p)

    def to_prompt_text(self) -> str:
        """
        Convert context to human-readable text suitable for VLM prompt injection.
        """
        lines = [
            f"Task: {self.task_instruction}",
            f"Task Type: {self.task_type or 'unknown'}",
            "",
            f"Visible Objects: {', '.join(self.visible_objects) or 'none'}",
            f"Current Location: {self.current_location}",
            f"Held Object: {self.held_object or 'none'}",
        ]

        # 可操作物体 + affordances
        if self.manipulable_objects:
            lines.append("Manipulable Objects (with affordances):")
            for obj, affs in list(self.manipulable_objects.items())[:8]:
                lines.append(f"  {obj}: [{', '.join(affs)}]")

        # 可用 skillset
        if self.available_skills:
            lines.append(
                f"Available Skills: {', '.join(self.available_skills)}"
            )

        # 物理约束
        if self.constraints:
            lines.append("Constraints:")
            cat_rules = self.constraints.get("category_rules", {})
            if cat_rules:
                lines.append("  Category Rules:")
                for cat, basket in cat_rules.items():
                    lines.append(f"    {cat} → {basket}")
            if self.constraints.get("no_category_mixing"):
                lines.append("  No category mixing allowed across baskets.")
            if self.constraints.get("collision_avoidance"):
                lines.append("  Collision avoidance is active.")
            occlusions = self.constraints.get("occlusions", [])
            if occlusions:
                lines.append(f"  Detected Occlusions ({len(occlusions)}):")
                for oc in occlusions:
                    on_list = oc.get("blocker_on_list", False)
                    action = f"→ move directly to {oc.get('target')}" if on_list else "→ temporarily move aside"
                    lines.append(
                        f"    {oc.get('blocker')} blocks {oc.get('blocked')} {action}"
                    )

        if self.recommended_skills:
            lines.append(
                f"Recommended Skill Sequence: {' → '.join(self.recommended_skills)}"
            )

        if self.task_success_rate is not None:
            lines.append(
                f"Historical Success Rate: {self.task_success_rate:.1%}"
            )

        if self.object_location_priors:
            lines.append("Known Object Locations:")
            for obj, locs in list(self.object_location_priors.items())[:5]:
                lines.append(f"  {obj}: likely at {', '.join(str(l) for l in locs[:2])}")

        if self.similar_episodes:
            lines.append(
                f"Similar Past Episodes ({len(self.similar_episodes)} found):"
            )
            for ep in self.similar_episodes[:2]:
                status = "SUCCESS" if ep.get("success") else "FAILED"
                task_str = ep.get("task_instruction", ep.get("task", ""))[:60]
                skills = ep.get("blueprint_skills") or ep.get("objects_encountered", [])
                lines.append(f"  [{status}] {task_str}")
                if isinstance(skills, list) and skills:
                    lines.append(f"    skills: {' → '.join(str(s) for s in skills[:6])}")

        if self.recent_actions:
            n = min(3, len(self.recent_actions))
            lines.append(f"Recent Actions (last {n}):")
            for act in self.recent_actions[-n:]:
                status = "ok" if act.get("success") else "fail"
                lines.append(
                    f"  step {act.get('step')}: {act.get('action')} [{status}]"
                    + (f" — {act.get('feedback', '')[:40]}" if act.get("feedback") else "")
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_occlusion_milestone(tag: str) -> Dict[str, Any]:
    """Parse 'occlusion:blocked=X,blocker=Y,blocker_on_list=True,target=Z' into dict."""
    result: Dict[str, Any] = {}
    content = tag[len("occlusion:"):]
    for part in content.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.lower() in ("true", "false"):
                result[k] = v.lower() == "true"
            else:
                result[k] = v
    return result


# ---------------------------------------------------------------------------
# PlanningMemoryInterface
# ---------------------------------------------------------------------------

class PlanningMemoryInterface:
    """
    Read-only interface into the three-layer memory system for the planning module.

    The planning module receives this object via dependency injection from
    PerceptionMemoryPipeline and calls it to get context, record actions, and
    export VLM-ready JSON files.

    Methods
    -------
    get_planning_context(task_instruction)  — primary: build full PlanningContext
    get_object_info(obj_name)               — object affordances + location history
    get_user_preferences(need)              — user preferences for a functional need
    record_action_result(action, ...)       — record executed action outcome
    export_context_for_vlm(task, path)      — write memory_context.json for VLM Brain
    get_memory_snapshot()                   — full debug snapshot
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    # ------------------------------------------------------------------
    # Primary planning query
    # ------------------------------------------------------------------

    def get_planning_context(
        self,
        task_instruction: str,
        top_k_episodes: int = 3,
        include_preferences: bool = True,
    ) -> PlanningContext:
        """
        Build a complete PlanningContext from the current memory state.

        This is the primary method the planning module should call each step.
        It aggregates data from all three memory layers into one structured object.

        Parameters
        ----------
        task_instruction   : natural language task for this episode
        top_k_episodes     : number of similar past episodes to retrieve
        include_preferences: whether to include user preference lookups

        Returns
        -------
        PlanningContext populated from working, episodic, and semantic memory
        """
        wm = self._memory.working

        # Semantic: task schema (success rate, common steps, blueprint skills)
        schema = self._memory.query_task_schema(task_instruction)

        # Episodic: similar past episodes
        similar = self._memory.query_similar_episodes(task_instruction, top_k=top_k_episodes)

        # All known object names (visible + in memory)
        all_object_names = list(dict.fromkeys(
            list(wm.observation.visible_objects)
            + list(wm.observation.memory_objects.keys())
        ))

        # Semantic: object location priors
        object_priors: Dict[str, List[str]] = {}
        for obj in all_object_names:
            kb = self._memory.query_object(obj)
            likely = [loc for loc, _ in kb.get("likely_locations", [])]
            if likely:
                object_priors[obj] = likely[:3]

        # User preferences for a subset of visible objects
        preferences: Dict[str, Any] = {}
        if include_preferences:
            for obj in all_object_names[:5]:
                pref = self._memory.query_user_preference(obj)
                if pref.get("preferred") or pref.get("avoided"):
                    preferences[obj] = pref

        # Action buffer → recent_actions (last 10)
        recent_actions = [
            {
                "step": r.step,
                "action": r.action,
                "action_id": r.action_id,
                "success": r.success,
                "feedback": r.feedback,
                "reasoning": r.reasoning,
            }
            for r in list(wm.action_buffer.records)[-10:]
        ]

        # ── 新增字段 1：manipulable_objects ─────────────────────────────
        # 查询语义记忆中每个物体的 affordances
        manipulable_objects: Dict[str, List[str]] = {}
        for obj in all_object_names:
            kb = self._memory.query_object(obj)
            affordances = kb.get("affordances", [])
            manipulable_objects[obj] = affordances if affordances else list(DEFAULT_AFFORDANCES)

        # ── 新增字段 2：available_skills ─────────────────────────────────
        # 脊脑静态 skillset + 语义记忆中历史出现过的 skills（动态补充）
        history_skills: List[str] = schema.get("blueprint_skills", [])
        available_skills = list(dict.fromkeys(history_skills + DEFAULT_AVAILABLE_SKILLS))

        # ── 新增字段 3：constraints ──────────────────────────────────────
        # 从 working memory goal.intent["constraints"] 读取（Planner 注入）
        constraints: Dict[str, Any] = dict(
            wm.goal.intent.get("constraints", {})
        )
        # 从 clock milestones 中收集 Simulation 写入的遮挡记录
        occlusion_records = [
            _parse_occlusion_milestone(m)
            for m in list(wm.clock.milestones)
            if isinstance(m, str) and m.startswith("occlusion:")
        ]
        if occlusion_records:
            constraints["occlusions"] = occlusion_records

        return PlanningContext(
            task_instruction=task_instruction,
            task_type=schema.get("task_type", ""),
            visible_objects=list(wm.observation.visible_objects),
            scene_description=wm.observation.text or "",
            object_attributes=dict(wm.observation.memory_objects),
            memory_objects=dict(wm.observation.memory_objects),
            current_location=wm.observation.current_location or "unknown",
            visited_locations=dict(wm.observation.visited_locations),
            held_object=wm.observation.held_object,
            task_success_rate=schema.get("success_rate"),
            common_steps=schema.get("common_steps", []),
            recommended_skills=schema.get("blueprint_skills", []),
            object_location_priors=object_priors,
            user_preferences=preferences,
            similar_episodes=similar,
            recent_actions=recent_actions,
            manipulable_objects=manipulable_objects,
            available_skills=available_skills,
            constraints=constraints,
        )

    # ------------------------------------------------------------------
    # Focused semantic queries
    # ------------------------------------------------------------------

    def get_object_info(self, obj_name: str) -> Dict[str, Any]:
        """
        Query object affordances and historical location frequencies.

        Returns dict with keys:
          affordances      : list[str]  — e.g., ["grasp", "place"]
          likely_locations : [(location, frequency), ...]
          grasp_count      : int
          total_count      : int
        """
        return self._memory.query_object(obj_name)

    def get_user_preferences(self, need: str) -> Dict[str, Any]:
        """Query user preferences for a functional need."""
        return self._memory.query_user_preference(need)

    def get_task_schema(self, task_instruction: str) -> Dict[str, Any]:
        """Query historical task execution statistics."""
        return self._memory.query_task_schema(task_instruction)

    def get_similar_episodes(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve lightweight summaries of similar past episodes."""
        return self._memory.query_similar_episodes(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Write-back: action recording
    # ------------------------------------------------------------------

    def record_action_result(
        self,
        action: str,
        action_id: Any = None,
        success: Optional[bool] = None,
        feedback: str = "",
        reasoning: str = "",
    ) -> None:
        """Record an executed action and its outcome into working memory."""
        self._memory.record_action(
            action=action,
            action_id=action_id,
            success=success,
            feedback=feedback,
            reasoning=reasoning,
        )

    def record_goal_intent(self, intent: Dict[str, Any]) -> None:
        """Push GoalReasoner output into working memory goal slot."""
        self._memory.record_goal_intent(intent)

    # ------------------------------------------------------------------
    # Planning module JSON export
    # ------------------------------------------------------------------

    def export_planning_input(
        self,
        task_instruction: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Write the three planning-essential fields to a JSON file.

        Output format:
            {
                "task_instruction": "...",
                "manipulable_objects": {"香蕉": ["grasp", "place"], ...},
                "available_skills":    ["move_above", "descend", ...],
                "constraints": {
                    "category_rules":      {"食品类": "篮子一", ...},
                    "no_category_mixing":  true,
                    "collision_avoidance": true,
                    "occlusions": [
                        {"blocked": "香蕉", "blocker": "大食品盒",
                         "blocker_on_list": true, "target": "篮子三"}
                    ]
                }
            }

        Parameters
        ----------
        task_instruction : natural language task for this episode
        output_path      : file path to write, e.g. "run_001/planning_input.json"

        Returns
        -------
        The dict that was written (for in-process use as well).
        """
        ctx = self.get_planning_context(task_instruction)
        payload = {
            "task_instruction": ctx.task_instruction,
            "manipulable_objects": ctx.manipulable_objects,
            "available_skills": ctx.available_skills,
            "constraints": ctx.constraints,
        }
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)
        return payload

    # ------------------------------------------------------------------
    # VLM Brain export
    # ------------------------------------------------------------------

    def export_context_for_vlm(
        self,
        task_instruction: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Write memory context to a JSON file for the VLM Brain to consume."""
        return self._memory.export_vlm_context(task_instruction, output_path)

    # ------------------------------------------------------------------
    # Debug / monitoring
    # ------------------------------------------------------------------

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Full JSON-serializable snapshot of the current memory state."""
        return self._memory.snapshot()
