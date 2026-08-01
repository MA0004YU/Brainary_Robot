from __future__ import annotations

from typing import Any, Dict, Optional

from .types import ActionNode, ActionResult, PMConfig


class SimActionExecutor:
    def __init__(self, sim: Any | None = None, config: PMConfig | None = None):
        self.sim = sim
        self.config = config or PMConfig()
        self.held_object: Optional[str] = None

    def execute(self, node: ActionNode) -> ActionResult:
        if self.config.dry_run or self.sim is None:
            return ActionResult(
                id=node.id,
                action=node.action,
                target=node.target,
                ok=True,
                reason="",
                steps=0,
                timed_out=False,
                skill_result={"dry_run": True, "resources": list(node.resources)},
                feedback="dry-run: 未调用仿真技能",
            )

        attempts = 0
        last_result: Dict[str, Any] = {}
        last_feedback = ""
        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                result = self._call_sim(node)
            except Exception as exc:
                result = {"ok": False, "reason": str(exc), "exception": exc.__class__.__name__}
            ok = bool(result.get("ok", False))
            last_result = result
            last_feedback = self._feedback_from_result(result)
            if ok:
                self._update_state(node)
                return ActionResult(
                    id=node.id,
                    action=node.action,
                    target=node.target,
                    ok=True,
                    reason="",
                    steps=int(result.get("steps", 0)),
                    timed_out=bool(result.get("timed_out", False)),
                    attempts=attempts,
                    skill_result=result,
                    feedback=last_feedback,
                )
        return ActionResult(
            id=node.id,
            action=node.action,
            target=node.target,
            ok=False,
            reason=str(last_result.get("reason", last_feedback)),
            steps=int(last_result.get("steps", 0)),
            timed_out=bool(last_result.get("timed_out", False)),
            attempts=attempts,
            skill_result=last_result,
            feedback=last_feedback,
        )

    def _call_sim(self, node: ActionNode) -> Dict[str, Any]:
        action = node.action
        target = node.target
        max_steps = self._max_steps_for(node)
        if action in {"grasp", "place", "open_drawer", "close_drawer", "open_door", "close_door"}:
            if target is None:
                raise ValueError(f"动作 {node.id}:{action} 缺少 target")
            self._validate_target(action, target)
            return getattr(self.sim, action)(target, max_steps=max_steps)
        if action == "operate_coffee":
            self._validate_target(action, target)
            return self.sim.operate_coffee(max_steps=max_steps)
        if action == "go_home":
            return self.sim.go_home(max_steps=max_steps)
        if action == "open_gripper":
            return self.sim.open_gripper()
        if action == "close_gripper":
            return self.sim.close_gripper()
        if action == "wait":
            return {"ok": True, "skill": "wait", "reason": ""}
        if hasattr(self.sim, action):
            method = getattr(self.sim, action)
            if target is None:
                return method()
            return method(target)
        if self.config.allow_unknown_actions:
            return {"ok": True, "skill": action, "target": target, "reason": "unknown action skipped by config"}
        raise ValueError(f"未知动作 {action!r}，无法映射到 SimInterface")

    def _max_steps_for(self, node: ActionNode) -> int:
        if "max_steps" in node.metadata:
            return int(node.metadata["max_steps"])
        if node.action in {"open_drawer", "close_drawer", "open_door", "close_door", "operate_coffee"}:
            return 15000
        if node.action == "go_home":
            return 1500
        return int(self.config.max_steps_per_action)

    def _validate_target(self, action: str, target: Optional[str]) -> None:
        if self.sim is None:
            return
        target_map = {
            "grasp": getattr(self.sim, "list_graspable", None),
            "place": getattr(self.sim, "list_place_baskets", None),
            "open_drawer": getattr(self.sim, "list_drawers", None),
            "close_drawer": getattr(self.sim, "list_drawers", None),
            "open_door": getattr(self.sim, "list_doors", None),
            "close_door": getattr(self.sim, "list_doors", None),
        }
        if action == "operate_coffee":
            skills = self.sim.list_skills() if hasattr(self.sim, "list_skills") else []
            available = any(skill.get("skill") == "operate_coffee" for skill in skills)
            if not available:
                raise ValueError("operate_coffee 当前场景不可用；请以 sim.list_skills() 返回为准")
            return
        getter = target_map.get(action)
        if getter is None:
            return
        valid = list(getter())
        if target not in valid:
            raise ValueError(f"{action}: 非法目标 {target!r}; 可选 {valid}")

    def _update_state(self, node: ActionNode) -> None:
        if node.action == "grasp":
            self.held_object = node.target
        elif node.action in {"place", "open_gripper"}:
            self.held_object = None

    @staticmethod
    def _feedback_from_result(result: Dict[str, Any]) -> str:
        for key in ("reason", "failure_reason", "status"):
            value = result.get(key)
            if value:
                return str(value)
        return "" if result.get("ok") else "执行失败但未返回原因"
