from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .executor import SimActionExecutor
from .optimizer import PlanOptimizer
from .scheduler import ParallelScheduler, group_by_wave
from .types import ExecutionReport, PMConfig


class ProjectManager:
    def __init__(
        self,
        sim: Any | None = None,
        config: PMConfig | None = None,
        object_aliases: Optional[Dict[str, str]] = None,
        object_alias_path: str | Path | None = None,
    ):
        self.config = config or PMConfig()
        self.optimizer = PlanOptimizer(object_aliases=object_aliases, alias_path=object_alias_path)
        self.scheduler = ParallelScheduler(self.config)
        self.executor = SimActionExecutor(sim=sim, config=self.config)

    def load_plan(self, path: str | Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            return data["steps"]
        if not isinstance(data, list):
            raise ValueError(f"plan 文件必须是 list 或包含 steps:list，实际是 {type(data).__name__}")
        return data

    def load_planning_input(self, path: str | Path | None) -> Optional[Dict[str, Any]]:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("planning_input 必须是 JSON object")
        return data

    def prepare_plan(
        self,
        plan: Sequence[Dict[str, Any]],
        planning_input: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        optimized = self.optimizer.normalize_plan(plan, planning_input=planning_input)
        if self.config.append_go_home and not any(item.get("action") == "go_home" for item in optimized):
            dependencies = [str(item.get("id")) for item in optimized if item.get("id")]
            optimized.append({"id": "PM_GO_HOME", "action": "go_home", "depends_on": dependencies})
        return optimized

    def schedule_plan(
        self,
        plan: Sequence[Dict[str, Any]],
        planning_input: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        prepared = self.prepare_plan(plan, planning_input=planning_input)
        return [item.to_dict() for item in self.scheduler.schedule(prepared)]

    def run(
        self,
        plan: Sequence[Dict[str, Any]],
        planning_input: Optional[Dict[str, Any]] = None,
    ) -> ExecutionReport:
        prepared = self.prepare_plan(plan, planning_input=planning_input)
        scheduled = self.scheduler.schedule(prepared)
        completed = set()
        failed = set()
        executed: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        feedback: List[str] = []

        for wave in group_by_wave(scheduled):
            for item in wave:
                node = item.node
                missing = [dep for dep in node.depends_on if dep not in completed]
                blocked = [dep for dep in node.depends_on if dep in failed]
                if missing or blocked:
                    skipped.append({
                        "id": node.id,
                        "action": node.action,
                        "target": node.target,
                        "reason": f"dependency_not_completed={missing}, dependency_failed={blocked}",
                    })
                    continue
                result = self.executor.execute(node)
                result_dict = result.to_dict()
                result_dict["wave"] = item.wave
                if result.ok:
                    completed.add(node.id)
                    executed.append(result_dict)
                else:
                    failed.add(node.id)
                    failures.append(result_dict)
                    msg = f"{node.id}:{node.action}({node.target}) failed: {result.feedback}"
                    feedback.append(msg)
                    if self.config.stop_on_failure:
                        self._skip_remaining_after_failure(scheduled, completed, failed, skipped)
                        return ExecutionReport(
                            ok=False,
                            mode="dry_run" if self.config.dry_run else "execute",
                            scheduled_plan=[item.to_dict() for item in scheduled],
                            executed=executed,
                            failed=failures,
                            skipped=skipped,
                            feedback=feedback,
                        )
        return ExecutionReport(
            ok=not failures,
            mode="dry_run" if self.config.dry_run else "execute",
            scheduled_plan=[item.to_dict() for item in scheduled],
            executed=executed,
            failed=failures,
            skipped=skipped,
            feedback=feedback,
        )

    def execute_plan(
        self,
        plan: Sequence[Dict[str, Any]],
        planning_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.run(plan, planning_input=planning_input).to_dict()

    def execute_plan_file(
        self,
        plan_path: str | Path,
        planning_input_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        return self.run_files(plan_path, planning_input_path=planning_input_path, output_path=output_path).to_dict()

    def run_files(
        self,
        plan_path: str | Path,
        planning_input_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> ExecutionReport:
        plan = self.load_plan(plan_path)
        planning_input = self.load_planning_input(planning_input_path)
        report = self.run(plan, planning_input=planning_input)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _skip_remaining_after_failure(scheduled, completed, failed, skipped) -> None:
        for item in scheduled:
            node = item.node
            if node.id in completed or node.id in failed:
                continue
            skipped.append({
                "id": node.id,
                "action": node.action,
                "target": node.target,
                "reason": "stop_on_failure",
            })
