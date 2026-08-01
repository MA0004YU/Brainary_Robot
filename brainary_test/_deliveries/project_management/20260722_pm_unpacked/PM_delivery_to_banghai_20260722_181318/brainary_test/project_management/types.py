from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PMConfig:
    max_concurrent_workers: int = 4
    default_duration: float = 1.0
    max_steps_per_action: int = 8000
    stop_on_failure: bool = True
    max_retries: int = 0
    dry_run: bool = True
    append_go_home: bool = False
    allow_unknown_actions: bool = False


@dataclass
class ActionNode:
    id: str
    action: str
    target: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    duration: float = 1.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], default_duration: float = 1.0) -> "ActionNode":
        metadata = {k: v for k, v in raw.items() if k not in {
            "id", "action", "target", "depends_on", "resources", "duration", "priority"
        }}
        return cls(
            id=str(raw.get("id", "")).strip(),
            action=str(raw.get("action", "")).strip(),
            target=raw.get("target"),
            depends_on=[str(v) for v in raw.get("depends_on", [])],
            resources=[str(v) for v in raw.get("resources", [])],
            duration=float(raw.get("duration", default_duration)),
            priority=int(raw.get("priority", 0)),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.target is None:
            data.pop("target", None)
        if not self.resources:
            data.pop("resources", None)
        if not self.metadata:
            data.pop("metadata", None)
        return data


@dataclass
class ScheduledAction:
    node: ActionNode
    wave: int
    start_time: float
    end_time: float
    parallel_group: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = self.node.to_dict()
        data.update({
            "wave": self.wave,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "parallel_group": list(self.parallel_group),
        })
        return data


@dataclass
class ActionResult:
    """单步执行结果，字段对齐 API.md §4 返回结构。"""
    id: str
    action: str
    target: Optional[str]
    ok: bool
    reason: str = ""
    steps: int = 0
    timed_out: bool = False
    attempts: int = 1
    skill_result: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionReport:
    ok: bool
    mode: str
    scheduled_plan: List[Dict[str, Any]] = field(default_factory=list)
    executed: List[Dict[str, Any]] = field(default_factory=list)
    failed: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
