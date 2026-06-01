"""Skill plan data structures for the Franka state-machine cerebellum.

This module is intentionally independent from Isaac Lab so JSON skill plans can
be parsed and validated before launching or stepping a simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class SkillCommand:
    """One skill invocation from a high-level JSON plan."""

    skill: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillCommand":
        if "skill" not in data:
            raise ValueError("Skill command is missing required field 'skill'.")
        if "target" not in data:
            raise ValueError("Skill command is missing required field 'target'.")
        return cls(skill=str(data["skill"]), target=str(data["target"]), params=dict(data.get("params", {})))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillPlan:
    """Episode-level skill plan loaded from JSON."""

    task: str
    episode_id: str
    skill_plan: list[SkillCommand]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillPlan":
        commands = [SkillCommand.from_dict(item) for item in data.get("skill_plan", [])]
        if not commands:
            raise ValueError("Skill plan must contain at least one skill command.")
        return cls(
            task=str(data.get("task", "")),
            episode_id=str(data.get("episode_id", "episode_000000")),
            skill_plan=commands,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SkillPlan":
        with Path(path).expanduser().open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def with_episode_id(self, episode_id: str) -> "SkillPlan":
        return SkillPlan(task=self.task, episode_id=episode_id, skill_plan=list(self.skill_plan))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "episode_id": self.episode_id,
            "skill_plan": [command.to_dict() for command in self.skill_plan],
        }


@dataclass
class SkillResult:
    """Execution result for a single skill."""

    skill: str
    target: str
    params: dict[str, Any]
    start_step: int
    end_step: int
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    success: bool
    failure_reason: str | None
    num_steps: int
    state_sequence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
