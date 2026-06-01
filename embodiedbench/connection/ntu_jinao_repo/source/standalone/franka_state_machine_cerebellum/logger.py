"""Structured JSONL logging for stage-1 Franka state-machine episodes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json


class StateMachineDatasetLogger:
    """Writes episode-level and optional trajectory-level JSONL datasets."""

    def __init__(self, output_dir: str | Path, save_trajectory: bool = True):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.save_trajectory = save_trajectory
        self.trajectory_dir = self.output_dir / "trajectories"
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_trajectory:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._episode_file = self.episodes_path.open("w", encoding="utf-8")
        self._trajectory_file = None
        self._trajectory_path: Path | None = None

    def close(self) -> None:
        if self._trajectory_file is not None:
            self._trajectory_file.close()
            self._trajectory_file = None
        self._episode_file.close()

    def start_trajectory(self, episode_id: str) -> str | None:
        if not self.save_trajectory:
            self._trajectory_path = None
            return None
        if self._trajectory_file is not None:
            self._trajectory_file.close()
        self._trajectory_path = self.trajectory_dir / f"{episode_id}.jsonl"
        self._trajectory_file = self._trajectory_path.open("w", encoding="utf-8")
        return self._trajectory_path.relative_to(self.output_dir).as_posix()

    def log_trajectory_step(self, row: dict[str, Any]) -> None:
        if self._trajectory_file is None:
            return
        self._trajectory_file.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")

    def finish_trajectory(self) -> None:
        if self._trajectory_file is not None:
            self._trajectory_file.close()
            self._trajectory_file = None

    def log_episode(self, episode: dict[str, Any]) -> None:
        self._episode_file.write(json.dumps(_json_safe(episode), ensure_ascii=False) + "\n")
        self._episode_file.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
            f.write("\n")


class SummaryAccumulator:
    """Collects aggregate metrics across generated episodes."""

    def __init__(self) -> None:
        self.total_episodes = 0
        self.successes = 0
        self.total_steps = 0
        self.skill_successes: Counter[str] = Counter()
        self.skill_totals: Counter[str] = Counter()
        self.failure_reason_counts: Counter[str] = Counter()

    def add_episode(self, episode: dict[str, Any]) -> None:
        self.total_episodes += 1
        final_result = episode.get("final_result", {})
        if final_result.get("success"):
            self.successes += 1

        episode_steps = 0
        counted_skill_failure = False
        for skill_trace in episode.get("execution_trace", []):
            skill_name = str(skill_trace.get("skill", "unknown"))
            self.skill_totals[skill_name] += 1
            if skill_trace.get("success"):
                self.skill_successes[skill_name] += 1
            skill_reason = skill_trace.get("failure_reason")
            if skill_reason:
                self.failure_reason_counts[str(skill_reason)] += 1
                counted_skill_failure = True
            episode_steps += int(skill_trace.get("num_steps", 0))
        reason = final_result.get("failure_reason")
        if reason and not counted_skill_failure:
            self.failure_reason_counts[str(reason)] += 1
        self.total_steps += episode_steps

    def to_dict(self) -> dict[str, Any]:
        total = max(self.total_episodes, 1)
        return {
            "total_episodes": self.total_episodes,
            "success_rate": self.successes / total,
            "pick_success_rate": self._skill_rate("pick"),
            "place_success_rate": self._skill_rate("place"),
            "average_steps": self.total_steps / total,
            "failure_reason_counts": dict(self.failure_reason_counts),
        }

    def _skill_rate(self, skill_name: str) -> float:
        total = self.skill_totals[skill_name]
        if total == 0:
            return 0.0
        return self.skill_successes[skill_name] / total


class BlueprintDatasetLogger(StateMachineDatasetLogger):
    """Writes stage-2 episode, trajectory, and predictor dataset JSONL files."""

    def __init__(self, output_dir: str | Path, save_trajectory: bool = True):
        super().__init__(output_dir=output_dir, save_trajectory=save_trajectory)
        self.predictor_path = self.output_dir / "predictor_dataset.jsonl"
        self._predictor_file = self.predictor_path.open("w", encoding="utf-8")
        self.ee_diagnostics_path = self.output_dir / "ee_target_diagnostics.jsonl"
        self._ee_diagnostics_file = self.ee_diagnostics_path.open("w", encoding="utf-8")

    def close(self) -> None:
        if getattr(self, "_ee_diagnostics_file", None) is not None:
            self._ee_diagnostics_file.close()
            self._ee_diagnostics_file = None
        if getattr(self, "_predictor_file", None) is not None:
            self._predictor_file.close()
            self._predictor_file = None
        super().close()

    def log_predictor_sample(self, sample: dict[str, Any]) -> None:
        self._predictor_file.write(json.dumps(_json_safe(sample), ensure_ascii=False) + "\n")
        self._predictor_file.flush()

    def log_ee_diagnostic(self, row: dict[str, Any]) -> None:
        self._ee_diagnostics_file.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
        self._ee_diagnostics_file.flush()


class BlueprintSummaryAccumulator:
    """Collects aggregate statistics for stage-2 blueprint execution."""

    def __init__(self) -> None:
        self.total_episodes = 0
        self.successes = 0
        self.total_steps = 0
        self.skill_successes: Counter[str] = Counter()
        self.skill_totals: Counter[str] = Counter()
        self.condition_counts: dict[str, Counter[str]] = {}
        self.failure_reason_counts: Counter[str] = Counter()
        self.predictor_sample_count = 0
        self.missing_metric_counts: Counter[str] = Counter()

    def add_episode(self, episode: dict[str, Any]) -> None:
        self.total_episodes += 1
        final_result = episode.get("final_result", {})
        if final_result.get("success"):
            self.successes += 1
        reason = final_result.get("failure_reason")
        if reason:
            self.failure_reason_counts[str(reason)] += 1

        episode_steps = 0
        for skill_trace in episode.get("skill_execution_trace", []):
            skill = str(skill_trace.get("skill", "unknown"))
            self.skill_totals[skill] += 1
            if skill_trace.get("success"):
                self.skill_successes[skill] += 1
            skill_reason = skill_trace.get("failure_reason")
            if skill_reason:
                self.failure_reason_counts[str(skill_reason)] += 1
            episode_steps += int(skill_trace.get("num_steps", 0))
            self.predictor_sample_count += 1
        self.total_steps += episode_steps

        for condition in episode.get("condition_trace", []):
            name = str(condition.get("condition", "unknown"))
            if name not in self.condition_counts:
                self.condition_counts[name] = Counter()
            self.condition_counts[name]["true" if condition.get("result") else "false"] += 1

    def add_missing_metric_counts(self, counts: Counter[str]) -> None:
        self.missing_metric_counts.update(counts)

    def to_dict(self) -> dict[str, Any]:
        total = max(self.total_episodes, 1)
        return {
            "total_episodes": self.total_episodes,
            "episode_success_rate": self.successes / total,
            "average_steps": self.total_steps / total,
            "skill_success_rate": {
                skill: self.skill_successes[skill] / self.skill_totals[skill]
                for skill in sorted(self.skill_totals)
                if self.skill_totals[skill] > 0
            },
            "condition_stats": {
                name: {"true": counts["true"], "false": counts["false"]}
                for name, counts in sorted(self.condition_counts.items())
            },
            "failure_reason_counts": dict(self.failure_reason_counts),
            "predictor_sample_count": self.predictor_sample_count,
            "missing_metric_counts": dict(self.missing_metric_counts),
        }


def _json_safe(value: Any) -> Any:
    """Convert tensors, numpy values, paths, and tuples into JSON-safe data."""

    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
