"""Lightweight training summary writer for constrained reach runs."""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


KEY_SCALARS = {
    "iteration_or_step": ["Train/iteration", "Perf/total_fps", "Total timesteps"],
    "mean_reward": ["Mean reward", "Train/mean_reward", "Episode_Reward/stage1_reward"],
    "mean_episode_length": ["Mean episode length", "Train/mean_episode_length"],
    "stage1/mean_position_error": ["stage1/mean_position_error"],
    "stage1/mean_position_tolerance": ["stage1/mean_position_tolerance"],
    "stage1/position_success_rate": ["stage1/position_success_rate"],
    "stage1/mean_orientation_error": ["stage1/mean_orientation_error"],
    "stage1/mean_orientation_tolerance": ["stage1/mean_orientation_tolerance"],
    "stage1/orientation_success_rate": ["stage1/orientation_success_rate"],
    "stage1/pose_success_rate": ["stage1/pose_success_rate"],
    "stage1/mean_normalized_position_error": ["stage1/mean_normalized_position_error"],
    "stage1/mean_normalized_orientation_error": ["stage1/mean_normalized_orientation_error"],
    "stage1/mean_quat_dot_abs": ["stage1/mean_quat_dot_abs"],
    "stage1/min_quat_dot_abs": ["stage1/min_quat_dot_abs"],
    "stage1/mean_action_magnitude": ["stage1/mean_action_magnitude"],
    "stage1/mean_joint_velocity": ["stage1/mean_joint_velocity"],
    "stage1/mean_joint_limit_margin": ["stage1/mean_joint_limit_margin"],
}


@dataclass
class _MetricPoint:
    step: int
    value: float


class TrainingReportWriter:
    """Periodically emits lightweight markdown/json/csv training summaries."""

    def __init__(self, log_dir: str | Path, interval_s: float = 60.0):
        self.log_dir = Path(log_dir)
        self.interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned_unavailable = False
        self._warned_write_error = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="stage1-training-report", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def flush_once(self) -> None:
        self._write_summary_once()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._write_summary_once()

    def _write_summary_once(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        summary_md = self.log_dir / "training_summary.md"
        latest_json = self.log_dir / "training_key_metrics_latest.json"
        sampled_csv = self.log_dir / "training_key_metrics_sampled.csv"

        metrics, err = self._read_latest_metrics()
        now = datetime.now(timezone.utc).isoformat()

        if metrics is None:
            msg = (
                "# Training Summary\n\n"
                f"- run path: `{self.log_dir}`\n"
                f"- last update time: `{now}`\n"
                f"- status: metric extraction unavailable\n"
                f"- reason: `{err}`\n"
            )
            try:
                summary_md.write_text(msg, encoding="utf-8")
            except Exception as write_err:
                if not self._warned_write_error:
                    print(f"[WARN] Failed to write training_summary.md: {write_err}")
                    self._warned_write_error = True
            return

        diagnosis = self._diagnose(metrics)
        try:
            latest_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            self._append_csv(sampled_csv, metrics)
            summary_md.write_text(self._build_markdown(metrics, diagnosis), encoding="utf-8")
        except Exception as write_err:
            if not self._warned_write_error:
                print(f"[WARN] Failed to write training summary artifacts: {write_err}")
                self._warned_write_error = True

    def _read_latest_metrics(self) -> tuple[dict[str, float | int | str] | None, str | None]:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception as exc:
            if not self._warned_unavailable:
                print(f"[WARN] TensorBoard event reader unavailable: {exc}")
                self._warned_unavailable = True
            return None, f"tensorboard event reader unavailable: {exc}"

        event_files = sorted(self.log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return None, "no TensorBoard event file found yet"

        event_file = str(event_files[-1])
        try:
            acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
            acc.Reload()
        except Exception as exc:
            return None, f"failed to parse events: {exc}"

        scalar_tags = set(acc.Tags().get("scalars", []))
        out: dict[str, float | int | str] = {
            "run_path": str(self.log_dir),
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "event_file": event_file,
        }

        latest_step = 0
        for out_key, candidates in KEY_SCALARS.items():
            point = self._latest_from_candidates(acc, scalar_tags, candidates)
            if point is not None:
                out[out_key] = float(point.value)
                latest_step = max(latest_step, int(point.step))
            else:
                out[out_key] = float("nan")

        out["latest_iteration_or_step"] = latest_step
        return out, None

    @staticmethod
    def _latest_from_candidates(acc, scalar_tags: set[str], candidates: list[str]) -> _MetricPoint | None:
        for tag in candidates:
            if tag not in scalar_tags:
                continue
            events = acc.Scalars(tag)
            if not events:
                continue
            last = events[-1]
            return _MetricPoint(step=int(last.step), value=float(last.value))
        return None

    @staticmethod
    def _safe_float(metrics: dict[str, float | int | str], key: str) -> float:
        try:
            value = metrics.get(key, float("nan"))
            return float(value)
        except Exception:
            return float("nan")

    def _diagnose(self, metrics: dict[str, float | int | str]) -> list[str]:
        notes: list[str] = []
        pos_succ = self._safe_float(metrics, "stage1/position_success_rate")
        ori_succ = self._safe_float(metrics, "stage1/orientation_success_rate")
        pose_succ = self._safe_float(metrics, "stage1/pose_success_rate")
        act_mag = self._safe_float(metrics, "stage1/mean_action_magnitude")
        joint_vel = self._safe_float(metrics, "stage1/mean_joint_velocity")

        if pos_succ > 0.8 and ori_succ < 0.1:
            notes.append("位置已学会，姿态未学会。")
        if ori_succ > 0.1 and pose_succ < 0.2:
            notes.append("姿态开始学习，但同步成功不足。")
        if pose_succ > 0.5:
            notes.append("位姿联合到达已初步学会。")
        if act_mag > 2.0 or joint_vel > 1.2:
            notes.append("动作可能过激。")
        if not notes:
            notes.append("训练进行中，当前未见明显异常模式。")
        return notes

    def _build_markdown(self, metrics: dict[str, float | int | str], diagnosis: list[str]) -> str:
        lines = ["# Training Summary", ""]
        lines.append(f"- run path: `{metrics.get('run_path', self.log_dir)}`")
        lines.append(f"- last update time: `{metrics.get('last_update_time', '')}`")
        lines.append(f"- latest iteration/step: `{int(self._safe_float(metrics, 'latest_iteration_or_step'))}`")
        lines.append("")
        lines.append("## Key Metrics")
        for key in (
            "mean_reward",
            "mean_episode_length",
            "stage1/mean_position_error",
            "stage1/mean_position_tolerance",
            "stage1/position_success_rate",
            "stage1/mean_orientation_error",
            "stage1/mean_orientation_tolerance",
            "stage1/orientation_success_rate",
            "stage1/pose_success_rate",
            "stage1/mean_normalized_position_error",
            "stage1/mean_normalized_orientation_error",
            "stage1/mean_quat_dot_abs",
            "stage1/min_quat_dot_abs",
            "stage1/mean_action_magnitude",
            "stage1/mean_joint_velocity",
            "stage1/mean_joint_limit_margin",
        ):
            lines.append(f"- {key}: `{self._safe_float(metrics, key):.6f}`")
        lines.append("")
        lines.append("## Diagnosis")
        for note in diagnosis:
            lines.append(f"- {note}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _append_csv(path: Path, metrics: dict[str, float | int | str]) -> None:
        fieldnames = [
            "last_update_time",
            "latest_iteration_or_step",
            "mean_reward",
            "mean_episode_length",
            "stage1/mean_position_error",
            "stage1/mean_position_tolerance",
            "stage1/position_success_rate",
            "stage1/mean_orientation_error",
            "stage1/mean_orientation_tolerance",
            "stage1/orientation_success_rate",
            "stage1/pose_success_rate",
            "stage1/mean_normalized_position_error",
            "stage1/mean_normalized_orientation_error",
            "stage1/mean_quat_dot_abs",
            "stage1/min_quat_dot_abs",
            "stage1/mean_action_magnitude",
            "stage1/mean_joint_velocity",
            "stage1/mean_joint_limit_margin",
        ]
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {k: metrics.get(k, "") for k in fieldnames}
            writer.writerow(row)

