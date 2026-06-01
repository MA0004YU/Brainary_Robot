"""JSONL logger for Stage V brain intervention records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BrainInterventionLogger:
    """Write one complete intervention record per VLM/governor event."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "brain_interventions.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._file.close()

    def log_intervention(
        self,
        episode_id: str,
        node_id: str,
        trigger_reason: str,
        governor_mode: str,
        scene_state: dict[str, Any],
        image_path: str | None,
        predictor_feedback: dict[str, Any] | None,
        vlm_revision_applied: bool,
        revised_params: dict[str, Any] | None,
    ) -> None:
        row = {
            "episode_id": episode_id,
            "node_id": node_id,
            "trigger_reason": trigger_reason,
            "governor_mode": governor_mode,
            "scene_state": scene_state,
            "image_path": image_path,
            "predictor_feedback": _compact_predictor_feedback(predictor_feedback or {}),
            "vlm_revision_applied": bool(vlm_revision_applied),
            "revised_params": revised_params or {},
        }
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()


def _compact_predictor_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "success_probability",
        "timeout_probability",
        "failure_reason",
        "final_ee_position",
        "target_position",
        "final_ee_position_error",
        "final_ee_linear_speed",
        "object_target_xy_error",
    )
    return {key: feedback.get(key) for key in keys if key in feedback}
