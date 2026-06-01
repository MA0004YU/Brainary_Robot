"""Unified experiment run directories under ntu_jinao_repo/data/."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

VLM_INPUTS_ROOT = DATA_ROOT / "01_vlm_inputs"
VLM_OUTPUTS_ROOT = DATA_ROOT / "02_vlm_outputs"
PREDICTOR_OUTPUTS_ROOT = DATA_ROOT / "03_predictor_outputs"
EXECUTION_DATA_ROOT = DATA_ROOT / "04_execution_data"
MANIFESTS_ROOT = DATA_ROOT / "manifests"


def _copy_if_different(src: Path, dst: Path) -> Path:
    """Copy ``src`` to ``dst`` unless they are already the same file."""

    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved == dst_resolved:
        return dst_resolved
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_resolved, dst_resolved)
    return dst_resolved


def new_run_id(when: datetime | None = None) -> str:
    """Minute-resolution run id, e.g. 2026-05-26_1549."""

    when = when or datetime.now()
    return when.strftime("%Y-%m-%d_%H%M")


@dataclass(frozen=True)
class ExperimentRun:
    """One logical experiment with the same run_id under four category folders."""

    run_id: str
    vlm_inputs: Path
    vlm_outputs: Path
    predictor_outputs: Path
    execution_data: Path
    manifest_path: Path

    @classmethod
    def create(cls, run_id: str | None = None, note: str = "") -> ExperimentRun:
        run_id = run_id or new_run_id()
        run = cls(
            run_id=run_id,
            vlm_inputs=VLM_INPUTS_ROOT / run_id,
            vlm_outputs=VLM_OUTPUTS_ROOT / run_id,
            predictor_outputs=PREDICTOR_OUTPUTS_ROOT / run_id,
            execution_data=EXECUTION_DATA_ROOT / run_id,
            manifest_path=MANIFESTS_ROOT / f"{run_id}.json",
        )
        for path in (run.vlm_inputs, run.vlm_outputs, run.predictor_outputs, run.execution_data, MANIFESTS_ROOT):
            path.mkdir(parents=True, exist_ok=True)
        if not run.manifest_path.exists():
            run._write_manifest({"note": note, "artifacts": {}})
        return run

    @classmethod
    def open(cls, run_id: str, create_if_missing: bool = True) -> ExperimentRun:
        run = cls(
            run_id=run_id,
            vlm_inputs=VLM_INPUTS_ROOT / run_id,
            vlm_outputs=VLM_OUTPUTS_ROOT / run_id,
            predictor_outputs=PREDICTOR_OUTPUTS_ROOT / run_id,
            execution_data=EXECUTION_DATA_ROOT / run_id,
            manifest_path=MANIFESTS_ROOT / f"{run_id}.json",
        )
        if create_if_missing:
            for path in (run.vlm_inputs, run.vlm_outputs, run.predictor_outputs, run.execution_data, MANIFESTS_ROOT):
                path.mkdir(parents=True, exist_ok=True)
        return run

    def archive_vlm_inputs(
        self,
        *,
        scene_state: dict[str, Any],
        task: str,
        image_path: str | Path | None = None,
        config_path: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Copy/save VLM inputs into 01_vlm_inputs/<run_id>/."""

        scene_path = self.vlm_inputs / "scene_state.json"
        self._write_json(scene_path, scene_state)
        task_path = self.vlm_inputs / "task.txt"
        task_path.write_text(task.strip() + "\n", encoding="utf-8")

        archived: dict[str, str] = {
            "scene_state": scene_path.as_posix(),
            "task": task_path.as_posix(),
        }
        if image_path is not None:
            src = Path(image_path).expanduser()
            if src.is_file():
                dst = self.vlm_inputs / f"image{src.suffix.lower()}"
                archived["image"] = _copy_if_different(src, dst).as_posix()
        if config_path is not None:
            src = Path(config_path).expanduser()
            if src.is_file():
                dst = self.vlm_inputs / "vlm_config.json"
                archived["vlm_config"] = _copy_if_different(src, dst).as_posix()
        if extra:
            meta_path = self.vlm_inputs / "input_meta.json"
            self._write_json(meta_path, extra)
            archived["input_meta"] = meta_path.as_posix()

        self.update_manifest("vlm_inputs", archived)
        return archived

    def vlm_output_dir(self) -> Path:
        self.vlm_outputs.mkdir(parents=True, exist_ok=True)
        return self.vlm_outputs

    def predictor_feedback_path(self) -> Path:
        self.predictor_outputs.mkdir(parents=True, exist_ok=True)
        return self.predictor_outputs / "predictor_feedback.json"

    def execution_output_dir(self) -> Path:
        self.execution_data.mkdir(parents=True, exist_ok=True)
        return self.execution_data

    def copy_blueprint_for_execution(self, blueprint_path: Path) -> Path:
        dst = self.execution_data / "blueprint_used.json"
        self.execution_data.mkdir(parents=True, exist_ok=True)
        copied = _copy_if_different(blueprint_path.expanduser(), dst)
        self.update_manifest("execution", {"blueprint_used": copied.as_posix()})
        return copied

    def update_manifest(self, section: str, payload: dict[str, Any]) -> None:
        manifest = self._read_manifest()
        artifacts = manifest.setdefault("artifacts", {})
        if section in artifacts and isinstance(artifacts[section], dict):
            artifacts[section].update(payload)
        else:
            artifacts[section] = payload
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_manifest(manifest)

    def _read_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {
                "vlm_inputs": self.vlm_inputs.as_posix(),
                "vlm_outputs": self.vlm_outputs.as_posix(),
                "predictor_outputs": self.predictor_outputs.as_posix(),
                "execution_data": self.execution_data.as_posix(),
            },
            "artifacts": {},
        }

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest.setdefault("run_id", self.run_id)
        manifest.setdefault(
            "paths",
            {
                "vlm_inputs": self.vlm_inputs.as_posix(),
                "vlm_outputs": self.vlm_outputs.as_posix(),
                "predictor_outputs": self.predictor_outputs.as_posix(),
                "execution_data": self.execution_data.as_posix(),
            },
        )
        MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")


def resolve_experiment_run_from_args(args) -> ExperimentRun | None:
    """Return an :class:`ExperimentRun` when CLI flags request the unified layout."""

    run_new = bool(getattr(args, "experiment_run_new", False))
    run_id = getattr(args, "experiment_run_id", None)
    note = getattr(args, "experiment_run_note", "") or ""

    if run_new:
        run = ExperimentRun.create(note=note)
        print(f"[INFO] Created experiment run: {run.run_id}")
        return run
    if run_id:
        run = ExperimentRun.open(run_id)
        print(f"[INFO] Using experiment run: {run.run_id}")
        return run
    return None


def add_experiment_run_args(parser) -> None:
    group = parser.add_argument_group("experiment data layout (data/)")
    group.add_argument(
        "--experiment_run_new",
        action="store_true",
        help="Create a new minute-stamped run under data/01..04 and data/manifests.",
    )
    group.add_argument(
        "--experiment_run_id",
        type=str,
        default=None,
        help="Use an existing run id (YYYY-MM-DD_HHMM). Implied by --experiment_run_new.",
    )
    group.add_argument(
        "--experiment_run_note",
        type=str,
        default="",
        help="Optional note stored in the run manifest.",
    )
