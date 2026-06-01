"""Run Qwen3-VL blueprint generation or a prompt-only dry run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from data.run_layout import add_experiment_run_args, resolve_experiment_run_from_args  # noqa: E402
from blueprint_parser import extract_json_from_text
from blueprint_validator import validate_blueprint
from qwen3vl_loader import generate_response, load_qwen3vl_model

VLM_DIR = Path(__file__).resolve().parent
REPO_ROOT = VLM_DIR.parents[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a skill_blueprint JSON with Qwen3-VL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--scene_state", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output_json", required=False, default=None)
    parser.add_argument("--validate", default="true")
    parser.add_argument("--dry_run", default="false")
    add_experiment_run_args(parser)
    args = parser.parse_args()

    config = _read_json(Path(args.config))
    scene_state = _read_json(Path(args.scene_state))
    experiment_run = resolve_experiment_run_from_args(args)
    if experiment_run is not None:
        experiment_run.archive_vlm_inputs(
            scene_state=scene_state,
            task=args.task,
            image_path=args.image,
            config_path=args.config,
            extra={"dry_run": _str_to_bool(args.dry_run)},
        )
        output_json = experiment_run.vlm_output_dir() / "generated_blueprint.json"
    elif args.output_json:
        output_json = Path(args.output_json).expanduser()
    else:
        parser.error("Provide --output_json or --experiment_run_id / --experiment_run_new.")
    output_dir = output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_generation_prompt(args.task, scene_state)

    if _str_to_bool(args.dry_run):
        (output_dir / "dry_run_prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"[INFO] Dry-run prompt written to: {(output_dir / 'dry_run_prompt.txt').resolve()}")
        if experiment_run is not None:
            experiment_run.update_manifest(
                "vlm_outputs",
                {"dry_run_prompt": (output_dir / "dry_run_prompt.txt").as_posix()},
            )
        return

    model, processor = load_qwen3vl_model(config)
    raw_response = generate_response(model, processor, args.image, prompt, config)
    # Re-create output dir in case it was removed during the long generation step.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    parse_result = extract_json_from_text(raw_response)
    if parse_result.extracted_text:
        (output_dir / "parsed_blueprint.json").write_text(parse_result.extracted_text + "\n", encoding="utf-8")

    validation_report: dict[str, Any] = {"parse_ok": parse_result.ok, "parse_error": parse_result.error}
    if parse_result.ok and parse_result.blueprint is not None and _str_to_bool(args.validate):
        validation = validate_blueprint(parse_result.blueprint)
        validation_report.update(validation.to_dict())
        if validation.valid:
            _write_json(output_json, parse_result.blueprint)
    elif parse_result.ok and parse_result.blueprint is not None:
        validation_report.update({"valid": None, "errors": [], "warnings": ["validation_skipped"]})
        _write_json(output_json, parse_result.blueprint)

    _write_json(output_dir / "validation_report.json", validation_report)
    if not parse_result.ok:
        raise RuntimeError(f"Failed to parse VLM output: {parse_result.error}")
    if _str_to_bool(args.validate) and not validation_report.get("valid"):
        raise RuntimeError(f"Generated blueprint failed validation: {validation_report.get('errors')}")

    if experiment_run is not None:
        experiment_run.update_manifest(
            "vlm_outputs",
            {
                "raw_response": (output_dir / "raw_response.txt").as_posix(),
                "parsed_blueprint": (output_dir / "parsed_blueprint.json").as_posix(),
                "validation_report": (output_dir / "validation_report.json").as_posix(),
                "generated_blueprint": output_json.as_posix(),
            },
        )


def build_generation_prompt(task: str, scene_state: dict[str, Any]) -> str:
    system_prompt = _read_text(VLM_DIR / "prompts" / "system_prompt.md")
    generation_prompt = _read_text(VLM_DIR / "prompts" / "blueprint_generation_prompt.md")
    schema = _read_json(VLM_DIR / "schemas" / "skill_blueprint_schema.json")
    example = _read_json(VLM_DIR / "examples" / "sample_blueprint.json")
    return (
        f"{system_prompt}\n\n"
        f"{generation_prompt}\n\n"
        f"Task instruction:\n{task}\n\n"
        f"Scene state JSON:\n{json.dumps(scene_state, indent=2, ensure_ascii=False)}\n\n"
        f"Available skills:\nmove_above, reach, descend, grasp, lift, place, retreat, wait, align_orientation\n\n"
        f"Available logic:\nskill, condition, parallel, terminal\n\n"
        f"Performance metrics:\nsuccess, execution_steps, execution_time, trajectory_length, final_ee_position, "
        f"target_position, final_ee_position_error, final_ee_orientation_error, final_ee_linear_speed, "
        f"average_ee_linear_speed, final_object_position, object_target_xy_error, object_target_position_error, "
        f"timeout, failure_reason\n\n"
        f"Output schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"
        f"Example blueprint:\n{json.dumps(example, indent=2, ensure_ascii=False)}\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    return path.expanduser().read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.expanduser().open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    main()
