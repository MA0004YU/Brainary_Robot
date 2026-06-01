"""Dry-run capable VLM generation -> predictor feedback -> revision loop."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

VLM_DIR = Path(__file__).resolve().parent
REPO_ROOT = VLM_DIR.parents[0]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from data.run_layout import add_experiment_run_args, resolve_experiment_run_from_args  # noqa: E402
from blueprint_parser import extract_json_from_text
from blueprint_validator import validate_blueprint
from predictor_bridge import build_predictor_feedback
from run_vlm_inference import build_generation_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM + predictor revision loop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--scene_state", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--predictor_checkpoint", default=None)
    parser.add_argument("--predictor_data_dir", default=None)
    parser.add_argument("--output_dir", required=False, default=None)
    parser.add_argument("--max_refine_iters", default=2, type=int)
    parser.add_argument("--dry_run", default="false")
    parser.add_argument("--mock_predictor", default="false")
    add_experiment_run_args(parser)
    args = parser.parse_args()

    experiment_run = resolve_experiment_run_from_args(args)
    if experiment_run is not None:
        experiment_run.archive_vlm_inputs(
            scene_state=_read_json(Path(args.scene_state)),
            task=args.task,
            image_path=args.image,
            config_path=args.config,
            extra={"dry_run": _str_to_bool(args.dry_run), "mock_predictor": _str_to_bool(args.mock_predictor)},
        )
        output_dir = experiment_run.vlm_output_dir()
    elif args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        parser.error("Provide --output_dir or --experiment_run_id / --experiment_run_new.")
    scene_state = _read_json(Path(args.scene_state))
    dry_run = _str_to_bool(args.dry_run)
    mock_predictor = _str_to_bool(args.mock_predictor)

    if dry_run:
        initial_blueprint_path = VLM_DIR / "examples" / "sample_blueprint.json"
        initial_blueprint = _read_json(initial_blueprint_path)
        shutil.copyfile(initial_blueprint_path, output_dir / "initial_blueprint.json")
        (output_dir / "initial_raw_response.txt").write_text(
            json.dumps(initial_blueprint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        initial_blueprint = _run_real_generation(args, scene_state, output_dir)
        _write_json(output_dir / "initial_blueprint.json", initial_blueprint)

    initial_validation = validate_blueprint(initial_blueprint)
    _write_json(output_dir / "initial_validation_report.json", initial_validation.to_dict())
    if not initial_validation.valid:
        _write_json(
            output_dir / "loop_summary.json",
            {
                "status": "initial_validation_failed",
                "errors": initial_validation.errors,
                "warnings": initial_validation.warnings,
            },
        )
        raise RuntimeError(f"Initial blueprint failed validation: {initial_validation.errors}")

    predictor_feedback = build_predictor_feedback(
        blueprint_path=output_dir / "initial_blueprint.json",
        checkpoint_path=Path(args.predictor_checkpoint).expanduser() if args.predictor_checkpoint else None,
        scene_state=scene_state,
        use_mock=mock_predictor or dry_run or not args.predictor_checkpoint,
    )
    _write_json(output_dir / "predictor_feedback_iter0.json", predictor_feedback)
    if experiment_run is not None:
        feedback_path = experiment_run.predictor_feedback_path()
        _write_json(feedback_path, predictor_feedback)
        experiment_run.update_manifest(
            "predictor",
            {
                "predictor_feedback": feedback_path.as_posix(),
                "predictor_feedback_iter0": (output_dir / "predictor_feedback_iter0.json").as_posix(),
            },
        )

    high_risk_nodes = predictor_feedback.get("overall_assessment", {}).get("high_risk_nodes", [])
    revised_paths: list[str] = []
    if high_risk_nodes and args.max_refine_iters > 0:
        revised = _dry_run_revision(initial_blueprint, predictor_feedback) if dry_run else _run_real_revision(args, initial_blueprint, predictor_feedback, scene_state, output_dir)
        revised_path = output_dir / "revised_blueprint_iter1.json"
        _write_json(revised_path, revised)
        revised_paths.append(revised_path.as_posix())
        revised_validation = validate_blueprint(revised)
        _write_json(output_dir / "revised_validation_iter1.json", revised_validation.to_dict())

    _write_json(
        output_dir / "loop_summary.json",
        {
            "status": "ok",
            "dry_run": dry_run,
            "mock_predictor": mock_predictor or dry_run,
            "initial_valid": initial_validation.valid,
            "high_risk_nodes": high_risk_nodes,
            "refine_iters_completed": len(revised_paths),
            "revised_blueprints": revised_paths,
        },
    )
    print(f"[INFO] Loop outputs written to: {output_dir.resolve()}")
    if experiment_run is not None:
        experiment_run.update_manifest(
            "vlm_outputs",
            {
                "loop_summary": (output_dir / "loop_summary.json").as_posix(),
                "initial_blueprint": (output_dir / "initial_blueprint.json").as_posix(),
                "revised_blueprints": revised_paths,
            },
        )


def _run_real_generation(args, scene_state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from qwen3vl_loader import generate_response, load_qwen3vl_model

    config = _read_json(Path(args.config))
    prompt = build_generation_prompt(args.task, scene_state)
    model, processor = load_qwen3vl_model(config)
    raw = generate_response(model, processor, args.image, prompt, config)
    (output_dir / "initial_raw_response.txt").write_text(raw, encoding="utf-8")
    parsed = extract_json_from_text(raw)
    if not parsed.ok or parsed.blueprint is None:
        raise RuntimeError(f"Failed to parse initial VLM response: {parsed.error}")
    return parsed.blueprint


def _run_real_revision(args, blueprint: dict[str, Any], feedback: dict[str, Any], scene_state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from qwen3vl_loader import generate_response, load_qwen3vl_model

    config = _read_json(Path(args.config))
    prompt = _revision_prompt(blueprint, feedback, scene_state)
    model, processor = load_qwen3vl_model(config)
    raw = generate_response(model, processor, args.image, prompt, config)
    (output_dir / "revision_raw_response_iter1.txt").write_text(raw, encoding="utf-8")
    parsed = extract_json_from_text(raw)
    if not parsed.ok or parsed.blueprint is None:
        raise RuntimeError(f"Failed to parse revision VLM response: {parsed.error}")
    return parsed.blueprint


def _dry_run_revision(blueprint: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    revised = json.loads(json.dumps(blueprint))
    risky = set(feedback.get("overall_assessment", {}).get("high_risk_nodes", []))
    for node_id, node in revised.get("execution_graph", {}).get("nodes", {}).items():
        if node_id not in risky or node.get("type") not in {"skill", "parallel"}:
            continue
        params = node.setdefault("params", {})
        if "timeout_steps" in params:
            params["timeout_steps"] = int(params["timeout_steps"] * 1.2)
        if node.get("skill") == "place":
            params["place_height"] = max(0.02, float(params.get("place_height", 0.045)) - 0.01)
            params["open_wait_steps"] = int(params.get("open_wait_steps", 60)) + 20
    return revised


def _revision_prompt(blueprint: dict[str, Any], feedback: dict[str, Any], scene_state: dict[str, Any]) -> str:
    prompt = (VLM_DIR / "prompts" / "blueprint_revision_prompt.md").read_text(encoding="utf-8")
    return (
        f"{prompt}\n\n"
        f"Original skill_blueprint:\n{json.dumps(blueprint, indent=2, ensure_ascii=False)}\n\n"
        f"Predictor feedback:\n{json.dumps(feedback, indent=2, ensure_ascii=False)}\n\n"
        f"Scene state:\n{json.dumps(scene_state, indent=2, ensure_ascii=False)}\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


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
