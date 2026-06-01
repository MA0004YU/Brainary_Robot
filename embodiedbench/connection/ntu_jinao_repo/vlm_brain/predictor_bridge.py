"""Build node-level predictor feedback for VLM blueprint revision.

The bridge can run with a trained predictor checkpoint or emit deterministic
mock feedback for wiring tests before predictor_v1 is trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from data.run_layout import add_experiment_run_args, resolve_experiment_run_from_args  # noqa: E402
PREDICTOR_DIR = REPO_ROOT / "skill_performance_predictor"
if PREDICTOR_DIR.as_posix() not in sys.path:
    sys.path.insert(0, PREDICTOR_DIR.as_posix())


DEFAULT_SCENE_STATE = {
    "ee_pose": [0.45, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0],
    "cube_pose": [0.45, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0],
    "target_pose": [0.30, -0.18, 0.035, 1.0, 0.0, 0.0, 0.0],
    "gripper_width": 0.08,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VLM-readable predictor feedback for a skill blueprint.")
    parser.add_argument("--blueprint_path", "--blueprint", dest="blueprint_path", required=True, type=str)
    parser.add_argument("--output_json", "--output_feedback", dest="output_json", required=False, type=str, default=None)
    parser.add_argument("--checkpoint", "--predictor_checkpoint", dest="checkpoint", default=None, type=str)
    parser.add_argument("--predictor_data_dir", default=None, type=str, help="Reserved for compatibility with Stage IV CLI.")
    parser.add_argument("--scene_state_json", "--scene_state", dest="scene_state_json", default=None, type=str)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--mock", default="false", help="Use deterministic mock predictions instead of a checkpoint.")
    add_experiment_run_args(parser)
    args = parser.parse_args()

    experiment_run = resolve_experiment_run_from_args(args)
    if experiment_run is not None and not args.output_json:
        args.output_json = str(experiment_run.predictor_feedback_path())
    if not args.output_json:
        parser.error("Provide --output_json/--output_feedback or --experiment_run_id / --experiment_run_new.")

    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    use_mock = _str_to_bool(args.mock) or checkpoint_path is None or not checkpoint_path.exists()
    if checkpoint_path is not None and not checkpoint_path.exists():
        print(f"[WARN] Predictor checkpoint not found; using mock feedback: {checkpoint_path}")
    feedback = build_predictor_feedback(
        blueprint_path=Path(args.blueprint_path).expanduser(),
        checkpoint_path=checkpoint_path,
        scene_state=_load_scene_state(args.scene_state_json),
        device_arg=args.device,
        use_mock=use_mock,
    )
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[INFO] Wrote predictor feedback to: {output_path.resolve()}")
    if experiment_run is not None:
        meta = {
            "mode": "mock" if use_mock else "checkpoint",
            "checkpoint": checkpoint_path.as_posix() if checkpoint_path else None,
            "blueprint": Path(args.blueprint_path).expanduser().as_posix(),
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta_path = experiment_run.predictor_outputs / "predictor_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
        experiment_run.update_manifest(
            "predictor",
            {"predictor_feedback": output_path.as_posix(), "predictor_meta": meta_path.as_posix()},
        )


def build_predictor_feedback(
    blueprint_path: Path,
    checkpoint_path: Path | None = None,
    scene_state: dict[str, Any] | None = None,
    device_arg: str = "auto",
    use_mock: bool = False,
) -> dict[str, Any]:
    with blueprint_path.open("r", encoding="utf-8") as f:
        blueprint = json.load(f)
    scene = _normalize_scene_state(scene_state or DEFAULT_SCENE_STATE)
    predictor = None
    checkpoint = None
    device = None
    if not use_mock:
        from evaluate_predictor import _build_model_from_checkpoint, _load_checkpoint

        device = _resolve_device(device_arg)
        checkpoint = _load_checkpoint(checkpoint_path, device)  # type: ignore[arg-type]
        predictor = _build_model_from_checkpoint(checkpoint).to(device)
        predictor.load_state_dict(checkpoint["model_state_dict"])
        predictor.eval()

    node_feedback = []
    nodes = blueprint.get("execution_graph", {}).get("nodes", {})
    for node_id, node in nodes.items():
        if node.get("type") not in {"skill", "parallel"}:
            continue
        predicted = _mock_prediction(node_id, node, scene) if use_mock else _predict_node(predictor, checkpoint, node_id, node, scene, device)
        risk = _risk_from_prediction(predicted)
        node_feedback.append(
            {
                "node_id": node_id,
                "skill": node.get("skill") if node.get("type") == "skill" else "parallel",
                "target": node.get("target") if node.get("type") == "skill" else _parallel_target(node),
                "predicted": predicted,
                "risk": risk,
                "suggested_revision": _suggest_revision(node, predicted, risk),
            }
        )

    high_risk_nodes = [row["node_id"] for row in node_feedback if row["risk"]["risk_level"] == "high"]
    success_values = [row["predicted"]["success_probability"] for row in node_feedback]
    timeout_values = [row["predicted"]["timeout_probability"] for row in node_feedback]
    overall_success = min(success_values) if success_values else 0.0
    overall_timeout = max(timeout_values) if timeout_values else 0.0
    failure_reason = next(
        (row["predicted"]["failure_reason"] for row in node_feedback if row["predicted"]["failure_reason"] != "none"),
        "none",
    )
    return {
        "overall_assessment": {
            "predicted_success_probability": overall_success,
            "predicted_timeout_probability": overall_timeout,
            "predicted_failure_reason": failure_reason,
            "high_risk_nodes": high_risk_nodes,
            "summary": _summary(high_risk_nodes, failure_reason),
        },
        "node_feedback": node_feedback,
    }


def _predict_node(model, checkpoint: dict[str, Any], node_id: str, node: dict[str, Any], scene: dict[str, Any], device: Any) -> dict[str, Any]:
    from feature_extractor import extract_sample
    from infer_predictor import _format_prediction, _make_batch

    raw_sample = {
        "sample_id": f"feedback_{node_id}",
        "episode_id": "feedback",
        "node_id": node_id,
        "skill": node.get("skill") if node.get("type") == "skill" else "parallel",
        "target": node.get("target") if node.get("type") == "skill" else _parallel_target(node),
        "scene_state_before": scene,
        "skill_params": _skill_params(node),
        "measured_performance": {},
        "performance_query": node.get("performance_query", []),
    }
    processed = extract_sample(raw_sample)
    batch = _make_batch(processed, checkpoint, device)
    import torch

    with torch.no_grad():
        outputs = model(batch["numeric_features"], batch["skill_id"], batch["target_id"])
    return _compact_prediction(_format_prediction(outputs, checkpoint))


def _mock_prediction(node_id: str, node: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    skill = node.get("skill") if node.get("type") == "skill" else "parallel"
    target_position = (scene.get("target_pose") or DEFAULT_SCENE_STATE["target_pose"])[:3]
    cube_position = (scene.get("cube_pose") or DEFAULT_SCENE_STATE["cube_pose"])[:3]
    ee_position = target_position if skill in {"place", "parallel"} else cube_position
    object_xy_error = 0.072 if skill == "place" or node_id == "s6" else 0.02
    success_probability = 0.48 if skill == "place" or node_id == "s6" else 0.9
    failure_reason = "object_not_near_target" if skill == "place" or node_id == "s6" else "none"
    return {
        "success_probability": success_probability,
        "timeout_probability": 0.12 if skill == "place" or node_id == "s6" else 0.03,
        "failure_reason": failure_reason,
        "final_ee_position": [float(v) for v in ee_position],
        "target_position": [float(v) for v in target_position],
        "final_ee_position_error": 0.012 if skill != "place" else 0.045,
        "final_ee_orientation_error": 0.06,
        "final_ee_linear_speed": 0.02,
        "average_ee_linear_speed": 0.05,
        "execution_steps": 280 if skill == "place" else 180,
        "execution_time": 2.8 if skill == "place" else 1.8,
        "trajectory_length": 0.32,
        "final_object_position": [float(cube_position[0] + object_xy_error), float(cube_position[1]), float(cube_position[2])],
        "object_target_xy_error": object_xy_error,
        "object_target_position_error": max(object_xy_error, 0.075 if skill == "place" else 0.025),
    }


def _compact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "success_probability",
        "timeout_probability",
        "failure_reason",
        "final_ee_position",
        "target_position",
        "final_ee_position_error",
        "final_ee_orientation_error",
        "final_ee_linear_speed",
        "average_ee_linear_speed",
        "execution_steps",
        "execution_time",
        "trajectory_length",
        "final_object_position",
        "object_target_position_error",
        "object_target_xy_error",
        "final_position_error",
    )
    regression = prediction.get("regression", {})
    compact = {name: prediction[name] for name in fields if name in prediction}
    for name in fields:
        if name in compact:
            continue
        if name in regression:
            compact[name] = regression[name]
    for name in (
        "final_ee_position_error",
        "final_ee_orientation_error",
        "final_ee_linear_speed",
        "average_ee_linear_speed",
        "execution_steps",
        "execution_time",
        "trajectory_length",
        "object_target_position_error",
        "object_target_xy_error",
        "final_position_error",
    ):
        if name in compact and compact[name] is not None:
            compact[name] = max(0.0, float(compact[name]))
    return compact


def _risk_from_prediction(predicted: dict[str, Any]) -> dict[str, Any]:
    if predicted.get("timeout_probability", 0.0) >= 0.35:
        return {"risk_level": "high", "risk_reason": "timeout_risk"}
    if predicted.get("failure_reason") not in {None, "none"}:
        return {"risk_level": "high", "risk_reason": predicted.get("failure_reason")}
    if predicted.get("object_target_xy_error", 0.0) > 0.06:
        return {"risk_level": "high", "risk_reason": "object_target_xy_error_too_large"}
    if predicted.get("final_ee_position_error", 0.0) > 0.04:
        return {"risk_level": "medium", "risk_reason": "final_position_error_too_large"}
    if predicted.get("final_ee_orientation_error", 0.0) > 0.12:
        return {"risk_level": "medium", "risk_reason": "orientation_not_converged"}
    return {"risk_level": "low", "risk_reason": None}


def _suggest_revision(node: dict[str, Any], predicted: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any] | None:
    if risk["risk_level"] == "low":
        return None
    reason = risk.get("risk_reason")
    if reason == "object_target_xy_error_too_large" or predicted.get("failure_reason") == "object_not_near_target":
        return {
            "params_to_adjust": ["place_height", "open_wait_steps", "target_tolerance", "move_above.target.height_offset"],
            "natural_language_hint": (
                "Predicted object-target XY error is high. Lower place height, increase open wait steps, "
                "and reduce target approach error."
            ),
        }
    if reason == "timeout_risk":
        return {
            "params_to_adjust": ["timeout_steps", "speed"],
            "natural_language_hint": "Execution may reach timeout. Increase timeout_steps or slightly increase speed away from contact phases.",
        }
    if reason == "orientation_not_converged":
        return {
            "params_to_adjust": ["orientation_tolerance", "angular_speed", "timeout_steps", "orientation_mode"],
            "natural_language_hint": "Predicted orientation error is high. Increase angular speed or timeout, relax tolerance, or set orientation_mode to none if orientation is unimportant.",
        }
    if predicted.get("failure_reason") == "object_not_in_gripper":
        return {
            "params_to_adjust": ["descend.target_height", "close_wait_steps", "move_above.cube.height_offset", "descend.speed"],
            "natural_language_hint": "Predicted grasp failure. Lower descend target height, increase close wait, and slow down descend.",
        }
    return {
        "params_to_adjust": ["timeout_steps", "speed", "position_tolerance", "height_offset", "xy_offset"],
        "natural_language_hint": "Predicted target reach error is high. Slow down, allow more steps, and adjust the position target offsets.",
    }


def _skill_params(node: dict[str, Any]) -> dict[str, Any]:
    params = dict(node.get("params") or {})
    if node.get("type") == "parallel":
        params["parallel_mode"] = node.get("parallel_mode", "all_success")
        params["goals"] = node.get("goals", {})
    return params


def _parallel_target(node: dict[str, Any]) -> str:
    labels = []
    for goal in (node.get("goals") or {}).values():
        if isinstance(goal, dict) and goal.get("target"):
            labels.append(str(goal["target"]))
    return "+".join(labels) if labels else "parallel"


def _summary(high_risk_nodes: list[str], failure_reason: str) -> str:
    if high_risk_nodes:
        return f"The plan needs revision around {', '.join(high_risk_nodes)}; predicted risk reason: {failure_reason}."
    return "The plan is mostly feasible under predictor feedback."


def _load_scene_state(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_scene_state(scene: dict[str, Any]) -> dict[str, Any]:
    if {"ee_pose", "cube_pose", "target_pose"} <= set(scene):
        return scene
    objects = scene.get("objects") or {}
    robot = scene.get("robot") or {}
    cube = objects.get("cube") or {}
    target = objects.get("target") or {}
    return {
        "ee_pose": robot.get("ee_pose", DEFAULT_SCENE_STATE["ee_pose"]),
        "cube_pose": cube.get("pose", DEFAULT_SCENE_STATE["cube_pose"]),
        "target_pose": target.get("pose", DEFAULT_SCENE_STATE["target_pose"]),
        "gripper_width": robot.get("gripper_width", DEFAULT_SCENE_STATE["gripper_width"]),
    }


def _resolve_device(device_arg: str) -> Any:
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    main()
