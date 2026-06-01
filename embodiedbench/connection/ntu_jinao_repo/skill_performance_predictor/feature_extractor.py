"""Feature extraction from one predictor JSON sample."""

from __future__ import annotations

from typing import Any
import math

from dataset_schema import (
    FAILURE_REASON_CLASSES,
    NUMERIC_FEATURE_NAMES,
    REGRESSION_TARGET_NAMES,
    SUPPORTED_SKILLS,
    SUPPORTED_TARGETS,
)


def extract_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract model-ready features and labels from one raw predictor sample."""

    scene = sample.get("scene_state_before") or {}
    params = sample.get("skill_params") or {}
    measured = sample.get("measured_performance") or {}

    ee = _pose_position(_first_present(scene, ["ee_pose", "ee_pose_w"]))
    cube = _pose_position(_first_present(scene, ["cube_pose", "cube_pose_w"]))
    target_pose = _pose_position(_first_present(scene, ["target_pose", "target_pose_w"]))
    gripper_width = _to_float(scene.get("gripper_width"), default=0.0)

    numeric = {name: 0.0 for name in NUMERIC_FEATURE_NAMES}
    numeric.update(
        {
            "ee_x": ee[0],
            "ee_y": ee[1],
            "ee_z": ee[2],
            "cube_x": cube[0],
            "cube_y": cube[1],
            "cube_z": cube[2],
            "target_x": target_pose[0],
            "target_y": target_pose[1],
            "target_z": target_pose[2],
            "gripper_width": gripper_width,
        }
    )

    _add_delta_features(numeric, "ee_cube", ee, cube)
    _add_delta_features(numeric, "ee_target", ee, target_pose)
    _add_delta_features(numeric, "cube_target", cube, target_pose)
    numeric["ee_cube_dist"] = _distance(ee, cube)
    numeric["ee_target_dist"] = _distance(ee, target_pose)
    numeric["cube_target_dist"] = _distance(cube, target_pose)
    numeric["cube_target_xy_dist"] = _distance(cube[:2], target_pose[:2])

    _extract_skill_params(numeric, params)
    _extract_parallel_params(numeric, params)

    skill = _normalize_skill(sample.get("skill"))
    target = _normalize_target(sample.get("target"))
    success = 1.0 if bool(measured.get("success", False)) else 0.0
    timeout = 1.0 if bool(measured.get("timeout", False)) else 0.0
    failure_reason = _normalize_failure_reason(measured.get("failure_reason"))

    regression_values = []
    regression_mask = []
    for name in REGRESSION_TARGET_NAMES:
        value = _lookup_regression_value(measured, name)
        if _is_valid_number(value):
            regression_values.append(float(value))
            regression_mask.append(1.0)
        else:
            regression_values.append(0.0)
            regression_mask.append(0.0)

    return {
        "sample_id": str(sample.get("sample_id", "")),
        "episode_id": str(sample.get("episode_id", "")),
        "node_id": str(sample.get("node_id", "")),
        "skill": skill,
        "target": target,
        "numeric_features": [float(numeric[name]) for name in NUMERIC_FEATURE_NAMES],
        "success": success,
        "timeout": timeout,
        "failure_reason": failure_reason,
        "regression_targets": regression_values,
        "regression_mask": regression_mask,
        "performance_query": list(sample.get("performance_query") or []),
    }


def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _pose_position(value: Any) -> list[float]:
    if isinstance(value, dict):
        value = value.get("position") or value.get("pos") or value.get("translation")
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [_to_float(value[0]), _to_float(value[1]), _to_float(value[2])]
    return [0.0, 0.0, 0.0]


def _add_delta_features(features: dict[str, float], prefix: str, a: list[float], b: list[float]) -> None:
    features[f"{prefix}_dx"] = float(b[0] - a[0])
    features[f"{prefix}_dy"] = float(b[1] - a[1])
    features[f"{prefix}_dz"] = float(b[2] - a[2])


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _extract_skill_params(features: dict[str, float], params: dict[str, Any]) -> None:
    simple_fields = [
        "height_offset",
        "speed",
        "position_tolerance",
        "timeout_steps",
        "target_height",
        "relative_z",
        "close_wait_steps",
        "lift_height",
        "place_height",
        "release_height",
        "open_wait_steps",
        "target_tolerance",
        "retreat_height",
        "wait_steps",
        "orientation_tolerance",
        "angular_speed",
    ]
    for field in simple_fields:
        _set_param_feature(features, field, _lookup_param(params, field))

    xy_offset = _lookup_param(params, "xy_offset")
    if isinstance(xy_offset, (list, tuple)) and len(xy_offset) >= 2:
        _set_param_feature(features, "xy_offset_x", xy_offset[0])
        _set_param_feature(features, "xy_offset_y", xy_offset[1])


def _extract_parallel_params(features: dict[str, float], params: dict[str, Any]) -> None:
    goals = params.get("goals") or {}
    position_goal = goals.get("position_goal") or {}
    orientation_goal = goals.get("orientation_goal") or {}
    position_params = position_goal.get("params") or {}
    orientation_params = orientation_goal.get("params") or {}

    features["has_position_goal"] = 1.0 if position_goal else 0.0
    features["has_orientation_goal"] = 1.0 if orientation_goal else 0.0
    features["position_goal_speed"] = _to_float(position_params.get("speed"), 0.0)
    features["orientation_goal_angular_speed"] = _to_float(orientation_params.get("angular_speed"), 0.0)
    features["position_goal_tolerance"] = _to_float(position_params.get("position_tolerance"), 0.0)
    features["orientation_goal_tolerance"] = _to_float(orientation_params.get("orientation_tolerance"), 0.0)


def _lookup_param(params: dict[str, Any], field: str) -> Any:
    if field in params:
        return params[field]
    goals = params.get("goals") or {}
    for goal_name in ("position_goal", "orientation_goal"):
        goal_params = (goals.get(goal_name) or {}).get("params") or {}
        if field in goal_params:
            return goal_params[field]
    return None


def _set_param_feature(features: dict[str, float], name: str, value: Any) -> None:
    if _is_valid_number(value):
        features[name] = float(value)
        features[f"has_{name}"] = 1.0
    else:
        features[name] = 0.0
        features[f"has_{name}"] = 0.0


def _normalize_skill(value: Any) -> str:
    skill = str(value or "unknown")
    return skill if skill in SUPPORTED_SKILLS else "unknown"


def _normalize_target(value: Any) -> str:
    target = str(value or "unknown")
    if target in SUPPORTED_TARGETS:
        return target
    parts = [part for part in target.split("+") if part]
    if parts and all(part == parts[0] for part in parts) and parts[0] in SUPPORTED_TARGETS:
        return parts[0]
    if target.startswith("custom"):
        return "custom_pose"
    return "unknown"


def _normalize_failure_reason(value: Any) -> str:
    if value is None:
        return "none"
    reason = str(value)
    return reason if reason in FAILURE_REASON_CLASSES else "unknown"


def _lookup_regression_value(measured: dict[str, Any], name: str) -> Any:
    if name in measured:
        return measured.get(name)

    vector_mappings = {
        "final_ee_x": ("final_ee_position", 0),
        "final_ee_y": ("final_ee_position", 1),
        "final_ee_z": ("final_ee_position", 2),
        "target_x": ("target_position", 0),
        "target_y": ("target_position", 1),
        "target_z": ("target_position", 2),
        "final_ee_roll": ("final_ee_rpy", 0),
        "final_ee_pitch": ("final_ee_rpy", 1),
        "final_ee_yaw": ("final_ee_rpy", 2),
        "final_object_x": ("final_object_position", 0),
        "final_object_y": ("final_object_position", 1),
        "final_object_z": ("final_object_position", 2),
    }
    if name not in vector_mappings:
        return None
    source_name, index = vector_mappings[name]
    value = measured.get(source_name)
    if isinstance(value, (list, tuple)) and len(value) > index:
        return value[index]
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if _is_valid_number(value):
        return float(value)
    return float(default)


def _is_valid_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(converted)
