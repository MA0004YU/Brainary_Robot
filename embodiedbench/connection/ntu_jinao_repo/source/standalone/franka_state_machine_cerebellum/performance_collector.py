"""Collect measured performance labels for stage-2 predictor datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any
import math

from skill_blueprint_schema import OPTIONAL_UNAVAILABLE_METRICS, PERFORMANCE_METRICS

_NULL_IS_VALID_METRICS = {"failure_reason", "performance_risk_reason"}


class PerformanceCollector:
    """Builds one predictor training sample per executed skill or parallel node."""

    def __init__(self, sim_dt: float, decimation: int):
        self.step_time = float(sim_dt) * int(decimation)
        self.missing_metric_counts: Counter[str] = Counter()

    def collect(self, episode_id: str, blueprint_id: str, node, result) -> dict[str, Any]:
        requested = list(node.performance_query)
        measured: dict[str, Any] = {}
        missing: dict[str, str] = {}
        available = self._compute_available_metrics(node, result)

        for metric in requested:
            if metric not in PERFORMANCE_METRICS:
                measured[metric] = None
                missing[metric] = "unsupported_metric"
            elif metric in available:
                measured[metric] = available[metric]
                if available[metric] is None and metric not in _NULL_IS_VALID_METRICS:
                    missing[metric] = self._missing_reason(metric, node, result)
            else:
                measured[metric] = None
                missing[metric] = self._missing_reason(metric, node, result)

        for metric in missing:
            self.missing_metric_counts[metric] += 1

        skill_name = node.skill if node.type == "skill" else "parallel"
        target = node.target if node.type == "skill" else getattr(result, "target", "parallel")
        skill_params = dict(node.params)
        if node.type == "parallel":
            skill_params = {
                **skill_params,
                "parallel_mode": node.parallel_mode,
                "goals": node.goals,
            }

        return {
            "sample_id": f"{episode_id}_{node.node_id}",
            "episode_id": episode_id,
            "blueprint_id": blueprint_id,
            "node_id": node.node_id,
            "node_type": node.type,
            "skill": skill_name,
            "target": target,
            "scene_state_before": result.pre_state,
            "skill_params": skill_params,
            "performance_query": requested,
            "measured_performance": measured,
            "measured_performance_missing": missing,
            "scene_state_after": result.post_state,
        }

    def _compute_available_metrics(self, node, result) -> dict[str, Any]:
        before = result.pre_state
        after = result.post_state
        steps = result.step_records
        cube_before = before["cube_pose"]
        cube_after = after["cube_pose"]
        target_after = after["target_pose"]
        ee_after = after["ee_pose"]
        ee_positions = [before["ee_pose"][:3]]
        ee_positions.extend(row["ee_pose"][:3] for row in steps)
        if not steps or steps[-1].get("ee_pose") != after["ee_pose"]:
            ee_positions.append(after["ee_pose"][:3])

        final_ee_error = None
        if result.target_pose is not None:
            final_ee_error = _distance(ee_after[:3], result.target_pose[:3])
        trajectory_length = _path_length(ee_positions)
        execution_time = result.num_steps * self.step_time
        target_position = _pose_position(result.target_pose)
        target_orientation = _pose_orientation_xyzw(result.target_orientation or result.target_pose)
        final_ee_orientation = _pose_orientation_xyzw(ee_after)
        final_object_orientation = _pose_orientation_xyzw(cube_after)
        final_ee_linear_velocity = _final_linear_velocity(ee_positions, self.step_time)
        final_ee_linear_speed = _norm(final_ee_linear_velocity) if final_ee_linear_velocity is not None else None
        average_ee_linear_speed = trajectory_length / execution_time if execution_time > 0.0 else None

        orientation_absent = self._orientation_goal_absent(node)
        orientation_mode_none = self._orientation_mode_none(node)

        final_ee_orientation_error = getattr(result, "final_ee_orientation_error", None)
        orientation_converged = getattr(result, "orientation_converged", None)
        position_converged = getattr(result, "position_converged", None)
        position_goal_success = getattr(result, "position_goal_success", None)
        orientation_goal_success = getattr(result, "orientation_goal_success", None)

        if orientation_absent or orientation_mode_none:
            orientation_converged = True if orientation_converged is None else orientation_converged
            orientation_goal_success = True if orientation_goal_success is None else orientation_goal_success
            if final_ee_orientation_error is None:
                final_ee_orientation_error = None

        parallel_mode = getattr(result, "parallel_mode", None)
        parallel_goal_count = getattr(result, "parallel_goal_count", None)
        if node.type == "parallel":
            parallel_mode = parallel_mode or node.parallel_mode or "all_success"
            if parallel_goal_count is None and node.goals:
                parallel_goal_count = int("position_goal" in node.goals) + int("orientation_goal" in node.goals)

        target_tolerance = _target_tolerance(node)
        reached_target_within_tolerance = None
        if final_ee_error is not None:
            reached_target_within_tolerance = final_ee_error <= target_tolerance
        elif position_converged is not None:
            reached_target_within_tolerance = bool(position_converged)

        object_target_position_error = _distance(cube_after[:3], target_after[:3])
        object_target_xy_error = _distance(cube_after[:2], target_after[:2])
        risk_level, risk_reason = _performance_risk(
            node=node,
            result=result,
            final_ee_error=final_ee_error,
            target_tolerance=target_tolerance,
            object_target_xy_error=object_target_xy_error,
            orientation_converged=orientation_converged,
        )

        metrics = {
            "success": result.success,
            "execution_steps": result.num_steps,
            "execution_time": execution_time,
            "trajectory_length": trajectory_length,
            "final_ee_position": _pose_position(ee_after),
            "final_ee_orientation": final_ee_orientation,
            "final_ee_rpy": _quat_xyzw_to_rpy(final_ee_orientation),
            "target_position": target_position,
            "target_orientation": target_orientation,
            "final_ee_position_error": final_ee_error,
            "final_ee_orientation_error": final_ee_orientation_error,
            "final_ee_linear_velocity": final_ee_linear_velocity,
            "final_ee_linear_speed": final_ee_linear_speed,
            "average_ee_linear_speed": average_ee_linear_speed,
            "final_ee_angular_velocity": None,
            "final_ee_angular_speed": None,
            "position_converged": position_converged,
            "orientation_converged": orientation_converged,
            "reached_target_within_tolerance": reached_target_within_tolerance,
            "parallel_mode": parallel_mode,
            "parallel_goal_count": parallel_goal_count,
            "position_goal_success": position_goal_success,
            "orientation_goal_success": orientation_goal_success,
            "timeout": result.timeout,
            "failure_reason": result.failure_reason,
            "final_object_position": _pose_position(cube_after),
            "final_object_orientation": final_object_orientation,
            "final_object_rpy": _quat_xyzw_to_rpy(final_object_orientation),
            "object_lift_delta": float(cube_after[2]) - float(cube_before[2]),
            "ee_object_distance": _distance(ee_after[:3], cube_after[:3]),
            "min_ee_object_distance": _min_ee_object_distance(steps, before, after),
            "object_target_xy_distance": _distance(cube_after[:2], target_after[:2]),
            "object_target_position_error": object_target_position_error,
            "object_target_xy_error": object_target_xy_error,
            "final_position_error": _distance(cube_after[:3], target_after[:3]),
            "object_displacement": _distance(cube_after[:3], cube_before[:3]),
            "object_stability": _object_stability([row["cube_pose"][:3] for row in steps]),
            "gripper_width_start": before["gripper_width"],
            "gripper_width_end": after["gripper_width"],
            "gripper_command_final": result.final_action[-1] if result.final_action else None,
            "performance_risk_level": risk_level,
            "performance_risk_reason": risk_reason,
            "max_contact_force": None,
            "collision_count": None,
            "collision_risk": None,
            "object_drop_risk": None,
        }
        if getattr(result, "orientation_error_type", None):
            metrics["orientation_error_type"] = result.orientation_error_type
        return metrics

    def _orientation_goal_absent(self, node) -> bool:
        if node.type != "parallel":
            return False
        goals = node.goals or {}
        return goals.get("orientation_goal") is None

    def _orientation_mode_none(self, node) -> bool:
        if node.type == "parallel":
            orientation_goal = (node.goals or {}).get("orientation_goal")
            if orientation_goal is None:
                return False
            mode = str((orientation_goal.get("params") or {}).get("orientation_mode", ""))
            return mode == "none"
        if node.type == "skill" and node.skill == "align_orientation":
            return str(node.params.get("orientation_mode", "")) == "none"
        return False

    def _missing_reason(self, metric: str, node, result) -> str:
        if metric == "final_ee_orientation_error":
            if self._orientation_goal_absent(node):
                return "orientation_goal_absent"
            if self._orientation_mode_none(node):
                return "orientation_mode_none"
            if getattr(result, "final_ee_orientation_error", None) is None:
                return "orientation_error_unavailable"
        if metric == "orientation_converged" and self._orientation_goal_absent(node):
            return "orientation_goal_absent"
        if metric == "orientation_goal_success" and self._orientation_goal_absent(node):
            return "orientation_goal_absent"
        if metric in OPTIONAL_UNAVAILABLE_METRICS:
            return "sensor_unavailable_todo"
        if metric == "final_ee_position_error":
            return "skill_has_no_explicit_target_pose"
        if metric == "gripper_command_final":
            return "no_action_emitted"
        if metric in {"target_position", "target_orientation"}:
            return "skill_has_no_explicit_target_pose"
        if metric in {"final_ee_linear_velocity", "final_ee_linear_speed"}:
            return "insufficient_motion_history"
        if metric in {"final_ee_angular_velocity", "final_ee_angular_speed"}:
            return "angular_velocity_unavailable"
        if metric == "reached_target_within_tolerance":
            return "target_tolerance_unavailable"
        if metric == "performance_risk_reason":
            return "no_risk_detected"
        if metric in {"parallel_mode", "parallel_goal_count", "position_converged", "position_goal_success"}:
            if node.type != "parallel" and metric.startswith("parallel"):
                return "not_parallel_node"
        return "not_available"


def _distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _norm(values: list[float] | None) -> float | None:
    if values is None:
        return None
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _pose_position(pose: Any) -> list[float] | None:
    if isinstance(pose, dict):
        pose = pose.get("position") or pose.get("pos") or pose.get("translation")
    if isinstance(pose, (list, tuple)) and len(pose) >= 3:
        return [float(pose[0]), float(pose[1]), float(pose[2])]
    return None


def _pose_orientation_xyzw(pose: Any) -> list[float] | None:
    if isinstance(pose, dict):
        quat = pose.get("orientation") or pose.get("quat") or pose.get("quaternion")
        if isinstance(quat, (list, tuple)) and len(quat) >= 4:
            return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    if isinstance(pose, (list, tuple)) and len(pose) >= 7:
        # Isaac Lab poses store quaternions as [w, x, y, z]; predictor feedback uses [x, y, z, w].
        return [float(pose[4]), float(pose[5]), float(pose[6]), float(pose[3])]
    if isinstance(pose, (list, tuple)) and len(pose) == 4:
        return [float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])]
    return None


def _quat_xyzw_to_rpy(quat: list[float] | None) -> list[float] | None:
    if quat is None or len(quat) < 4:
        return None
    x, y, z, w = [float(value) for value in quat[:4]]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def _final_linear_velocity(positions: list[list[float]], dt: float) -> list[float] | None:
    if dt <= 0.0 or len(positions) < 2:
        return None
    window = min(5, len(positions) - 1)
    if window <= 0:
        return None
    start = positions[-1 - window]
    end = positions[-1]
    elapsed = window * dt
    if elapsed <= 0.0:
        return None
    return [(float(end[idx]) - float(start[idx])) / elapsed for idx in range(3)]


def _target_tolerance(node) -> float:
    params = getattr(node, "params", {}) or {}
    if "target_tolerance" in params:
        return float(params["target_tolerance"])
    if "position_tolerance" in params:
        return float(params["position_tolerance"])
    goals = getattr(node, "goals", None) or {}
    position_params = ((goals.get("position_goal") or {}).get("params") or {})
    if "position_tolerance" in position_params:
        return float(position_params["position_tolerance"])
    return 0.05


def _performance_risk(
    node,
    result,
    final_ee_error: float | None,
    target_tolerance: float,
    object_target_xy_error: float,
    orientation_converged: bool | None,
) -> tuple[str, str | None]:
    if result.timeout:
        return "high", "timeout_risk"
    if result.failure_reason:
        return "high", str(result.failure_reason)
    if orientation_converged is False:
        return "medium", "orientation_not_converged"
    if final_ee_error is not None and final_ee_error > max(target_tolerance * 1.5, 0.03):
        return "medium", "final_position_error_too_large"
    if getattr(node, "skill", None) == "place" and object_target_xy_error > max(target_tolerance * 1.5, 0.06):
        return "medium", "object_target_xy_error_too_large"
    return "low", None


def _path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_distance(prev, cur) for prev, cur in zip(points[:-1], points[1:]))


def _min_ee_object_distance(steps: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any]) -> float:
    distances = [_distance(before["ee_pose"][:3], before["cube_pose"][:3]), _distance(after["ee_pose"][:3], after["cube_pose"][:3])]
    distances.extend(_distance(row["ee_pose"][:3], row["cube_pose"][:3]) for row in steps)
    return min(distances)


def _object_stability(cube_positions: list[list[float]]) -> float:
    if not cube_positions:
        return 0.0
    tail = cube_positions[-min(10, len(cube_positions)) :]
    final = tail[-1]
    return max(_distance(pos, final) for pos in tail)
