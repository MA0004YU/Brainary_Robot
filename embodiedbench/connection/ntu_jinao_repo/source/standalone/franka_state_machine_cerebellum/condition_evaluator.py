"""Condition evaluation for stage-2 skill blueprint execution."""

from __future__ import annotations

from typing import Any
import math


class ConditionEvaluator:
    """Evaluates blueprint conditions from current simulator state."""

    def __init__(self, cerebellum, episode_initial_state: dict[str, Any]):
        self.cerebellum = cerebellum
        self.episode_initial_state = episode_initial_state
        self.last_skill_result = None

    def set_last_skill_result(self, result) -> None:
        self.last_skill_result = result

    def evaluate(self, condition: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        name = str(condition.get("name", ""))
        if name == "object_in_gripper":
            return self._object_in_gripper(condition)
        if name == "object_near_target":
            return self._object_near_target(condition)
        if name == "ee_reached_target":
            return self._ee_reached_target(condition)
        if name == "timeout":
            result = bool(getattr(self.last_skill_result, "timeout", False))
            return result, {"timeout": result}
        if name == "collision_detected":
            # TODO: wire this to a contact force/collision sensor when one is
            # added to the Franka Lift environment configuration.
            return False, {"collision_detected": False, "note": "collision sensor unavailable in current setup"}
        raise ValueError(f"Unsupported condition: {name}")

    def _object_in_gripper(self, condition: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        state = self._scene_state()
        initial_cube_z = float(self.episode_initial_state["cube_pose"][2])
        cube = state["cube_pose"]
        ee = state["ee_pose"]
        lift_delta = float(cube[2]) - initial_cube_z
        distance = _distance(cube[:3], ee[:3])
        lifted_and_near = (
            lift_delta >= float(condition.get("min_lift_delta", 0.03))
            and distance <= float(condition.get("max_ee_object_distance", 0.08))
        )
        gripper_width = float(state["gripper_width"])
        closed_near_object = gripper_width < 0.06 and distance <= float(condition.get("max_ee_object_distance", 0.08))
        result = lifted_and_near or closed_near_object
        return result, {
            "object_lift_delta": lift_delta,
            "ee_object_distance": distance,
            "gripper_width": gripper_width,
            "lifted_and_near": lifted_and_near,
            "closed_near_object": closed_near_object,
        }

    def _object_near_target(self, condition: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        state = self._scene_state()
        cube = state["cube_pose"]
        target = state["target_pose"]
        xy_error = _distance(cube[:2], target[:2])
        z_error = abs(float(cube[2]) - float(target[2]))
        result = (
            xy_error <= float(condition.get("max_xy_error", 0.05))
            and z_error <= float(condition.get("max_z_error", 0.05))
        )
        return result, {"object_target_xy_distance": xy_error, "object_target_z_error": z_error}

    def _ee_reached_target(self, condition: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        state = self._scene_state()
        ee = state["ee_pose"]
        target_name = str(condition.get("target", "target"))
        if target_name == "cube":
            target = state["cube_pose"]
        elif target_name == "target":
            target = state["target_pose"]
        elif target_name == "custom_pose":
            target = condition.get("target_pose", [0.0, 0.0, 0.0])
        else:
            raise ValueError(f"Unsupported ee_reached_target target: {target_name}")
        error = _distance(ee[:3], target[:3])
        result = error <= float(condition.get("max_position_error", 0.03))
        return result, {"final_ee_position_error": error}

    def _scene_state(self) -> dict[str, Any]:
        state = self.cerebellum.get_scene_state()
        return {
            "ee_pose": state["ee_pose_w"],
            "cube_pose": state["cube_pose_w"],
            "target_pose": state["target_pose_w"],
            "gripper_width": state["gripper_width"],
        }


def _distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
