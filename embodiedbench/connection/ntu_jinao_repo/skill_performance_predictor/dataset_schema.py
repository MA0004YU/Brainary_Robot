"""Schema constants for the skill performance predictor dataset."""

from __future__ import annotations

PREDICTOR_SCHEMA_VERSION = "predictor_v1"

SUPPORTED_SKILLS = [
    "move_above",
    "reach",
    "descend",
    "grasp",
    "lift",
    "place",
    "retreat",
    "wait",
    "align_orientation",
    "parallel",
    "unknown",
]

SUPPORTED_TARGETS = [
    "cube",
    "target",
    "current",
    "custom_pose",
    "unknown",
]

FAILURE_REASON_CLASSES = [
    "none",
    "skill_failed",
    "object_not_in_gripper",
    "object_not_near_target",
    "timeout",
    "parallel_timeout",
    "orientation_not_converged",
    "reach_failed",
    "place_failed",
    "unknown",
]

BASE_NUMERIC_FEATURES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "cube_x",
    "cube_y",
    "cube_z",
    "target_x",
    "target_y",
    "target_z",
    "gripper_width",
    "ee_cube_dx",
    "ee_cube_dy",
    "ee_cube_dz",
    "ee_target_dx",
    "ee_target_dy",
    "ee_target_dz",
    "cube_target_dx",
    "cube_target_dy",
    "cube_target_dz",
    "ee_cube_dist",
    "ee_target_dist",
    "cube_target_dist",
    "cube_target_xy_dist",
]

SKILL_PARAM_FEATURES = [
    "height_offset",
    "xy_offset_x",
    "xy_offset_y",
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

SKILL_PARAM_FEATURES_WITH_MASKS = []
for _name in SKILL_PARAM_FEATURES:
    SKILL_PARAM_FEATURES_WITH_MASKS.extend([_name, f"has_{_name}"])

PARALLEL_FEATURES = [
    "has_position_goal",
    "has_orientation_goal",
    "position_goal_speed",
    "orientation_goal_angular_speed",
    "position_goal_tolerance",
    "orientation_goal_tolerance",
]

NUMERIC_FEATURE_NAMES = BASE_NUMERIC_FEATURES + SKILL_PARAM_FEATURES_WITH_MASKS + PARALLEL_FEATURES

REGRESSION_TARGET_NAMES = [
    "execution_steps",
    "execution_time",
    "trajectory_length",
    "final_ee_x",
    "final_ee_y",
    "final_ee_z",
    "target_x",
    "target_y",
    "target_z",
    "final_ee_position_error",
    "final_ee_roll",
    "final_ee_pitch",
    "final_ee_yaw",
    "final_ee_orientation_error",
    "final_ee_linear_speed",
    "average_ee_linear_speed",
    "final_object_x",
    "final_object_y",
    "final_object_z",
    "object_target_position_error",
    "object_target_xy_error",
    "object_lift_delta",
    "ee_object_distance",
    "min_ee_object_distance",
    "object_target_xy_distance",
    "final_position_error",
    "object_displacement",
    "gripper_width_start",
    "gripper_width_end",
]

CLASSIFICATION_TARGET_NAMES = [
    "success",
    "timeout",
    "failure_reason",
]

OPTIONAL_FUTURE_TARGETS = [
    "max_contact_force",
    "collision_count",
    "collision_risk",
    "object_drop_risk",
]


def default_vocab() -> dict[str, dict[str, int]]:
    """Return deterministic vocabularies used by all splits."""

    return {
        "skill": {name: idx for idx, name in enumerate(SUPPORTED_SKILLS)},
        "target": {name: idx for idx, name in enumerate(SUPPORTED_TARGETS)},
        "failure_reason": {name: idx for idx, name in enumerate(FAILURE_REASON_CLASSES)},
    }
