"""Schema constants for stage-2 skill blueprint execution.

This module is intentionally simulator-independent so blueprints can be loaded
and validated before launching Isaac Sim.
"""

from __future__ import annotations

LOGIC_TYPES = {"sequence", "condition", "parallel"}

NODE_TYPES = {"skill", "condition", "terminal", "parallel"}

PARALLEL_MODES = {"all_success"}

POSITION_GOAL_SKILLS = {"move_above", "reach"}

SKILL_TYPES = {
    "move_above",
    "reach",
    "descend",
    "grasp",
    "lift",
    "place",
    "retreat",
    "wait",
    "align_orientation",
}

CONDITION_TYPES = {
    "object_in_gripper",
    "object_near_target",
    "ee_reached_target",
    "timeout",
    "collision_detected",
}

PERFORMANCE_METRICS = {
    "success",
    "execution_steps",
    "execution_time",
    "trajectory_length",
    "final_ee_position",
    "final_ee_orientation",
    "final_ee_rpy",
    "target_position",
    "target_orientation",
    "final_ee_position_error",
    "final_ee_orientation_error",
    "final_ee_linear_velocity",
    "final_ee_linear_speed",
    "average_ee_linear_speed",
    "final_ee_angular_velocity",
    "final_ee_angular_speed",
    "position_converged",
    "orientation_converged",
    "reached_target_within_tolerance",
    "parallel_mode",
    "parallel_goal_count",
    "position_goal_success",
    "orientation_goal_success",
    "timeout",
    "failure_reason",
    "final_object_position",
    "final_object_orientation",
    "final_object_rpy",
    "object_lift_delta",
    "ee_object_distance",
    "min_ee_object_distance",
    "object_target_xy_distance",
    "object_target_position_error",
    "object_target_xy_error",
    "final_position_error",
    "object_displacement",
    "object_stability",
    "gripper_width_start",
    "gripper_width_end",
    "gripper_command_final",
    "performance_risk_level",
    "performance_risk_reason",
    "max_contact_force",
    "collision_count",
    "collision_risk",
    "object_drop_risk",
}

OPTIONAL_UNAVAILABLE_METRICS = {
    "max_contact_force",
    "collision_count",
    "collision_risk",
    "object_drop_risk",
}

SKILL_PARAM_FIELDS = {
    "move_above": {
        "height_offset",
        "xy_offset",
        "speed",
        "position_tolerance",
        "orientation_tolerance",
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "timeout_steps",
    },
    "reach": {
        "target_pose",
        "target_ref",
        "offset",
        "speed",
        "position_tolerance",
        "orientation_tolerance",
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "timeout_steps",
    },
    "descend": {
        "target_height",
        "relative_z",
        "speed",
        "position_tolerance",
        "orientation_tolerance",
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "timeout_steps",
    },
    "grasp": {
        "close_wait_steps",
        "check",
        "timeout_steps",
    },
    "lift": {
        "lift_height",
        "speed",
        "position_tolerance",
        "orientation_tolerance",
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "timeout_steps",
    },
    "place": {
        "place_height",
        "release_height",
        "open_wait_steps",
        "position_tolerance",
        "orientation_tolerance",
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "target_tolerance",
        "timeout_steps",
    },
    "retreat": {
        "retreat_height",
        "speed",
        "position_tolerance",
        "orientation_tolerance",
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "timeout_steps",
    },
    "wait": {
        "wait_steps",
        "gripper",
    },
    "align_orientation": {
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "orientation_tolerance",
        "angular_speed",
        "timeout_steps",
    },
}

PARALLEL_NODE_PARAM_FIELDS = {
    "timeout_steps",
    "gripper",
}

PARALLEL_GOAL_PARAM_FIELDS = {
    "move_above": {
        "height_offset",
        "xy_offset",
        "speed",
        "position_tolerance",
    },
    "reach": {
        "target_pose",
        "target_ref",
        "offset",
        "speed",
        "position_tolerance",
    },
    "align_orientation": {
        "orientation_mode",
        "keep_top_down",
        "fixed_yaw",
        "target_rpy",
        "orientation_tolerance",
        "angular_speed",
    },
}

RESERVED_LOGIC_TODO = {
    "jump",
    "loop",
    "fallback",
    "retry",
}
