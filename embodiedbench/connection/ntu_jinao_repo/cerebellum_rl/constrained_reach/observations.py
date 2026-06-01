"""Observation construction for Stage 1.1 constrained position reach."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .utils import (
    canonicalize_quat_largest_abs,
    get_ee_and_target_position_env,
    get_joint_limit_terms,
    get_stage1_orientation_terms,
    get_stage1_orientation_tolerance,
    get_stage1_position_tolerance,
)


def constrained_reach_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Build fixed 126-dim observation in the required order."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    # A) robot state (28)
    q = robot.data.joint_pos[:, joint_ids]
    joint_lower, joint_upper, joint_range = get_joint_limit_terms(robot, joint_ids)
    q_norm = 2.0 * (q - joint_lower) / joint_range - 1.0
    q_norm = torch.clamp(q_norm, -1.5, 1.5)

    dq = robot.data.joint_vel[:, joint_ids]
    dq_scale = 5.0
    dq_scaled = torch.clamp(dq / dq_scale, -5.0, 5.0)

    prev_action = env.action_manager.prev_action  # 7
    margin_to_lower = q - joint_lower
    margin_to_upper = joint_upper - q
    joint_limit_margin = torch.minimum(margin_to_lower, margin_to_upper) / joint_range
    joint_limit_margin = torch.clamp(joint_limit_margin, -1.0, 1.0)

    # B) ee current state (13)
    ee_position, target_position = get_ee_and_target_position_env(env, asset_cfg, "ee_pose")
    orientation_error, _, ee_quat_w, target_quat_w, _ = get_stage1_orientation_terms(env, asset_cfg, "ee_pose")
    ee_orientation = canonicalize_quat_largest_abs(ee_quat_w)
    ee_linear_velocity = torch.zeros(env.num_envs, 3, device=env.device)  # 3 (placeholder)
    ee_angular_velocity = torch.zeros(env.num_envs, 3, device=env.device)  # 3 (placeholder)

    # C) target pose/vel/error (25)
    target_orientation = canonicalize_quat_largest_abs(target_quat_w)
    target_linear_velocity = torch.zeros(env.num_envs, 3, device=env.device)  # 3
    target_angular_velocity = torch.zeros(env.num_envs, 3, device=env.device)  # 3
    position_error = target_position - ee_position  # 3
    orientation_error = torch.clamp(orientation_error, -3.14, 3.14)
    linear_velocity_error = torch.zeros(env.num_envs, 3, device=env.device)  # 3
    angular_velocity_error = torch.zeros(env.num_envs, 3, device=env.device)  # 3

    # D) tolerance constraints (12)
    pos_tol = get_stage1_position_tolerance(env)
    ori_tol = get_stage1_orientation_tolerance(env)
    lin_vel_tol = torch.zeros(env.num_envs, 3, device=env.device)
    ang_vel_tol = torch.zeros(env.num_envs, 3, device=env.device)

    # E) obstacle constraints (30) placeholder
    obstacles = torch.zeros(env.num_envs, 30, device=env.device)

    # F) time info (1)
    episode_progress = (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)

    # G) force/torque/contact reserve (7)
    ee_force = torch.zeros(env.num_envs, 3, device=env.device)
    ee_torque = torch.zeros(env.num_envs, 3, device=env.device)
    contact_enabled = torch.zeros(env.num_envs, 1, device=env.device)

    # H) extra reserve (10)
    reserved = torch.zeros(env.num_envs, 10, device=env.device)

    obs = torch.cat(
        [
            q_norm,
            dq_scaled,
            prev_action,
            joint_limit_margin,
            ee_position,
            ee_orientation,
            ee_linear_velocity,
            ee_angular_velocity,
            target_position,
            target_orientation,
            target_linear_velocity,
            target_angular_velocity,
            position_error,
            orientation_error,
            linear_velocity_error,
            angular_velocity_error,
            pos_tol,
            ori_tol,
            lin_vel_tol,
            ang_vel_tol,
            obstacles,
            episode_progress,
            ee_force,
            ee_torque,
            contact_enabled,
            reserved,
        ],
        dim=-1,
    )
    assert obs.shape[-1] == 126
    return obs

