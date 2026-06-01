"""Reward terms for Stage 1.1 constrained position reach."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .utils import (
    get_ee_and_target_position_env,
    get_joint_limit_terms,
    get_stage1_orientation_terms,
    get_stage1_orientation_tolerance,
    get_stage1_position_tolerance,
)


def stage1_position_reach_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Stage 1.2A tolerance-conditioned pose reach reward."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    ee_position, target_position = get_ee_and_target_position_env(env, asset_cfg, command_name)
    pos_tol_xyz = get_stage1_position_tolerance(env)
    ori_tol_xyz = get_stage1_orientation_tolerance(env)

    pos_error_vec = target_position - ee_position
    pos_error = torch.norm(pos_error_vec, dim=1)

    eps = 1e-6
    normalized_pos_error_vec = pos_error_vec / torch.clamp(pos_tol_xyz, min=eps)
    normalized_pos_error = torch.norm(normalized_pos_error_vec, dim=1)

    pos_success = normalized_pos_error < 1.0

    _, orientation_angle_error, _, _, quat_dot_abs = get_stage1_orientation_terms(env, asset_cfg, command_name)
    ori_tol_scalar = torch.clamp(ori_tol_xyz[:, 0], min=eps)
    normalized_ori_error = orientation_angle_error / ori_tol_scalar
    ori_success = normalized_ori_error < 1.0
    pose_success = pos_success & ori_success

    r_pos = torch.exp(-2.0 * normalized_pos_error)
    r_dist = -0.5 * normalized_pos_error
    r_pos_success = torch.where(
        pos_success,
        torch.full_like(pos_error, 3.0),
        torch.zeros_like(pos_error),
    )
    r_ori = torch.exp(-2.0 * normalized_ori_error)
    r_ori_dist = -0.12 * normalized_ori_error
    r_ori_success = torch.where(
        ori_success,
        torch.full_like(normalized_ori_error, 3.0),
        torch.zeros_like(normalized_ori_error),
    )
    r_pose_success = torch.where(
        pose_success,
        torch.full_like(pos_error, 10.0),
        torch.zeros_like(pos_error),
    )

    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    dq = robot.data.joint_vel[:, joint_ids]

    q = robot.data.joint_pos[:, joint_ids]
    joint_lower, joint_upper, joint_range = get_joint_limit_terms(robot, joint_ids)
    margin_to_lower = q - joint_lower
    margin_to_upper = joint_upper - q
    joint_limit_margin = torch.minimum(margin_to_lower, margin_to_upper) / joint_range

    r_action = -0.005 * torch.mean(torch.square(action), dim=1)
    r_smooth = -0.005 * torch.mean(torch.square(action - prev_action), dim=1)
    r_dq = -0.0005 * torch.mean(torch.square(dq), dim=1)
    margin_threshold = 0.08
    limit_violation = torch.relu(margin_threshold - joint_limit_margin)
    r_limit = -0.05 * torch.mean(torch.square(limit_violation), dim=1)

    if (
        not hasattr(env, "stage1_prev_normalized_pos_error")
        or env.stage1_prev_normalized_pos_error is None
        or env.stage1_prev_normalized_pos_error.shape != normalized_pos_error.shape
    ):
        env.stage1_prev_normalized_pos_error = normalized_pos_error.detach().clone()
    reset_mask = env.episode_length_buf == 0
    env.stage1_prev_normalized_pos_error[reset_mask] = normalized_pos_error.detach()[reset_mask]
    pos_progress = env.stage1_prev_normalized_pos_error - normalized_pos_error.detach()
    pos_progress = torch.clamp(pos_progress, -0.5, 0.5)
    r_pos_progress = 1.0 * pos_progress
    env.stage1_prev_normalized_pos_error = normalized_pos_error.detach().clone()

    if (
        not hasattr(env, "stage1_prev_normalized_ori_error")
        or env.stage1_prev_normalized_ori_error is None
        or env.stage1_prev_normalized_ori_error.shape != normalized_ori_error.shape
    ):
        env.stage1_prev_normalized_ori_error = normalized_ori_error.detach().clone()
    env.stage1_prev_normalized_ori_error[reset_mask] = normalized_ori_error.detach()[reset_mask]
    ori_progress = env.stage1_prev_normalized_ori_error - normalized_ori_error.detach()
    ori_progress = torch.clamp(ori_progress, -0.5, 0.5)
    r_ori_progress = 1.0 * ori_progress
    env.stage1_prev_normalized_ori_error = normalized_ori_error.detach().clone()

    reward = (
        2.0 * r_pos
        + 1.0 * r_dist
        + r_pos_progress
        + r_pos_success
        + 1.0 * r_ori
        + 1.0 * r_ori_dist
        + r_ori_progress
        + r_ori_success
        + r_pose_success
        + r_action
        + r_smooth
        + r_dq
        + r_limit
    )

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["stage1/mean_position_error"] = pos_error.mean()
    env.extras["log"]["stage1/mean_position_tolerance"] = pos_tol_xyz[:, 0].mean()
    env.extras["log"]["stage1/mean_normalized_position_error"] = normalized_pos_error.mean()
    env.extras["log"]["stage1/position_success_rate"] = pos_success.float().mean()
    env.extras["log"]["stage1/mean_orientation_error"] = orientation_angle_error.mean()
    env.extras["log"]["stage1/mean_orientation_tolerance"] = ori_tol_scalar.mean()
    env.extras["log"]["stage1/mean_normalized_orientation_error"] = normalized_ori_error.mean()
    env.extras["log"]["stage1/orientation_success_rate"] = ori_success.float().mean()
    env.extras["log"]["stage1/pose_success_rate"] = pose_success.float().mean()
    env.extras["log"]["stage1/success_rate"] = pose_success.float().mean()
    env.extras["log"]["stage1/mean_quat_dot_abs"] = quat_dot_abs.mean()
    env.extras["log"]["stage1/min_quat_dot_abs"] = quat_dot_abs.min()
    env.extras["log"]["stage1/position_progress_reward_mean"] = r_pos_progress.mean()
    env.extras["log"]["stage1/orientation_progress_reward_mean"] = r_ori_progress.mean()
    env.extras["log"]["stage1/mean_action_magnitude"] = torch.mean(torch.abs(action))
    env.extras["log"]["stage1/mean_joint_velocity"] = torch.mean(torch.abs(dq))
    env.extras["log"]["stage1/mean_joint_limit_margin"] = joint_limit_margin.mean()

    return reward
