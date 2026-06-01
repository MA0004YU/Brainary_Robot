"""Termination terms for Stage 1.1 constrained position reach."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .utils import (
    get_ee_and_target_position_env,
    get_stage1_orientation_terms,
    get_stage1_orientation_tolerance,
    get_stage1_position_tolerance,
)


def position_success(
    env,
    threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Pose success when normalized position and orientation errors are below 1.0."""
    del threshold  # kept for config compatibility; Stage 1.1B uses dynamic tolerance
    ee_position, target_position = get_ee_and_target_position_env(env, asset_cfg, command_name)
    pos_tol_xyz = get_stage1_position_tolerance(env)
    pos_error_vec = target_position - ee_position
    normalized_pos_error = torch.norm(pos_error_vec / torch.clamp(pos_tol_xyz, min=1e-6), dim=1)
    _, orientation_angle_error, _, _, _ = get_stage1_orientation_terms(env, asset_cfg, command_name)
    ori_tol_xyz = get_stage1_orientation_tolerance(env)
    ori_tol_scalar = torch.clamp(ori_tol_xyz[:, 0], min=1e-6)
    normalized_ori_error = orientation_angle_error / ori_tol_scalar
    return (normalized_pos_error < 1.0) & (normalized_ori_error < 1.0)


def nan_or_inf_abort(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Optional safety termination for NaN/Inf state."""
    robot: Articulation = env.scene[asset_cfg.name]
    invalid_q = ~torch.isfinite(robot.data.joint_pos).all(dim=1)
    invalid_dq = ~torch.isfinite(robot.data.joint_vel).all(dim=1)
    return torch.logical_or(invalid_q, invalid_dq)
