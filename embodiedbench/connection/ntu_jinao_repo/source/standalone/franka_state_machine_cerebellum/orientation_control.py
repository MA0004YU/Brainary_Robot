"""Orientation helpers for stage-2 IK-Rel control.

All deltas are computed in the robot base frame to match Franka IK-Rel action
semantics: action[3:6] = (droll, dpitch, dyaw) as axis-angle increments.
World-frame quaternions from the simulator use Isaac Lab (w, x, y, z) convention.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)

ORIENTATION_MODES = {
    "none",
    "keep_current",
    "fixed_top_down",
    "align_yaw_with_target",
    "custom_rpy",
}


def wrap_to_pi(angle: float) -> float:
    """Wrap scalar angle to [-pi, pi]."""

    return (angle + math.pi) % (2 * math.pi) - math.pi


def top_down_quat_base(device: torch.device, batch: int = 1) -> torch.Tensor:
    """Default Franka top-down grasp quaternion in robot base frame (w, x, y, z)."""

    quat = torch.zeros((batch, 4), device=device)
    quat[:, 1] = 1.0
    return quat


def quat_w_to_base(cerebellum, quat_w: torch.Tensor) -> torch.Tensor:
    """Transform a world-frame quaternion into the robot base frame."""

    robot = cerebellum.unwrapped.scene["robot"]
    _, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        None,
        quat_w,
    )
    return quat_b


def compute_target_orientation_w(
    cerebellum,
    target_ref: str,
    orientation_mode: str,
    params: dict[str, Any],
    current_quat_w: torch.Tensor,
    target_pose_fn,
) -> torch.Tensor | None:
    """Return desired end-effector orientation in world frame, or None if inactive."""

    mode = str(orientation_mode)
    device = cerebellum.device
    if mode == "none":
        return None
    if mode == "keep_current":
        return current_quat_w.clone()
    if mode == "fixed_top_down":
        return _top_down_in_world(cerebellum, params.get("fixed_yaw"))
    if mode == "align_yaw_with_target":
        ref_pose = target_pose_fn(str(target_ref))
        return _align_yaw_top_down_in_world(cerebellum, ref_pose[:, 3:7], params.get("keep_top_down", True))
    if mode == "custom_rpy":
        # TODO: full custom RPY in world frame; first version uses base-frame Euler.
        rpy = params.get("target_rpy", [math.pi, 0.0, 0.0])
        roll = torch.tensor([float(rpy[0])], device=device)
        pitch = torch.tensor([float(rpy[1])], device=device)
        yaw = torch.tensor([float(rpy[2])], device=device)
        desired_b = quat_from_euler_xyz(roll, pitch, yaw)
        return _base_quat_to_world(cerebellum, desired_b)
    raise ValueError(f"Unsupported orientation_mode: {orientation_mode}")


def compute_orientation_delta_in_base_frame(
    cerebellum,
    current_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    angular_speed: float,
) -> torch.Tensor:
    """Compute clipped IK-Rel orientation delta (droll, dpitch, dyaw) in base frame."""

    current_b = quat_w_to_base(cerebellum, current_quat_w)
    target_b = quat_w_to_base(cerebellum, target_quat_w)
    delta_quat = quat_mul(target_b, quat_conjugate(current_b))
    delta_axis_angle = axis_angle_from_quat(delta_quat)
    limit = abs(float(angular_speed))
    return torch.clamp(delta_axis_angle, min=-limit, max=limit)


def compute_orientation_error(
    current_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    yaw_only: bool = False,
) -> tuple[float, str]:
    """Return scalar orientation error and error type label."""

    if yaw_only:
        _, _, cur_yaw = euler_xyz_from_quat(current_quat_w)
        _, _, tgt_yaw = euler_xyz_from_quat(target_quat_w)
        error = abs(wrap_to_pi(float(cur_yaw[0].item() - tgt_yaw[0].item())))
        return error, "yaw_only"
    delta_quat = quat_mul(target_quat_w, quat_conjugate(current_quat_w))
    axis_angle = axis_angle_from_quat(delta_quat)
    error = float(torch.norm(axis_angle[0]).item())
    return error, "axis_angle"


def orientation_converged(
    current_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    tolerance: float,
    yaw_only: bool = False,
) -> bool:
    error, _ = compute_orientation_error(current_quat_w, target_quat_w, yaw_only=yaw_only)
    return error <= float(tolerance)


def _top_down_in_world(cerebellum, fixed_yaw: float | None) -> torch.Tensor:
    desired_b = top_down_quat_base(cerebellum.device)
    if fixed_yaw is not None:
        yaw = torch.tensor([float(fixed_yaw)], device=cerebellum.device)
        yaw_quat = quat_from_euler_xyz(
            torch.zeros(1, device=cerebellum.device),
            torch.zeros(1, device=cerebellum.device),
            yaw,
        )
        desired_b = quat_mul(yaw_quat, desired_b)
    return _base_quat_to_world(cerebellum, desired_b)


def _align_yaw_top_down_in_world(
    cerebellum,
    ref_quat_w: torch.Tensor,
    keep_top_down: bool,
) -> torch.Tensor:
    ref_b = quat_w_to_base(cerebellum, ref_quat_w)
    _, _, ref_yaw = euler_xyz_from_quat(ref_b)
    if keep_top_down:
        desired_b = top_down_quat_base(cerebellum.device)
        yaw_quat = quat_from_euler_xyz(
            torch.zeros(1, device=cerebellum.device),
            torch.zeros(1, device=cerebellum.device),
            ref_yaw,
        )
        desired_b = quat_mul(yaw_quat, desired_b)
    else:
        desired_b = quat_from_euler_xyz(
            torch.zeros(1, device=cerebellum.device),
            torch.zeros(1, device=cerebellum.device),
            ref_yaw,
        )
    return _base_quat_to_world(cerebellum, desired_b)


def _base_quat_to_world(cerebellum, quat_b: torch.Tensor) -> torch.Tensor:
    """Convert base-frame orientation to world frame using robot root pose."""

    from isaaclab.utils.math import combine_frame_transforms

    robot = cerebellum.unwrapped.scene["robot"]
    _, quat_w = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        torch.zeros((1, 3), device=cerebellum.device),
        quat_b,
    )
    return quat_w
