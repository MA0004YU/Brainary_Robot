"""Primitive skill executor for stage-2 Franka skill blueprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from orientation_control import (
    compute_orientation_delta_in_base_frame,
    compute_orientation_error,
    compute_target_orientation_w,
    orientation_converged,
)
from state_machine_cerebellum import CLOSE_GRIPPER, OPEN_GRIPPER, _first_bool, _first_scalar, _tensor_to_list


@dataclass
class PrimitiveSkillResult:
    """Execution result for one blueprint skill or parallel node."""

    node_id: str
    skill: str
    target: str
    params: dict[str, Any]
    start_step: int
    end_step: int
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    success: bool
    failure_reason: str | None
    timeout: bool
    num_steps: int
    target_pose: list[float] | None = None
    final_action: list[float] | None = None
    step_records: list[dict[str, Any]] = field(default_factory=list)
    node_type: str = "skill"
    position_converged: bool | None = None
    orientation_converged: bool | None = None
    final_ee_orientation_error: float | None = None
    orientation_error_type: str | None = None
    position_goal_success: bool | None = None
    orientation_goal_success: bool | None = None
    parallel_mode: str | None = None
    parallel_goal_count: int | None = None
    target_orientation: list[float] | None = None

    def to_trace(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("step_records", None)
        data["pre_state"] = self.pre_state
        data["post_state"] = self.post_state
        return data


class PrimitiveSkillExecutor:
    """Executes blueprint primitive skills through the existing IK-Rel environment."""

    def __init__(self, cerebellum, logger, skill_target_marker=None):
        self.cerebellum = cerebellum
        self.logger = logger
        self.device = cerebellum.device
        self.env = cerebellum.env
        self.unwrapped = cerebellum.unwrapped
        self.last_gripper_command = OPEN_GRIPPER
        self.skill_target_marker = skill_target_marker

    def execute(self, node) -> PrimitiveSkillResult:
        self._log_skill_start(node)
        if node.type == "parallel":
            return self.execute_parallel(node)
        skill = str(node.skill)
        if skill == "move_above":
            return self._execute_move_above(node)
        if skill == "reach":
            return self._execute_reach(node)
        if skill == "descend":
            return self._execute_descend(node)
        if skill == "grasp":
            return self._execute_grasp(node)
        if skill == "lift":
            return self._execute_lift(node)
        if skill == "place":
            return self._execute_place(node)
        if skill == "retreat":
            return self._execute_retreat(node)
        if skill == "wait":
            return self._execute_wait(node)
        if skill == "align_orientation":
            return self._execute_align_orientation(node)
        return self._empty_result(node, success=False, failure_reason="unsupported_skill")

    def execute_parallel(self, node) -> PrimitiveSkillResult:
        """Execute a parallel node: one env.step per loop, converging position and orientation."""

        goals = node.goals or {}
        position_goal = goals.get("position_goal")
        orientation_goal = goals.get("orientation_goal")
        node_params = dict(node.params)
        timeout_steps = int(node_params.get("timeout_steps", 400))
        gripper_mode = str(node_params.get("gripper", "keep"))
        gripper_command = self._resolve_gripper_command(gripper_mode)

        has_position = position_goal is not None
        has_orientation = (
            orientation_goal is not None
            and str((orientation_goal.get("params") or {}).get("orientation_mode", "")) != "none"
        )
        goal_count = int(position_goal is not None) + int(orientation_goal is not None)
        # Require stable simultaneous convergence before leaving a parallel node.
        confirm_steps_required = 3

        position_params = dict((position_goal or {}).get("params") or {})
        orientation_params = dict((orientation_goal or {}).get("params") or {})
        position_tolerance = float(position_params.get("position_tolerance", 0.025)) if has_position else 0.0
        position_speed = float(position_params.get("speed", 0.08)) if has_position else 0.0
        orientation_tolerance = float(orientation_params.get("orientation_tolerance", 0.08)) if orientation_goal else 0.0
        angular_speed = float(orientation_params.get("angular_speed", 0.08)) if orientation_goal else 0.0
        yaw_only = False if orientation_goal else False

        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        target_position = self._compute_position_target_from_goal(position_goal) if has_position else None
        target_orientation_w: torch.Tensor | None = None
        final_action: list[float] | None = None

        position_ok = not has_position
        orientation_ok = not has_orientation
        pos_error = 0.0
        orient_error: float | None = None
        orient_error_type: str | None = None
        timed_out = False
        failure_reason: str | None = None
        success_streak = 0

        for _ in range(max(0, timeout_steps)):
            ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
            if has_position and target_position is not None:
                pos_error = float(torch.norm(target_position[:, :3] - ee_pos, dim=-1)[0].item())
                position_ok = pos_error <= position_tolerance
            if orientation_goal is not None:
                if not has_orientation:
                    orientation_ok = True
                else:
                    target_orientation_w = compute_target_orientation_w(
                        self.cerebellum,
                        str(orientation_goal.get("target", "")),
                        str(orientation_params.get("orientation_mode", "none")),
                        orientation_params,
                        ee_quat,
                        self._target_pose,
                    )
                    if target_orientation_w is None:
                        orientation_ok = True
                        orient_error = None
                    else:
                        orient_error, orient_error_type = compute_orientation_error(ee_quat, target_orientation_w, yaw_only=yaw_only)
                        orientation_ok = orient_error <= orientation_tolerance
            self._visualize_skill_target(target_position, target_orientation_w, state_name="PARALLEL_TARGET")

            if position_ok and orientation_ok:
                success_streak += 1
                if success_streak >= confirm_steps_required:
                    break
            else:
                success_streak = 0

            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            if has_position and target_position is not None:
                delta_b = self.cerebellum._compute_position_delta_in_base_frame(target_position[:, :3], ee_pos)
                action[:, :3] = torch.clamp(delta_b, min=-abs(position_speed), max=abs(position_speed))
            if has_orientation and target_orientation_w is not None and not orientation_ok:
                action[:, 3:6] = compute_orientation_delta_in_base_frame(
                    self.cerebellum, ee_quat, target_orientation_w, angular_speed
                )
            action[:, 6] = gripper_command
            final_action = _tensor_to_list(action[0])

            position_goal_log = None
            if has_position and target_position is not None:
                position_goal_log = {
                    "target": str((position_goal or {}).get("target", "")),
                    "target_position": _tensor_to_list(target_position[0, :3]),
                    "position_error": pos_error,
                }
            orientation_goal_log = None
            if orientation_goal is not None:
                orientation_goal_log = {
                    "target": str(orientation_goal.get("target", "")),
                    "orientation_mode": str(orientation_params.get("orientation_mode", "none")),
                    "target_orientation": _tensor_to_list(target_orientation_w[0]) if target_orientation_w is not None else None,
                    "orientation_error": orient_error,
                }

            self._step(
                action,
                node.node_id,
                "parallel",
                "PARALLEL",
                step_records,
                node_type="parallel",
                position_goal=position_goal_log,
                orientation_goal=orientation_goal_log,
                target_pose_for_diag=target_position,
                target_orientation_for_diag=target_orientation_w,
            )
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                failure_reason = "env_done"
                break
        else:
            timed_out = not (position_ok and orientation_ok)
            if timed_out:
                if not orientation_ok and has_orientation:
                    failure_reason = "orientation_not_converged"
                elif not position_ok and has_position:
                    failure_reason = "position_not_converged"
                else:
                    failure_reason = "parallel_timeout"

        success = position_ok and orientation_ok and success_streak >= confirm_steps_required and failure_reason is None
        post_state = self._scene_state()
        parallel_mode = str(node.parallel_mode or "all_success")
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill="parallel",
            target=self._parallel_target_label(goals),
            params=dict(node_params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=success,
            failure_reason=None if success else failure_reason,
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=_tensor_to_list(target_position[0]) if target_position is not None else None,
            target_orientation=_tensor_to_list(target_orientation_w[0]) if target_orientation_w is not None else None,
            final_action=final_action,
            step_records=step_records,
            node_type="parallel",
            position_converged=position_ok,
            orientation_converged=orientation_ok,
            final_ee_orientation_error=orient_error,
            orientation_error_type=orient_error_type,
            position_goal_success=position_ok,
            orientation_goal_success=orientation_ok if orientation_goal is not None else None,
            parallel_mode=parallel_mode,
            parallel_goal_count=goal_count,
        )

    def _execute_align_orientation(self, node) -> PrimitiveSkillResult:
        params = node.params
        orientation_mode = str(params.get("orientation_mode", "none"))
        if orientation_mode == "none":
            state = self._scene_state()
            step = self.cerebellum.global_step
            return PrimitiveSkillResult(
                node_id=node.node_id,
                skill=str(node.skill),
                target=str(node.target),
                params=dict(params),
                start_step=step,
                end_step=step,
                pre_state=state,
                post_state=state,
                success=True,
                failure_reason=None,
                timeout=False,
                num_steps=0,
                orientation_converged=True,
                orientation_goal_success=True,
                final_ee_orientation_error=None,
                orientation_error_type="none",
            )

        tolerance = float(params.get("orientation_tolerance", 0.08))
        angular_speed = float(params.get("angular_speed", 0.08))
        timeout_steps = int(params.get("timeout_steps", 200))
        yaw_only = False
        gripper_command = self.last_gripper_command

        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        final_action: list[float] | None = None
        target_orientation_w: torch.Tensor | None = None
        orient_error: float | None = None
        orient_error_type: str | None = None
        converged = False
        timed_out = False

        for _ in range(max(0, timeout_steps)):
            ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
            target_orientation_w = compute_target_orientation_w(
                self.cerebellum,
                str(node.target),
                orientation_mode,
                params,
                ee_quat,
                self._target_pose,
            )
            if target_orientation_w is None or orientation_mode == "keep_current":
                converged = True
                break
            orient_error, orient_error_type = compute_orientation_error(ee_quat, target_orientation_w, yaw_only=yaw_only)
            if orient_error <= tolerance:
                converged = True
                break
            self._visualize_skill_target(None, target_orientation_w, state_name="ALIGN_ORIENTATION_TARGET")
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, 3:6] = compute_orientation_delta_in_base_frame(
                self.cerebellum, ee_quat, target_orientation_w, angular_speed
            )
            action[:, 6] = gripper_command
            final_action = _tensor_to_list(action[0])
            orientation_goal_log = {
                "target": str(node.target),
                "orientation_mode": orientation_mode,
                "target_orientation": _tensor_to_list(target_orientation_w[0]),
                "orientation_error": orient_error,
            }
            self._step(
                action,
                node.node_id,
                str(node.skill),
                "ALIGN_ORIENTATION",
                step_records,
                orientation_goal=orientation_goal_log,
                target_orientation_for_diag=target_orientation_w,
            )
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                break
        else:
            timed_out = not converged

        post_state = self._scene_state()
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target),
            params=dict(params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=converged,
            failure_reason=None if converged else ("orientation_not_converged" if not timed_out else "timeout"),
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_orientation=_tensor_to_list(target_orientation_w[0]) if target_orientation_w is not None else None,
            final_action=final_action,
            step_records=step_records,
            orientation_converged=converged,
            orientation_goal_success=converged,
            final_ee_orientation_error=orient_error,
            orientation_error_type=orient_error_type,
        )

    def _execute_move_above(self, node) -> PrimitiveSkillResult:
        params = node.params
        target_pose = self._target_pose(str(node.target))
        xy_offset = params.get("xy_offset", [0.0, 0.0])
        target_pose[:, 0] += float(xy_offset[0])
        target_pose[:, 1] += float(xy_offset[1])
        target_pose[:, 2] += float(params.get("height_offset", 0.12))
        target_pose[:, 3:7] = self._target_orientation_for_motion(node=node, target_pose=target_pose, current_quat_w=None)
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.025)),
            orientation_tolerance=float(params.get("orientation_tolerance", 0.08)),
            timeout_steps=int(params.get("timeout_steps", 250)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=self.last_gripper_command,
            state_name="MOVE_ABOVE",
        )

    def _execute_reach(self, node) -> PrimitiveSkillResult:
        params = node.params
        if "target_pose" in params:
            target_pose = self._pose_from_xyz(params["target_pose"][:3])
        else:
            target_ref = str(params.get("target_ref", node.target or "target"))
            target_pose = self._target_pose(target_ref)
            offset = params.get("offset", [0.0, 0.0, 0.0])
            target_pose[:, :3] += torch.tensor(offset[:3], device=self.device).reshape(1, 3)
        target_pose[:, 3:7] = self._target_orientation_for_motion(node=node, target_pose=target_pose, current_quat_w=None)
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.025)),
            orientation_tolerance=float(params.get("orientation_tolerance", 0.08)),
            timeout_steps=int(params.get("timeout_steps", 250)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=self.last_gripper_command,
            state_name="REACH",
        )

    def _execute_descend(self, node) -> PrimitiveSkillResult:
        params = node.params
        ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
        target_pose = torch.cat([ee_pos.clone(), ee_quat.clone()], dim=-1)
        ref_pose = self._target_pose(str(node.target)) if str(node.target) != "current" else None
        # Default behavior: during descend, keep x/y aligned with the target reference
        # (typically cube) so the gripper does not descend with residual lateral offset.
        if ref_pose is not None:
            target_pose[:, 0] = ref_pose[:, 0]
            target_pose[:, 1] = ref_pose[:, 1]
        if params.get("target_height") is not None:
            target_pose[:, 2] = float(params["target_height"])
        elif params.get("relative_z") is not None:
            target_pose[:, 2] = ee_pos[:, 2] - abs(float(params["relative_z"]))
        else:
            if ref_pose is not None:
                target_pose[:, 2] = ref_pose[:, 2]
        target_pose[:, 3:7] = self._target_orientation_for_motion(node=node, target_pose=target_pose, current_quat_w=ee_quat)

        def _refresh_descend_target(pose: torch.Tensor) -> torch.Tensor:
            if ref_pose is None:
                return pose
            refreshed = pose.clone()
            cur_ref = self._target_pose(str(node.target))
            refreshed[:, 0] = cur_ref[:, 0]
            refreshed[:, 1] = cur_ref[:, 1]
            return refreshed

        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.015)),
            orientation_tolerance=float(params.get("orientation_tolerance", 0.08)),
            timeout_steps=int(params.get("timeout_steps", 150)),
            speed=float(params.get("speed", 0.035)),
            gripper_command=self.last_gripper_command,
            state_name="DESCEND",
            dynamic_target_fn=_refresh_descend_target,
        )

    def _execute_grasp(self, node) -> PrimitiveSkillResult:
        params = node.params
        wait_steps = min(int(params.get("close_wait_steps", 30)), int(params.get("timeout_steps", 80)))
        result = self._run_hold_skill(node, CLOSE_GRIPPER, wait_steps, "GRASP")
        self.last_gripper_command = CLOSE_GRIPPER
        return result

    def _execute_lift(self, node) -> PrimitiveSkillResult:
        params = node.params
        ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
        target_pose = torch.cat([ee_pos.clone(), ee_quat.clone()], dim=-1)
        target_pose[:, 2] = ee_pos[:, 2] + float(params.get("lift_height", 0.18))
        target_pose[:, 3:7] = self._target_orientation_for_motion(node=node, target_pose=target_pose, current_quat_w=ee_quat)
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.025)),
            orientation_tolerance=float(params.get("orientation_tolerance", 0.08)),
            timeout_steps=int(params.get("timeout_steps", 200)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=CLOSE_GRIPPER,
            state_name="LIFT",
        )

    def _execute_place(self, node) -> PrimitiveSkillResult:
        params = node.params
        target_pose = self._target_pose(str(node.target))
        target_pose[:, 2] = target_pose[:, 2] + float(params.get("place_height", 0.045))
        target_pose[:, 3:7] = self._target_orientation_for_motion(node=node, target_pose=target_pose, current_quat_w=None)
        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        final_action: list[float] | None = None
        timeout_steps = int(params.get("timeout_steps", 300))
        move_budget = max(1, timeout_steps - int(params.get("open_wait_steps", 30)))
        reached = self._move_to_pose(
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.035)),
            orientation_tolerance=float(params.get("orientation_tolerance", 0.08)),
            max_steps=move_budget,
            speed=float(params.get("speed", 0.08)),
            gripper_command=CLOSE_GRIPPER,
            node_id=node.node_id,
            active_skill=str(node.skill),
            state_name="PLACE_MOVE",
            step_records=step_records,
        )
        if step_records:
            final_action = step_records[-1].get("action")
        timed_out = not reached
        failure_reason = None if reached else "timeout"
        if reached:
            hold_steps = min(int(params.get("open_wait_steps", 30)), max(0, timeout_steps - len(step_records)))
            self.last_gripper_command = OPEN_GRIPPER
            final_action = self._hold(OPEN_GRIPPER, hold_steps, node.node_id, str(node.skill), "PLACE_OPEN", step_records)
        post_state = self._scene_state()
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target),
            params=dict(node.params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=reached,
            failure_reason=failure_reason,
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=_tensor_to_list(target_pose[0]),
            final_action=final_action,
            step_records=step_records,
        )

    def _execute_retreat(self, node) -> PrimitiveSkillResult:
        params = node.params
        ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
        target_pose = torch.cat([ee_pos.clone(), ee_quat.clone()], dim=-1)
        target_pose[:, 2] = ee_pos[:, 2] + float(params.get("retreat_height", 0.15))
        target_pose[:, 3:7] = self._target_orientation_for_motion(node=node, target_pose=target_pose, current_quat_w=ee_quat)
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.035)),
            orientation_tolerance=float(params.get("orientation_tolerance", 0.08)),
            timeout_steps=int(params.get("timeout_steps", 150)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=OPEN_GRIPPER,
            state_name="RETREAT",
        )

    def _execute_wait(self, node) -> PrimitiveSkillResult:
        params = node.params
        gripper = str(params.get("gripper", "keep"))
        if gripper == "open":
            command = OPEN_GRIPPER
        elif gripper == "close":
            command = CLOSE_GRIPPER
        else:
            command = self.last_gripper_command
        return self._run_hold_skill(node, command, int(params.get("wait_steps", 30)), "WAIT")

    def _run_motion_skill(
        self,
        node,
        target_pose: torch.Tensor,
        tolerance: float,
        orientation_tolerance: float,
        timeout_steps: int,
        speed: float,
        gripper_command: float,
        state_name: str,
        dynamic_target_fn=None,
    ) -> PrimitiveSkillResult:
        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        reached = self._move_to_pose(
            target_pose=target_pose,
            tolerance=tolerance,
            orientation_tolerance=orientation_tolerance,
            max_steps=timeout_steps,
            speed=speed,
            gripper_command=gripper_command,
            node_id=node.node_id,
            active_skill=str(node.skill),
            state_name=state_name,
            step_records=step_records,
            dynamic_target_fn=dynamic_target_fn,
        )
        self.last_gripper_command = gripper_command
        post_state = self._scene_state()
        timed_out = not reached
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target),
            params=dict(node.params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=reached,
            failure_reason=None if reached else "timeout",
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=_tensor_to_list(target_pose[0]),
            final_action=step_records[-1].get("action") if step_records else None,
            step_records=step_records,
        )

    def _run_hold_skill(self, node, gripper_command: float, steps: int, state_name: str) -> PrimitiveSkillResult:
        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        final_action = self._hold(gripper_command, steps, node.node_id, str(node.skill), state_name, step_records)
        post_state = self._scene_state()
        self.last_gripper_command = gripper_command
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target or "current"),
            params=dict(node.params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=not self.cerebellum._episode_done(),
            failure_reason="env_done" if self.cerebellum._episode_done() else None,
            timeout=False,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=None,
            final_action=final_action,
            step_records=step_records,
        )

    def _move_to_pose(
        self,
        target_pose: torch.Tensor,
        tolerance: float,
        orientation_tolerance: float,
        max_steps: int,
        speed: float,
        gripper_command: float,
        node_id: str,
        active_skill: str,
        state_name: str,
        step_records: list[dict[str, Any]],
        dynamic_target_fn=None,
    ) -> bool:
        self._visualize_skill_target(target_pose, None, state_name=state_name)
        for _ in range(max(0, max_steps)):
            if dynamic_target_fn is not None:
                target_pose = dynamic_target_fn(target_pose)
            ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
            pos_error = torch.norm(target_pose[:, :3] - ee_pos, dim=-1)[0].item()
            orientation_error, _ = compute_orientation_error(
                ee_quat,
                target_pose[:, 3:7],
                yaw_only=False,
            )
            if pos_error < tolerance and orientation_error < orientation_tolerance:
                return True
            delta_b = self.cerebellum._compute_position_delta_in_base_frame(target_pose[:, :3], ee_pos)
            delta_b = torch.clamp(delta_b, min=-abs(speed), max=abs(speed))
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, :3] = delta_b
            orientation_delta = compute_orientation_delta_in_base_frame(
                self.cerebellum,
                ee_quat,
                target_pose[:, 3:7],
                angular_speed=float(min(abs(speed), 0.10)),
            )
            action[:, 3:6] = orientation_delta
            action[:, 6] = gripper_command
            self._step(
                action,
                node_id,
                active_skill,
                state_name,
                step_records,
                target_pose_for_diag=target_pose,
                target_orientation_for_diag=target_pose[:, 3:7],
            )
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                return False
        ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
        pos_error = torch.norm(target_pose[:, :3] - ee_pos, dim=-1)[0].item()
        orientation_error, _ = compute_orientation_error(
            ee_quat,
            target_pose[:, 3:7],
            yaw_only=False,
        )
        return pos_error < tolerance and orientation_error < orientation_tolerance

    def _hold(
        self,
        gripper_command: float,
        steps: int,
        node_id: str,
        active_skill: str,
        state_name: str,
        step_records: list[dict[str, Any]],
    ) -> list[float] | None:
        final_action = None
        for _ in range(max(0, steps)):
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, 6] = gripper_command
            final_action = _tensor_to_list(action[0])
            self._step(
                action,
                node_id,
                active_skill,
                state_name,
                step_records,
            )
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                break
        return final_action

    def _step(
        self,
        action: torch.Tensor,
        node_id: str,
        active_skill: str,
        state_name: str,
        step_records: list[dict[str, Any]],
        node_type: str | None = None,
        position_goal: dict[str, Any] | None = None,
        orientation_goal: dict[str, Any] | None = None,
        target_pose_for_diag: torch.Tensor | None = None,
        target_orientation_for_diag: torch.Tensor | None = None,
    ) -> None:
        result = self.env.step(action)
        if len(result) == 5:
            _, reward, terminated, truncated, _ = result
        elif len(result) == 4:
            _, reward, dones, _ = result
            terminated = dones
            truncated = torch.zeros_like(dones, dtype=torch.bool)
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(result)}")
        self.cerebellum.last_reward = _first_scalar(reward)
        self.cerebellum.last_terminated = _first_bool(terminated)
        self.cerebellum.last_truncated = _first_bool(truncated)
        scene_state = self._scene_state()
        row = {
            "global_step": self.cerebellum.global_step,
            "node_id": node_id,
            "node_type": node_type,
            "active_skill": active_skill,
            "state_name": state_name,
            "ee_pose": scene_state["ee_pose"],
            "cube_pose": scene_state["cube_pose"],
            "target_pose": scene_state["target_pose"],
            "gripper_width": scene_state["gripper_width"],
            "action": _tensor_to_list(action[0]),
            "reward": self.cerebellum.last_reward,
            "terminated": self.cerebellum.last_terminated,
            "truncated": self.cerebellum.last_truncated,
            "position_goal": position_goal,
            "orientation_goal": orientation_goal,
        }
        step_records.append(row)
        self.logger.log_trajectory_step(row)
        self._log_ee_target_diagnostic(
            row=row,
            node_id=node_id,
            active_skill=active_skill,
            state_name=state_name,
            target_pose_for_diag=target_pose_for_diag,
            target_orientation_for_diag=target_orientation_for_diag,
        )
        self.cerebellum.global_step += 1

    def _compute_position_target_from_goal(self, position_goal: dict[str, Any] | None) -> torch.Tensor | None:
        if position_goal is None:
            return None
        skill = str(position_goal.get("skill", ""))
        target = str(position_goal.get("target", ""))
        params = dict(position_goal.get("params") or {})
        if skill == "move_above":
            target_pose = self._target_pose(target)
            xy_offset = params.get("xy_offset", [0.0, 0.0])
            target_pose[:, 0] += float(xy_offset[0])
            target_pose[:, 1] += float(xy_offset[1])
            target_pose[:, 2] += float(params.get("height_offset", 0.12))
            return target_pose
        if skill == "reach":
            if "target_pose" in params:
                return self._pose_from_xyz(params["target_pose"][:3])
            target_ref = str(params.get("target_ref", target))
            target_pose = self._target_pose(target_ref)
            offset = params.get("offset", [0.0, 0.0, 0.0])
            target_pose[:, :3] += torch.tensor(offset[:3], device=self.device).reshape(1, 3)
            return target_pose
        raise ValueError(f"Unsupported position_goal skill: {skill}")

    def _resolve_gripper_command(self, gripper_mode: str) -> float:
        if gripper_mode == "open":
            return OPEN_GRIPPER
        if gripper_mode == "close":
            return CLOSE_GRIPPER
        return self.last_gripper_command

    def _parallel_target_label(self, goals: dict[str, Any]) -> str:
        labels = []
        position_goal = goals.get("position_goal")
        orientation_goal = goals.get("orientation_goal")
        if position_goal:
            labels.append(str(position_goal.get("target", "")))
        if orientation_goal:
            labels.append(str(orientation_goal.get("target", "")))
        return "+".join(labels) if labels else "parallel"

    def _target_pose(self, target: str) -> torch.Tensor:
        if target == "cube":
            return self.cerebellum._read_cube_pose_w().clone()
        if target == "target":
            return self.cerebellum._get_target_pose_w().clone()
        if target == "current":
            ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
            return torch.cat([ee_pos, ee_quat], dim=-1)
        raise ValueError(f"Unsupported target reference: {target}")

    def _pose_from_xyz(self, xyz: list[float]) -> torch.Tensor:
        pose = torch.zeros((1, 7), device=self.device)
        pose[:, :3] = torch.tensor(xyz[:3], device=self.device).reshape(1, 3)
        pose[:, 3] = 1.0
        return pose

    def _scene_state(self) -> dict[str, Any]:
        state = self.cerebellum.get_scene_state()
        return {
            "ee_pose": state["ee_pose_w"],
            "cube_pose": state["cube_pose_w"],
            "target_pose": state["target_pose_w"],
            "gripper_width": state["gripper_width"],
            "robot_joint_pos": state["robot_joint_pos"],
            "step_index": state["step_index"],
        }

    def _target_orientation_for_motion(
        self,
        node,
        target_pose: torch.Tensor,
        current_quat_w: torch.Tensor | None,
    ) -> torch.Tensor:
        params = dict(node.params or {})
        skill_name = str(getattr(node, "skill", "") or "")
        default_mode = "align_yaw_with_target"
        if skill_name in {"descend", "lift", "retreat"}:
            default_mode = "keep_current"
        mode = str(params.get("orientation_mode", default_mode))
        if mode == "none":
            if current_quat_w is None:
                _, ee_quat = self.cerebellum._read_ee_pose_w()
                return ee_quat.clone()
            return current_quat_w.clone()
        if current_quat_w is None:
            _, current_quat_w = self.cerebellum._read_ee_pose_w()
        orientation_params = {
            "orientation_mode": mode,
            "keep_top_down": bool(params.get("keep_top_down", True)),
            "fixed_yaw": params.get("fixed_yaw"),
            "target_rpy": params.get("target_rpy"),
        }
        target_orientation_w = compute_target_orientation_w(
            self.cerebellum,
            str(node.target),
            mode,
            orientation_params,
            current_quat_w,
            self._target_pose,
        )
        if target_orientation_w is None:
            return current_quat_w.clone()
        return target_orientation_w

    def _log_skill_start(self, node) -> None:
        skill_name = "parallel" if node.type == "parallel" else str(node.skill)
        target = ""
        if node.type == "parallel":
            goals = node.goals or {}
            position_goal = goals.get("position_goal") or {}
            orientation_goal = goals.get("orientation_goal") or {}
            pos_target = str(position_goal.get("target", ""))
            ori_target = str(orientation_goal.get("target", ""))
            target = f"{pos_target}+{ori_target}".strip("+")
        else:
            target = str(node.target or "")
        params_repr = node.params if isinstance(node.params, dict) else {}
        print(
            f"[SKILL] node={node.node_id} type={node.type} skill={skill_name} "
            f"target={target} params={params_repr}"
        )

    def _visualize_skill_target(
        self,
        target_pose: torch.Tensor | None,
        target_orientation_w: torch.Tensor | None,
        state_name: str,
    ) -> None:
        if self.skill_target_marker is None:
            return
        if target_pose is None and target_orientation_w is None:
            return
        if target_pose is None:
            ee_pos, _ = self.cerebellum._read_ee_pose_w()
            translations = ee_pos
        else:
            translations = target_pose[:, :3]
        if target_orientation_w is not None:
            orientations = target_orientation_w
        elif target_pose is not None:
            orientations = target_pose[:, 3:7]
        else:
            _, ee_quat = self.cerebellum._read_ee_pose_w()
            orientations = ee_quat
        self.skill_target_marker.visualize(
            translations=translations,
            orientations=orientations,
        )

    def _log_ee_target_diagnostic(
        self,
        row: dict[str, Any],
        node_id: str,
        active_skill: str,
        state_name: str,
        target_pose_for_diag: torch.Tensor | None,
        target_orientation_for_diag: torch.Tensor | None,
    ) -> None:
        if not hasattr(self.logger, "log_ee_diagnostic"):
            return
        ee_pose = row.get("ee_pose")
        if not isinstance(ee_pose, list) or len(ee_pose) < 7:
            return
        ee_pos = [float(ee_pose[0]), float(ee_pose[1]), float(ee_pose[2])]
        ee_quat = torch.tensor([ee_pose[3:7]], device=self.device, dtype=torch.float32)

        target_pos = None
        target_ori = None
        if target_pose_for_diag is not None:
            target_pos = [float(x) for x in _tensor_to_list(target_pose_for_diag[0, :3])]
            target_ori = [float(x) for x in _tensor_to_list(target_pose_for_diag[0, 3:7])]
        if target_orientation_for_diag is not None:
            target_ori = [float(x) for x in _tensor_to_list(target_orientation_for_diag[0])]

        xy_error = None
        z_error = None
        orientation_error = None
        if target_pos is not None:
            dx = ee_pos[0] - target_pos[0]
            dy = ee_pos[1] - target_pos[1]
            xy_error = (dx * dx + dy * dy) ** 0.5
            z_error = abs(ee_pos[2] - target_pos[2])
        if target_ori is not None:
            target_quat = torch.tensor([target_ori], device=self.device, dtype=torch.float32)
            orientation_error, _ = compute_orientation_error(ee_quat, target_quat, yaw_only=False)

        payload = {
            "global_step": row.get("global_step"),
            "node_id": node_id,
            "active_skill": active_skill,
            "state_name": state_name,
            "ee_pose": [float(x) for x in ee_pose[:7]],
            "target_pose": target_pos + target_ori if (target_pos is not None and target_ori is not None) else None,
            "xy_error": xy_error,
            "z_error": z_error,
            "orientation_error": orientation_error,
        }
        self.logger.log_ee_diagnostic(payload)

    def _empty_result(self, node, success: bool, failure_reason: str) -> PrimitiveSkillResult:
        state = self._scene_state()
        step = self.cerebellum.global_step
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target or ""),
            params=dict(node.params),
            start_step=step,
            end_step=step,
            pre_state=state,
            post_state=state,
            success=success,
            failure_reason=failure_reason,
            timeout=False,
            num_steps=0,
        )
