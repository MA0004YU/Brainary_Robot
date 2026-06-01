"""State-machine cerebellum for Franka pick/place data generation.

The controller targets Isaac Lab v2.0.x/v2.1.x manager-based Franka IK-Rel
manipulation environments. It only emits relative task-space deltas and binary
gripper commands, and does not invoke any learning or demonstration pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.utils.math import axis_angle_from_quat, combine_frame_transforms, quat_conjugate, quat_mul, subtract_frame_transforms

from skill_interface import SkillCommand, SkillPlan, SkillResult


OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0


class StateMachineCerebellum:
    """Executes JSON skills with a simple Franka IK-Rel state machine."""

    def __init__(
        self,
        env,
        skill_plan: SkillPlan,
        device: str | torch.device,
        logger,
        env_id: str,
        target_mode: str = "custom_tabletop",
        seed: int = 0,
        max_episode_steps: int = 3000,
        max_delta_pos: float = 0.08,
        max_delta_rot: float = 0.20,
        settle_steps: int = 30,
        target_marker=None,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.skill_plan = skill_plan
        self.device = torch.device(device)
        self.logger = logger
        self.env_id = env_id
        self.target_mode = target_mode
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.max_delta_pos = max_delta_pos
        self.max_delta_rot = max_delta_rot
        self.settle_steps = settle_steps
        self.target_marker = target_marker
        self.global_step = 0
        self.target_pose_w: torch.Tensor | None = None
        self.last_reward: float | None = None
        self.last_terminated = False
        self.last_truncated = False
        self._finger_joint_ids: Sequence[int] | slice | None = None

        self._validate_action_space()

    def execute_episode(self, episode_id: str, episode_seed: int) -> dict[str, Any]:
        """Reset the environment, execute the skill plan, and return one episode record."""

        self.global_step = 0
        self.last_reward = None
        self.last_terminated = False
        self.last_truncated = False
        self.env.reset(seed=episode_seed)
        self._settle_scene()

        cube_pose = self._read_cube_pose_w()
        if self.target_mode == "custom_tabletop":
            self.target_pose_w = self.sample_tabletop_target_pose(cube_pose, episode_seed)
        elif self.target_mode == "official_command":
            self.target_pose_w = self._read_official_command_pose_w()
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")
        self._visualize_target_pose()

        trajectory_file = self.logger.start_trajectory(episode_id)
        initial_scene = self.get_scene_state()

        execution_trace: list[dict[str, Any]] = []
        episode_failure: str | None = None
        for command in self.skill_plan.skill_plan:
            if self.global_step >= self.max_episode_steps:
                episode_failure = "timeout"
                break
            result = self.execute_skill(command)
            execution_trace.append(result.to_dict())
            if not result.success:
                episode_failure = result.failure_reason or f"{command.skill}_failed"
                break

        self.logger.finish_trajectory()
        final_state = self.get_scene_state()
        pick_success = _skill_success(execution_trace, "pick")
        place_success = _skill_success(execution_trace, "place")
        episode_success = pick_success and place_success and episode_failure is None
        if not episode_success and episode_failure is None:
            episode_failure = "skill_failed"

        episode = {
            "episode_id": episode_id,
            "task": self.skill_plan.task,
            "env_id": self.env_id,
            "target_mode": self.target_mode,
            "seed": episode_seed,
            "initial_scene": initial_scene,
            "skill_plan": [command.to_dict() for command in self.skill_plan.skill_plan],
            "execution_trace": execution_trace,
            "final_result": {
                "success": episode_success,
                "failure_reason": None if episode_success else episode_failure,
                "final_cube_pose": final_state["cube_pose_w"],
                "final_relation": self._final_relation(episode_success),
            },
            "trajectory_file": trajectory_file,
        }
        self.logger.log_episode(episode)
        return episode

    def execute_skill(self, command: SkillCommand) -> SkillResult:
        """Dispatch a skill command."""

        skill_name = command.skill.lower()
        if skill_name == "pick":
            return self._execute_pick(command)
        if skill_name == "place":
            return self._execute_place(command)
        start_step = self.global_step
        state = self.get_scene_state()
        return SkillResult(
            skill=command.skill,
            target=command.target,
            params=command.params,
            start_step=start_step,
            end_step=self.global_step,
            pre_state=state,
            post_state=state,
            success=False,
            failure_reason="unsupported_skill",
            num_steps=0,
            state_sequence=["UNSUPPORTED_SKILL"],
        )

    def get_scene_state(self) -> dict[str, Any]:
        """Read stable world-frame state fields for logging and decisions."""

        robot = self.unwrapped.scene["robot"]
        ee_pos, ee_quat = self._read_ee_pose_w()
        cube_pose = self._read_cube_pose_w()
        target_pose = self._get_target_pose_w()
        return {
            "ee_pose_w": _tensor_to_list(torch.cat([ee_pos, ee_quat], dim=-1)[0]),
            "cube_pose_w": _tensor_to_list(cube_pose[0]),
            "target_pose_w": _tensor_to_list(target_pose[0]),
            "gripper_width": self._read_gripper_width(),
            "robot_joint_pos": _tensor_to_list(robot.data.joint_pos[0]),
            "step_index": self.global_step,
        }

    def sample_tabletop_target_pose(self, cube_pose_w: torch.Tensor, seed: int) -> torch.Tensor:
        """Sample a reachable tabletop placement target away from the cube."""

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        cube_xy = cube_pose_w[0, :2]
        x_range = (0.38, 0.62)
        y_range = (-0.22, 0.22)
        min_distance = 0.12
        target_xy = None
        for _ in range(100):
            x = _uniform(x_range[0], x_range[1], generator, self.device)
            y = _uniform(y_range[0], y_range[1], generator, self.device)
            candidate = torch.stack([x, y])
            if torch.norm(candidate - cube_xy) > min_distance:
                target_xy = candidate
                break
        if target_xy is None:
            target_xy = torch.tensor([0.55, -0.18], device=self.device)

        target_pose = torch.zeros((1, 7), device=self.device)
        target_pose[0, 0:2] = target_xy
        target_pose[0, 2] = cube_pose_w[0, 2]
        target_pose[0, 3] = 1.0
        return target_pose

    def move_ee_to_pose(
        self,
        target_pose_w: torch.Tensor,
        tolerance: float,
        max_steps: int,
        gripper_command: float,
        active_skill: str,
        state_name: str,
        control_orientation: bool = True,
    ) -> bool:
        """Move the end effector toward a world-frame target using IK-Rel position deltas."""

        for _ in range(max_steps):
            ee_pos, ee_quat = self._read_ee_pose_w()
            pos_error_w = target_pose_w[:, :3] - ee_pos
            if torch.norm(pos_error_w[0]).item() < tolerance:
                return True
            delta_b = self._compute_position_delta_in_base_frame(target_pose_w[:, :3], ee_pos)
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, :3] = delta_b
            if control_orientation:
                action[:, 3:6] = self._compute_top_down_orientation_delta_in_base_frame(ee_quat)
            action[:, 6] = gripper_command
            self._step_env(action, active_skill=active_skill, state_name=state_name)
            if self._episode_done() or self.global_step >= self.max_episode_steps:
                return False
        return False

    def _compute_position_delta_in_base_frame(self, target_pos_w: torch.Tensor, ee_pos_w: torch.Tensor) -> torch.Tensor:
        """Convert world-frame position error into robot-base-frame IK-Rel deltas."""

        robot = self.unwrapped.scene["robot"]
        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            ee_pos_w,
        )
        target_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            target_pos_w,
        )
        delta_b = target_pos_b - ee_pos_b
        return torch.clamp(delta_b, min=-self.max_delta_pos, max=self.max_delta_pos)

    def _compute_top_down_orientation_delta_in_base_frame(self, ee_quat_w: torch.Tensor) -> torch.Tensor:
        """Rotate the Franka TCP toward a top-down grasp orientation."""

        robot = self.unwrapped.scene["robot"]
        _, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            None,
            ee_quat_w,
        )
        desired_quat_b = torch.zeros_like(ee_quat_b)
        desired_quat_b[:, 1] = 1.0  # Franka's standard downward grasp quaternion in the robot base frame.
        delta_quat = quat_mul(desired_quat_b, quat_conjugate(ee_quat_b))
        delta_axis_angle = axis_angle_from_quat(delta_quat)
        return torch.clamp(delta_axis_angle, min=-self.max_delta_rot, max=self.max_delta_rot)

    def _visualize_target_pose(self) -> None:
        """Show the actual custom tabletop target, not the Lift task command target."""

        if self.target_marker is None or self.target_pose_w is None:
            return
        self.target_marker.visualize(
            translations=self.target_pose_w[:, :3],
            orientations=self.target_pose_w[:, 3:7],
        )

    def _settle_scene(self) -> None:
        """Let reset objects settle before sampling targets or logging initial state."""

        action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
        action[:, 6] = OPEN_GRIPPER
        for _ in range(max(0, self.settle_steps)):
            result = self.env.step(action)
            if len(result) == 5:
                _, _, terminated, truncated, _ = result
            elif len(result) == 4:
                _, _, dones, _ = result
                terminated = dones
                truncated = torch.zeros_like(dones, dtype=torch.bool)
            else:
                raise RuntimeError(f"Unexpected env.step return length during settle: {len(result)}")
            if _first_bool(terminated) or _first_bool(truncated):
                break

    def _execute_pick(self, command: SkillCommand) -> SkillResult:
        params = command.params
        approach_height = float(params.get("approach_height", 0.12))
        grasp_height = float(params.get("grasp_height", 0.0))
        lift_height = float(params.get("lift_height", 0.18))
        tolerance = float(params.get("position_tolerance", 0.03))
        grasp_tolerance = float(params.get("grasp_tolerance", min(tolerance, 0.015)))
        approach_steps = int(params.get("approach_steps", 500))
        descend_steps = int(params.get("descend_steps", 250))
        grasp_steps = int(params.get("grasp_steps", 60))
        lift_steps = int(params.get("lift_steps", 350))

        start_step = self.global_step
        pre_state = self.get_scene_state()
        initial_cube_z = self._read_cube_pose_w()[0, 2].item()
        failure_reason: str | None = None

        cube_pose = self._read_cube_pose_w()
        pre_grasp_pose = cube_pose.clone()
        pre_grasp_pose[:, 2] = cube_pose[:, 2] + approach_height
        if not self._move_phase(pre_grasp_pose, tolerance, approach_steps, OPEN_GRIPPER, "pick", "MOVE_ABOVE_OBJECT"):
            failure_reason = "reach_failed"

        if failure_reason is None:
            cube_pose = self._read_cube_pose_w()
            grasp_pose = cube_pose.clone()
            grasp_pose[:, 2] = cube_pose[:, 2] + grasp_height
            if not self._move_phase(grasp_pose, grasp_tolerance, descend_steps, OPEN_GRIPPER, "pick", "DESCEND_TO_GRASP"):
                failure_reason = "reach_failed"

        if failure_reason is None:
            self._hold_gripper(CLOSE_GRIPPER, grasp_steps, "pick", "CLOSE_GRIPPER")

        if failure_reason is None:
            lift_pose = self._read_cube_pose_w().clone()
            lift_pose[:, 2] = initial_cube_z + lift_height
            if not self._move_phase(lift_pose, tolerance, lift_steps, CLOSE_GRIPPER, "pick", "LIFT_OBJECT"):
                failure_reason = "object_not_lifted"

        success = False if failure_reason else self._check_pick_success(initial_cube_z)
        if not success and failure_reason is None:
            failure_reason = "grasp_failed"
        post_state = self.get_scene_state()
        return SkillResult(
            skill=command.skill,
            target=command.target,
            params=params,
            start_step=start_step,
            end_step=self.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=success,
            failure_reason=None if success else failure_reason,
            num_steps=self.global_step - start_step,
            state_sequence=[
                "PICK_START",
                "MOVE_ABOVE_OBJECT",
                "DESCEND_TO_GRASP",
                "CLOSE_GRIPPER",
                "LIFT_OBJECT",
                "CHECK_GRASP",
                "PICK_SUCCESS" if success else "PICK_FAILED",
            ],
        )

    def _execute_place(self, command: SkillCommand) -> SkillResult:
        params = command.params
        place_height = float(params.get("place_height", 0.05))
        release_height = float(params.get("release_height", 0.08))
        retreat_height = float(params.get("retreat_height", 0.15))
        tolerance = float(params.get("position_tolerance", 0.04))
        approach_steps = int(params.get("approach_steps", 500))
        descend_steps = int(params.get("descend_steps", 300))
        release_steps = int(params.get("release_steps", 50))
        retreat_steps = int(params.get("retreat_steps", 250))

        start_step = self.global_step
        pre_state = self.get_scene_state()
        failure_reason: str | None = None
        target_pose = self._get_target_pose_w()

        if self.target_mode == "official_command":
            success = self._move_phase(
                target_pose, tolerance, approach_steps, CLOSE_GRIPPER, "place", "MOVE_TO_OFFICIAL_TARGET"
            )
            if not success:
                failure_reason = "target_unreachable"
            success = False if failure_reason else self._check_official_target_success(tolerance)
            if not success and failure_reason is None:
                failure_reason = "place_failed"
            post_state = self.get_scene_state()
            return SkillResult(
                skill=command.skill,
                target=command.target,
                params=params,
                start_step=start_step,
                end_step=self.global_step,
                pre_state=pre_state,
                post_state=post_state,
                success=success,
                failure_reason=None if success else failure_reason,
                num_steps=self.global_step - start_step,
                state_sequence=[
                    "PLACE_START",
                    "MOVE_TO_OFFICIAL_TARGET",
                    "CHECK_PLACE",
                    "PLACE_SUCCESS" if success else "PLACE_FAILED",
                ],
            )

        above_target = target_pose.clone()
        above_target[:, 2] = target_pose[:, 2] + max(release_height, retreat_height)
        if not self._move_phase(above_target, tolerance, approach_steps, CLOSE_GRIPPER, "place", "MOVE_ABOVE_TARGET"):
            failure_reason = "target_unreachable"

        if failure_reason is None:
            place_pose = target_pose.clone()
            place_pose[:, 2] = target_pose[:, 2] + place_height
            if not self._move_phase(place_pose, tolerance, descend_steps, CLOSE_GRIPPER, "place", "DESCEND_TO_PLACE"):
                failure_reason = "target_unreachable"

        if failure_reason is None:
            self._hold_gripper(OPEN_GRIPPER, release_steps, "place", "OPEN_GRIPPER")

        if failure_reason is None:
            retreat_pose = target_pose.clone()
            retreat_pose[:, 2] = target_pose[:, 2] + retreat_height
            self._move_phase(retreat_pose, tolerance, retreat_steps, OPEN_GRIPPER, "place", "RETREAT")

        success = False if failure_reason else self._check_place_success(tolerance)
        if not success and failure_reason is None:
            failure_reason = "place_failed"
        post_state = self.get_scene_state()
        return SkillResult(
            skill=command.skill,
            target=command.target,
            params=params,
            start_step=start_step,
            end_step=self.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=success,
            failure_reason=None if success else failure_reason,
            num_steps=self.global_step - start_step,
            state_sequence=[
                "PLACE_START",
                "MOVE_ABOVE_TARGET",
                "DESCEND_TO_PLACE",
                "OPEN_GRIPPER",
                "RETREAT",
                "CHECK_PLACE",
                "PLACE_SUCCESS" if success else "PLACE_FAILED",
            ],
        )

    def _move_phase(
        self,
        target_pose_w: torch.Tensor,
        tolerance: float,
        phase_max_steps: int,
        gripper_command: float,
        active_skill: str,
        state_name: str,
        control_orientation: bool = True,
    ) -> bool:
        """Run one motion phase with its own step budget."""

        remaining = min(phase_max_steps, self.max_episode_steps - self.global_step)
        if remaining <= 0:
            return False
        return self.move_ee_to_pose(
            target_pose_w,
            tolerance,
            remaining,
            gripper_command,
            active_skill,
            state_name,
            control_orientation=control_orientation,
        )

    def _hold_gripper(self, gripper_command: float, steps: int, active_skill: str, state_name: str) -> None:
        for _ in range(max(0, steps)):
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, 6] = gripper_command
            self._step_env(action, active_skill=active_skill, state_name=state_name)
            if self._episode_done() or self.global_step >= self.max_episode_steps:
                break

    def _step_env(self, action: torch.Tensor, active_skill: str, state_name: str) -> None:
        result = self.env.step(action)
        if len(result) == 5:
            _, reward, terminated, truncated, _ = result
        elif len(result) == 4:
            _, reward, dones, _ = result
            terminated = dones
            truncated = torch.zeros_like(dones, dtype=torch.bool)
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(result)}")

        self.last_reward = _first_scalar(reward)
        self.last_terminated = _first_bool(terminated)
        self.last_truncated = _first_bool(truncated)
        self.logger.log_trajectory_step(
            {
                "global_step": self.global_step,
                "active_skill": active_skill,
                "state_name": state_name,
                "ee_pose": self.get_scene_state()["ee_pose_w"],
                "cube_pose": self.get_scene_state()["cube_pose_w"],
                "target_pose": self.get_scene_state()["target_pose_w"],
                "gripper_width": self._read_gripper_width(),
                "action": _tensor_to_list(action[0]),
                "reward": self.last_reward,
                "terminated": self.last_terminated,
                "truncated": self.last_truncated,
            }
        )
        self.global_step += 1

    def _check_pick_success(self, initial_cube_z: float) -> bool:
        if self._official_object_grasped():
            return True
        cube_pose = self._read_cube_pose_w()
        ee_pos, _ = self._read_ee_pose_w()
        cube_lifted = cube_pose[0, 2].item() - initial_cube_z > 0.05
        cube_near_ee = torch.norm(cube_pose[0, :3] - ee_pos[0]).item() < 0.08
        return cube_lifted and cube_near_ee

    def _check_place_success(self, tolerance: float) -> bool:
        cube_pose = self._read_cube_pose_w()
        target_pose = self._get_target_pose_w()
        xy_distance = torch.norm(cube_pose[0, :2] - target_pose[0, :2]).item()
        z_error = abs(cube_pose[0, 2].item() - target_pose[0, 2].item())
        gripper_open = self._read_gripper_width() > 0.035
        cube_reasonable_height = z_error < 0.08 and cube_pose[0, 2].item() > -0.02
        cube_not_flown = cube_pose[0, 2].item() < target_pose[0, 2].item() + 0.18
        return xy_distance < tolerance and cube_reasonable_height and cube_not_flown and gripper_open

    def _check_official_target_success(self, tolerance: float) -> bool:
        cube_pose = self._read_cube_pose_w()
        target_pose = self._get_target_pose_w()
        ee_pos, _ = self._read_ee_pose_w()
        cube_to_target = torch.norm(cube_pose[0, :3] - target_pose[0, :3]).item()
        cube_to_ee = torch.norm(cube_pose[0, :3] - ee_pos[0]).item()
        return cube_to_target < max(tolerance, 0.05) and cube_to_ee < 0.08

    def _official_object_grasped(self) -> bool:
        return False

    def _episode_done(self) -> bool:
        return self.last_terminated or self.last_truncated

    def _validate_action_space(self) -> None:
        if list(self.unwrapped.action_manager.active_terms)[:2] != ["arm_action", "gripper_action"]:
            raise RuntimeError(
                "Expected action terms ['arm_action', 'gripper_action']. "
                f"Got {self.unwrapped.action_manager.active_terms}."
            )
        dims = self.unwrapped.action_manager.action_term_dim
        if dims != [6, 1]:
            raise RuntimeError(
                "This script requires Franka IK-Rel action space [dx, dy, dz, droll, dpitch, dyaw, gripper]. "
                f"Got action term dims {dims}. Use Isaac-Lift-Cube-Franka-IK-Rel-v0."
            )

    def _read_ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_frame = self.unwrapped.scene["ee_frame"]
        ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone()
        ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
        return ee_pos, ee_quat

    def _read_cube_pose_w(self) -> torch.Tensor:
        obj = self.unwrapped.scene["object"]
        return torch.cat([obj.data.root_pos_w.clone(), obj.data.root_quat_w.clone()], dim=-1)

    def _get_target_pose_w(self) -> torch.Tensor:
        if self.target_mode == "official_command":
            self.target_pose_w = self._read_official_command_pose_w()
        if self.target_pose_w is None:
            raise RuntimeError("Target pose has not been initialized.")
        return self.target_pose_w.clone()

    def _read_official_command_pose_w(self) -> torch.Tensor:
        if not hasattr(self.unwrapped, "command_manager") or self.unwrapped.command_manager is None:
            raise RuntimeError("Environment does not expose command_manager for target_mode='official_command'.")
        command = self.unwrapped.command_manager.get_command("object_pose")
        robot = self.unwrapped.scene["robot"]
        command_quat = torch.zeros((self.unwrapped.num_envs, 4), device=self.device)
        command_quat[:, 0] = 1.0
        if command.shape[-1] >= 7:
            command_quat = command[:, 3:7]
        pos_w, quat_w = combine_frame_transforms(
            robot.data.root_state_w[:, :3],
            robot.data.root_state_w[:, 3:7],
            command[:, :3],
            command_quat,
        )
        return torch.cat([pos_w, quat_w], dim=-1)

    def _read_gripper_width(self) -> float:
        robot = self.unwrapped.scene["robot"]
        if self._finger_joint_ids is None:
            try:
                finger_ids, _ = robot.find_joints(["panda_finger.*"])
                self._finger_joint_ids = finger_ids
            except Exception:
                self._finger_joint_ids = slice(-2, None)
        finger_pos = robot.data.joint_pos[0, self._finger_joint_ids]
        return float(torch.sum(finger_pos).item())

    def _final_relation(self, success: bool) -> str | None:
        if not success:
            return None
        if self.target_mode == "official_command":
            return "cube_at_official_target"
        return "cube_on_target"


def _skill_success(execution_trace: list[dict[str, Any]], skill_name: str) -> bool:
    return any(trace.get("skill") == skill_name and trace.get("success") for trace in execution_trace)


def _uniform(low: float, high: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand((), generator=generator, device=device) * (high - low) + low


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def _first_scalar(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return float(value.detach().flatten()[0].cpu().item())
    return float(value)


def _first_bool(value) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        return bool(value.detach().flatten()[0].cpu().item())
    return bool(value)
