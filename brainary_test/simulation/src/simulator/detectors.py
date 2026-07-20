import numpy as np
import sapien.core as sapien
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class DestructionFeedback:
    error_type: str = "MECHANICAL_DESTRUCTION"
    timestamp: float = 0.0
    culprit_actor: str = ""
    max_normal_force_N: float = 0.0
    safety_threshold_N: float = 0.0
    contact_pair: List[str] = None
    force_direction: List[float] = None
    # 👇 新增遥测反馈参数 (供 LLM 反思用)
    tcp_offset_x_mm: Optional[float] = None
    tcp_offset_y_mm: Optional[float] = None
    target_width_mm: Optional[float] = None
    actual_width_mm: Optional[float] = None
    predicted_force_N: Optional[float] = None

    def to_dict(self): return asdict(self)


# 👇 新增滑脱探针
@dataclass
class SlipFeedback:
    error_type: str = "KINEMATIC_SLIP"
    timestamp: float = 0.0
    culprit_actor: str = ""
    min_required_grip_force_N: float = 0.0
    predicted_force_N: float = 0.0
    tcp_offset_x_mm: Optional[float] = None
    tcp_offset_y_mm: Optional[float] = None
    target_width_mm: Optional[float] = None
    actual_width_mm: Optional[float] = None

    def to_dict(self): return asdict(self)


@dataclass
class TippingFeedback:
    error_type: str = "TOPOLOGICAL_TIPPING"
    timestamp: float = 0.0
    culprit_actor: str = ""
    tilt_angle_deg: float = 0.0
    max_allowed_angle: float = 0.0

    def to_dict(self): return asdict(self)


@dataclass
class IntrusionFeedback:
    error_type: str = "UNEXPECTED_COLLISION"
    timestamp: float = 0.0
    collided_pair: List[str] = None
    collision_xyz: List[float] = None

    def to_dict(self): return asdict(self)


@dataclass
class DeadlockKinematicFeedback:
    error_type: str = "KINEMATIC_DYNAMIC_DEADLOCK"
    timestamp: float = 0.0
    saturated_joint_id: int = 0
    current_torque_Nm: float = 0.0
    max_torque_limit_Nm: float = 0.0
    tracking_error_rad: float = 0.0

    def to_dict(self): return asdict(self)


class PhysicsBoundaryDetectors:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config["detector_thresholds"]
        self.robot_limits = config["robot_config"]
        self.dt = config["simulator_config"]["time_step"]
        self.strict_safety_margin = 0.85
        self.initial_stable_z_axes = {}

    def check_stiffness_and_destruction(self, scene: sapien.Scene, physics_dict: Dict[str, Any], current_time: float) -> \
    Optional[DestructionFeedback]:
        for contact in scene.get_contacts():
            actor0, actor1 = contact.bodies[0].entity, contact.bodies[1].entity
            name0, name1 = actor0.name, actor1.name

            if name0 in ["ground", "camera_mount", "operating_table"] or name1 in ["ground", "camera_mount",
                                                                                   "operating_table"]:
                continue

            # 保留语义强匹配逻辑
            label0 = name0.rsplit('_', 1)[0].replace('_', ' ')
            label1 = name1.rsplit('_', 1)[0].replace('_', ' ')

            limit0 = physics_dict.get(label0, {}).get("virtual_pain_limit", float('inf')) * self.strict_safety_margin
            limit1 = physics_dict.get(label1, {}).get("virtual_pain_limit", float('inf')) * self.strict_safety_margin

            for point in contact.points:
                impulse_norm = np.linalg.norm(point.impulse)
                if impulse_norm == 0: continue
                normal_force_N = impulse_norm / self.dt

                if normal_force_N > limit0:
                    return DestructionFeedback(
                        timestamp=current_time,
                        culprit_actor=name0,
                        max_normal_force_N=float(normal_force_N),
                        safety_threshold_N=float(limit0),
                        contact_pair=[name0, name1],
                        force_direction=(point.impulse / impulse_norm).tolist()
                    )
                elif normal_force_N > limit1:
                    return DestructionFeedback(
                        timestamp=current_time,
                        culprit_actor=name1,
                        max_normal_force_N=float(normal_force_N),
                        safety_threshold_N=float(limit1),
                        contact_pair=[name1, name0],
                        force_direction=(-point.impulse / impulse_norm).tolist()
                    )
        return None

    def check_stability_and_tipping(self, scene: sapien.Scene, non_target_actors: List["Any"], current_time: float) -> \
    Optional[TippingFeedback]:
        max_angle = self.thresholds["max_allowed_angle"] * self.strict_safety_margin
        for actor in non_target_actors:
            if not any("RigidDynamicComponent" in type(c).__name__ for c in actor.components): continue
            local_z = actor.get_pose().to_transformation_matrix()[:3, 2]
            if actor.name not in self.initial_stable_z_axes:
                self.initial_stable_z_axes[actor.name] = local_z
                continue
            baseline_z = self.initial_stable_z_axes[actor.name]
            relative_tilt_deg = np.degrees(np.arccos(np.clip(np.dot(local_z, baseline_z), -1.0, 1.0)))
            if relative_tilt_deg > max_angle:
                return TippingFeedback(
                    timestamp=current_time,
                    culprit_actor=actor.name,
                    tilt_angle_deg=float(relative_tilt_deg),
                    max_allowed_angle=float(max_angle)
                )
        return None

    def check_unexpected_collision(self, scene: sapien.Scene, robot_name_prefix: str, expected_target: str,
                                   current_time: float) -> Optional[IntrusionFeedback]:
        max_force = self.thresholds["unplanned_collision_force"] * self.strict_safety_margin
        for contact in scene.get_contacts():
            name0, name1 = contact.bodies[0].entity.name, contact.bodies[1].entity.name
            is_0_robot = robot_name_prefix in name0
            is_1_robot = robot_name_prefix in name1

            if not is_0_robot and not is_1_robot: continue

            env_actor = name1 if is_0_robot else name0
            if env_actor in [expected_target, "ground", "operating_table"] or robot_name_prefix in env_actor:
                continue

            for point in contact.points:
                force_N = np.linalg.norm(point.impulse) / self.dt
                if force_N > max_force:
                    return IntrusionFeedback(
                        timestamp=current_time,
                        collided_pair=[name0, name1],
                        collision_xyz=point.position.tolist()
                    )
        return None

    def check_feasibility_and_deadlock(self, robot_articulation: "Any", current_time: float) -> Optional[
        DeadlockKinematicFeedback]:
        current_qf = robot_articulation.get_qf()
        current_qpos = robot_articulation.get_qpos()
        target_qpos = [j.get_drive_target() for j in robot_articulation.get_active_joints()]
        max_torque = self.robot_limits["max_joint_torque"] * self.strict_safety_margin
        max_err = self.robot_limits["tracking_error_threshold"]

        for j_id, (torque, q_curr, q_target) in enumerate(zip(current_qf, current_qpos, target_qpos)):
            if abs(torque) >= max_torque and abs(q_target - q_curr) > max_err:
                return DeadlockKinematicFeedback(
                    timestamp=current_time,
                    saturated_joint_id=j_id,
                    current_torque_Nm=float(abs(torque)),
                    max_torque_limit_Nm=float(max_torque),
                    tracking_error_rad=float(abs(q_target - q_curr))
                )
        return None
