# simulation/src/simulator/detectors.py

import numpy as np
import sapien.core as sapien
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DestructionFeedback:
    error_type: str = "MECHANICAL_DESTRUCTION"
    timestamp: float
    culprit_actor: str
    max_normal_force_N: float
    safety_threshold_N: float
    contact_pair: List[str]
    force_direction: List[float]


@dataclass
class TippingFeedback:
    error_type: str = "TOPOLOGICAL_TIPPING"
    timestamp: float
    culprit_actor: str
    tilt_angle_deg: float
    max_allowed_angle: float


@dataclass
class IntrusionFeedback:
    error_type: str = "UNEXPECTED_COLLISION"
    timestamp: float
    collided_pair: List[str]
    collision_xyz: List[float]


@dataclass
class DeadlockKinematicFeedback:
    error_type: str = "KINEMATIC_DYNAMIC_DEADLOCK"
    timestamp: float
    saturated_joint_id: int
    current_torque_Nm: float
    max_torque_limit_Nm: float
    tracking_error_rad: float


class PhysicsBoundaryDetectors:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config["detector_thresholds"]
        self.robot_limits = config["robot_config"]
        self.dt = config["simulator_config"]["time_step"]
        # 引入严格的收缩系数，杜绝物理违规漏报
        self.strict_safety_margin = 0.85

    def check_stiffness_and_destruction(self, scene: sapien.Scene, physics_dict: Dict[str, Any], current_time: float) -> \
    Optional[DestructionFeedback]:
        """计算接触冲量推导受力"""
        contacts = scene.get_contacts()
        for contact in contacts:
            actor0, actor1 = contact.actor0, contact.actor1
            name0, name1 = actor0.name, actor1.name

            if name0 in ["ground", "camera_mount"] or name1 in ["ground", "camera_mount"]:
                continue

            limit0 = physics_dict.get(name0.split('_')[0], {}).get("yield_stress_N",
                                                                   float('inf')) * self.strict_safety_margin
            limit1 = physics_dict.get(name1.split('_')[0], {}).get("yield_stress_N",
                                                                   float('inf')) * self.strict_safety_margin

            for point in contact.points:
                impulse_norm = np.linalg.norm(point.impulse)
                if impulse_norm == 0: continue
                normal_force_N = impulse_norm / self.dt

                if normal_force_N > limit0:
                    return DestructionFeedback(current_time, name0, float(normal_force_N), float(limit0),
                                               [name0, name1], (point.impulse / impulse_norm).tolist())
                elif normal_force_N > limit1:
                    return DestructionFeedback(current_time, name1, float(normal_force_N), float(limit1),
                                               [name1, name0], (-point.impulse / impulse_norm).tolist())
        return None

    def check_stability_and_tipping(self, scene: sapien.Scene, non_target_actors: List[sapien.Actor],
                                    current_time: float) -> Optional[TippingFeedback]:
        max_angle = self.thresholds["max_allowed_angle"] * self.strict_safety_margin
        world_z = np.array([0.0, 0.0, 1.0])

        for actor in non_target_actors:
            if actor.is_kinematic: continue
            local_z = actor.get_pose().to_transformation_matrix()[:3, 2]
            tilt_angle_deg = np.degrees(np.arccos(np.clip(np.dot(local_z, world_z), -1.0, 1.0)))

            if tilt_angle_deg > max_angle:
                return TippingFeedback(current_time, actor.name, float(tilt_angle_deg), float(max_angle))
        return None

    def check_unexpected_collision(self, scene: sapien.Scene, robot_name_prefix: str, expected_target: str,
                                   current_time: float) -> Optional[IntrusionFeedback]:
        max_force = self.thresholds["unplanned_collision_force"] * self.strict_safety_margin
        for contact in scene.get_contacts():
            name0, name1 = contact.actor0.name, contact.actor1.name
            is_0_robot = robot_name_prefix in name0
            is_1_robot = robot_name_prefix in name1

            if not is_0_robot and not is_1_robot: continue

            env_actor = name1 if is_0_robot else name0
            if env_actor in [expected_target, "ground"] or robot_name_prefix in env_actor:
                continue

            for point in contact.points:
                force_N = np.linalg.norm(point.impulse) / self.dt
                if force_N > max_force:
                    return IntrusionFeedback(current_time, [name0, name1], point.position.tolist())
        return None

    def check_feasibility_and_deadlock(self, robot_articulation: sapien.Articulation, current_time: float) -> Optional[
        DeadlockKinematicFeedback]:
        current_qf = robot_articulation.get_qf()
        current_qpos = robot_articulation.get_qpos()
        target_qpos = robot_articulation.get_drive_target()

        # 严苛的防死锁红线
        max_torque = self.robot_limits["max_joint_torque"] * self.strict_safety_margin
        max_err = self.robot_limits["tracking_error_threshold"]

        for j_id, (torque, q_curr, q_target) in enumerate(zip(current_qf, current_qpos, target_qpos)):
            if abs(torque) >= max_torque and abs(q_target - q_curr) > max_err:
                return DeadlockKinematicFeedback(current_time, j_id, float(abs(torque)), float(max_torque),
                                                 float(abs(q_target - q_curr)))
        return None
