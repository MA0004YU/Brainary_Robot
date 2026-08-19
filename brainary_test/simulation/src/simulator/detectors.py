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
    note: str = ""

    def to_dict(self): return asdict(self)


@dataclass
class IntrusionFeedback:
    error_type: str = "UNEXPECTED_COLLISION"
    timestamp: float = 0.0
    collided_pair: List[str] = None
    collision_xyz: List[float] = None
    # 量化字段(供 planner 精准重规划):
    max_force_N: float = 0.0            # 碰撞瞬时接触力
    penetration_mm: float = 0.0        # 穿透深度(几何干涉多少 mm)
    nearest_obstacle: str = ""         # 规划失败时:最近障碍物
    obstacle_dist_mm: float = 0.0      # 目标到最近障碍物中心距离
    clearance_mm: float = 0.0          # 净空(<0=重叠;需至少腾出多少 mm)
    reach_mm: float = 0.0              # 目标到机器人 base 的水平距离(判断是否超臂展)
    displacement_cm: float = 0.0       # 目标相对初始位置被撞动多少 cm(级联)
    note: str = ""

    def to_dict(self): return asdict(self)


@dataclass
class DeadlockKinematicFeedback:
    error_type: str = "KINEMATIC_DYNAMIC_DEADLOCK"
    timestamp: float = 0.0
    saturated_joint_id: int = 0
    current_torque_Nm: float = 0.0
    max_torque_limit_Nm: float = 0.0
    tracking_error_rad: float = 0.0
    note: str = ""

    def to_dict(self): return asdict(self)


class PhysicsBoundaryDetectors:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config["detector_thresholds"]
        self.robot_limits = config["robot_config"]
        self.dt = config["simulator_config"]["time_step"]
        self.strict_safety_margin = 0.95   # 触发点=真限值×此系数(<1 会提前触发/易误报);0.95=留5%缓冲,贴近真限值
        self.initial_stable_z_axes = {}

    def check_stiffness_and_destruction(self, scene: sapien.Scene, physics_dict: Dict[str, Any], current_time: float,
                                        skip_names=()) -> Optional[DestructionFeedback]:
        # skip_names:当前正在【抓取/夹持】的目标——魔法吸附靠运动学持物、从不挤压它,所以夹爪/手指与该物体
        # 的接触是抓取本身(预期的),不算"夹碎";放置沉降/搬运撞别的物体仍照常检测。
        skip = set(n for n in skip_names if n)
        for contact in scene.get_contacts():
            actor0, actor1 = contact.bodies[0].entity, contact.bodies[1].entity
            name0, name1 = actor0.name, actor1.name

            if name0 in skip or name1 in skip:
                continue
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
                                   current_time: float, held_name: str = "") -> Optional[IntrusionFeedback]:
        # 侵入源 = 【整条机械臂】(名字含 robot_name_prefix="panda":link0-7 + hand + 双指)+【被夹持的物体】
        # (held_name;魔法吸附后它随臂移动,撞到任何东西也要停)。任一侵入源与"意料之外"的物体发生
        # 超力接触即报错 -> 满足"任何时刻、任何位置的碰撞都及时停下"。
        max_force = self.thresholds["unplanned_collision_force"] * self.strict_safety_margin
        for contact in scene.get_contacts():
            name0, name1 = contact.bodies[0].entity.name, contact.bodies[1].entity.name
            is_0_robot = robot_name_prefix in name0 or (held_name and name0 == held_name)
            is_1_robot = robot_name_prefix in name1 or (held_name and name1 == held_name)

            if not is_0_robot and not is_1_robot: continue

            env_actor = name1 if is_0_robot else name0
            # 允许的接触:正在抓/放的目标、地面、桌面、机械臂自身、被夹物体自身(夹爪抱着它/它就是被夹的)
            if env_actor in [expected_target, "ground", "operating_table", held_name] \
                    or robot_name_prefix in env_actor:
                continue

            for point in contact.points:
                force_N = np.linalg.norm(point.impulse) / self.dt
                if force_N > max_force:
                    pen_mm = max(0.0, -float(getattr(point, "separation", 0.0))) * 1000.0
                    return IntrusionFeedback(
                        timestamp=current_time,
                        collided_pair=[name0, name1],
                        collision_xyz=point.position.tolist(),
                        max_force_N=float(force_N),
                        penetration_mm=float(pen_mm),
                        note=f"[{name0}] 与 [{name1}] 超力接触({force_N:.1f}N > 红线 {max_force:.1f}N),几何干涉约 {pen_mm:.1f}mm",
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
