import os
import time
import sys
import math
import faulthandler
import numpy as np
import sapien.core as sapien
import mplib
from typing import List, Dict, Any

from src.simulator.detectors import DestructionFeedback, SlipFeedback, IntrusionFeedback

faulthandler.enable(sys.stderr)


def log(msg):
    print(f"👉 [诊断雷达] {msg}")
    sys.stdout.flush()


class DAGSimulationEngine:
    def __init__(self, scene: sapien.Scene, config: dict, detectors, physics_dict):
        log("进入 DAGSimulationEngine 初始化...")
        self.scene = scene
        self.robot_config = config.get("robot_config", {})
        self.detectors = detectors
        self.physics_dict = physics_dict

        urdf_relative = self.robot_config.get("urdf_path", "assets/robots/panda/panda.urdf")
        urdf_path = os.path.abspath(urdf_relative)
        srdf_path = urdf_path.replace(".urdf", ".srdf")

        log("正在加载 URDF 和 SRDF 模型...")
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf_path)
        self.robot.set_name("panda")

        try:
            high_fric_mat = self.scene.create_physical_material(100.0, 100.0, 0.0)
            for link in self.robot.get_links():
                if "finger" not in link.name.lower():
                    continue
                shapes = []
                if hasattr(link, 'get_collision_shapes'):
                    shapes.extend(link.get_collision_shapes())
                elif hasattr(link, 'components'):
                    for comp in link.components:
                        if hasattr(comp, 'get_collision_shapes'):
                            shapes.extend(comp.get_collision_shapes())
                for shape in shapes:
                    if hasattr(shape, 'set_physical_material'):
                        shape.set_physical_material(high_fric_mat)
        except Exception as e:
            pass

        init_qpos = self.robot_config.get("home_qpos", [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
        self.robot.set_qpos(init_qpos)

        [j.set_drive_target(t) for j, t in zip(self.robot.get_active_joints()[:7], init_qpos[:7])]
        for joint in self.robot.get_active_joints()[:7]:
            joint.set_drive_property(
                stiffness=self.robot_config.get("arm_stiffness", 4000.0),
                damping=self.robot_config.get("arm_damping", 400.0)
            )

        self.gripper_joints = self.robot.get_active_joints()[7:9]
        for joint in self.gripper_joints:
            joint.set_drive_property(
                stiffness=self.robot_config.get("gripper_stiffness", 2000.0),
                damping=self.robot_config.get("gripper_damping", 50.0),
                force_limit=self.robot_config.get("gripper_force_limit", 150.0)
            )
            joint.set_drive_target(self.robot_config.get("gripper_max_width", 0.08) / 2.0)

        try:
            self.planner = mplib.Planner(
                urdf=urdf_path, srdf=srdf_path,
                user_link_names=[link.name for link in self.robot.get_links()],
                user_joint_names=[joint.name for joint in self.robot.get_active_joints()],
                move_group="panda_hand",
                joint_vel_limits=np.ones(7, dtype=np.float64),
                joint_acc_limits=np.ones(7, dtype=np.float64)
            )
        except RuntimeError:
            self.planner = mplib.Planner(
                urdf=urdf_path, srdf="",
                user_link_names=[link.name for link in self.robot.get_links()],
                user_joint_names=[joint.name for joint in self.robot.get_active_joints()],
                move_group="panda_hand",
                joint_vel_limits=np.ones(7, dtype=np.float64),
                joint_acc_limits=np.ones(7, dtype=np.float64)
            )

        self.held_target = None
        self.safe_overhead_qpos = np.array(init_qpos)
        self._safe_dummy_pc = np.array([[0.0, 0.0, -10.0]], dtype=np.float64)

        self.scene.step()
        self.downward_quat = [0.0, 1.0, 0.0, 0.0]

    def _get_actor_half_size(self, actor) -> np.ndarray:
        for comp in actor.components:
            if hasattr(comp, 'get_collision_shapes'):
                shapes = comp.get_collision_shapes()
                if shapes:
                    shape = shapes[0]
                    if hasattr(shape, 'half_size'): return np.array(shape.half_size)
                    if hasattr(shape, 'geometry'):
                        geom = shape.geometry
                        if hasattr(geom, 'half_lengths'): return np.array(geom.half_lengths)
                        if hasattr(geom, 'half_size'): return np.array(geom.half_size)
        return np.array([0.03, 0.03, 0.03])

    def _sync_planner_point_cloud(self, target_names: List[str] = None):
        all_points = []
        grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 30), np.linspace(-0.6, 0.6, 30))
        ground_pc = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros_like(grid_x.ravel())))
        all_points.append(ground_pc)

        for actor in self.scene.get_all_actors():
            if "panda" in actor.name or "ground" in actor.name: continue
            if target_names and any(
                    t.lower().replace(" ", "_") in actor.name.lower().replace(" ", "_") for t in target_names): continue

            half_size = self._get_actor_half_size(actor)
            pose = actor.get_pose()
            noise_local = np.random.uniform(-half_size, half_size, size=(300, 3))
            noise_world = (pose.to_transformation_matrix()[:3, :3] @ noise_local.T).T + pose.p
            all_points.append(noise_world)

        if all_points:
            self.planner.update_point_cloud(np.vstack(all_points).astype(np.float64))
        else:
            self.planner.update_point_cloud(self._safe_dummy_pc)

    def _drive_trajectory(self, trajectory: List[np.ndarray], check_probes: bool = False,
                          exact_target_name: str = "") -> tuple:
        if trajectory is None or len(trajectory) == 0: return True, None
        gripper_targets = [j.get_drive_target() for j in self.gripper_joints]
        active_actors = [a for a in self.scene.get_all_actors() if
                         any("RigidDynamicComponent" in type(c).__name__ for c in a.components) and a.name != "panda"]

        for qpos_target in trajectory:
            for j, t in zip(self.robot.get_active_joints()[:7], qpos_target[:7]):
                j.set_drive_target(float(t))
            for j, t in zip(self.gripper_joints, gripper_targets):
                j.set_drive_target(float(t))

            self.scene.step()
            self.scene.step()

            if getattr(self, 'viewer', None): self.viewer.render()

            if check_probes:
                t = time.time()
                violation = (
                        self.detectors.check_stiffness_and_destruction(self.scene, self.physics_dict, t) or
                        self.detectors.check_stability_and_tipping(self.scene, active_actors, t) or
                        self.detectors.check_unexpected_collision(self.scene, "panda", exact_target_name, t) or
                        self.detectors.check_feasibility_and_deadlock(self.robot, t)
                )
                if violation: return False, violation

        for _ in range(100):
            self.scene.step()
            if getattr(self, 'viewer', None): self.viewer.render()
            if check_probes:
                t = time.time()
                violation = (
                        self.detectors.check_stiffness_and_destruction(self.scene, self.physics_dict, t) or
                        self.detectors.check_stability_and_tipping(self.scene, active_actors, t) or
                        self.detectors.check_unexpected_collision(self.scene, "panda", exact_target_name, t) or
                        self.detectors.check_feasibility_and_deadlock(self.robot, t)
                )
                if violation: return False, violation
        return True, None

    # 🚀 改造点：不仅 print，还要将多行文本返回，用于附加到 error 对象的黑匣子里
    def _dump_god_mode_telemetry(self, stage_name: str, target_actor) -> str:
        log_lines = []
        log_lines.append(f"================ 🚨 [全景物理诊断: {stage_name}] 🚨 ================")
        pos = target_actor.get_pose().p
        hs = self._get_actor_half_size(target_actor)
        log_lines.append(f"📦 【目标积木】 {target_actor.name} | 中心XYZ: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        log_lines.append(f"   -> 物理高度范围: Z_min={pos[2] - hs[2]:.4f}, Z_max={pos[2] + hs[2]:.4f}")

        hand_link = next((l for l in self.robot.get_links() if l.name == "panda_hand"), None)
        l_finger = next((l for l in self.robot.get_links() if l.name == "panda_leftfinger"), None)
        r_finger = next((l for l in self.robot.get_links() if l.name == "panda_rightfinger"), None)

        if hand_link:
            hp = hand_link.get_pose().p
            log_lines.append(f"🤖 【手腕基座】 中心XYZ: ({hp[0]:.4f}, {hp[1]:.4f}, {hp[2]:.4f})")
        if l_finger and r_finger:
            lf_p, rf_p = l_finger.get_pose().p, r_finger.get_pose().p
            tcp_x = (lf_p[0] + rf_p[0]) / 2.0
            tcp_y = (lf_p[1] + rf_p[1]) / 2.0
            tcp_z = (lf_p[2] + rf_p[2]) / 2.0
            log_lines.append(f"🎯 【夹爪 TCP】 XYZ: ({tcp_x:.4f}, {tcp_y:.4f}, {tcp_z:.4f})")
            log_lines.append(f"   -> 📐 水平偏差: X差 {tcp_x - pos[0]:.4f} m, Y差 {tcp_y - pos[1]:.4f} m")

        contacts = self.scene.get_contacts()
        contact_found = False
        if contacts:
            for c in contacts:
                name0 = c.bodies[0].entity.name
                name1 = c.bodies[1].entity.name
                if target_actor.name not in [name0, name1]: continue
                if "ground" in name0 or "ground" in name1: continue

                for point in c.points:
                    imp_norm = np.linalg.norm(point.impulse)
                    if imp_norm > 1e-4:
                        contact_found = True
                        force_N = imp_norm * 240.0
                        log_lines.append(f"💥 [{name0}] 接触 [{name1}] | 瞬时力: {force_N:.2f} N")
        if not contact_found: log_lines.append("🟢 目标与机械臂无受力接触。")
        log_lines.append("==============================================================================\n")

        full_log = "\n".join(log_lines)
        print(f"\n{full_log}")
        return full_log

    def _mock_grasp(self, target_name: str, params: dict = None) -> dict:
        if params is None: params = {}
        log(f"进入 _mock_grasp, 目标: {target_name}")

        clean_target = target_name.lower().replace(" ", "_")
        target_actor = next(
            (a for a in self.scene.get_all_actors() if a.name and clean_target in a.name.lower().replace(" ", "_")),
            None)

        if not target_actor:
            err = IntrusionFeedback(error_type="TARGET_NOT_FOUND", collided_pair=[target_name, "scene_missing"])
            return {"ok": False, "reason": err}

        exact_actor_name = target_actor.name
        self._sync_planner_point_cloud([target_name])
        target_pos = np.array(target_actor.get_pose().p)
        initial_target_z = target_pos[2]
        current_qpos = self.robot.get_qpos()

        # 安全回零逻辑，排除奇异点爆炸
        if np.linalg.norm(current_qpos[:7] - self.safe_overhead_qpos[:7]) > 0.02:
            try:
                target_home = np.copy(self.safe_overhead_qpos)
                target_home[7:9] = current_qpos[7:9]
                res_ready = self.planner.plan_qpos([target_home], current_qpos, time_step=0.002)
                if res_ready['status'] == 'Success':
                    self._drive_trajectory(res_ready['position'], True, "")
            except Exception:
                pass

        tcp_offset = self.robot_config.get("tcp_offset_z", 0.1034)
        h = params.get("hover_height", 0.20)
        safe_hover_z = target_pos[2] + tcp_offset + h
        hover_pose = sapien.Pose([target_pos[0], target_pos[1], safe_hover_z], self.downward_quat)
        res_hover = self.planner.plan_pose(hover_pose, self.robot.get_qpos(), time_step=0.002)

        if res_hover['status'] != 'Success':
            err = IntrusionFeedback(error_type="KINEMATIC_OCCLUSION")
            return {"ok": False, "reason": err}

        status, viol = self._drive_trajectory(res_hover['position'], True, exact_actor_name)
        if not status: return {"ok": False, "reason": viol}

        try:
            clearance = self.robot_config.get("approach_clearance", 0.005)
            safe_descend_z = target_pos[2] + tcp_offset + clearance
            descend_pose = sapien.Pose([target_pos[0], target_pos[1], safe_descend_z], self.downward_quat)
            res_descend = self.planner.plan_pose(descend_pose, self.robot.get_qpos(), time_step=0.002)

            if res_descend['status'] != 'Success':
                err = IntrusionFeedback(error_type="Z_APPROACH_BLOCKED")
                return {"ok": False, "reason": err}
        except Exception:
            err = IntrusionFeedback(error_type="Z_APPROACH_EXCEPTION")
            return {"ok": False, "reason": err}

        for j in self.gripper_joints: j.set_drive_target(self.robot_config.get("gripper_max_width", 0.08) / 2.0)
        for _ in range(80): self.scene.step()

        status, viol = self._drive_trajectory(res_descend['position'], True, exact_actor_name)
        if not status: return {"ok": False, "reason": viol}

        self._dump_god_mode_telemetry("下探完毕，准备夹持", target_actor)

        l_finger = next((l for l in self.robot.get_links() if l.name == "panda_leftfinger"), None)
        r_finger = next((l for l in self.robot.get_links() if l.name == "panda_rightfinger"), None)
        offset_x_mm, offset_y_mm = 0.0, 0.0
        if l_finger and r_finger:
            lf_p, rf_p = l_finger.get_pose().p, r_finger.get_pose().p
            offset_x_mm = (((lf_p[0] + rf_p[0]) / 2.0) - target_pos[0]) * 1000
            offset_y_mm = (((lf_p[1] + rf_p[1]) / 2.0) - target_pos[1]) * 1000

        total_target_width = params.get("target_width", 0.04)
        hs = self._get_actor_half_size(target_actor)
        physical_width = hs[0] * 2

        phys_props = self.physics_dict.get(clean_target, {"mass": 0.05, "friction": 1.0, "virtual_pain_limit": 34.0})
        obj_mass = phys_props.get("mass", 0.05)
        obj_fric = phys_props.get("friction", 1.0)
        destruction_threshold = phys_props.get("virtual_pain_limit", 34.0)

        k_gripper = self.robot_config.get("gripper_stiffness", 2000.0)
        squeeze_depth = physical_width - total_target_width
        predicted_force = squeeze_depth * k_gripper if squeeze_depth > 0 else 0.0
        min_grip_force = (obj_mass * 9.81) / (2.0 * obj_fric)

        print(
            f"\n   [🧠 业务逻辑] 预估计算: 设定宽 {total_target_width * 1000:.1f}mm vs 物理宽 {physical_width * 1000:.1f}mm")
        print(
            f"   [🧠 业务逻辑] 预估产生 {predicted_force:.1f}N 挤压力 (维持所需: {min_grip_force:.1f}N, 破坏阈值: {destruction_threshold}N)")

        # 🚀 改造点：发生错误时，把遥测日志塞进错误对象中
        if predicted_force > destruction_threshold:
            telemetry_str = self._dump_god_mode_telemetry("预期阶段触发破坏性拦截！", target_actor)
            err = DestructionFeedback(
                culprit_actor=exact_actor_name,
                safety_threshold_N=destruction_threshold,
                predicted_force_N=predicted_force,
                tcp_offset_x_mm=offset_x_mm,
                tcp_offset_y_mm=offset_y_mm,
                target_width_mm=total_target_width * 1000,
                actual_width_mm=physical_width * 1000
            )
            err.blackbox_log = telemetry_str
            return {"ok": False, "reason": err}

        elif predicted_force < min_grip_force and total_target_width >= physical_width:
            telemetry_str = self._dump_god_mode_telemetry("预期阶段触发滑脱拦截！", target_actor)
            err = SlipFeedback(
                culprit_actor=exact_actor_name,
                min_required_grip_force_N=min_grip_force,
                predicted_force_N=predicted_force,
                tcp_offset_x_mm=offset_x_mm,
                tcp_offset_y_mm=offset_y_mm,
                target_width_mm=total_target_width * 1000,
                actual_width_mm=physical_width * 1000
            )
            err.blackbox_log = telemetry_str
            return {"ok": False, "reason": err}

        # 核心 1：使用标准容差，绝不随意放大，防止穿模
        tolerance = self.robot_config.get("penetration_tolerance_m", 0.002)
        safe_physx_width = max(total_target_width, physical_width - tolerance)
        gripper_target = safe_physx_width / 2.0

        print(f"   [🛡️ 物理映射] 为防止底层矩阵数值爆炸，C++ 驱动器目标钳制为: {safe_physx_width * 1000:.1f}mm")

        # =========================================================================
        # ✅ 终极防崩溃：柔顺平滑插值 (Soft Closing Interpolation)
        # =========================================================================
        for j in self.gripper_joints:
            j.set_drive_property(stiffness=500.0, damping=20.0, force_limit=100.0)

        start_g_targets = [j.get_drive_target() for j in self.gripper_joints]

        for step_idx in range(150):
            alpha = min(1.0, step_idx / 50.0)
            for j, st in zip(self.gripper_joints, start_g_targets):
                j.set_drive_target(st + (gripper_target - st) * alpha)

            self.scene.step()
            t = time.time()
            viol = self.detectors.check_stiffness_and_destruction(self.scene, self.physics_dict, t)
            # 🚀 改造点：动态检测出物理破坏时，收集瞬间的遥测黑匣子
            if viol:
                for j in self.gripper_joints:
                    j.set_drive_property(stiffness=self.robot_config.get("gripper_stiffness", 2000.0),
                                         damping=self.robot_config.get("gripper_damping", 50.0),
                                         force_limit=self.robot_config.get("gripper_force_limit", 150.0))
                viol.tcp_offset_x_mm = offset_x_mm
                viol.tcp_offset_y_mm = offset_y_mm
                telemetry_str = self._dump_god_mode_telemetry("物体破碎瞬间！", target_actor)
                viol.blackbox_log = telemetry_str
                return {"ok": False, "reason": viol}

        for j in self.gripper_joints:
            j.set_drive_property(stiffness=self.robot_config.get("gripper_stiffness", 2000.0),
                                 damping=self.robot_config.get("gripper_damping", 50.0),
                                 force_limit=self.robot_config.get("gripper_force_limit", 150.0))
        for _ in range(20): self.scene.step()

        self._dump_god_mode_telemetry("夹爪闭合, 静息期结束", target_actor)
        self.held_target = exact_actor_name

        print("   [Debug] 夹取成功，执行原路倒放抬升...")
        retreat_trajectory = res_descend['position'][::-1]
        status, viol = self._drive_trajectory(retreat_trajectory, True, exact_actor_name)
        if not status: return {"ok": False, "reason": viol}

        current_target_z = target_actor.get_pose().p[2]
        success_lift = self.robot_config.get("lift_success_threshold_m", 0.035)
        if (current_target_z - initial_target_z) < success_lift:
            # 🚀 改造点：滑脱时注入
            telemetry_str = self._dump_god_mode_telemetry("抬升滑脱瞬间！", target_actor)
            err = SlipFeedback(
                culprit_actor=exact_actor_name,
                min_required_grip_force_N=min_grip_force,
                predicted_force_N=predicted_force,
                tcp_offset_x_mm=offset_x_mm,
                tcp_offset_y_mm=offset_y_mm,
                target_width_mm=total_target_width * 1000,
                actual_width_mm=physical_width * 1000
            )
            err.blackbox_log = telemetry_str
            return {"ok": False, "reason": err}

        return {"ok": True, "reason": ""}

    def _mock_place(self, basket_name: str, params: dict = None) -> dict:
        if params is None: params = {}
        log(f"进入 _mock_place, 目标容器: {basket_name}")

        clean_basket = basket_name.lower().replace(" ", "_")
        basket_actor = next(
            (a for a in self.scene.get_all_actors() if clean_basket in a.name.lower().replace(" ", "_")), None)

        if basket_actor:
            basket_pos = basket_actor.get_pose().p
            basket_hs = self._get_actor_half_size(basket_actor)
            target_pos = np.array([basket_pos[0], basket_pos[1], basket_pos[2] + basket_hs[2]])
        else:
            err = IntrusionFeedback(error_type="TARGET_NOT_FOUND", collided_pair=[basket_name, "scene_missing"])
            return {"ok": False, "reason": err}

        self._sync_planner_point_cloud([self.held_target])

        tcp_offset = self.robot_config.get("tcp_offset_z", 0.1034)
        hover_h = params.get("hover_height", 0.20)
        drop_h = params.get("drop_height", 0.02)

        held_actor = next((a for a in self.scene.get_all_actors() if a.name == self.held_target), None)
        held_offset_z = 0.0
        if held_actor:
            held_hs = self._get_actor_half_size(held_actor)
            held_offset_z = held_hs[2]

        hover_z = target_pos[2] + tcp_offset + held_offset_z + hover_h
        drop_z = target_pos[2] + tcp_offset + held_offset_z + drop_h

        hover_pose = sapien.Pose([target_pos[0], target_pos[1], hover_z], self.downward_quat)
        res_hover = self.planner.plan_pose(hover_pose, self.robot.get_qpos(), time_step=0.002)

        if res_hover['status'] != 'Success':
            err = IntrusionFeedback(error_type="CARRY_PATH_BLOCKED")
            return {"ok": False, "reason": err}

        status, viol = self._drive_trajectory(res_hover['position'], True, self.held_target)
        if not status: return {"ok": False, "reason": viol}

        drop_pose = sapien.Pose([target_pos[0], target_pos[1], drop_z], self.downward_quat)
        res_drop = self.planner.plan_pose(drop_pose, self.robot.get_qpos(), time_step=0.002)

        if res_drop['status'] == 'Success':
            status, viol = self._drive_trajectory(res_drop['position'], True, self.held_target)
            if not status: return {"ok": False, "reason": viol}
        else:
            err = IntrusionFeedback(error_type="UNEXPECTED_COLLISION", collided_pair=["panda_fingers", basket_name])
            return {"ok": False, "reason": err}

        tracking_actor = next((a for a in self.scene.get_all_actors() if a.name == self.held_target), None)

        start_g_targets = [j.get_drive_target() for j in self.gripper_joints]
        open_target = self.robot_config.get("gripper_max_width", 0.08) / 2.0
        for step_idx in range(40):
            alpha = step_idx / 40.0
            for j, st in zip(self.gripper_joints, start_g_targets):
                j.set_drive_target(st + (open_target - st) * alpha)
            self.scene.step()

        retreat_pose = sapien.Pose([target_pos[0], target_pos[1], drop_z + 0.15], self.downward_quat)
        res_retreat = self.planner.plan_pose(retreat_pose, self.robot.get_qpos(), time_step=0.002)
        if res_retreat['status'] == 'Success':
            self._drive_trajectory(res_retreat['position'], False, "")

        active_actors = [a for a in self.scene.get_all_actors() if
                         any("RigidDynamicComponent" in type(c).__name__ for c in a.components)]

        print(f"\n🔍 [高频遥测] 开始追踪 [{self.held_target}] 释放落地的 120 步连续动态...")
        from scipy.spatial.transform import Rotation as R

        # 🚀 改造点：收集时间序列数组
        radar_logs = []

        for step_idx in range(120):
            self.scene.step()
            if tracking_actor and step_idx % 10 == 0:
                pos = tracking_actor.get_pose().p
                q = tracking_actor.get_pose().q
                r = R.from_quat([q[1], q[2], q[3], q[0]])
                up_vector = r.apply([0, 0, 1])
                tilt_deg = math.degrees(math.acos(np.clip(up_vector[2], -1.0, 1.0)))

                contact_names = []
                for c in self.scene.get_contacts():
                    n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
                    if tracking_actor.name == n0: contact_names.append(n1)
                    if tracking_actor.name == n1: contact_names.append(n0)
                contact_names = list(set(contact_names))

                msg = f"   ⏳ Step {step_idx:03d} | Z高度: {pos[2]:.4f}m | 倾角: {tilt_deg:.2f}° | 当前受力接触: {contact_names}"
                print(msg)
                radar_logs.append(msg)

            t = time.time()
            violation = self.detectors.check_stability_and_tipping(self.scene, active_actors, t)
            # 🚀 改造点：发生倾覆时，注入之前收集的所有帧的历史记录
            if violation:
                print(f"💥 [高频遥测] 捕获倾覆违规！终止于 Step {step_idx}。")
                violation.blackbox_log = "\n".join(radar_logs)
                return {"ok": False, "reason": violation}

        self.held_target = None
        return {"ok": True, "reason": ""}

    def execute_dag_plan(self, plan_dag: List[Dict[str, Any]]) -> dict:
        log("=== 启动 DAG 拓扑执行 ===")
        execution_history = []
        completed_nodes = set()

        for node in plan_dag:
            node_id = node["id"]
            action = node["action"]
            target = node["target"]
            depends_on = node.get("depends_on", [])

            if not all(dep in completed_nodes for dep in depends_on):
                err = IntrusionFeedback(error_type="DEPENDENCY_VIOLATION")
                return {
                    "evaluation_status": "REPLAN_REQUIRED",
                    "failed_node": node_id,
                    "error_type": "DEPENDENCY_VIOLATION",
                    "physics_feedback": err
                }

            if action == "grasp":
                res = self._mock_grasp(target, node.get("parameters", {}))
            elif action == "place":
                res = self._mock_place(target, node.get("parameters", {}))
            else:
                err = IntrusionFeedback(error_type="UNIMPLEMENTED_SKILL")
                res = {"ok": False, "reason": err}

            execution_history.append(
                {"node": node_id, "action": action, "status": "SUCCESS" if res["ok"] else "FAILED"})

            if not res["ok"]:
                return {
                    "evaluation_status": "REPLAN_REQUIRED",
                    "failed_node": node_id,
                    "failed_action": action,
                    "failed_target": target,
                    "error_type": "PHYSICS_REJECTED",
                    "physics_feedback": res["reason"],
                    "history": execution_history
                }

            completed_nodes.add(node_id)

        log("=== DAG 拓扑全绿通关 ===")
        return {
            "evaluation_status": "PASS",
            "optimized_plan": plan_dag,
            "history": execution_history
        }
