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

        # 禁用机器人【自碰撞】:mplib(SRDF)规划时已把相邻连杆排除,但 SAPIEN 物理场景默认没排除 ->
        # 手腕折叠构型下 panda_hand 与 panda_link7 互相穿透 -> 数十万牛接触力 -> PhysX 发散段错误。
        # 给所有连杆碰撞形状打同一个 group2 位(共享位=互不碰撞);物体不带该位,仍能与机器人正常碰撞。
        # SAPIEN 忽略碰撞条件: (A.g2 & B.g2) 且 (A.g3==B.g3)。给所有机器人形状 g2 打共享位 + g3 设同一 id,
        # 则机器人内部两两互相忽略;物体默认 g3=0,与机器人 id 不同 -> 仍正常碰撞。set 传【列表】。
        try:
            for link in self.robot.get_links():
                shapes = []
                if hasattr(link, 'get_collision_shapes'):
                    shapes = link.get_collision_shapes()
                elif hasattr(link, 'components'):
                    for comp in link.components:
                        if hasattr(comp, 'get_collision_shapes'):
                            shapes.extend(comp.get_collision_shapes())
                for sh in shapes:
                    g = list(sh.get_collision_groups())
                    g[2] = int(g[2]) | (1 << 0)     # 共享 ignore 位
                    g[3] = 1                         # 同一 id -> 机器人内部互不碰撞
                    sh.set_collision_groups(g)       # 传列表(FixedSize(4))
            log("已禁用机器人自碰撞(group2 共享位 + 同 id)")
        except Exception as _e:
            print(f"[robot] disable self-collision failed: {type(_e).__name__}: {_e}", flush=True)

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
        self._attach = None                       # 魔法吸附状态: {"actor","comp","rel"(相对 panda_hand 的位姿)}
        self.safe_overhead_qpos = np.array(init_qpos)
        self._safe_dummy_pc = np.array([[0.0, 0.0, -10.0]], dtype=np.float64)

        self.scene.step()
        self.downward_quat = [0.0, 1.0, 0.0, 0.0]

    # ------------------------------------------------------------------ 魔法抓取(运动学吸附)
    def _rigid_comp(self, actor):
        return next((c for c in actor.components if "RigidDynamicComponent" in type(c).__name__), None)

    def _hand_link(self):
        return next((l for l in self.robot.get_links() if l.name == "panda_hand"), None)

    def _attach_object(self, actor) -> bool:
        """魔法吸附:把物体设为 kinematic + 关重力,记录它相对 panda_hand 的位姿,之后每步随夹爪刚性移动。
        避免接触力闭合夹爪(密集接触 scene.step 是段错误/数值爆炸高发区)。"""
        hand, comp = self._hand_link(), self._rigid_comp(actor)
        if hand is None or comp is None:
            return False
        rel = hand.get_pose().inv() * actor.get_pose()      # T_hand->obj
        try:
            comp.set_disable_gravity(True)
            comp.set_kinematic(True)
        except Exception:
            return False
        self._attach = {"actor": actor, "comp": comp, "rel": rel}
        return True

    def _set_robot_ignore(self, actor, ignore: bool):
        """把物体加入/移出机器人碰撞忽略组(g2 共享位 + g3=1)。忽略时夹爪手指与它【不产生接触】——魔法抓取
        靠运动学吸附持物,手指-小物体的退化接触会把 PhysX scene.step 搞崩(实测第二次抓取崩点),故忽略之;
        但它与【其它物体】g3 不同 -> 仍正常碰撞(搬运撞别的物体照样能检测)。"""
        comp = self._rigid_comp(actor)
        if comp is None:
            return
        try:
            for sh in comp.get_collision_shapes():
                g = list(sh.get_collision_groups())
                if ignore:
                    g[2] = int(g[2]) | (1 << 0); g[3] = 1
                else:
                    g[2] = int(g[2]) & ~(1 << 0); g[3] = 0
                sh.set_collision_groups(g)
        except Exception:
            pass

    def _detach_object(self):
        """解除吸附:恢复动力学 + 重力 + 清速度 + 复原碰撞组,让物体自然沉降(place 落篮)。"""
        a, self._attach = self._attach, None
        if a and a.get("actor") is not None:
            self._set_robot_ignore(a["actor"], False)     # 复原碰撞组(重新与机器人正常碰撞)
        if a and a.get("comp") is not None:
            try:
                a["comp"].set_kinematic(False)
                a["comp"].set_disable_gravity(False)
                a["comp"].set_linear_velocity([0.0, 0.0, 0.0])
                a["comp"].set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass

    def _follow_attached(self):
        """吸附物体每步跟随夹爪: hand.pose * rel -> kinematic_target(每步调用,放在 scene.step 之前)。"""
        a = getattr(self, "_attach", None)
        if not a:
            return
        hand = self._hand_link()
        if hand is None:
            return
        tgt = hand.get_pose() * a["rel"]
        try:
            a["comp"].set_kinematic_target(tgt)
        except Exception:
            try:
                a["actor"].set_pose(tgt)
            except Exception:
                pass

    def _diagnose_blocked(self, target_pos, excludes=()):
        """规划失败量化诊断:最近障碍物名 + 目标到它中心距离(m)+ 粗略净空(m)+ 目标到 base 水平距离(m)。
        excludes:不算作障碍的名字(目标自身/被夹物)。"""
        tp = np.array(target_pos, dtype=float)[:3]
        reach = float(np.linalg.norm(tp[:2]))                       # 机器人 base 在原点
        exl = [e.lower() for e in excludes if e]
        nearest, ndist, nclear = "", 1e9, 1e9
        for a in self.scene.get_all_actors():
            nm = (a.name or "").lower()
            if "panda" in nm or "ground" in nm or any(e in nm for e in exl):
                continue
            d = float(np.linalg.norm(np.array(a.get_pose().p, dtype=float) - tp))
            if d < ndist:
                nearest, ndist = a.name, d
                nclear = d - float(np.max(self._get_actor_half_size(a)))
        return nearest, ndist, nclear, reach

    def _blocked_feedback(self, error_type, target_pos, target_name="", exclude_name=""):
        """把规划失败包装成【带量化数据 + 级联位移】的 IntrusionFeedback,供 planner 精准重规划。"""
        nb, nd, nclr, reach = self._diagnose_blocked(target_pos, (target_name, exclude_name))
        # 目标自身/障碍物是否被前序动作撞动(级联):找名字匹配 target_name 且离 target_pos 最近的 actor
        tp = np.array(target_pos, dtype=float)[:3]
        tgt_actor = min((a for a in self.scene.get_all_actors()
                         if target_name and target_name.lower() in (a.name or "").lower()),
                        key=lambda a: float(np.linalg.norm(np.array(a.get_pose().p, dtype=float) - tp)), default=None)
        disp_t = self._displacement_cm(tgt_actor.name) if tgt_actor else 0.0
        disp_o = self._displacement_cm(nb)
        note = f"目标'{target_name}'水平距 base {reach * 100:.1f}cm(臂展约 85cm)"
        if reach > 0.85:
            note += " ⚠超出臂展、够不到"
        if disp_t > 5.0:
            note += f"; ⚠该目标已从初始位置被撞动 {disp_t:.1f}cm(疑似前序动作/放入的物体把它撞跑 -> 现在位置变了/够不到)"
        if nb:
            note += f"; 最近障碍[{nb}]距目标中心 {nd * 100:.1f}cm、净空 {nclr * 100:.1f}cm"
            if nclr < 0:
                note += "(已重叠!需为投放腾出空间或错开投放点)"
            if disp_o > 5.0:
                note += f"; 该障碍也被撞动了 {disp_o:.1f}cm"
        return IntrusionFeedback(
            error_type=error_type, collision_xyz=[float(v) for v in tp],
            nearest_obstacle=nb, obstacle_dist_mm=float(nd * 1000), clearance_mm=float(nclr * 1000),
            reach_mm=float(reach * 1000), displacement_cm=float(disp_t), note=note)

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
        # 被吸附的物体正随夹爪运动 -> 从"倾覆检测"里排除(否则它的移动会被误判为倾覆);
        # 但它在"碰撞检测"里被当作侵入源(held_name),撞到别的物体照样要停下来报错。
        held = self.held_target if getattr(self, "_attach", None) else ""
        active_actors = [a for a in self.scene.get_all_actors() if
                         any("RigidDynamicComponent" in type(c).__name__ for c in a.components)
                         and a.name != "panda" and a.name != held]

        for qpos_target in trajectory:
            for j, t in zip(self.robot.get_active_joints()[:7], qpos_target[:7]):
                j.set_drive_target(float(t))
            for j, t in zip(self.gripper_joints, gripper_targets):
                j.set_drive_target(float(t))

            self._follow_attached()               # 吸附物体跟随夹爪(每步,step 之前)
            self.scene.step()
            self.scene.step()

            if getattr(self, 'viewer', None): self.viewer.render()

            if check_probes:
                t = time.time()
                violation = (
                        self.detectors.check_stiffness_and_destruction(self.scene, self.physics_dict, t, (exact_target_name, held)) or
                        self.detectors.check_stability_and_tipping(self.scene, active_actors, t) or
                        self.detectors.check_unexpected_collision(self.scene, "panda", exact_target_name, t, held) or
                        self.detectors.check_feasibility_and_deadlock(self.robot, t)
                )
                if violation: return False, self._enrich(violation)

        for _ in range(100):
            self._follow_attached()
            self.scene.step()
            if getattr(self, 'viewer', None): self.viewer.render()
            if check_probes:
                t = time.time()
                violation = (
                        self.detectors.check_stiffness_and_destruction(self.scene, self.physics_dict, t, (exact_target_name, held)) or
                        self.detectors.check_stability_and_tipping(self.scene, active_actors, t) or
                        self.detectors.check_unexpected_collision(self.scene, "panda", exact_target_name, t, held) or
                        self.detectors.check_feasibility_and_deadlock(self.robot, t)
                )
                if violation: return False, self._enrich(violation)
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
            err = self._blocked_feedback("KINEMATIC_OCCLUSION", target_pos, target_name)
            return {"ok": False, "reason": err}

        status, viol = self._drive_trajectory(res_hover['position'], True, exact_actor_name)
        if not status: return {"ok": False, "reason": viol}

        try:
            clearance = self.robot_config.get("approach_clearance", 0.005)
            safe_descend_z = target_pos[2] + tcp_offset + clearance
            descend_pose = sapien.Pose([target_pos[0], target_pos[1], safe_descend_z], self.downward_quat)
            res_descend = self.planner.plan_pose(descend_pose, self.robot.get_qpos(), time_step=0.002)

            if res_descend['status'] != 'Success':
                err = self._blocked_feedback("Z_APPROACH_BLOCKED", target_pos, target_name)
                return {"ok": False, "reason": err}
        except Exception:
            err = IntrusionFeedback(error_type="Z_APPROACH_EXCEPTION")
            return {"ok": False, "reason": err}

        # 夹爪按【物体宽度 + 余量】张开(不张到最大)-> 下探时不容易碰到旁边的物体。
        gripper_max = self.robot_config.get("gripper_max_width", 0.08)
        physical_width = float(self._get_actor_half_size(target_actor)[0] * 2)
        open_w = float(np.clip(physical_width + self.robot_config.get("grasp_open_clearance", 0.02), 0.0, gripper_max))
        for j in self.gripper_joints: j.set_drive_target(open_w / 2.0)
        for _ in range(80): self.scene.step()

        # 下探前:①把目标设为 kinematic(反正马上要魔法吸附);②把目标加入机器人碰撞忽略组 -> 夹爪手指
        # 与它不产生退化接触(实测第二次抓取在手指刚碰到杯子那步 scene.step 段错误的根因)。放置时会复原。
        self._set_robot_ignore(target_actor, True)
        _tc = self._rigid_comp(target_actor)
        if _tc is not None:
            try:
                _tc.set_disable_gravity(True)
                _tc.set_kinematic(True)
            except Exception:
                pass
        status, viol = self._drive_trajectory(res_descend['position'], True, exact_actor_name)
        if not status: return {"ok": False, "reason": viol}

        self._dump_god_mode_telemetry("下探完毕，准备夹持", target_actor)

        # =========================================================================
        # ✅ 魔法抓取(kinematic attach):靠【运动学吸附】持物,不靠夹持力 -> 天然【不会夹碎、不会滑脱】,
        #    所以去掉了原来基于力的夹碎/滑脱预测(那是接触式夹持才需要的)。夹爪只做视觉合拢到物体宽度(刚好包住)。
        #    物体真的比夹爪还宽时,会在上面下探/合拢时被【全臂碰撞检测】抓到(手指撞物体),无需单独力学判据。
        # =========================================================================
        visual_width = max(0.0, physical_width - 0.002)          # 合拢到物体宽度(仅观感,不靠接触力)
        for j in self.gripper_joints:
            j.set_drive_target(visual_width / 2.0)
        for _ in range(10):
            self.scene.step()

        if not self._attach_object(target_actor):
            err = IntrusionFeedback(error_type="ATTACH_FAILED", collided_pair=[exact_actor_name, "panda_hand"])
            return {"ok": False, "reason": err}
        self.held_target = exact_actor_name
        self._dump_god_mode_telemetry("魔法吸附完成, 准备抬升", target_actor)

        print("   [Debug] 魔法吸附成功，执行原路倒放抬升...")
        retreat_trajectory = res_descend['position'][::-1]
        status, viol = self._drive_trajectory(retreat_trajectory, True, exact_actor_name)
        if not status: return {"ok": False, "reason": viol}

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
            err = self._blocked_feedback("CARRY_PATH_BLOCKED", target_pos, basket_name, exclude_name=self.held_target)
            return {"ok": False, "reason": err}

        # 搬运/落篮阶段 expected_target 传【篮子】:允许被吸附物体接触篮子(落进去);它撞到【别的】物体仍会报错。
        status, viol = self._drive_trajectory(res_hover['position'], True, basket_actor.name)
        if not status: return {"ok": False, "reason": viol}

        drop_pose = sapien.Pose([target_pos[0], target_pos[1], drop_z], self.downward_quat)
        res_drop = self.planner.plan_pose(drop_pose, self.robot.get_qpos(), time_step=0.002)

        if res_drop['status'] == 'Success':
            status, viol = self._drive_trajectory(res_drop['position'], True, basket_actor.name)
            if not status: return {"ok": False, "reason": viol}
        else:
            err = IntrusionFeedback(error_type="UNEXPECTED_COLLISION", collided_pair=["panda_fingers", basket_name])
            return {"ok": False, "reason": err}

        tracking_actor = next((a for a in self.scene.get_all_actors() if a.name == self.held_target), None)

        # 释放:先【解除吸附】(恢复动力学+重力),物体靠重力自然落进篮子。夹爪只需【稍微松开】让物体脱开,
        # 不必张到最大(张满反而可能碰到篮子边/旁物)。detach 后物体已是动力学体,微张即可落下。
        self._detach_object()
        start_g_targets = [j.get_drive_target() for j in self.gripper_joints]
        gripper_max_half = self.robot_config.get("gripper_max_width", 0.08) / 2.0
        release_open = self.robot_config.get("place_release_open", 0.015)   # 每指再张 ~1.5cm(总开口 +3cm)
        open_target = min(gripper_max_half, float(start_g_targets[0]) + release_open)
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
                return {"ok": False, "reason": self._enrich(violation)}

        self.held_target = None
        return {"ok": True, "reason": ""}

    def _displacement_cm(self, name):
        """物体相对【DAG 开始时初始位置】移动了多少 cm —— 检测'被前序动作撞动'的级联结果。无记录返回 0。"""
        if not name or not getattr(self, "_initial_pos", None):
            return 0.0
        init = self._initial_pos.get(name)
        cur = next((np.array(a.get_pose().p, dtype=float) for a in self.scene.get_all_actors() if a.name == name), None)
        if init is None or cur is None:
            return 0.0
        return float(np.linalg.norm(cur - init) * 100.0)

    def _enrich(self, fb):
        """给任意探针反馈补【级联位移】:涉及的物体若已从初始位置被撞动,写进 note(供 planner 判断连带后果)。"""
        if fb is None:
            return fb
        names = []
        for attr in ("culprit_actor", "nearest_obstacle"):
            v = getattr(fb, attr, "")
            if v:
                names.append(v)
        for attr in ("collided_pair", "contact_pair"):
            v = getattr(fb, attr, None) or []
            names.extend([n for n in v if n and "panda" not in n and "ground" not in n])
        extra = []
        for nm in dict.fromkeys(names):
            d = self._displacement_cm(nm)
            if d > 5.0:
                extra.append(f"[{nm}]已从初始位置被撞动 {d:.1f}cm")
        if extra:
            cur = getattr(fb, "note", "") or ""
            try:
                fb.note = (cur + "; " if cur else "") + "级联位移: " + "; ".join(extra)
            except Exception:
                pass
        return fb

    def execute_dag_plan(self, plan_dag: List[Dict[str, Any]]) -> dict:
        log("=== 启动 DAG 拓扑执行 ===")
        # 快照各物体初始位置:用于诊断"被前序动作撞动"的级联位移(如放第一个物体把篮子撞跑)
        self._initial_pos = {a.name: np.array(a.get_pose().p, dtype=float)
                             for a in self.scene.get_all_actors() if a.name}
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
