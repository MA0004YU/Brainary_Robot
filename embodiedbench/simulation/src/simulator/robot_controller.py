# simulation/src/simulator/robot_controller.py

import numpy as np
import sapien.core as sapien
import mplib
import time

class BlueprintGraphEngine:
    def __init__(self, scene: sapien.Scene, config: dict, detectors, physics_dict):
        self.scene = scene
        self.config = config["robot_config"]
        self.detectors = detectors
        self.physics_dict = physics_dict

        urdf_path = self.config.get("urdf_path", "assets/robots/panda/panda.urdf")
        srdf_path = urdf_path.replace(".urdf", ".srdf")

        # 1. 物理层机械臂实例
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf_path)
        self.robot.set_name("panda")

        init_qpos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
        self.robot.set_qpos(init_qpos)
        self.robot.set_drive_target(init_qpos)
        for joint in self.robot.get_active_joints():
            joint.set_drive_property(stiffness=1000.0, damping=100.0)
        self.gripper_joints = self.robot.get_active_joints()[7:9]

        # 2. 运动学规划器
        self.planner = mplib.Planner(
            urdf=urdf_path, srdf=srdf_path,
            user_link_names=[link.name for link in self.robot.get_links()],
            user_joint_names=[joint.name for joint in self.robot.get_active_joints()],
            move_group="panda_hand",
            joint_vel_limits=np.ones(7), joint_acc_limits=np.ones(7)
        )

    def _sync_planner_point_cloud(self):
        """将 SAPIEN 中的所有障碍物 OBB 同步给 mplib 避障引擎"""
        # (在此处接入您已有的多机位点云提取逻辑即可，维持原样)
        pass

    def _step_physics_with_probes(self, trajectory: List[np.ndarray], target_name: str) -> tuple:
        """核心考场：在轨迹回放中开启 500Hz 探针全覆盖扫描"""
        active_actors = [a for a in self.scene.get_all_actors() if not a.is_kinematic and a.name != "panda"]
        
        for qpos_target in trajectory:
            self.robot.set_drive_target(np.concatenate([
                qpos_target, 
                [self.gripper_joints[0].get_drive_target()], 
                [self.gripper_joints[1].get_drive_target()]
            ]))
            self.scene.step()
            t = time.time()

            # 探针阵列全量拦截
            violation = (
                self.detectors.check_stiffness_and_destruction(self.scene, self.physics_dict, t) or
                self.detectors.check_stability_and_tipping(self.scene, active_actors, t) or
                self.detectors.check_unexpected_collision(self.scene, "panda", target_name, t) or
                self.detectors.check_feasibility_and_deadlock(self.robot, t)
            )
            
            if violation:
                return False, violation
        return True, None

    def _execute_skill(self, skill_name: str, target_name: str, params: dict) -> tuple:
        """将 9 大原子接口编译为底层 mplib IK 动作流"""
        self._sync_planner_point_cloud()
        current_qpos = self.robot.get_qpos()[:7]
        target_actor = next((a for a in self.scene.get_all_actors() if a.name == target_name), None)
        
        # 夹爪动作不涉及本体移动
        if skill_name == "grasp":
            for j in self.gripper_joints: j.set_drive_target(0.0) # 合拢
            for _ in range(params.get("close_wait_steps", 80)): self.scene.step()
            return True, None
            
        elif skill_name == "wait" or skill_name == "place":
            if skill_name == "place" or params.get("gripper") == "open":
                for j in self.gripper_joints: j.set_drive_target(0.04)
            wait_steps = params.get("open_wait_steps", 60) if skill_name == "place" else params.get("wait_steps", 100)
            for _ in range(wait_steps): self.scene.step()
            return True, None

        # 空间规划动作
        if not target_actor: return False, {"error_type": "TARGET_NOT_FOUND"}
        t_pose = target_actor.get_pose()
        
        target_ik_pose = None
        if skill_name == "move_above":
            z_off = params.get("height_offset", 0.1)
            target_ik_pose = sapien.Pose(t_pose.p + np.array([0, 0, z_off]), [0, 1, 0, 0]) # 保持 Top-Down 约束
        elif skill_name == "descend":
            z_tgt = params.get("target_height", t_pose.p[2])
            ee_pose = self.robot.get_links()[-1].get_pose()
            target_ik_pose = sapien.Pose([ee_pose.p[0], ee_pose.p[1], z_tgt], ee_pose.q)
        elif skill_name == "lift" or skill_name == "retreat":
            z_lift = params.get("lift_height", params.get("retreat_height", 0.15))
            ee_pose = self.robot.get_links()[-1].get_pose()
            target_ik_pose = sapien.Pose(ee_pose.p + np.array([0, 0, z_lift]), ee_pose.q)

        if target_ik_pose:
            res = self.planner.plan_qpos_to_pose(target_ik_pose, current_qpos, time_step=0.002)
            if res['status'] == 'Success':
                return self._step_physics_with_probes(res['position'], target_name)
            else:
                return False, {"error_type": "KINEMATIC_UNREACHABLE"}
                
        return False, {"error_type": "SKILL_NOT_IMPLEMENTED"}

    def execute_blueprint(self, blueprint_json: dict) -> dict:
        """核心图遍历器：严密解析执行图，遇到物理阻碍立刻熔断转入 on_failure 分支"""
        graph = blueprint_json["execution_graph"]
        current_node_id = graph["start"]
        nodes = graph["nodes"]
        
        execution_history = []
        final_violation = None

        while current_node_id:
            node = nodes[current_node_id]
            node_type = node["type"]
            
            if node_type in ["skill", "parallel"]:
                # 当前简化 parallel 统一走 skill 的物理通道进行位置验证
                skill_val = node.get("skill") or node.get("goals", {}).get("position_goal", {}).get("skill")
                target_val = node.get("target") or node.get("goals", {}).get("position_goal", {}).get("target")
                params = node.get("params", {})
                
                status, violation = self._execute_skill(skill_val, target_val, params)
                execution_history.append({"node": current_node_id, "status": "SUCCESS" if status else "FAILED"})
                
                if status:
                    current_node_id = node.get("next")
                else:
                    final_violation = violation
                    current_node_id = node.get("on_failure")

            elif node_type == "condition":
                cond_name = node["condition"]["name"]
                is_true = False
                
                # 读取当前的物理常识状态
                if cond_name == "object_in_gripper":
                    width = self.gripper_joints[0].get_qpos() + self.gripper_joints[1].get_qpos()
                    is_true = (width > 0.005) # 大于闭合死区，判定为抓取实体
                
                current_node_id = node["if_true"] if is_true else node["if_false"]

            elif node_type == "terminal":
                if node["result"] == "failure":
                    return {
                        "evaluation_status": "REJECTED_BY_PHYSICS",
                        "failed_node": current_node_id,
                        "physics_feedback": final_violation.__dict__ if final_violation else node.get("failure_reason"),
                        "history": execution_history
                    }
                else:
                    return {"evaluation_status": "SUCCESS_VERIFIED", "history": execution_history}

        return {"evaluation_status": "ERROR_DAG_CORRUPTED"}
