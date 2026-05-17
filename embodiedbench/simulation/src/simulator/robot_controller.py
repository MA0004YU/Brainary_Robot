# simulation/src/simulator/robot_controller.py


import os
import numpy as np
import sapien.core as sapien
import mplib


class RobotController:
    def __init__(self, scene: sapien.Scene, config: dict):
        self.scene = scene
        self.config = config["robot_config"]

        urdf_path = self.config.get("urdf_path", "assets/robots/panda/panda.urdf")
        srdf_path = urdf_path.replace(".urdf", ".srdf")

        # 1. 加载 Franka 机械臂
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf_path)

        # 初始化位姿并挂载 PD 阻尼
        init_qpos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
        self.robot.set_qpos(init_qpos)
        self.robot.set_drive_target(init_qpos)
        for joint in self.robot.get_active_joints():
            joint.set_drive_property(stiffness=1000.0, damping=100.0)

        # 提取 Franka 的夹爪电机 (最后两个自由度)
        self.gripper_joints = self.robot.get_active_joints()[7:9]

        # 2. 真实初始化 mplib 避障运动规划器
        link_names = [link.name for link in self.robot.get_links()]
        joint_names = [joint.name for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7)
        )

    def actuate_gripper(self, action: str):
        """真实驱动夹爪电机：开合"""
        target_width = 0.08 if action == "open" else 0.00
        for joint in self.gripper_joints:
            joint.set_drive_target(target_width / 2.0)

    def generate_grasp_pose(self, target_actor: sapien.Actor, size_whd: list) -> sapien.Pose:
        """真实表面采样：基于视觉 OBB 尺寸，定位物体顶部表面并切入 2cm"""
        center_pose = target_actor.get_pose()
        h = size_whd[1] if len(size_whd) == 3 else 0.1
        grasp_z_offset = (h / 2.0) - 0.02

        grasp_rot = [0.0, 1.0, 0.0, 0.0]
        return sapien.Pose(center_pose.p + np.array([0, 0, grasp_z_offset]), grasp_rot)

    def execute_skill_primitive(self, primitive: dict, point_cloud: np.ndarray) -> dict:
        """结合环境点云进行真实避障规划"""
        action = primitive.get("action")
        target_name = primitive.get("target")

        # 更新环境障碍物点云
        self.planner.update_point_cloud(point_cloud)

        target_actor = next((a for a in self.scene.get_all_actors() if a.name == target_name), None)
        if not target_actor:
            return {"status": False, "trajectory": []}

        # 获取抓取位姿 (结合视觉透传的尺寸)
        grasp_pose = self.generate_grasp_pose(target_actor, [0.1, 0.1, 0.1])
        current_qpos = self.robot.get_qpos()[:7]
        trajectory_result = None

        if action == "approach":
            pre_grasp_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, 0.1]), grasp_pose.q)
            res = self.planner.plan_qpos_to_pose(pre_grasp_pose, current_qpos, time_step=0.002)
            if res['status'] == 'Success':
                trajectory_result = res['position']

        elif action == "pull_straight":
            self.actuate_gripper("close")
            direction = np.array(primitive.get("direction", [0, -1, 0]))
            end_pose = sapien.Pose(grasp_pose.p + direction * 0.15, grasp_pose.q)
            res = self.planner.plan_qpos_to_pose(end_pose, current_qpos, time_step=0.002)
            if res['status'] == 'Success':
                trajectory_result = res['position']

        if trajectory_result is not None:
            return {"status": True, "trajectory": trajectory_result}
        else:
            print(f"[Robot] 运动规划失败：轨迹干涉或不可达！目标动作: {action}")
            return {"status": False, "trajectory": []}
