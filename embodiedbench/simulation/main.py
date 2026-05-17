# simulation/main.py

import os
import cv2
import time
import numpy as np
import dataclasses
import sapien.core as sapien

from src.perception import PerceptionPipeline
from src.simulator import SceneBuilder, RobotController, PhysicsBoundaryDetectors
from src.memory import WorkingMemory, PrincipleExtractor, PHYSICS_DICTIONARY
from src.plan import LLMPlanner


def extract_topology_from_perception(vision_out: dict) -> list:
    """
    从视觉管线输出的 JSON 字典中解析空间几何干涉，
    """
    topology = []
    entities = vision_out["scene_entities"]

    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i == j:
                continue
            # 提取视觉解算出的世界系 3D 质心坐标
            p1 = np.array(e1["pose"]["pos"])
            p2 = np.array(e2["pose"]["pos"])

            # 空间水平距离与垂直高度差判定
            if np.linalg.norm(p1[:2] - p2[:2]) < 0.15 and (p2[2] - p1[2]) > 0.05:
                topology.append({"id": e1["id"], "spatial_state": f"compressed_by_{e2['id']}"})
                topology.append({"id": e2["id"], "spatial_state": f"stacking_on_top_of_{e1['id']}"})
    return topology


def load_real_world_three_views(data_dir: str):
    """从输入目录抓取真实的前、上、侧三视图 RGB-D 对"""
    views = ["front", "top", "side"]
    rgb_dict = {}
    depth_dict = {}

    for v in views:
        rgb_path = os.path.join(data_dir, f"{v}.png")
        depth_path = os.path.join(data_dir, f"{v}_depth.npy")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"[Main ERROR] 缺失关键多目视图对: {v} (期待路径: {rgb_path} 或 {depth_path})")

        img = cv2.imread(rgb_path)
        rgb_dict[v] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth_dict[v] = np.load(depth_path)

    return rgb_dict, depth_dict


def extract_omnidirectional_point_cloud(scene: sapien.Scene) -> np.ndarray:
    """在 SAPIEN 沙盒中动态架设虚拟多机位相机，为 mplib 避障算法融合出无死角点云"""
    cameras_config = [
        {"name": "cam_top", "pose": sapien.Pose([0.5, 0.0, 1.2], [0.707, 0, 0.707, 0])},
        {"name": "cam_front", "pose": sapien.Pose([1.2, 0.0, 0.5], [0.5, -0.5, 0.5, -0.5])},
        {"name": "cam_side", "pose": sapien.Pose([0.5, 0.8, 0.5], [0.0, 0.0, -0.707, 0.707])}
    ]

    virtual_cameras = []
    for cfg in cameras_config:
        cams = [c for c in scene.get_cameras() if c.name == cfg["name"]]
        if not cams:
            cam = scene.add_camera(name=cfg["name"], width=640, height=480, fovy=np.deg2rad(60), near=0.1, far=10.0)
            cam.set_pose(cfg["pose"])
            virtual_cameras.append(cam)
        else:
            virtual_cameras.append(cams[0])

    scene.update_render()
    fused_valid_xyz = []

    for cam in virtual_cameras:
        cam.take_picture()
        pos_rgba = cam.get_position_rgba()
        xyz_world = pos_rgba[..., :3].reshape(-1, 3)
        valid_mask = (pos_rgba[..., 3].flatten() > 0) & (xyz_world[:, 2] > 0.01)
        fused_valid_xyz.append(xyz_world[valid_mask])

    final_point_cloud = np.vstack(fused_valid_xyz)
    if len(final_point_cloud) > 10000:
        indices = np.random.choice(len(final_point_cloud), 10000, replace=False)
        final_point_cloud = final_point_cloud[indices]

    return final_point_cloud


def map_to_proprioceptive_frame(global_primitive: dict, robot_base_pose: sapien.Pose) -> dict:
    """将全局世界系下的规划动作，严格变换至机械臂本体感觉坐标系，确保底层控制精度"""
    mapped_primitive = global_primitive.copy()
    T_world_to_base = robot_base_pose.inv().to_transformation_matrix()

    if "direction" in mapped_primitive:
        dir_world = np.array(mapped_primitive["direction"] + [0.0])
        dir_base = (T_world_to_base @ dir_world)[:3]
        norm = np.linalg.norm(dir_base)
        mapped_primitive["direction"] = (dir_base / norm).tolist() if norm > 0 else dir_base.tolist()

    if "placement_offset" in mapped_primitive:
        offset_world = np.array(mapped_primitive["placement_offset"] + [0.0])
        offset_base = (T_world_to_base @ offset_world)[:3]
        mapped_primitive["placement_offset"] = offset_base.tolist()

    return mapped_primitive


def run_end_to_end_tamp_loop():
    config_path = "config/global_config.yaml"
    perception_module = PerceptionPipeline(config_path)

    # 读取现实三视图
    try:
        rgb_views, depth_views = load_real_world_three_views("inputs/real_world_data/")
    except Exception as e:
        print(f"[Main FATAL] 真实世界图像读取失败: {e}")
        return

    # 执行多目融合解算，吐出物体方块 JSON
    vision_out = perception_module.process_scene(rgb_views, depth_views)

    print("=================== 文本拓扑提取与工作记忆入库 ===================")
    # 提取物体之间的语义树关系
    scene_topology = extract_topology_from_perception(vision_out)

    # 强注入初始目标指令与视觉拓扑拓扑关系
    wm = WorkingMemory(task_name="dense_object_extraction", target_id="plastic_box_1")
    wm.update_scene_topology(scene_topology)

    print("=================== 开环长序列规划 ===================")
    # 大模型在除了视觉输入和拓扑入库之外的所有流程开始前，率先阅读任务并给出第一版plan
    planner_llm = LLMPlanner(perception_module.config)
    global_action_plan = planner_llm.generate_initial_global_plan(wm.get_full_context())

    print("=================== 虚拟物理沙盒实例化与计划接收 ===================")
    # simulator根据视觉 JSON 主动建模
    scene_simulator = SceneBuilder(config_path, PHYSICS_DICTIONARY)
    scene_simulator.build_twin_world(vision_out)

    # 初始化小脑控制器和探针，接收plan
    robot_agent = RobotController(scene_simulator.scene, scene_simulator.config)
    detectors = PhysicsBoundaryDetectors(scene_simulator.config)
    extractor = PrincipleExtractor(scene_simulator.config)

    print("=================== 虚拟演练与物理熔断闭环 ===================")
    success = False

    while not success and wm.memory["task_context"]["attempt_count"] < 5:
        triggered_fb = None

        # 抽取沙盒避障点云
        scene_point_cloud = extract_omnidirectional_point_cloud(scene_simulator.scene)

        for primitive in global_action_plan:
            robot_base_pose = robot_agent.robot.get_pose()

            # 将大模型动作转换为机械臂本体感觉坐标系蓝图
            proprioceptive_blueprint = map_to_proprioceptive_frame(primitive, robot_base_pose)

            # 喂给小脑执行避障运动规划
            plan_result = robot_agent.execute_skill_primitive(proprioceptive_blueprint, scene_point_cloud)
            if not plan_result["status"]:
                print(f"[Main] 技能动作 {primitive['action']} 运动学不可达，跳过。")
                continue

            trajectory = plan_result["trajectory"]

            # 500Hz 物理高频步进演练
            for qpos_target in trajectory:
                robot_agent.robot.set_drive_target(np.concatenate([
                    qpos_target,
                    [robot_agent.gripper_joints[0].get_drive_target()],
                    [robot_agent.gripper_joints[1].get_drive_target()]
                ]))
                scene_simulator.scene.step()
                t = time.time()

                # 探针高频全量轮询
                active_actors = [a for a in scene_simulator.scene.get_all_actors() if not a.is_kinematic]
                triggered_fb = (
                        detectors.check_stiffness_and_destruction(scene_simulator.scene, PHYSICS_DICTIONARY, t) or
                        detectors.check_stability_and_tipping(scene_simulator.scene, active_actors, t) or
                        detectors.check_unexpected_collision(scene_simulator.scene, "panda", "plastic_box_1", t) or
                        detectors.check_feasibility_and_deadlock(robot_agent.robot, t)
                )
                if triggered_fb:
                    break

            if triggered_fb:
                print(f"🛑 [Main Hub] 检测到物理违规，熔断当前计划！故障类型: {triggered_fb.error_type}")
                wm.append_action_history(primitive, "FAILED")
                break
            else:
                wm.append_action_history(primitive, "SUCCESS")

        if triggered_fb:
            print("=================== 事件驱动反思与重规划 ===================")
            # 物理违规触发反思大模型升维提炼
            rule = extractor.extract_scene_based_principle(dataclasses.asdict(triggered_fb), wm.get_full_context())
            wm.mount_failure_payload(triggered_fb, rule)
            # 重新唤醒大模型进行局部重规划，新 Plan 会在下一轮循环中被 SAPIEN 重新接收演练
            global_action_plan = planner_llm.replan_triggered_by_event(wm.get_full_context())
        else:
            success = True
            print("🎉 通过虚拟沙盒全量物理验证，任务全链路安全达成！")


if __name__ == "__main__":
    run_end_to_end_tamp_loop()
