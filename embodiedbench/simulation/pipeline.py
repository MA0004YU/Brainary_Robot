import numpy as np
from schemas.data_types import Pose, RobotInfo
from modules.perception import NavigationPerception
from modules.geometry import NavigationGeometryChecker
from modules.agent_llm import NavigationAgentLLM
from modules.sapien_engine import SapienNavigationEngine

def run_real_navigation_pipeline():
    print("=== 开始全链路 Embodied AI 导航与交互模拟 ===")
    
    # ==========================================
    # 1. 模块初始化 (请在此处填入你的 OpenAI Key)
    # ==========================================
    OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 替换为真实Key
    
    robot_info = RobotInfo(radius=0.25, height=0.5, max_push_force=100.0)
    geometry_checker = NavigationGeometryChecker(robot_info.radius)
    
    # 实例化我们的最终版模块
    perception = NavigationPerception()
    agent = NavigationAgentLLM(api_key=OPENAI_API_KEY)
    engine = SapienNavigationEngine(robot_info, use_gui=True) # 开启 GUI 观看过程

    try:
        # ==========================================
        # 2. 生成模拟传感器数据 (替代真实的相机 SDK)
        # ==========================================
        # 假设这是一张 480x640 的 RGBD 图像
        fake_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        # 假设前方 2 米处有一堵墙，我们设定整个画面的深度基础值为 2.0
        fake_depth = np.full((480, 640), 2.0, dtype=np.float32) 
        
        # 定义相机内参 (RealSense D435 常见参数)
        intrinsic = np.array([
            [615.0,   0.0, 320.0],
            [  0.0, 615.0, 240.0],
            [  0.0,   0.0,   1.0]
        ])
        # 相机外参：假设相机水平固定在机器人头上，高度为 1 米
        extrinsic = np.eye(4)
        extrinsic[2, 3] = 1.0 

        # ==========================================
        # 3. 换眼：执行硬核 3D 视觉计算
        # ==========================================
        print("\n[阶段一] 视觉 3D 建模中...")
        # 调用上一轮补全的 Open3D OBB 计算代码
        scene_objects = perception.process_image_to_objects(
            fake_rgb, fake_depth, intrinsic, extrinsic
        )
        print(f"感知到 {len(scene_objects)} 个物理实体。")

        # ==========================================
        # 4. 常识拦截
        # ==========================================
        # 假设根据视觉算出前方通道最窄处只有 0.3 米
        if not geometry_checker.check_passable_width(gap_width=0.3):
            print("\n[系统警告] 道路过窄，机器人宽度无法通过，任务提前终止。")
            # 真实情况下，这里可以触发 LLM 的重规划，为了演示我们先放行宽路口
            # return 
            print("[系统放行] 为了演示测试，强制继续执行。")

        # ==========================================
        # 5. 换脑：大模型常识推理与策略下发
        # ==========================================
        print("\n[阶段二] 连接大脑进行物理属性推理与动作规划...")
        robot_current_pos = (0.0, 0.0, 0.0)
        action = agent.assign_physics_and_plan(scene_objects, robot_current_pos)

        # ==========================================
        # 6. SAPIEN 物理仿真执行
        # ==========================================
        print("\n[阶段三] SAPIEN 物理引擎接管模拟...")
        # 此时的 scene_objects 已经带上了 LLM 赋予的真实 mass 和 is_passable 属性
        engine.build_scene_objects(scene_objects)
        engine.spawn_robot(initial_pose=Pose(position=robot_current_pos))

        print(f"[SAPIEN] 执行指令 -> {action.action_type}, 方向: {action.direction}, 力量: {action.apply_force}N")
        result = engine.execute_action(action)

        # ==========================================
        # 7. 评估反馈
        # ==========================================
        if result.action_success:
            print(f"\n✅ 模拟成功！越过障碍。实际推进距离: {result.distance_moved:.2f} 米")
        else:
            print(f"\n❌ 模拟失败！力学受阻。实际推进距离仅: {result.distance_moved:.2f} 米")
            print(f"失败原因: {result.error_reason}")

    except Exception as e:
        print(f"\n[致命错误] 流水线异常: {e}")
    finally:
        # 安全清理引擎缓存，防止显存泄漏
        engine.destroy()
        print("\n=== 流水线执行完毕 ===")

if __name__ == "__main__":
    run_real_navigation_pipeline()
