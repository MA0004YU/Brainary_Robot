def run_real_navigation():
    # 初始化
    api_key = "sk-xxxxxxxxxxxxxxxxxxxx" # 替换你的 Key
    agent = NavigationAgentLLM(api_key=api_key)
    perception = NavigationPerception()
    engine = SapienNavigationEngine(RobotInfo(), use_gui=True)

    # 1. 获取真实传感器数据（此处以 numpy 占位）
    # 在实际任务中，你会从相机 SDK 中读取这些数据
    fake_rgb = np.zeros((480, 640, 3))
    fake_depth = np.ones((480, 640))
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]) # 示例相机内参

    # 2. 换眼：从图像中提取物体
    scene_objects = perception.process_image_to_objects(fake_rgb, fake_depth, K, np.eye(4))
    
    # 3. 构建场景
    engine.build_scene_objects(scene_objects)
    engine.spawn_robot(initial_pose=Pose(position=(0.0, 0.0, 0.0)))

    # 4. 换脑：请求 AI 决策
    action = agent.assign_physics_and_plan(scene_objects, robot_pos=(0, 0))

    # 5. 执行
    result = engine.execute_action(action)
    print(f"执行结果: {result.action_success}, 移动距离: {result.distance_moved}")
