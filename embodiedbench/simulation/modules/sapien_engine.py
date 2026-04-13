import sapien.core as sapien
import numpy as np
import math
from typing import Dict, List
from schemas.data_types import SceneObject, NavigationAction, SimulationResult, Pose, RobotInfo

class SapienNavigationEngine:
    def __init__(self, robot_info: RobotInfo, use_gui: bool = False):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer() if use_gui else None
        if self.renderer:
            self.engine.set_renderer(self.renderer)

        self.scene_config = sapien.SceneConfig()
        self.scene_config.default_static_friction = 0.5
        self.scene_config.default_dynamic_friction = 0.5
        self.scene_config.solver_iterations = 20
        
        self.scene = self.engine.create_scene(self.scene_config)
        self.scene.set_timestep(1 / 240.0)
        self.scene.add_ground(altitude=0.0)
        
        self.actors: Dict[str, sapien.Actor] = {}
        self.robot_info = robot_info
        self.robot_actor = None

    def _convert_pose_to_sapien(self, pose: Pose) -> sapien.Pose:
        return sapien.Pose(p=np.array(pose.position), q=np.array(pose.quaternion))

    def build_scene_objects(self, objects: List[SceneObject]):
        """构建障碍物、纸箱和门帘"""
        for obj in objects:
            builder = self.scene.create_actor_builder()
            half_size = [d / 2.0 for d in obj.dimensions]
            
            # 如果是门帘 (is_passable=True)，不添加物理碰撞体，只加视觉体
            if obj.physics and obj.physics.is_passable:
                builder.add_box_visual(half_size=half_size, color=[0.2, 0.5, 0.8, 0.5]) # 半透明蓝色
                actor = builder.build_kinematic(name=obj.object_id) # 设置为 Kinematic 防止掉落
            else:
                material = self.scene.create_physical_material(
                    static_friction=obj.physics.friction_static if obj.physics else 0.5,
                    dynamic_friction=obj.physics.friction_dynamic if obj.physics else 0.5,
                    restitution=0.0
                )
                builder.add_box_collision(half_size=half_size, material=material)
                builder.add_box_visual(half_size=half_size, color=[0.7, 0.5, 0.3]) # 纸箱色
                actor = builder.build(name=obj.object_id)
                if obj.physics and obj.physics.mass:
                    actor.set_mass(obj.physics.mass)
            
            actor.set_pose(self._convert_pose_to_sapien(obj.current_pose))
            self.actors[obj.object_id] = actor

    def spawn_robot(self, initial_pose: Pose):
        """生成代表机器人的圆柱体"""
        builder = self.scene.create_actor_builder()
        # 为防止圆柱体翻倒，我们在高级仿真中通常会锁住旋转轴，这里为了简单将其重心下压或使用较低高度
        builder.add_cylinder_collision(radius=self.robot_info.radius, half_length=self.robot_info.height / 2.0)
        builder.add_cylinder_visual(radius=self.robot_info.radius, half_length=self.robot_info.height / 2.0, color=[0.8, 0.1, 0.1])
        
        self.robot_actor = builder.build(name="robot")
        self.robot_actor.set_mass(50.0) # 机器人比较重
        # 抬高一点防止卡地
        p = list(initial_pose.position)
        p[2] = self.robot_info.height / 2.0 + 0.01 
        self.robot_actor.set_pose(sapien.Pose(p=p))

    def execute_action(self, action: NavigationAction) -> SimulationResult:
        """执行推进动作"""
        start_pose = self.robot_actor.get_pose().p.copy()
        
        # 限制力不超过机器人的最大推力
        applied_force = min(action.apply_force, self.robot_info.max_push_force)
        force_vec = np.array([action.direction[0] * applied_force, action.direction[1] * applied_force, 0])
        
        time_step = self.scene.get_timestep()
        max_steps = int(action.duration / time_step)
        
        for _ in range(max_steps):
            # 在质心持续施加推力
            self.robot_actor.add_force_at_point(force_vec, self.robot_actor.get_pose().p)
            # 锁定 Z 轴旋转和倾角，确保是个稳固的底盘
            v = self.robot_actor.get_velocity()
            self.robot_actor.set_velocity([v[0], v[1], 0]) 
            self.robot_actor.set_angular_velocity([0, 0, 0])
            
            self.scene.step()

        end_pose = self.robot_actor.get_pose().p
        distance = float(np.linalg.norm(end_pose[:2] - start_pose[:2]))
        
        # 如果施加了力但是没怎么移动，说明撞到重物被卡住了
        success = distance > 0.1 # 移动超过 10cm 算成功推进
        
        return SimulationResult(
            action_success=success,
            final_robot_pose=Pose(position=tuple(end_pose)),
            distance_moved=distance,
            error_reason="被障碍物卡住，推力不足" if not success else ""
        )
