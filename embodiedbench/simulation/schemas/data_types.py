from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class PhysicalProperty:
    mass: float = 1.0
    friction_static: float = 0.5
    friction_dynamic: float = 0.5
    restitution: float = 0.0
    is_passable: bool = False  

@dataclass
class Pose:
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

@dataclass
class SceneObject:
    object_id: str
    semantic_label: str
    dimensions: Tuple[float, float, float]
    current_pose: Pose
    physics: Optional[PhysicalProperty] = None

@dataclass
class RobotInfo:
    """机器人自身的物理参数"""
    radius: float = 0.25      # 圆柱体底盘半径
    height: float = 0.5       # 圆柱体高度
    max_push_force: float = 100.0  # 最大推力 (牛顿)

@dataclass
class NavigationAction:
    """导航指令：不再是绝对坐标，而是方向和推力"""
    action_type: str          # "move_forward", "push"
    direction: Tuple[float, float] # 2D 移动向量 (dx, dy)，需归一化
    duration: float = 2.0     # 持续施加力/移动的时间 (秒)
    apply_force: float = 50.0 # 预期施加的力大小

@dataclass
class SimulationResult:
    action_success: bool
    final_robot_pose: Pose
    distance_moved: float     # 实际移动的距离，用来判断是否被卡住
    error_reason: str = ""
