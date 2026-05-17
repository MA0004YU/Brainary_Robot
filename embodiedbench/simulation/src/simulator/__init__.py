# simulation/src/simulator/__init__.py

from .scene_builder import SceneBuilder
from .robot_controller import RobotController
from .detectors import (
    PhysicsBoundaryDetectors,
    DestructionFeedback,
    TippingFeedback,
    IntrusionFeedback,
    DeadlockKinematicFeedback
)

__all__ = [
    "SceneBuilder",
    "RobotController",
    "PhysicsBoundaryDetectors",
    # 暴露四类结构体，方便外部进行 isinstance() 判断或类型注解
    "DestructionFeedback",
    "TippingFeedback",
    "IntrusionFeedback",
    "DeadlockKinematicFeedback"
]
