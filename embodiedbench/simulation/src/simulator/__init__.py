# simulation/src/simulator/__init__.py

from .scene_builder import SceneBuilder
from .robot_controller import BlueprintGraphEngine 
from .detectors import (
    PhysicsBoundaryDetectors,
    DestructionFeedback,
    TippingFeedback,
    IntrusionFeedback,
    DeadlockKinematicFeedback
)

__all__ = [
    "SceneBuilder",
    "BlueprintGraphEngine",  # 同步更新白名单
    "PhysicsBoundaryDetectors",
    # 暴露四类结构体，方便外部 API 进行 isinstance() 判断或记录错误日志
    "DestructionFeedback",
    "TippingFeedback",
    "IntrusionFeedback",
    "DeadlockKinematicFeedback"
]
