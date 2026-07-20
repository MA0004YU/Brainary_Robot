# simulation/src/simulator/__init__.py

from .scene_builder import SceneBuilder
from .robot_controller import DAGSimulationEngine
from .detectors import (
    PhysicsBoundaryDetectors,
    DestructionFeedback,
    TippingFeedback,
    IntrusionFeedback,
    DeadlockKinematicFeedback,
    SlipFeedback  # 🚀 新增：导入物理滑脱探针反馈
)

__all__ = [
    "SceneBuilder",
    "DAGSimulationEngine",  # 同步更新白名单：替换为全新的 DAG 拓扑执行引擎
    "PhysicsBoundaryDetectors",
    # 暴露五类结构体，方便外部 API 进行 isinstance() 判断或记录错误日志
    "DestructionFeedback",
    "TippingFeedback",
    "IntrusionFeedback",
    "DeadlockKinematicFeedback",
    "SlipFeedback"          # 🚀 新增：暴露滑脱探针
]
