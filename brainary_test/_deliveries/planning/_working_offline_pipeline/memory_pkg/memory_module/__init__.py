"""
memory_module: perception → memory → planning bridge for Brainary_Robot.

This package connects QwenVLMPerception (perception_vlm.py) to the three-layer
memory system (embodiedbench/memory_manip/) and exposes a clean interface for
the planning module.

Public API:
    PerceptionMemoryPipeline  — main orchestrator: perception → memory → planning
    VisionPerceptionAdapter   — adapts QwenVLMPerception to PerceptionInterface
    PlanningMemoryInterface   — read-only query interface for the planning module
    PlanningContext           — structured context bundle consumed by planning
    SceneStateBuilder         — converts perception outputs to spine brain format
"""
from memory_module.pipeline import PerceptionMemoryPipeline
from memory_module.perception_adapter import VisionPerceptionAdapter, encode_recognition_results
from memory_module.planning_interface import PlanningMemoryInterface, PlanningContext
from memory_module.scene_state_builder import build_stub_scene_state, merge_with_sensor_data

__all__ = [
    "PerceptionMemoryPipeline",
    "VisionPerceptionAdapter",
    "encode_recognition_results",
    "PlanningMemoryInterface",
    "PlanningContext",
    "build_stub_scene_state",
    "merge_with_sensor_data",
]
