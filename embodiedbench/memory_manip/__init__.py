"""
Three-layer memory system for embodied manipulation.

Layers:
  WorkingMemory   - volatile per-episode state (goal, observation, actions, robot state)
  EpisodicMemory  - persistent episode records with JSONL storage and retrieval
  SemanticMemory  - persistent world knowledge (objects, preferences, task schemata, topology)

Entry point: EmbodiedManipulationMemorySystem (agent_memory.py)
Configuration: MemorySystemConfig (config.py)
I/O interfaces: PerceptionInterface, SimulationInterface, MonitorInterface (interfaces.py)
Planning_module bridge: PlanningModuleBridge (bridges.py)
  - PlanningAgentBridge is a backward-compatible alias for PlanningModuleBridge
"""
from embodiedbench.memory_manip.config import MemorySystemConfig
from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
from embodiedbench.memory_manip.interfaces import (
    PerceptionInterface,
    SimulationInterface,
    MonitorInterface,
)
from embodiedbench.memory_manip.working_memory import WorkingMemory
from embodiedbench.memory_manip.episodic_memory import EpisodicMemory, EpisodeRecord, StepRecord
from embodiedbench.memory_manip.semantic_memory import SemanticMemory
from embodiedbench.memory_manip.bridges import PlanningModuleBridge, PlanningAgentBridge

__all__ = [
    "MemorySystemConfig",
    "EmbodiedManipulationMemorySystem",
    "PerceptionInterface",
    "SimulationInterface",
    "MonitorInterface",
    "WorkingMemory",
    "EpisodicMemory",
    "EpisodeRecord",
    "StepRecord",
    "SemanticMemory",
    "PlanningModuleBridge",
    "PlanningAgentBridge",  # backward-compatible alias
]
