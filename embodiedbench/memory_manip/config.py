"""Configuration for the manipulation memory stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MemorySystemConfig:
    """Tunable knobs for buffers, consolidation, and optional persistence."""

    # Cap how many discrete items are kept per working-memory lane before FIFO eviction.
    symbolic_queue_max: int = 64
    abstract_queue_max: int = 32
    semantic_fact_max: int = 128
    spatial_3d_pose_max: int = 32
    task_event_max: int = 64

    # Long-term graph limits (nodes / edges) — coarse guards for prototyping.
    conceptual_entity_cap: int = 512
    temporal_event_cap: int = 2048
    spatial_node_cap: int = 1024

    # When True, `consolidate_to_long_term` writes summaries into LTM structures.
    auto_consolidate_on_episode_end: bool = True

    # Optional JSON-serializable metadata for sim2real (robot name, workspace ID, etc.).
    rig_metadata: Dict[str, Any] = field(default_factory=dict)

    checkpoint_path: Optional[str] = None
