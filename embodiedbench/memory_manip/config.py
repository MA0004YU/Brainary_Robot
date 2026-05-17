"""Configuration for the three-layer manipulation memory stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class MemorySystemConfig:
    # --- Working Memory ---
    action_buffer_max: int = 32
    observation_history_max: int = 16
    episode_max_steps: int = 50

    # --- Episodic Memory ---
    episodic_max_episodes: int = 1000
    # Batch generalization: push episodic -> semantic every N completed episodes
    episodic_generalize_every_n: int = 5

    # --- Semantic Memory ---
    semantic_object_cap: int = 512
    semantic_location_cap: int = 256

    # --- Persistence ---
    # None = default to <package_dir>/store/
    store_dir: Optional[str] = None

    # --- Consolidation ---
    auto_consolidate_on_episode_end: bool = True

    # --- Real-robot metadata ---
    rig_metadata: Dict[str, Any] = field(default_factory=dict)
    robot_dof: int = 6

    # --- Planning_agent bridge ---
    # Path to the Planning_agent directory; None = auto-detect sibling dir
    planning_agent_dir: Optional[str] = None

    # --- EmbodiedLTM remote service ---
    embodiedltm_base_url: str = "http://127.0.0.1:8000"
    embodiedltm_timeout_sec: float = 8.0

    def get_store_dir(self) -> Path:
        if self.store_dir:
            return Path(self.store_dir)
        return Path(__file__).parent / "store"
