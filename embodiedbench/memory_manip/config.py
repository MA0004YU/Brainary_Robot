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

    # --- Planning_module bridge ---
    # Path to the Planning_module directory; None = auto-detect sibling dir
    planning_module_dir: Optional[str] = None

    # --- EmbodiedLTM remote service ---
    # Set to None to disable all remote LTM calls (safe for offline / unit-test runs).
    embodiedltm_base_url: Optional[str] = "http://127.0.0.1:8000"
    embodiedltm_timeout_sec: float = 8.0

    # --- Episode importance score weights (must sum to 1.0) ---
    score_weight_endogenous_intent: float = 0.20   # A3: current task novelty
    score_weight_endogenous_longterm: float = 0.15  # A2: historically hard tasks
    score_weight_endogenous_value: float = 0.10     # A1: value/identity alignment (stub)
    score_weight_exogenous: float = 0.25            # B:  novelty + violation signals
    score_weight_reward_external: float = 0.20      # C1: task success + surprise
    score_weight_reward_internal: float = 0.10      # C2: internal goal alignment (stub)

    # --- Activation decay (ACT-R Base-Level Learning) ---
    activation_decay_rate: float = 0.5    # d in exp(-d * age); higher → faster forgetting
    activation_access_boost: float = 0.3  # log1p(access_count) coefficient
    activation_prune_threshold: float = 0.05  # episodes below this + abstracted are pruned
    activation_min_retrieval: float = 0.0      # 0.0 = no retrieval filter

    def get_store_dir(self) -> Path:
        if self.store_dir:
            return Path(self.store_dir)
        return Path(__file__).parent / "store"
