"""
EpisodeScorer: computes initial importance score for a completed episode.

Score components (all in [0, 1]):
  A3  endogenous_intent    – current task type novelty (top-down, immediate)
  A2  endogenous_longterm  – targets historically hard task types (strategic)
  A1  endogenous_value     – value / identity alignment [stub → 0.5]
  B   exogenous            – novelty + violation signals (bottom-up)
  C1  reward_external      – task success + surprise value
  C2  reward_internal      – internal goal-alignment estimate [stub → 0.5]

Subclass EpisodeScorer and override individual score_* methods to specialise
for a new robot type (manipulation arm, humanoid, navigation agent, …).
ManipulationScorer provides arm-specific physics-violation weighting.

Usage:
    from embodiedbench.memory_manip.episode_scorer import ManipulationScorer
    scorer = ManipulationScorer(config)
    breakdown = scorer.compute(record, semantic=semantic_memory)
    record.initial_score = breakdown.total
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from embodiedbench.memory_manip.config import MemorySystemConfig
    from embodiedbench.memory_manip.episodic_memory import EpisodeRecord
    from embodiedbench.memory_manip.semantic_memory import SemanticMemory


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------

@dataclass
class EpisodeScoreBreakdown:
    """Per-component score breakdown for a single episode (all values in [0, 1])."""

    endogenous_intent: float = 0.0    # A3
    endogenous_longterm: float = 0.0  # A2
    endogenous_value: float = 0.5     # A1 (stub default = neutral)
    exogenous: float = 0.0            # B
    reward_external: float = 0.0      # C1
    reward_internal: float = 0.5      # C2 (stub default = neutral)

    # Weights — populated from MemorySystemConfig at compute time
    _w_intent: float = field(default=0.20, repr=False)
    _w_longterm: float = field(default=0.15, repr=False)
    _w_value: float = field(default=0.10, repr=False)
    _w_exogenous: float = field(default=0.25, repr=False)
    _w_reward_ext: float = field(default=0.20, repr=False)
    _w_reward_int: float = field(default=0.10, repr=False)

    @property
    def total(self) -> float:
        return (
            self.endogenous_intent   * self._w_intent
            + self.endogenous_longterm * self._w_longterm
            + self.endogenous_value    * self._w_value
            + self.exogenous           * self._w_exogenous
            + self.reward_external     * self._w_reward_ext
            + self.reward_internal     * self._w_reward_int
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "endogenous_intent":   round(self.endogenous_intent, 4),
            "endogenous_longterm": round(self.endogenous_longterm, 4),
            "endogenous_value":    round(self.endogenous_value, 4),
            "exogenous":           round(self.exogenous, 4),
            "reward_external":     round(self.reward_external, 4),
            "reward_internal":     round(self.reward_internal, 4),
            "total":               round(self.total, 4),
        }


# ---------------------------------------------------------------------------
# Base scorer
# ---------------------------------------------------------------------------

class EpisodeScorer:
    """
    Base class for episode importance scoring.

    All score_* methods return a float in [0, 1].  Override any subset to
    specialise for a new robot type without touching the rest of the system.
    """

    def __init__(self, config: "MemorySystemConfig") -> None:
        self.config = config

    # --- A3: endogenous / current intent ------------------------------------

    def score_endogenous_intent(
        self,
        record: "EpisodeRecord",
        semantic: Optional["SemanticMemory"] = None,
    ) -> float:
        """
        Higher score when this task type has little prior data.
        Data-sparse tasks are more informative to retain.
        """
        if semantic is None:
            return 0.5
        task_type = semantic.task_schema.infer_type(record.task_instruction)
        task_data = semantic.task_schema._data.get(task_type, {})
        task_total = task_data.get("success_count", 0) + task_data.get("fail_count", 0)
        all_total = sum(
            v.get("success_count", 0) + v.get("fail_count", 0)
            for v in semantic.task_schema._data.values()
        )
        if all_total == 0:
            return 1.0
        density = task_total / all_total
        return max(0.0, 1.0 - density)

    # --- A2: endogenous / long-term goal ------------------------------------

    def score_endogenous_longterm(
        self,
        record: "EpisodeRecord",
        semantic: Optional["SemanticMemory"] = None,
    ) -> float:
        """
        Higher score when the task type has a low historical success rate.
        Hard tasks are strategically valuable to retain for future planning.
        """
        if semantic is None:
            return 0.5
        rate = semantic.task_schema.get_success_rate(record.task_instruction)
        if rate is None:
            return 0.8  # No prior data → treat as potentially hard
        return 1.0 - rate

    # --- A1: endogenous / value & identity (stub) ---------------------------

    def score_endogenous_value(
        self,
        record: "EpisodeRecord",
        **kwargs: Any,
    ) -> float:
        """
        Stub — returns 0.5 (neutral).
        Override to implement value / identity alignment scoring.
        Example: a humanoid scorer can check whether the episode matches
        explicit user-preference records in semantic memory.
        """
        return 0.5

    # --- B: exogenous / bottom-up surprise ----------------------------------

    def score_exogenous(
        self,
        record: "EpisodeRecord",
        semantic: Optional["SemanticMemory"] = None,
    ) -> float:
        """
        Combines novelty (unseen objects) and violation / failure signals.
        Override _novelty_score or _violation_score for robot-specific sensors.
        """
        novelty = self._novelty_score(record, semantic)
        violation = self._violation_score(record)
        return 0.5 * novelty + 0.5 * violation

    def _novelty_score(
        self,
        record: "EpisodeRecord",
        semantic: Optional["SemanticMemory"],
    ) -> float:
        """Fraction of encountered objects unseen in ObjectKB."""
        objs = list(record.objects_encountered.keys())
        if not objs:
            return 0.0
        if semantic is None:
            return 0.5
        new_count = sum(
            1 for o in objs
            if semantic.object_kb._data.get(o, {}).get("total_count", 0) == 0
        )
        return new_count / len(objs)

    def _violation_score(self, record: "EpisodeRecord") -> float:
        """Fraction of steps that contain failure or violation feedback."""
        if not record.steps:
            return 0.0
        _VIOLATION = {
            "MECHANICAL_DESTRUCTION", "TOPOLOGICAL_TIPPING",
            "UNEXPECTED_COLLISION", "KINEMATIC_DYNAMIC_DEADLOCK",
            "FAILED", "ERROR", "INVALID",
        }
        failed = sum(
            1 for s in record.steps
            if s.success is False
            or any(kw in (s.feedback or "").upper() for kw in _VIOLATION)
        )
        return failed / len(record.steps)

    # --- C1: external reward ------------------------------------------------

    def score_reward_external(
        self,
        record: "EpisodeRecord",
        semantic: Optional["SemanticMemory"] = None,
    ) -> float:
        """
        Combines absolute success value with outcome surprise.
        Unexpected success or unexpected failure carries more information.
        """
        success_val = 1.0 if record.success else 0.0
        if semantic is None:
            return success_val
        expected = semantic.task_schema.get_success_rate(record.task_instruction)
        if expected is None:
            return success_val
        surprise = abs(success_val - expected)
        return 0.5 * success_val + 0.5 * surprise

    # --- C2: internal reward / value estimate (stub) ------------------------

    def score_reward_internal(
        self,
        record: "EpisodeRecord",
        **kwargs: Any,
    ) -> float:
        """
        Stub — returns 0.5 (neutral).
        Override to evaluate whether the episode advances long-term goals,
        reduces world-model uncertainty, or improves self-consistency.
        """
        return 0.5

    # --- Combined -----------------------------------------------------------

    def compute(
        self,
        record: "EpisodeRecord",
        semantic: Optional["SemanticMemory"] = None,
    ) -> EpisodeScoreBreakdown:
        """Compute and return the full score breakdown for one episode."""
        cfg = self.config
        return EpisodeScoreBreakdown(
            endogenous_intent=self.score_endogenous_intent(record, semantic),
            endogenous_longterm=self.score_endogenous_longterm(record, semantic),
            endogenous_value=self.score_endogenous_value(record),
            exogenous=self.score_exogenous(record, semantic),
            reward_external=self.score_reward_external(record, semantic),
            reward_internal=self.score_reward_internal(record),
            _w_intent=cfg.score_weight_endogenous_intent,
            _w_longterm=cfg.score_weight_endogenous_longterm,
            _w_value=cfg.score_weight_endogenous_value,
            _w_exogenous=cfg.score_weight_exogenous,
            _w_reward_ext=cfg.score_weight_reward_external,
            _w_reward_int=cfg.score_weight_reward_internal,
        )


# ---------------------------------------------------------------------------
# ManipulationScorer  (arm-specific)
# ---------------------------------------------------------------------------

class ManipulationScorer(EpisodeScorer):
    """
    Manipulation-arm specialisation of EpisodeScorer.

    Physics violations (MECHANICAL_DESTRUCTION, TOPOLOGICAL_TIPPING, etc.)
    are weighted 2× generic failures because they carry precise causal
    information about the physical boundary of safe operation.
    """

    def _violation_score(self, record: "EpisodeRecord") -> float:
        if not record.steps:
            return 0.0
        _PHYSICS = {
            "MECHANICAL_DESTRUCTION", "TOPOLOGICAL_TIPPING",
            "UNEXPECTED_COLLISION", "KINEMATIC_DYNAMIC_DEADLOCK",
        }
        _GENERIC = {"FAILED", "ERROR", "INVALID"}
        physics_hits = sum(
            1 for s in record.steps
            if any(kw in (s.feedback or "").upper() for kw in _PHYSICS)
        )
        generic_hits = sum(
            1 for s in record.steps
            if (s.success is False)
            or any(kw in (s.feedback or "").upper() for kw in _GENERIC)
        )
        weighted = physics_hits * 2 + generic_hits
        max_possible = len(record.steps) * 2
        return min(1.0, weighted / max_possible) if max_possible > 0 else 0.0
