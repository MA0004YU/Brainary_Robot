"""
Tests for the episodic memory scoring and activation decay system.

Covers:
  - EpisodeScoreBreakdown.total and to_dict()
  - EpisodeScorer (base class, all score_* methods)
  - ManipulationScorer._violation_score (physics weighting)
  - compute_activation (decay + boost)
  - EpisodicMemory: save, query_similar, reactivation, mark_abstracted, prune
  - MemorySystemConfig: new weight and activation fields with defaults
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from embodiedbench.memory_manip.config import MemorySystemConfig
from embodiedbench.memory_manip.episode_scorer import (
    EpisodeScoreBreakdown,
    EpisodeScorer,
    ManipulationScorer,
)
from embodiedbench.memory_manip.episodic_memory import (
    EpisodeRecord,
    EpisodicMemory,
    StepRecord,
    compute_activation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(
    episode_id: str = "ep1",
    task_instruction: str = "pick up the red cube",
    success: bool = True,
    steps: list[StepRecord] | None = None,
    initial_score: float = 1.0,
    creation_episode: int = 0,
) -> EpisodeRecord:
    r = EpisodeRecord(
        episode_id=episode_id,
        task_instruction=task_instruction,
        task_variation="v0",
        scene_id="scene_01",
        success=success,
        total_steps=len(steps or []),
        duration_sec=5.0,
        initial_score=initial_score,
        creation_episode=creation_episode,
    )
    for s in steps or []:
        r.steps.append(s)
    return r


def make_step(success: bool = True, feedback: str = "") -> StepRecord:
    return StepRecord(
        step_idx=0,
        obs_summary="",
        visible_objects=[],
        current_location="table",
        action="pick",
        action_id=0,
        success=success,
        feedback=feedback,
    )


def default_config() -> MemorySystemConfig:
    return MemorySystemConfig(embodiedltm_base_url=None)


# ---------------------------------------------------------------------------
# EpisodeScoreBreakdown
# ---------------------------------------------------------------------------

class TestEpisodeScoreBreakdown:
    def test_total_weighted_sum(self):
        b = EpisodeScoreBreakdown(
            endogenous_intent=1.0,
            endogenous_longterm=1.0,
            endogenous_value=1.0,
            exogenous=1.0,
            reward_external=1.0,
            reward_internal=1.0,
        )
        assert abs(b.total - 1.0) < 1e-9

    def test_total_zero_when_all_zero(self):
        b = EpisodeScoreBreakdown(
            endogenous_intent=0.0,
            endogenous_longterm=0.0,
            endogenous_value=0.0,
            exogenous=0.0,
            reward_external=0.0,
            reward_internal=0.0,
        )
        assert b.total == 0.0

    def test_to_dict_has_all_keys(self):
        b = EpisodeScoreBreakdown()
        d = b.to_dict()
        for key in ("endogenous_intent", "endogenous_longterm", "endogenous_value",
                    "exogenous", "reward_external", "reward_internal", "total"):
            assert key in d

    def test_custom_weights_applied(self):
        b = EpisodeScoreBreakdown(
            reward_external=1.0,
            _w_intent=0.0, _w_longterm=0.0, _w_value=0.0,
            _w_exogenous=0.0, _w_reward_ext=0.5, _w_reward_int=0.0,
        )
        assert abs(b.total - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# EpisodeScorer (base)
# ---------------------------------------------------------------------------

class TestEpisodeScorerBase:
    def test_returns_neutral_without_semantic(self):
        cfg = default_config()
        scorer = EpisodeScorer(cfg)
        record = make_record()
        assert scorer.score_endogenous_intent(record) == 0.5
        assert scorer.score_endogenous_longterm(record) == 0.5
        assert scorer.score_endogenous_value(record) == 0.5
        assert scorer.score_reward_internal(record) == 0.5

    def test_exogenous_no_steps(self):
        cfg = default_config()
        scorer = EpisodeScorer(cfg)
        record = make_record(steps=[])
        assert scorer.score_exogenous(record) == 0.0

    def test_exogenous_all_violations(self):
        cfg = default_config()
        scorer = EpisodeScorer(cfg)
        steps = [make_step(success=False, feedback="FAILED") for _ in range(4)]
        record = make_record(steps=steps)
        score = scorer.score_exogenous(record)
        assert score > 0.0

    def test_reward_external_success_no_semantic(self):
        cfg = default_config()
        scorer = EpisodeScorer(cfg)
        record = make_record(success=True)
        assert scorer.score_reward_external(record) == 1.0

    def test_reward_external_failure_no_semantic(self):
        cfg = default_config()
        scorer = EpisodeScorer(cfg)
        record = make_record(success=False)
        assert scorer.score_reward_external(record) == 0.0

    def test_compute_returns_breakdown(self):
        cfg = default_config()
        scorer = EpisodeScorer(cfg)
        record = make_record(success=True)
        b = scorer.compute(record)
        assert isinstance(b, EpisodeScoreBreakdown)
        assert 0.0 <= b.total <= 1.0

    def test_compute_uses_config_weights(self):
        cfg = default_config()
        cfg.score_weight_endogenous_intent = 0.0
        cfg.score_weight_endogenous_longterm = 0.0
        cfg.score_weight_endogenous_value = 0.0
        cfg.score_weight_exogenous = 0.0
        cfg.score_weight_reward_external = 1.0
        cfg.score_weight_reward_internal = 0.0
        scorer = EpisodeScorer(cfg)
        record = make_record(success=True)
        b = scorer.compute(record)
        assert abs(b.total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# ManipulationScorer
# ---------------------------------------------------------------------------

class TestManipulationScorer:
    def test_physics_violation_weighted_higher(self):
        cfg = default_config()
        scorer = ManipulationScorer(cfg)

        physics_steps = [make_step(feedback="MECHANICAL_DESTRUCTION") for _ in range(2)]
        generic_steps = [make_step(success=False) for _ in range(2)]

        physics_record = make_record(steps=physics_steps)
        generic_record = make_record(steps=generic_steps)

        physics_score = scorer._violation_score(physics_record)
        generic_score = scorer._violation_score(generic_record)
        assert physics_score >= generic_score

    def test_no_steps_returns_zero(self):
        cfg = default_config()
        scorer = ManipulationScorer(cfg)
        record = make_record(steps=[])
        assert scorer._violation_score(record) == 0.0

    def test_violation_score_capped_at_one(self):
        cfg = default_config()
        scorer = ManipulationScorer(cfg)
        steps = [
            make_step(feedback="MECHANICAL_DESTRUCTION")
            for _ in range(10)
        ]
        record = make_record(steps=steps)
        assert scorer._violation_score(record) <= 1.0

    def test_combined_physics_and_generic(self):
        cfg = default_config()
        scorer = ManipulationScorer(cfg)
        steps = [
            make_step(feedback="TOPOLOGICAL_TIPPING"),
            make_step(success=False),
            make_step(success=True),
        ]
        record = make_record(steps=steps)
        score = scorer._violation_score(record)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# compute_activation
# ---------------------------------------------------------------------------

class TestComputeActivation:
    def test_no_decay_at_zero_age(self):
        act = compute_activation(
            initial_score=1.0, access_count=0,
            creation_episode=0, current_episode=0,
            decay_rate=0.5, access_boost=0.3,
        )
        assert abs(act - 1.0) < 1e-9

    def test_decays_with_age(self):
        act0 = compute_activation(1.0, 0, 0, 0)
        act10 = compute_activation(1.0, 0, 0, 10)
        assert act10 < act0

    def test_boost_from_access(self):
        act_no_access = compute_activation(1.0, 0, 0, 5)
        act_with_access = compute_activation(1.0, 10, 0, 5)
        assert act_with_access > act_no_access

    def test_boost_is_logarithmic(self):
        # Equal intervals (0→10 vs 10→20) should show diminishing returns.
        a0  = compute_activation(0.0, 0,  0, 0, access_boost=0.3)
        a10 = compute_activation(0.0, 10, 0, 0, access_boost=0.3)
        a20 = compute_activation(0.0, 20, 0, 0, access_boost=0.3)
        assert a10 - a0 > a20 - a10

    def test_older_creation_episode_than_current(self):
        # creation_episode > current_episode should treat age as 0
        act = compute_activation(1.0, 0, creation_episode=10, current_episode=5)
        assert abs(act - 1.0) < 1e-9

    def test_higher_decay_rate_decays_faster(self):
        slow = compute_activation(1.0, 0, 0, 5, decay_rate=0.1)
        fast = compute_activation(1.0, 0, 0, 5, decay_rate=1.0)
        assert fast < slow


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def _make_mem(self, tmp_path: Path, **kwargs) -> EpisodicMemory:
        return EpisodicMemory(
            store_path=tmp_path / "episodes.jsonl",
            max_episodes=100,
            **kwargs,
        )

    def test_save_and_total_episodes(self, tmp_path):
        mem = self._make_mem(tmp_path)
        r = make_record()
        mem.save_episode(r)
        assert mem.total_episodes == 1

    def test_episode_counter_advances(self, tmp_path):
        mem = self._make_mem(tmp_path)
        assert mem._episode_counter == 0
        mem.save_episode(make_record("e1"))
        assert mem._episode_counter == 1
        mem.save_episode(make_record("e2"))
        assert mem._episode_counter == 2

    def test_meta_initialized_on_save(self, tmp_path):
        mem = self._make_mem(tmp_path)
        r = make_record("ep42")
        mem.save_episode(r)
        assert "ep42" in mem._meta
        assert mem._meta["ep42"]["access_count"] == 0
        assert mem._meta["ep42"]["abstracted"] is False

    def test_query_similar_returns_results(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.save_episode(make_record("e1", task_instruction="pick up the red cube"))
        mem.save_episode(make_record("e2", task_instruction="place the blue sphere"))
        results = mem.query_similar("red cube", top_k=3)
        assert len(results) >= 1
        assert results[0].episode_id == "e1"

    def test_query_reactivates_hit(self, tmp_path):
        mem = self._make_mem(tmp_path)
        r = make_record("e1", task_instruction="pick up the red cube")
        mem.save_episode(r)
        mem.query_similar("red cube", top_k=1)
        assert mem._meta["e1"]["access_count"] == 1

    def test_activation_filter_excludes_low(self, tmp_path):
        mem = self._make_mem(tmp_path, min_activation_retrieval=0.9, decay_rate=5.0)
        # Save with old creation_episode so activation will be very low
        r = make_record("old_ep", initial_score=0.1, creation_episode=0)
        mem._records.append(r)
        mem._meta["old_ep"] = {"access_count": 0, "abstracted": False, "activation": 0.1}
        mem._episode_counter = 20
        results = mem.query_similar("pick up red cube")
        assert all(ri.episode_id != "old_ep" for ri in results)

    def test_decay_activations_updates_meta(self, tmp_path):
        mem = self._make_mem(tmp_path, decay_rate=0.5)
        r = make_record("e1", initial_score=1.0, creation_episode=0)
        mem.save_episode(r)
        mem._episode_counter = 5
        mem.decay_activations()
        expected = compute_activation(1.0, 0, 0, 5, decay_rate=0.5, access_boost=0.3)
        assert abs(mem._meta["e1"]["activation"] - expected) < 1e-6

    def test_mark_abstracted(self, tmp_path):
        mem = self._make_mem(tmp_path)
        r = make_record("e1")
        mem.save_episode(r)
        mem.mark_abstracted(["e1"])
        assert mem._meta["e1"]["abstracted"] is True

    def test_prune_removes_eligible_episodes(self, tmp_path):
        mem = self._make_mem(tmp_path, prune_threshold=0.5, decay_rate=10.0)
        r = make_record("e_old", initial_score=0.01, creation_episode=0)
        mem.save_episode(r)
        mem._meta["e_old"]["abstracted"] = True
        mem._meta["e_old"]["activation"] = 0.01  # below threshold
        pruned = mem.prune()
        assert pruned == 1
        assert mem.total_episodes == 0

    def test_prune_keeps_non_abstracted(self, tmp_path):
        mem = self._make_mem(tmp_path, prune_threshold=0.9)
        r = make_record("e_keep", initial_score=0.1)
        mem.save_episode(r)
        mem._meta["e_keep"]["activation"] = 0.01  # low but NOT abstracted
        pruned = mem.prune()
        assert pruned == 0
        assert mem.total_episodes == 1

    def test_prune_rewrites_jsonl(self, tmp_path):
        mem = self._make_mem(tmp_path, prune_threshold=0.9)
        for i in range(3):
            mem.save_episode(make_record(f"e{i}", initial_score=0.01))
        # mark e0 and e1 abstracted + low activation; keep e2
        for eid in ("e0", "e1"):
            mem._meta[eid]["abstracted"] = True
            mem._meta[eid]["activation"] = 0.001
        pruned = mem.prune()
        assert pruned == 2
        # Reload from disk
        mem2 = self._make_mem(tmp_path, prune_threshold=0.9)
        assert mem2.total_episodes == 1
        assert mem2._records[0].episode_id == "e2"

    def test_meta_persisted_across_reload(self, tmp_path):
        mem = self._make_mem(tmp_path)
        r = make_record("e1")
        mem.save_episode(r)
        mem.mark_abstracted(["e1"])
        # Reload
        mem2 = self._make_mem(tmp_path)
        assert mem2._meta["e1"]["abstracted"] is True
        assert mem2._episode_counter == 1

    def test_get_activation_stats(self, tmp_path):
        mem = self._make_mem(tmp_path)
        for i in range(3):
            mem.save_episode(make_record(f"e{i}", initial_score=float(i + 1) * 0.3))
        stats = mem.get_activation_stats()
        assert stats["count"] == 3
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_summary_contains_expected_keys(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.save_episode(make_record())
        s = mem.summary()
        for key in ("total_episodes", "abstracted_episodes", "episode_counter",
                    "success_rate", "activation_stats", "store_path"):
            assert key in s


# ---------------------------------------------------------------------------
# MemorySystemConfig defaults
# ---------------------------------------------------------------------------

class TestMemorySystemConfigDefaults:
    def test_score_weight_defaults_sum_to_one(self):
        cfg = MemorySystemConfig()
        total = (
            cfg.score_weight_endogenous_intent
            + cfg.score_weight_endogenous_longterm
            + cfg.score_weight_endogenous_value
            + cfg.score_weight_exogenous
            + cfg.score_weight_reward_external
            + cfg.score_weight_reward_internal
        )
        assert abs(total - 1.0) < 1e-9

    def test_activation_defaults_present(self):
        cfg = MemorySystemConfig()
        assert cfg.activation_decay_rate > 0
        assert cfg.activation_access_boost > 0
        assert cfg.activation_prune_threshold > 0
        assert cfg.activation_min_retrieval == 0.0
