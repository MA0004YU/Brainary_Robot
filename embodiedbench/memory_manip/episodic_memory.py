"""
Episodic Memory: persistent record of complete episodes.

Each episode is stored as an EpisodeRecord containing the full step-by-step
trajectory, task metadata, aggregated object observations, and an importance
score computed by EpisodeScorer at write time.

Persistence
-----------
episodes.jsonl    : append-only episode records (one JSON object per line).
episodes_meta.json: mutable index of per-episode activation, access_count,
                    and abstracted flag.  Kept separate so the JSONL can stay
                    append-only while activation evolves over time.

Activation model
----------------
activation = initial_score * exp(-decay_rate * age) + log1p(access_count) * access_boost

  age          : current_episode_num - creation_episode (in episode units)
  initial_score: computed by EpisodeScorer at write time
  access_count : incremented each time the episode is returned by a query

On every semantic generalization pass the processed episodes are marked
abstracted=True.  Episodes with activation < prune_threshold AND
abstracted=True are eligible for deletion (prune()).

Retrieval
---------
StringRetriever uses keyword overlap.  Results are filtered by a minimum
activation threshold and trigger a reactivation boost on the matched records.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

def compute_activation(
    initial_score: float,
    access_count: int,
    creation_episode: int,
    current_episode: int,
    decay_rate: float = 0.5,
    access_boost: float = 0.3,
) -> float:
    """
    Compute current activation level.

    activation = initial_score * exp(-decay_rate * age)
                 + log1p(access_count) * access_boost

    Properties:
      - Decays exponentially with episode age.
      - Each retrieval adds a diminishing log boost.
      - Initial score scales the baseline (higher importance → slower decay).
    """
    age = max(0, current_episode - creation_episode)
    decay = math.exp(-decay_rate * age)
    boost = math.log1p(access_count) * access_boost
    return initial_score * decay + boost


# ---------------------------------------------------------------------------
# StepRecord
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """A single (observation summary, action, outcome) tuple within an episode."""

    step_idx: int
    obs_summary: str
    visible_objects: List[str]
    current_location: str
    action: str
    action_id: Any
    success: Optional[bool] = None
    feedback: str = ""
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "obs_summary": self.obs_summary,
            "visible_objects": self.visible_objects,
            "current_location": self.current_location,
            "action": self.action,
            "action_id": self.action_id,
            "success": self.success,
            "feedback": self.feedback,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepRecord":
        return cls(
            step_idx=d.get("step_idx", 0),
            obs_summary=d.get("obs_summary", ""),
            visible_objects=d.get("visible_objects", []),
            current_location=d.get("current_location", ""),
            action=d.get("action", ""),
            action_id=d.get("action_id", None),
            success=d.get("success", None),
            feedback=d.get("feedback", ""),
            reasoning=d.get("reasoning", ""),
        )


# ---------------------------------------------------------------------------
# EpisodeRecord
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """Complete record of a single episode."""

    episode_id: str
    task_instruction: str
    task_variation: str
    scene_id: str
    success: bool
    total_steps: int
    duration_sec: float
    timestamp: float = field(default_factory=time.time)

    # Importance score computed by EpisodeScorer at write time.
    # Persisted so decay can be computed correctly after process restart.
    initial_score: float = 1.0
    # Episode number (counter) when this record was created.
    creation_episode: int = 0

    # Full step-by-step trajectory
    steps: List[StepRecord] = field(default_factory=list)

    # Aggregated object encounters:
    # {obj_name: {"locations": [str, …], "grasped": bool, "count": int}}
    objects_encountered: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Intent dict from GoalReasoner (empty if not available)
    intent: Dict[str, Any] = field(default_factory=dict)

    extra: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: StepRecord) -> None:
        """Append a step and update the objects_encountered aggregate."""
        self.steps.append(step)
        for obj in step.visible_objects:
            rec = self.objects_encountered.setdefault(
                obj, {"locations": [], "grasped": False, "count": 0}
            )
            rec["count"] += 1
            if step.current_location and step.current_location not in rec["locations"]:
                rec["locations"].append(step.current_location)
            if "pick" in step.action.lower() and obj.lower() in step.action.lower():
                rec["grasped"] = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_instruction": self.task_instruction,
            "task_variation": self.task_variation,
            "scene_id": self.scene_id,
            "success": self.success,
            "total_steps": self.total_steps,
            "duration_sec": self.duration_sec,
            "timestamp": self.timestamp,
            "initial_score": self.initial_score,
            "creation_episode": self.creation_episode,
            "steps": [s.to_dict() for s in self.steps],
            "objects_encountered": self.objects_encountered,
            "intent": self.intent,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeRecord":
        steps = [StepRecord.from_dict(s) for s in d.get("steps", [])]
        record = cls(
            episode_id=d.get("episode_id", ""),
            task_instruction=d.get("task_instruction", ""),
            task_variation=d.get("task_variation", ""),
            scene_id=d.get("scene_id", ""),
            success=d.get("success", False),
            total_steps=d.get("total_steps", 0),
            duration_sec=d.get("duration_sec", 0.0),
            timestamp=d.get("timestamp", time.time()),
            initial_score=d.get("initial_score", 1.0),
            creation_episode=d.get("creation_episode", 0),
            intent=d.get("intent", {}),
            extra=d.get("extra", {}),
        )
        record.steps = steps
        record.objects_encountered = d.get("objects_encountered", {})
        return record


# ---------------------------------------------------------------------------
# StringRetriever
# ---------------------------------------------------------------------------

class StringRetriever:
    """
    Keyword-overlap episode retriever.

    Scores each record by the fraction of query tokens that appear in the
    episode's instruction (weight 0.7) and objects_encountered keys (weight 0.3).
    """

    def score(self, query: str, record: EpisodeRecord) -> float:
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return 0.0
        instr_tokens = set(record.task_instruction.lower().split())
        obj_tokens = set(o.lower() for o in record.objects_encountered)
        instr_overlap = len(query_tokens & instr_tokens) / len(query_tokens)
        obj_overlap = len(query_tokens & obj_tokens) / len(query_tokens)
        return instr_overlap * 0.7 + obj_overlap * 0.3

    def retrieve(
        self,
        query: str,
        records: List[EpisodeRecord],
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> List[EpisodeRecord]:
        scored = [(self.score(query, r), r) for r in records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in scored if s >= min_score][:top_k]


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Persistent episodic memory backed by an append-only JSONL file.

    Activation state (access_count, abstracted flag, current activation) is
    maintained in a separate episodes_meta.json file so the JSONL can remain
    strictly append-only.

    Parameters
    ----------
    store_path       : path to episodes.jsonl
    max_episodes     : maximum episodes kept in memory
    decay_rate       : d in exp(-d * age); higher → faster forgetting
    access_boost     : log1p(access_count) coefficient in activation formula
    prune_threshold  : episodes below this activation AND abstracted are pruned
    min_activation_retrieval : episodes below this activation are excluded from
                               query results (set to 0.0 to disable filtering)
    """

    def __init__(
        self,
        store_path: Path,
        max_episodes: int = 1000,
        decay_rate: float = 0.5,
        access_boost: float = 0.3,
        prune_threshold: float = 0.05,
        min_activation_retrieval: float = 0.0,
    ) -> None:
        self.store_path = store_path
        self.max_episodes = max_episodes
        self.decay_rate = decay_rate
        self.access_boost = access_boost
        self.prune_threshold = prune_threshold
        self.min_activation_retrieval = min_activation_retrieval

        self._meta_path = store_path.parent / "episodes_meta.json"
        self._records: List[EpisodeRecord] = []
        self._retriever = StringRetriever()

        # episode counter: incremented by tick() after each save_episode call
        self._episode_counter: int = 0

        # per-episode mutable state: {episode_id: {"access_count", "abstracted", "activation"}}
        self._meta: Dict[str, Dict[str, Any]] = {}

        self._load()
        self._load_meta()

    # --- persistence: JSONL ------------------------------------------------

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(EpisodeRecord.from_dict(json.loads(line)))
                except Exception:
                    pass
        if len(self._records) > self.max_episodes:
            self._records = self._records[-self.max_episodes:]

    def _append_to_disk(self, record: EpisodeRecord) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    # --- persistence: meta -------------------------------------------------

    def _load_meta(self) -> None:
        if not self._meta_path.exists():
            return
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._meta = data.get("meta", {})
            self._episode_counter = data.get("episode_counter", 0)
        except Exception:
            pass

    def _save_meta(self) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._meta_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"episode_counter": self._episode_counter, "meta": self._meta},
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(str(tmp), str(self._meta_path))

    # --- write -------------------------------------------------------------

    def save_episode(self, record: EpisodeRecord) -> None:
        """Persist an episode: append to JSONL, initialise meta, advance counter."""
        self._records.append(record)
        self._append_to_disk(record)
        if len(self._records) > self.max_episodes:
            self._records = self._records[-self.max_episodes:]

        self._meta[record.episode_id] = {
            "access_count": 0,
            "abstracted": False,
            "activation": record.initial_score,
        }
        self.tick()

    def tick(self) -> None:
        """Advance the internal episode counter and persist meta."""
        self._episode_counter += 1
        self._save_meta()

    # --- activation --------------------------------------------------------

    def _get_activation(self, record: EpisodeRecord) -> float:
        """Return current activation for a record (computed on the fly)."""
        meta = self._meta.get(record.episode_id, {})
        return compute_activation(
            initial_score=record.initial_score,
            access_count=meta.get("access_count", 0),
            creation_episode=record.creation_episode,
            current_episode=self._episode_counter,
            decay_rate=self.decay_rate,
            access_boost=self.access_boost,
        )

    def _reactivate(self, record: EpisodeRecord) -> None:
        """Increment access count and refresh stored activation for a record."""
        meta = self._meta.setdefault(record.episode_id, {"access_count": 0, "abstracted": False})
        meta["access_count"] = meta.get("access_count", 0) + 1
        meta["activation"] = compute_activation(
            initial_score=record.initial_score,
            access_count=meta["access_count"],
            creation_episode=record.creation_episode,
            current_episode=self._episode_counter,
            decay_rate=self.decay_rate,
            access_boost=self.access_boost,
        )
        self._save_meta()

    def decay_activations(self) -> None:
        """
        Recompute activation for every record based on current episode count.
        Call after semantic generalization or periodically to age the memory.
        """
        for record in self._records:
            meta = self._meta.setdefault(record.episode_id, {"access_count": 0, "abstracted": False})
            meta["activation"] = compute_activation(
                initial_score=record.initial_score,
                access_count=meta.get("access_count", 0),
                creation_episode=record.creation_episode,
                current_episode=self._episode_counter,
                decay_rate=self.decay_rate,
                access_boost=self.access_boost,
            )
        self._save_meta()

    # --- forgetting --------------------------------------------------------

    def mark_abstracted(self, episode_ids: List[str]) -> None:
        """
        Mark episodes as abstracted (their key facts are in SemanticMemory).
        Abstracted + low-activation episodes become eligible for pruning.
        """
        for eid in episode_ids:
            self._meta.setdefault(eid, {"access_count": 0, "abstracted": False})
            self._meta[eid]["abstracted"] = True
        self._save_meta()

    def prune(self) -> int:
        """
        Remove episodes that are both abstracted and below prune_threshold.
        Returns the number of episodes deleted.
        """
        to_remove: List[str] = []
        for record in self._records:
            meta = self._meta.get(record.episode_id, {})
            activation = meta.get("activation", record.initial_score)
            abstracted = meta.get("abstracted", False)
            if abstracted and activation < self.prune_threshold:
                to_remove.append(record.episode_id)

        if not to_remove:
            return 0

        remove_set = set(to_remove)
        self._records = [r for r in self._records if r.episode_id not in remove_set]
        for eid in to_remove:
            self._meta.pop(eid, None)

        # Rewrite JSONL without the pruned records
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        os.replace(str(tmp), str(self.store_path))

        self._save_meta()
        return len(to_remove)

    # --- query -------------------------------------------------------------

    def query_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> List[EpisodeRecord]:
        """
        Return up to top_k episodes most similar to the query string.
        Episodes below min_activation_retrieval are excluded.
        Retrieved episodes receive a reactivation boost.
        """
        candidates = self._records
        if self.min_activation_retrieval > 0.0:
            candidates = [
                r for r in self._records
                if self._get_activation(r) >= self.min_activation_retrieval
            ]
        results = self._retriever.retrieve(query, candidates, top_k=top_k, min_score=min_score)
        for r in results:
            self._reactivate(r)
        return results

    def query_by_object(self, obj_name: str) -> List[EpisodeRecord]:
        """Return all episodes where obj_name was encountered."""
        name = obj_name.lower()
        return [r for r in self._records if any(o.lower() == name for o in r.objects_encountered)]

    def query_by_scene(self, scene_id: str) -> List[EpisodeRecord]:
        """Return all episodes recorded in the given scene."""
        return [r for r in self._records if r.scene_id == scene_id]

    def get_object_location_history(self, obj_name: str) -> Dict[str, int]:
        """Return {location: frequency} for obj_name across all stored episodes."""
        freq: Dict[str, int] = {}
        name = obj_name.lower()
        for r in self._records:
            for obj, info in r.objects_encountered.items():
                if obj.lower() == name:
                    for loc in info.get("locations", []):
                        freq[loc] = freq.get(loc, 0) + info.get("count", 1)
        return freq

    def get_success_rate(self, task_instruction: str, top_k: int = 20) -> Optional[float]:
        """Return historical success rate for tasks similar to task_instruction."""
        similar = self.query_similar(task_instruction, top_k=top_k)
        if not similar:
            return None
        return sum(1 for r in similar if r.success) / len(similar)

    def get_activation_stats(self) -> Dict[str, Any]:
        """Return min/max/mean activation across all records (for monitoring)."""
        if not self._records:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
        acts = [self._get_activation(r) for r in self._records]
        return {
            "min": round(min(acts), 4),
            "max": round(max(acts), 4),
            "mean": round(sum(acts) / len(acts), 4),
            "count": len(acts),
        }

    @property
    def total_episodes(self) -> int:
        return len(self._records)

    def summary(self) -> Dict[str, Any]:
        total = self.total_episodes
        abstracted = sum(
            1 for eid, m in self._meta.items() if m.get("abstracted", False)
        )
        return {
            "total_episodes": total,
            "abstracted_episodes": abstracted,
            "episode_counter": self._episode_counter,
            "success_rate": round(
                sum(1 for r in self._records if r.success) / max(total, 1), 3
            ),
            "activation_stats": self.get_activation_stats(),
            "store_path": str(self.store_path),
        }
