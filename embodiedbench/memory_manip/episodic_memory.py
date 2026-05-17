"""
Episodic Memory: persistent record of complete episodes.

Each episode is stored as an EpisodeRecord containing the full step-by-step
trajectory, task metadata, and aggregated object observations.

Persistence  : append-only JSONL file (one JSON object per line).
               New episodes are appended atomically; the in-memory list is
               trimmed to max_episodes (most-recent kept).

Retrieval    : StringRetriever uses keyword overlap between the query string
               and an episode's instruction + objects_encountered.
               The retriever is accessed only through the EpisodicMemory API,
               so it can be swapped for an embedding-based implementation
               without touching callers.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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
            # Mark as grasped if the action string indicates a pick
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

    This class is the default retriever slot; replace it with an embedding-based
    implementation by subclassing and overriding score() + retrieve().
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

    On init the JSONL file is loaded into memory (most-recent max_episodes kept).
    New episodes are appended to disk immediately; in-memory list is trimmed.
    """

    def __init__(self, store_path: Path, max_episodes: int = 1000) -> None:
        self.store_path = store_path
        self.max_episodes = max_episodes
        self._records: List[EpisodeRecord] = []
        self._retriever = StringRetriever()
        self._load()

    # --- persistence -------------------------------------------------------

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
                    pass  # Skip corrupted lines silently
        # Keep only most-recent episodes in memory
        if len(self._records) > self.max_episodes:
            self._records = self._records[-self.max_episodes:]

    def _append_to_disk(self, record: EpisodeRecord) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    # --- write -------------------------------------------------------------

    def save_episode(self, record: EpisodeRecord) -> None:
        """Persist an episode: append to JSONL and update in-memory list."""
        self._records.append(record)
        self._append_to_disk(record)
        if len(self._records) > self.max_episodes:
            self._records = self._records[-self.max_episodes:]

    # --- query -------------------------------------------------------------

    def query_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> List[EpisodeRecord]:
        """Return up to top_k episodes most similar to the query string."""
        return self._retriever.retrieve(query, self._records, top_k=top_k, min_score=min_score)

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

    @property
    def total_episodes(self) -> int:
        return len(self._records)

    def summary(self) -> Dict[str, Any]:
        total = self.total_episodes
        return {
            "total_episodes": total,
            "success_rate": (
                round(sum(1 for r in self._records if r.success) / max(total, 1), 3)
            ),
            "store_path": str(self.store_path),
        }
