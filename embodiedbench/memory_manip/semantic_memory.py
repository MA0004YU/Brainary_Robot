"""
Semantic Memory: persistent world knowledge accumulated across episodes.

Four knowledge stores:
  ObjectKB      - per-object affordances and historically observed locations
  UserPreference - user needs mapped to preferred / avoided objects
  TaskSchema    - per-task-type step templates and success statistics
  SpatialTopo   - location graph with neighbour links and object frequency

Persistence: single JSON file.
Writes are atomic (write to <file>.tmp then os.replace) to avoid corruption
on unexpected power loss during real-robot operation.

Generalization is triggered by EmbodiedManipulationMemorySystem every N
completed episodes (see MemorySystemConfig.episodic_generalize_every_n).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ObjectKB
# ---------------------------------------------------------------------------

class ObjectKB:
    """Per-object knowledge accumulated from episodic observations."""

    def __init__(self, data: Dict[str, Any] = None) -> None:
        # {obj_name: {"affordances": [str], "location_freq": {loc: count},
        #             "grasp_count": int, "total_count": int}}
        self._data: Dict[str, Any] = data or {}

    def update_from_episode(
        self, obj: str, locations: List[str], grasped: bool, count: int
    ) -> None:
        rec = self._data.setdefault(
            obj, {"affordances": [], "location_freq": {}, "grasp_count": 0, "total_count": 0}
        )
        for loc in locations:
            rec["location_freq"][loc] = rec["location_freq"].get(loc, 0) + count
        if grasped:
            rec["grasp_count"] += 1
            if "pick" not in rec["affordances"]:
                rec["affordances"].append("pick")
        rec["total_count"] += count

    def get_likely_locations(self, obj: str, top_k: int = 3) -> List[Tuple[str, int]]:
        """Return [(location, frequency), …] sorted descending by frequency."""
        freq = self._data.get(obj, {}).get("location_freq", {})
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_affordances(self, obj: str) -> List[str]:
        return self._data.get(obj, {}).get("affordances", [])

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# UserPreference
# ---------------------------------------------------------------------------

class UserPreference:
    """
    Maps user functional needs to preferred and avoided objects.

    Two provenance levels are tracked:
      "explicit"  - stated directly by the user (via Planning_agent bridge)
      "episodic"  - inferred from successful episode outcomes
    Explicit entries get a higher default weight so they override inferred ones.
    """

    def __init__(self, data: Dict[str, Any] = None) -> None:
        # {need: {"preferred": [{"obj": str, "source": str, "weight": float}],
        #         "avoided":   [{"obj": str, "source": str}]}}
        self._data: Dict[str, Any] = data or {}

    def add_preference(
        self, need: str, obj: str, source: str = "episodic", weight: float = 1.0
    ) -> None:
        rec = self._data.setdefault(need, {"preferred": [], "avoided": []})
        existing = next((p for p in rec["preferred"] if p["obj"] == obj), None)
        if existing:
            existing["weight"] = min(existing["weight"] + 0.1, 5.0)
        else:
            rec["preferred"].append({"obj": obj, "source": source, "weight": weight})

    def add_avoidance(self, need: str, obj: str, source: str = "explicit") -> None:
        rec = self._data.setdefault(need, {"preferred": [], "avoided": []})
        if not any(a["obj"] == obj for a in rec["avoided"]):
            rec["avoided"].append({"obj": obj, "source": source})

    def query(self, need: str) -> Dict[str, Any]:
        """Return preference record sorted by weight descending."""
        rec = self._data.get(need, {"preferred": [], "avoided": []})
        preferred_sorted = sorted(rec["preferred"], key=lambda x: x["weight"], reverse=True)
        return {"preferred": preferred_sorted, "avoided": rec["avoided"]}

    def get_preferred_objects(self, need: str, top_k: int = 3) -> List[str]:
        return [p["obj"] for p in self.query(need)["preferred"][:top_k]]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# TaskSchema
# ---------------------------------------------------------------------------

class TaskSchema:
    """
    Per-task-type statistics and successful step templates.

    Task type is inferred from the instruction via keyword matching.
    A future version can replace this heuristic with an LLM classifier
    without changing the rest of the system.
    """

    _TYPE_KEYWORDS: Dict[str, List[str]] = {
        "pick_and_place": ["pick", "bring", "fetch", "get", "grab", "take"],
        "open_container": ["open", "drawer", "cabinet", "fridge", "door"],
        "navigate_and_fetch": ["navigate", "go to", "find", "look for", "search"],
        "place_object": ["place", "put", "set", "drop"],
    }

    def __init__(self, data: Dict[str, Any] = None) -> None:
        # {task_type: {"success_count": int, "fail_count": int,
        #              "common_steps": [str]}}
        self._data: Dict[str, Any] = data or {}

    @classmethod
    def infer_type(cls, instruction: str) -> str:
        instr_lower = instruction.lower()
        for ttype, keywords in cls._TYPE_KEYWORDS.items():
            if any(kw in instr_lower for kw in keywords):
                return ttype
        return "general"

    def record_episode(
        self, instruction: str, success: bool, steps: List[str]
    ) -> None:
        ttype = self.infer_type(instruction)
        rec = self._data.setdefault(
            ttype, {"success_count": 0, "fail_count": 0, "common_steps": []}
        )
        if success:
            rec["success_count"] += 1
            for s in steps:
                if s not in rec["common_steps"]:
                    rec["common_steps"].append(s)
                    if len(rec["common_steps"]) > 20:
                        rec["common_steps"].pop(0)
        else:
            rec["fail_count"] += 1

    def get_success_rate(self, instruction: str) -> Optional[float]:
        ttype = self.infer_type(instruction)
        rec = self._data.get(ttype)
        if not rec:
            return None
        total = rec["success_count"] + rec["fail_count"]
        return rec["success_count"] / total if total else None

    def get_common_steps(self, instruction: str) -> List[str]:
        ttype = self.infer_type(instruction)
        return self._data.get(ttype, {}).get("common_steps", [])

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# SpatialTopo
# ---------------------------------------------------------------------------

class SpatialTopo:
    """
    Persistent spatial topology: location nodes with neighbour edges
    and per-location object frequency counters.

    Compatible with pose-graph SLAM: each node carries an optional metric
    pose that a SLAM module can fill in at deployment time.
    """

    def __init__(self, data: Dict[str, Any] = None) -> None:
        d = data or {}
        # {loc_name: {"neighbours": [str], "object_freq": {obj: count},
        #             "pose": [x, y, z] | null}}
        self._nodes: Dict[str, Any] = d.get("nodes", {})
        # [[loc_a, loc_b], …] undirected edges stored as sorted string pairs
        self._edges: List[List[str]] = d.get("edges", [])

    def add_location(self, location: str, pose: Optional[List[float]] = None) -> None:
        if location not in self._nodes:
            self._nodes[location] = {"neighbours": [], "object_freq": {}, "pose": pose}
        elif pose is not None:
            self._nodes[location]["pose"] = pose

    def record_object_at(self, location: str, obj: str, count: int = 1) -> None:
        self.add_location(location)
        freq = self._nodes[location]["object_freq"]
        freq[obj] = freq.get(obj, 0) + count

    def add_edge(self, loc_a: str, loc_b: str) -> None:
        pair = sorted([loc_a, loc_b])
        if pair not in self._edges:
            self._edges.append(pair)
            self.add_location(loc_a)
            self.add_location(loc_b)
            if loc_b not in self._nodes[loc_a]["neighbours"]:
                self._nodes[loc_a]["neighbours"].append(loc_b)
            if loc_a not in self._nodes[loc_b]["neighbours"]:
                self._nodes[loc_b]["neighbours"].append(loc_a)

    def set_pose(self, location: str, pose: List[float]) -> None:
        """Attach a metric pose to a location node (called by SLAM module)."""
        self.add_location(location, pose=pose)

    def get_neighbours(self, location: str) -> List[str]:
        return self._nodes.get(location, {}).get("neighbours", [])

    def get_frequent_objects(self, location: str, top_k: int = 5) -> List[Tuple[str, int]]:
        freq = self._nodes.get(location, {}).get("object_freq", {})
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self._nodes, "edges": self._edges}


# ---------------------------------------------------------------------------
# SemanticMemory  (top-level container)
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Top-level semantic memory container backed by a single JSON file.

    Writes are atomic: content is first written to a .tmp file, then
    renamed over the target path to avoid partial writes on power loss.
    """

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.object_kb = ObjectKB()
        self.user_pref = UserPreference()
        self.task_schema = TaskSchema()
        self.spatial_topo = SpatialTopo()
        self._load()

    # --- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.object_kb = ObjectKB(d.get("object_kb", {}))
            self.user_pref = UserPreference(d.get("user_pref", {}))
            self.task_schema = TaskSchema(d.get("task_schema", {}))
            self.spatial_topo = SpatialTopo(d.get("spatial_topo", {}))
        except Exception:
            pass  # Corrupted or absent file: start with empty stores

    def save(self) -> None:
        """Atomic write: serialize to .tmp then rename."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "object_kb": self.object_kb.to_dict(),
            "user_pref": self.user_pref.to_dict(),
            "task_schema": self.task_schema.to_dict(),
            "spatial_topo": self.spatial_topo.to_dict(),
        }
        tmp_path = self.store_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp_path), str(self.store_path))

    # --- generalization from episodic records ------------------------------

    def generalize_from_episodes(self, records: list) -> None:
        """
        Update object knowledge, task schemata, and spatial topology from a
        batch of EpisodeRecord objects. Called by the memory facade every N
        completed episodes.
        """
        for rec in records:
            for obj, info in rec.objects_encountered.items():
                self.object_kb.update_from_episode(
                    obj=obj,
                    locations=info.get("locations", []),
                    grasped=info.get("grasped", False),
                    count=info.get("count", 1),
                )
                for loc in info.get("locations", []):
                    self.spatial_topo.record_object_at(loc, obj, count=info.get("count", 1))

            action_sequence = [s.action for s in rec.steps]
            self.task_schema.record_episode(rec.task_instruction, rec.success, action_sequence)

            # Build spatial edges from consecutive location changes
            prev_loc: Optional[str] = None
            for step in rec.steps:
                loc = step.current_location
                if loc:
                    self.spatial_topo.add_location(loc)
                    if prev_loc and prev_loc != loc:
                        self.spatial_topo.add_edge(prev_loc, loc)
                    prev_loc = loc

        self.save()

    # --- query API ---------------------------------------------------------

    def query_object(self, obj: str) -> Dict[str, Any]:
        return {
            "affordances": self.object_kb.get_affordances(obj),
            "likely_locations": self.object_kb.get_likely_locations(obj),
        }

    def query_user_preference(self, need: str) -> Dict[str, Any]:
        return self.user_pref.query(need)

    def query_task_schema(self, instruction: str) -> Dict[str, Any]:
        return {
            "task_type": TaskSchema.infer_type(instruction),
            "success_rate": self.task_schema.get_success_rate(instruction),
            "common_steps": self.task_schema.get_common_steps(instruction),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "objects_known": len(self.object_kb._data),
            "user_needs_known": len(self.user_pref._data),
            "task_types_known": len(self.task_schema._data),
            "locations_known": len(self.spatial_topo._nodes),
            "store_path": str(self.store_path),
        }
