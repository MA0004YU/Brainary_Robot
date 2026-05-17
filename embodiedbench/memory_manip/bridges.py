"""
PlanningAgentBridge: one-way read bridge from Planning_agent MD files into memory layers.

Reads persistent_memory.md (user preferences + scene knowledge) and
session_context.md (current goal intent + visited locations) from the
Planning_agent directory, then populates SemanticMemory and WorkingMemory.

Design constraints:
  - Strictly read-only with respect to Planning_agent files.
  - Never writes back to persistent_memory.md or session_context.md.
  - Does not import from Planning_agent modules; relies only on file parsing.
  - Safe to call even when the files do not exist (returns gracefully).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from embodiedbench.memory_manip.working_memory import WorkingMemory
    from embodiedbench.memory_manip.semantic_memory import SemanticMemory


class PlanningAgentBridge:
    """
    Reads Planning_agent MD files and pushes their data into the memory system.

    Call sync() once per session start to bring the memory layers up to date
    with whatever the Planning_agent has already learned.
    """

    def __init__(self, planning_agent_dir: Optional[str] = None) -> None:
        if planning_agent_dir:
            self.agent_dir = Path(planning_agent_dir)
        else:
            # Default: Planning_agent is a sibling of memory_manip inside embodiedbench
            self.agent_dir = Path(__file__).parent.parent / "Planning_agent"

        self.persistent_file = self.agent_dir / "persistent_memory.md"
        self.session_file = self.agent_dir / "session_context.md"

    def is_available(self) -> bool:
        return self.persistent_file.exists() or self.session_file.exists()

    # --- file I/O ----------------------------------------------------------

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    # --- parsing -----------------------------------------------------------

    def _parse_preferences(self, md_text: str) -> List[Dict[str, str]]:
        """
        Extract preference entries from the ## Preferences section.

        Handles two formats:
          "- <need>: <preferred_object>"   → structured
          "- <free text bullet>"           → tagged as need="general"
        """
        prefs: List[Dict[str, str]] = []
        in_section = False
        _SKIP = {"暂无偏好记录。", "no preferences recorded.", "none"}

        for line in md_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## Preferences"):
                in_section = True
                continue
            if in_section and stripped.startswith("## "):
                in_section = False
            if not in_section or not stripped.startswith("- "):
                continue

            raw = stripped[2:].strip()
            if not raw or raw.lower() in _SKIP:
                continue

            if ":" in raw:
                parts = raw.split(":", 1)
                prefs.append({"need": parts[0].strip().lower(), "obj": parts[1].strip(), "raw": raw})
            else:
                prefs.append({"need": "general", "obj": raw, "raw": raw})

        return prefs

    def _parse_scene_knowledge(self, md_text: str) -> List[Dict[str, str]]:
        """
        Extract object-location pairs from the ## Scene Knowledge section.

        Recognised patterns (case-insensitive):
          "<object> is [in|at|near|on|inside] [the] <location>"
          "<object>: <location>"
        """
        knowledge: List[Dict[str, str]] = []
        in_section = False
        _LOC_PATTERN = re.compile(
            r"(.+?)\s+(?:is|are)\s+(?:in|at|near|on|inside)\s+(?:the\s+)?(.+)",
            re.IGNORECASE,
        )
        _COLON_PATTERN = re.compile(r"(.+?):\s+(.+)")

        for line in md_text.splitlines():
            stripped = line.strip()
            if "## Scene Knowledge" in stripped:
                in_section = True
                continue
            if in_section and stripped.startswith("## "):
                in_section = False
            if not in_section or not stripped.startswith("- "):
                continue

            raw = stripped[2:].strip()
            if not raw:
                continue

            m = _LOC_PATTERN.match(raw) or _COLON_PATTERN.match(raw)
            if m:
                knowledge.append({"obj": m.group(1).strip(), "location": m.group(2).strip(), "raw": raw})
            else:
                knowledge.append({"obj": raw, "location": "unknown", "raw": raw})

        return knowledge

    def _parse_session_intent(self, md_text: str) -> Optional[Dict[str, Any]]:
        """Extract the global_intent JSON object from session_context.md."""
        m = re.search(r"\*\*global_intent\*\*:\s*(\{.+?\})", md_text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    def _parse_visited_locations(self, md_text: str) -> Dict[str, str]:
        """Extract {location: state} pairs from session_context.md."""
        locations: Dict[str, str] = {}
        for m in re.finditer(r"-\s+(.+?):\s*(visited_\w+|fully_explored)", md_text):
            locations[m.group(1).strip()] = m.group(2).strip()
        return locations

    # --- sync --------------------------------------------------------------

    def sync_to_semantic(self, semantic: "SemanticMemory") -> int:
        """
        Read persistent_memory.md and push preferences + scene knowledge
        into SemanticMemory. Returns the number of entries synced.
        """
        if not self.persistent_file.exists():
            return 0

        text = self._read_file(self.persistent_file)
        count = 0

        for pref in self._parse_preferences(text):
            semantic.user_pref.add_preference(
                need=pref["need"],
                obj=pref["obj"],
                source="explicit",
                weight=2.0,  # Explicit user statements outweigh inferred episodic prefs
            )
            count += 1

        for entry in self._parse_scene_knowledge(text):
            if entry["location"] != "unknown":
                semantic.spatial_topo.record_object_at(entry["location"], entry["obj"])
                semantic.object_kb.update_from_episode(
                    obj=entry["obj"],
                    locations=[entry["location"]],
                    grasped=False,
                    count=1,
                )
                count += 1

        return count

    def sync_to_working(self, working: "WorkingMemory") -> bool:
        """
        Read session_context.md and update WorkingMemory goal intent and
        visited locations. Returns True if any data was synced.
        """
        if not self.session_file.exists():
            return False

        text = self._read_file(self.session_file)
        synced = False

        intent = self._parse_session_intent(text)
        if intent:
            working.goal.intent = intent
            synced = True

        visited = self._parse_visited_locations(text)
        if visited:
            working.observation.visited_locations.update(visited)
            synced = True

        return synced

    def sync(self, working: "WorkingMemory", semantic: "SemanticMemory") -> Dict[str, Any]:
        """Full sync: push both MD files into the appropriate memory layers."""
        sem_count = self.sync_to_semantic(semantic)
        wm_synced = self.sync_to_working(working)
        return {
            "semantic_entries_synced": sem_count,
            "working_memory_synced": wm_synced,
            "bridge_available": self.is_available(),
        }
