"""
PlanningModuleBridge: one-way read bridge from Planning_module into memory layers.

Reads persistent_memory.md (user preferences + scene knowledge) and
session_context.md (current goal intent + visited locations) from the
Planning_module directory if they exist, then populates SemanticMemory and
WorkingMemory.

As of the Planning_module refactor the MD files are no longer written by the
Planning_module (it stores everything in the EmbodiedLTM HTTP service instead).
sync_from_ltm() queries the shared EmbodiedLTM service to populate memory from
whatever the Planning_module has already learned in this session.

Design constraints:
  - Strictly read-only with respect to Planning_module files.
  - Never writes back to persistent_memory.md or session_context.md.
  - Does not import from Planning_module modules; relies only on file parsing
    and the shared EmbodiedLTM HTTP endpoint.
  - Safe to call even when the files do not exist (returns gracefully).

Backward compatibility:
  PlanningAgentBridge is kept as an alias so existing code that references
  the old class name continues to work without changes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from embodiedbench.memory_manip.working_memory import WorkingMemory
    from embodiedbench.memory_manip.semantic_memory import SemanticMemory
    from embodiedbench.memory_manip.longterm_memory import EmbodiedLTMClient


class PlanningModuleBridge:
    """
    Reads Planning_module data and pushes it into the memory system.

    Two sync paths:
      1. MD files (legacy): reads persistent_memory.md / session_context.md
         if they happen to exist in the Planning_module directory.
      2. EmbodiedLTM HTTP (new): queries the shared LTM service that the
         Planning_module writes to, then parses the response into structured
         SemanticMemory and WorkingMemory entries.

    Call sync() once per session start.
    """

    def __init__(self, planning_module_dir: Optional[str] = None) -> None:
        if planning_module_dir:
            self.module_dir = Path(planning_module_dir)
        else:
            # Default: Planning_module is a sibling of memory_manip inside embodiedbench
            self.module_dir = Path(__file__).parent.parent / "Planning_module"

        self.persistent_file = self.module_dir / "persistent_memory.md"
        self.session_file = self.module_dir / "session_context.md"

    def is_available(self) -> bool:
        return self.persistent_file.exists() or self.session_file.exists()

    # --- file I/O ----------------------------------------------------------

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    # --- MD parsing (legacy) -----------------------------------------------

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

    # --- MD sync (legacy) --------------------------------------------------

    def sync_to_semantic(self, semantic: "SemanticMemory") -> int:
        """
        Read persistent_memory.md and push preferences + scene knowledge into
        SemanticMemory. Returns the number of entries synced.
        No-op (returns 0) when the file does not exist.
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
                weight=2.0,
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
        No-op (returns False) when the file does not exist.
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

    # --- EmbodiedLTM HTTP sync (new) ---------------------------------------

    def sync_from_ltm(
        self,
        ltm_client: "EmbodiedLTMClient",
        working: "WorkingMemory",
        semantic: "SemanticMemory",
    ) -> Dict[str, Any]:
        """
        Query the shared EmbodiedLTM service for accumulated Planning_module
        knowledge, then parse the response into SemanticMemory and WorkingMemory.

        The EmbodiedLTM service is the canonical store once MD files are no
        longer written (Planning_module refactor). This method is a best-effort
        sync: it silently returns an empty result if the service is offline.

        Returns a status dict describing what was synced.
        """
        pref_count = 0
        scene_count = 0
        intent_synced = False
        location_count = 0

        # --- preferences ---
        pref_resp = ltm_client.query_memory(
            "What are the user's object preferences and what objects do they prefer?",
            mode="hybrid",
        )
        pref_text = pref_resp.get("final_answer", "") if isinstance(pref_resp, dict) else ""
        if pref_text and "error" not in pref_text.lower():
            for pref in self._parse_preferences(_wrap_as_preferences_section(pref_text)):
                semantic.user_pref.add_preference(
                    need=pref["need"],
                    obj=pref["obj"],
                    source="ltm",
                    weight=1.5,
                )
                pref_count += 1

        # --- scene / object locations ---
        scene_resp = ltm_client.query_memory(
            "What objects have been found and where are they located in the scene?",
            mode="hybrid",
        )
        scene_text = scene_resp.get("final_answer", "") if isinstance(scene_resp, dict) else ""
        if scene_text and "error" not in scene_text.lower():
            for entry in self._parse_scene_knowledge(
                _wrap_as_scene_section(scene_text)
            ):
                if entry["location"] != "unknown":
                    semantic.spatial_topo.record_object_at(entry["location"], entry["obj"])
                    semantic.object_kb.update_from_episode(
                        obj=entry["obj"],
                        locations=[entry["location"]],
                        grasped=False,
                        count=1,
                    )
                    scene_count += 1

        # --- explored locations for working memory ---
        loc_resp = ltm_client.query_memory(
            "Which locations has the agent already visited and fully explored?",
            mode="hybrid",
        )
        loc_text = loc_resp.get("final_answer", "") if isinstance(loc_resp, dict) else ""
        if loc_text and "error" not in loc_text.lower():
            for m in re.finditer(
                r"(?:visited|explored|navigated to)[^\n]*?['\"]?([a-zA-Z0-9_ ]+(?:counter|table|cabinet|refrigerator|drawer|shelf|bin|box|sofa|sink|microwave|oven|dishwasher)[a-zA-Z0-9_ ]*)['\"]?",
                loc_text,
                re.IGNORECASE,
            ):
                loc = m.group(1).strip()
                working.observation.visited_locations.setdefault(loc, "visited_open")
                location_count += 1

        return {
            "pref_count": pref_count,
            "scene_count": scene_count,
            "intent_synced": intent_synced,
            "location_count": location_count,
            "ltm_available": True,
        }

    # --- unified sync -------------------------------------------------------

    def sync(
        self,
        working: "WorkingMemory",
        semantic: "SemanticMemory",
        ltm_client: Optional["EmbodiedLTMClient"] = None,
    ) -> Dict[str, Any]:
        """
        Full sync: try MD files first (legacy path), then EmbodiedLTM HTTP.

        ltm_client: pass the EmbodiedLTMClient from EmbodiedManipulationMemorySystem
                    to enable the HTTP sync path. When None only the MD path runs.
        """
        sem_count = self.sync_to_semantic(semantic)
        wm_synced = self.sync_to_working(working)

        ltm_result: Dict[str, Any] = {}
        if ltm_client is not None:
            try:
                ltm_result = self.sync_from_ltm(ltm_client, working, semantic)
            except Exception:
                ltm_result = {"ltm_available": False}

        return {
            "md_semantic_entries": sem_count,
            "md_working_synced": wm_synced,
            "md_available": self.is_available(),
            **ltm_result,
        }


# ---------------------------------------------------------------------------
# Helpers for wrapping free-text LTM answers into parseable pseudo-MD
# ---------------------------------------------------------------------------

def _wrap_as_preferences_section(text: str) -> str:
    """Wrap LTM free-text answer into a ## Preferences section for the parser."""
    lines = ["## Preferences"]
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(f"- {line}" if not line.startswith("-") else line)
    return "\n".join(lines)


def _wrap_as_scene_section(text: str) -> str:
    """Wrap LTM free-text answer into a ## Scene Knowledge section for the parser."""
    lines = ["## Scene Knowledge"]
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(f"- {line}" if not line.startswith("-") else line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

PlanningAgentBridge = PlanningModuleBridge
