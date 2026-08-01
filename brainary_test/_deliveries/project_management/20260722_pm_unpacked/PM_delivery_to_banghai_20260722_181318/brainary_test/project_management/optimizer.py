from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


DEFAULT_ALIAS_PATH = Path(__file__).with_name("object_aliases.json")

CATEGORY_SYNONYMS = {
    "杯子": "杯具",
    "水杯": "杯具",
    "水果": "水果",
    "食品": "水果",
    "工具": "工具",
    "文具": "工具",
}

TEMPORARY_ACTIONS = {"move", "remove", "temporary_place", "move_aside", "clear_path"}


class PlanOptimizer:
    """PM 核心优化器：目标名转换 + 遮挡冗余消除（一步到位）。"""

    def __init__(self, object_aliases: Optional[Dict[str, str]] = None, alias_path: Optional[str | Path] = None):
        self.object_aliases = self._load_aliases(alias_path)
        if object_aliases:
            self.object_aliases.update(object_aliases)

    def normalize_plan(self, plan: Sequence[Dict[str, Any]], planning_input: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw in plan:
            item = deepcopy(raw)
            action = str(item.get("action", "")).strip()
            target = item.get("target")
            if action == "grasp" and isinstance(target, str):
                item["target"] = self.object_aliases.get(target, target)
                if target != item["target"]:
                    item.setdefault("source_target", target)
            normalized.append(item)
        return self.eliminate_redundancy(normalized, planning_input)

    def eliminate_redundancy(
        self,
        plan: Sequence[Dict[str, Any]],
        planning_input: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not planning_input:
            return list(plan)

        category_rules = planning_input.get("constraints", {}).get("category_rules", {})
        objects = planning_input.get("manipulable_objects", {})

        temp_objects: Dict[str, str] = {}
        for item in plan:
            action = str(item.get("action", "")).strip()
            target = item.get("target")
            if action in TEMPORARY_ACTIONS and isinstance(target, str):
                basket = self.goal_basket_for_object(target, objects, category_rules)
                if basket:
                    temp_objects[target] = basket

        if not temp_objects:
            return list(plan)

        redundant_ids, id_remap = self._find_redundant_pairs(plan, temp_objects)
        result: List[Dict[str, Any]] = []

        for item in plan:
            node_id = str(item.get("id", ""))
            action = str(item.get("action", "")).strip()
            target = item.get("target")

            if node_id in redundant_ids:
                continue

            new_item = deepcopy(item)
            if action in TEMPORARY_ACTIONS and isinstance(target, str) and target in temp_objects:
                new_item["action"] = "place"
                new_item["target"] = temp_objects[target]
                new_item["pm_optimized_from"] = action
                new_item["pm_reason"] = f"遮挡物 '{target}' 属于任务清单，一步到位直接放入 {temp_objects[target]}"
            result.append(new_item)

        return self._fix_dependencies(result, id_remap)

    def _find_redundant_pairs(
        self,
        plan: Sequence[Dict[str, Any]],
        temp_objects: Dict[str, str],
    ) -> Tuple[Set[str], Dict[str, str]]:
        redundant: Set[str] = set()
        id_remap: Dict[str, str] = {}
        plan_list = list(plan)

        for obj_name, goal_basket in temp_objects.items():
            temp_idx = self._find_temp_action_index(plan_list, obj_name)
            if temp_idx is None:
                continue
            temp_id = str(plan_list[temp_idx].get("id", ""))

            obj_sim_name = self.object_aliases.get(obj_name, obj_name)
            grasp_idx = None
            for i in range(temp_idx + 1, len(plan_list)):
                item = plan_list[i]
                action = str(item.get("action", "")).strip()
                target = item.get("target")
                if action == "grasp" and target in {obj_name, obj_sim_name}:
                    grasp_idx = i
                    break

            if grasp_idx is None:
                continue

            place_idx = None
            for i in range(grasp_idx + 1, len(plan_list)):
                item = plan_list[i]
                action = str(item.get("action", "")).strip()
                target = item.get("target")
                if action == "place" and target == goal_basket:
                    place_idx = i
                    break
                if action == "grasp":
                    break

            if place_idx is not None:
                grasp_id = str(plan_list[grasp_idx].get("id", ""))
                place_id = str(plan_list[place_idx].get("id", ""))
                if grasp_id:
                    redundant.add(grasp_id)
                    if temp_id:
                        id_remap[grasp_id] = temp_id
                if place_id:
                    redundant.add(place_id)
                    if temp_id:
                        id_remap[place_id] = temp_id

        return redundant, id_remap

    def _find_temp_action_index(self, plan: List[Dict[str, Any]], obj_name: str) -> Optional[int]:
        obj_sim_name = self.object_aliases.get(obj_name, obj_name)
        for i, item in enumerate(plan):
            action = str(item.get("action", "")).strip()
            target = item.get("target")
            if action in TEMPORARY_ACTIONS and target in {obj_name, obj_sim_name}:
                return i
        return None

    def _fix_dependencies(self, plan: List[Dict[str, Any]], id_remap: Dict[str, str]) -> List[Dict[str, Any]]:
        if not id_remap:
            return plan
        for item in plan:
            deps = item.get("depends_on", [])
            if not deps:
                continue
            new_deps: List[str] = []
            for dep in deps:
                if dep in id_remap:
                    replacement = id_remap[dep]
                    if replacement and replacement not in new_deps:
                        new_deps.append(replacement)
                else:
                    if dep not in new_deps:
                        new_deps.append(dep)
            item["depends_on"] = new_deps
        return plan

    def goal_basket_for_object(
        self,
        object_name: str,
        manipulable_objects: Dict[str, Dict[str, Any]],
        category_rules: Dict[str, str],
    ) -> Optional[str]:
        info = manipulable_objects.get(object_name)
        if info is None:
            for zh_name, prop_name in self.object_aliases.items():
                if object_name in {zh_name, prop_name}:
                    info = manipulable_objects.get(zh_name)
                    break
        if not info:
            return None
        category = str(info.get("category", ""))
        candidates = [category, CATEGORY_SYNONYMS.get(category, category)]
        for candidate in candidates:
            if candidate in category_rules:
                return category_rules[candidate]
        return None

    def _load_aliases(self, alias_path: Optional[str | Path]) -> Dict[str, str]:
        path = Path(alias_path) if alias_path is not None else DEFAULT_ALIAS_PATH
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"目标别名配置必须是 JSON object: {path}")
        return {str(key): str(value) for key, value in data.items()}
