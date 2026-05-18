import json
import os
import re
from .llm_client import LLMClient
from .ltm_client import LTMClient
from .prompt_templates import (
    SOLUTION_SPACE_SYSTEM_PROMPT, 
    SOLUTION_SPACE_USER_PROMPT,
    SOLUTION_SPACE_FILTER_SYSTEM_PROMPT,
    SOLUTION_SPACE_FILTER_USER_PROMPT
)
from .logger_utils import AgentLogger

class SolutionSpaceAnalyzer:
    def __init__(self, llm_client: LLMClient, logger: AgentLogger):
        self.llm = llm_client
        self.ltm = LTMClient()
        self.logger = logger
        self.world_model = {"explored_locations": [], "objects": {}}
        self.memory_objects = {}  # Retained for compatibility during refactoring
        self.current_visible_objects = []
        self.capabilities = []
        self.legal_combinations = {}
        
        self.composition_elements = {"actions": set(), "items": set(), "locations": set()}
        self.refined_items = {}
        self.relevant_objects = []
        self.relevant_locations = []
        
        self.priors_initialized = False
        self.global_intent = {}
        self.exploration_context = {}

    @staticmethod
    def _location_matches(current_location: str, target_location: str) -> bool:
        cur = (current_location or "").lower().strip()
        loc = (target_location or "").lower().strip()
        if not cur or not loc:
            return False
        return loc in cur or cur in loc

    def _compute_exploration_context(
        self,
        current_location: str,
        visited_locations_state: dict,
        held_object: str,
    ) -> dict:
        """
        检测当前位置是否存在「已到达但未 open」的容器。
        用于引导 Ranker 优先执行 open，避免到门口又走开。
        """
        if held_object:
            return {"pending_open_location": None, "pending_open_action_id": None}

        pending_open_location = None
        pending_open_action_id = None

        for aid, cap in self.legal_combinations.items():
            if aid == 9999:
                continue
            cap_lower = cap.lower()
            if not cap_lower.startswith("open "):
                continue
            loc = cap_lower.replace("open the ", "").replace("open ", "").strip()
            if not self._location_matches(current_location, loc):
                continue
            if self.receptacle_states.get(loc) == "open":
                continue
            already_opened = any(
                k.lower() == loc and v == "visited_open"
                for k, v in visited_locations_state.items()
            )
            if already_opened:
                continue
            pending_open_location = loc
            pending_open_action_id = aid
            break

        ctx = {
            "pending_open_location": pending_open_location,
            "pending_open_action_id": pending_open_action_id,
        }
        if pending_open_location:
            self.logger.info(
                f"[SolutionSpace] Pending open at current location: '{pending_open_location}' "
                f"(action_id={pending_open_action_id})"
            )
        return ctx

    def _sync_memory_from_ltm(self):
        """从 LTM 拉取世界模型快照，包含已探索的地点和所有已知物品的位置"""
        query = "Based on the agent's memory, summarize the exact current state. Return ONLY a valid JSON dictionary matching exactly this structure: {\"explored_locations\": [\"cabinet 5\", \"refrigerator\"], \"objects\": {\"apple\": \"refrigerator\", \"mug\": \"cabinet 5\"}}. If no locations are explored, leave the list empty. If no objects are known, leave the dict empty. Do not include any other keys."
        # 使用 global 模式拉取全量图谱知识，确保检索到所有地点的物品
        response_text = self.ltm.query(query, mode="global")
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                self.world_model = {
                    "explored_locations": [loc.lower() for loc in parsed.get("explored_locations", [])],
                    "objects": parsed.get("objects", {})
                }
                # 为了向下兼容后续代码，也更新一下 memory_objects（无需 requires_open）
                self.memory_objects = {}
                for obj, loc in self.world_model["objects"].items():
                    self.memory_objects[obj] = {"location": loc}
                    
                self.logger.info(f"[SolutionSpace] Synced World Model from LTM. Explored locations: {self.world_model['explored_locations']}. Objects: {list(self.world_model['objects'].keys())}")
        except Exception as e:
            self.logger.error(f"[SolutionSpace] Failed to parse LTM World Model: {e}")

    def _detect_composition(self, skill_set: list):
        """1. 组成判断层"""
        self.composition_elements = {"actions": set(), "locations": set(), "objects": set(), "pickable_objects": set()}
        for cap in skill_set:
            cap_clean = cap.replace("navigate to the ", "nav_").replace("navigate to ", "nav_")
            if cap_clean.startswith("nav_"):
                loc = cap_clean[4:].strip()
                if "push point" in loc.lower():
                    continue
                self.composition_elements["actions"].add("navigate")
                self.composition_elements["locations"].add(loc)
            else:
                for prefix in ["pick up the ", "open the ", "close the ", "interact with the ", "place at the "]:
                    if cap.startswith(prefix):
                        action = prefix.split()[0].strip()
                        obj = cap[len(prefix):].strip()
                        self.composition_elements["actions"].add(action)
                        self.composition_elements["objects"].add(obj)
                        if action == "pick":
                            self.composition_elements["pickable_objects"].add(obj)
                        break

    def _refine_and_filter(self, global_intent: dict, held_object: str = None):
        """2. 选项细化与筛选层（物体相关性完全由 LLM 判定）"""
        all_locs = list(self.composition_elements["locations"])
        mem_objs = list(self.memory_objects.keys())
        
        filter_user_prompt = SOLUTION_SPACE_FILTER_USER_PROMPT.format(
            global_intent=json.dumps(global_intent, ensure_ascii=False) if global_intent else "{}",
            visible_objects=json.dumps(self.current_visible_objects, ensure_ascii=False),
            memory_objects=json.dumps(mem_objs, ensure_ascii=False),
            all_locations=json.dumps(all_locs, ensure_ascii=False),
            held_object=held_object or "None",
        )
        
        filter_result = self.llm.generate_json(
            system_prompt=SOLUTION_SPACE_FILTER_SYSTEM_PROMPT,
            user_prompt=filter_user_prompt
        )
        
        if filter_result:
            self.relevant_objects = filter_result.get("relevant_objects", self.current_visible_objects + mem_objs)
            self.relevant_locations = filter_result.get("relevant_locations", all_locs)
        else:
            self.relevant_objects = self.current_visible_objects + mem_objs
            self.relevant_locations = all_locs

        if not self.relevant_objects:
            self.logger.info(
                "[SolutionSpace] LLM filter returned empty relevant_objects; "
                "pick and propose-alternative (9999) are disabled until exploration finds candidates."
            )

        self.refined_items = {
            "visible": self.current_visible_objects,
            "in_memory_only": [obj for obj in self.memory_objects.keys() if obj not in self.current_visible_objects],
            "all_locations": all_locs,
            "relevant_objects": self.relevant_objects,
            "relevant_locations": self.relevant_locations
        }

    def _apply_combination_constraints(self, skill_set: list, current_location: str, visited_locations_state: dict, held_object: str):
        """3. 组合约束层 (基于 World Model 彻底阻断无限循环)"""
        valid_combinations = {}
        blacklist_lower = [b.lower() for b in getattr(self, "blacklisted_actions", [])]
        relevant_lower = [o.lower() for o in self.relevant_objects]
        
        explored_locs = list(set(self.world_model.get("explored_locations", [])) | set([loc.lower() for loc in visited_locations_state.keys()]))
        known_objects_map = self.world_model.get("objects", {})
        
        # 判断当前目标物品是否在我们想去的某个 explored_loc 里
        allowed_explored_locs = set()
        for obj, loc in known_objects_map.items():
            if obj.lower() in relevant_lower:
                allowed_explored_locs.add(loc.lower())
                
        for i, cap in enumerate(skill_set):
            cap_lower = cap.lower()
            cap_clean = cap_lower.replace("navigate to the ", "nav_").replace("navigate to ", "nav_")
            
            # 过滤黑名单
            if cap_lower in blacklist_lower or cap_clean in blacklist_lower:
                continue
                
            is_valid = False
            
            if cap_clean.startswith("nav_"):
                loc = cap_clean[4:].strip()
                if "push point" in loc:
                    continue
                if loc != current_location.lower() and loc in [l.lower() for l in self.relevant_locations]:
                    is_valid = True
                    
                    # World Model 死循环拦截规则 1:
                    # 如果目的地已经 explored，且里面没有我们要找的目标物品，强行屏蔽 nav_
                    if loc in explored_locs and loc not in allowed_explored_locs:
                        is_valid = False
            else:
                # 交互动作的物理硬约束
                if cap_lower.startswith("pick "):
                    obj = cap_lower.replace("pick up the ", "").replace("pick up ", "").strip()
                    visible_lower = [v.lower() for v in getattr(self, "current_visible_objects", [])]
                    if obj in visible_lower and obj in relevant_lower and not held_object:
                        is_valid = True
                elif cap_lower.startswith("place ") or cap_lower.startswith("put ") or cap_lower.startswith("drop "):
                    if held_object:
                        is_valid = True
                elif cap_lower.startswith("open "):
                    loc = cap_lower.replace("open the ", "").replace("open ", "").strip()
                    if loc in current_location.lower() or current_location.lower() in loc:
                        # 物理状态硬拦截：如果已经是 open 状态，不允许再 open
                        if self.receptacle_states.get(loc) == "open":
                            is_valid = False
                        else:
                            is_valid = True
                            # World Model 死循环拦截规则 2:
                            # 如果它已经被 explored，且里面没有我们要的目标，没必要再 open
                            if loc in explored_locs and loc not in allowed_explored_locs:
                                is_valid = False
                elif cap_lower.startswith("close "):
                    loc = cap_lower.replace("close the ", "").replace("close ", "").strip()
                    if loc in current_location.lower() or current_location.lower() in loc:
                        # 物理状态硬拦截：如果已经是 closed 状态，不允许再 close
                        if self.receptacle_states.get(loc) == "closed":
                            is_valid = False
                        elif self.receptacle_states.get(loc) == "open":
                            # 只有在 open 状态下才允许 close
                            is_valid = True
                            # World Model 死循环拦截规则 3:
                            # 如果它在 explored 列表里，永远不准再 close（防止拿东西时手贱关上）
                            if loc in explored_locs:
                                is_valid = False
                        else:
                            # 状态未知，允许 close（保守策略）
                            is_valid = True
                elif cap_lower.startswith("interact "):
                    loc = cap_lower.replace("interact with the ", "").replace("interact with ", "").strip()
                    if loc in current_location.lower() or current_location.lower() in loc:
                        is_valid = True
                elif cap_lower == "done":
                    is_valid = True
                    
            if is_valid:
                valid_combinations[i] = cap
                
        all_locs_count = len(self.composition_elements.get("locations", []))
        all_explored = len(explored_locs) >= all_locs_count if all_locs_count else False
        held_in_relevant = bool(
            held_object and held_object.lower() in relevant_lower
        )

        # 9999：仅当 LLM 认定手持物在 relevant_objects 中，或全部探索完且空手
        if held_in_relevant:
            valid_combinations[9999] = "communicate with user and propose alternative"
        elif all_explored and not held_object:
            valid_combinations[9999] = "communicate with user and propose alternative"
        elif held_object and not held_in_relevant:
            self.logger.info(
                f"[SolutionSpace] Suppressed Action 9999: '{held_object}' not in LLM relevant_objects {self.relevant_objects}"
            )

        self.legal_combinations = valid_combinations
        self.exploration_context = self._compute_exploration_context(
            current_location, visited_locations_state, held_object
        )

    def update_capabilities(self, skill_set: list, inject_priors: bool = False):
        """更新动作能力并提取原子要素，且仅在第一次调用时初始化环境先验至 LTM"""
        self.capabilities = [str(skill) for skill in skill_set]
        self._detect_composition(self.capabilities)
        
        # 将静态的 Locations 和基础 Actions 作为先验知识写入 LTM（仅一次）
        if inject_priors and not self.priors_initialized and self.composition_elements["locations"]:
            locs = list(self.composition_elements["locations"])
            acts = ["navigate", "pick up", "place", "open", "close"]
            self.ltm.initialize_environment_priors(locs, acts)
            self.priors_initialized = True

    def update_observation(self, observation_text: str, current_location: str, visited_locations_state: dict, step: int, global_intent: dict = None, img_path: str = None, dual_img_path: str = None, held_object: str = None, receptacle_states: dict = None, blacklisted_actions: list = None):
        """解析环境反馈，从 LTM 同步记忆，拼装解空间"""
        self.receptacle_states = receptacle_states or {}
        self.blacklisted_actions = blacklisted_actions or []
        self.global_intent = global_intent or {}
        user_prompt = SOLUTION_SPACE_USER_PROMPT.format(
            observation_text=observation_text,
            legal_objects=json.dumps(list(self.composition_elements.get("pickable_objects", self.composition_elements["objects"])), ensure_ascii=False)
        )
        
        if img_path and os.path.exists(img_path):
            result_dict = self.llm.generate_vision_json(
                system_prompt=SOLUTION_SPACE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_path=img_path
            )
        else:
            result_dict = self.llm.generate_json(
                system_prompt=SOLUTION_SPACE_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
        
        visible = result_dict.get("visible_objects", [])
        self.current_visible_objects = visible
        
        # 我们现在从 LTM 同步旧物品，而不是依赖本地累加
        self._sync_memory_from_ltm()
        
        # --- 始终记录当前地点为"已访问" ---
        # 无论是否发现新物品，都主动向 LTM 写入一条访问记录，
        # 这样 LTM 就能把该地点纳入 explored_locations，防止重复探索死循环。
        if current_location and current_location not in self.world_model.get("explored_locations", []):
            new_observations = []
            for obj in visible:
                if obj not in self.memory_objects:
                    self.memory_objects[obj] = {"location": current_location}
                    new_observations.append(f"'{obj}'")
                    
            if new_observations:
                obs_str = ", ".join(new_observations)
                self.ltm.update_state(f"Visited '{current_location}'. Observed {obs_str} here.")
                self.logger.info(f"[SolutionSpace] Sent STATE UPDATE to LTM: Visited '{current_location}', found {obs_str}")
            else:
                # 即使没有新物品，也要标记该地点为已探索
                self.ltm.update_state(f"Visited '{current_location}'. No new objects found here (location is now fully explored).")
                self.logger.info(f"[SolutionSpace] Sent STATE UPDATE to LTM: Visited '{current_location}', no new objects found.")
        else:
            # 已探索过，但仍需处理本轮新视野中的物品（容器打开后新出现的物品）
            new_observations = []
            for obj in visible:
                if obj not in self.memory_objects:
                    self.memory_objects[obj] = {"location": current_location}
                    new_observations.append(f"'{obj}'")
            if new_observations:
                obs_str = ", ".join(new_observations)
                self.ltm.update_state(f"Observed {obs_str} at '{current_location}'.")
                self.logger.info(f"[SolutionSpace] Sent STATE UPDATE to LTM: Observed {obs_str} at '{current_location}'")
            
        self._refine_and_filter(global_intent, held_object=held_object)
        self._apply_combination_constraints(
            self.capabilities, current_location, visited_locations_state, held_object
        )
        
        output_data = {
            "composition": {
                "actions": list(self.composition_elements["actions"]),
                "objects": list(self.composition_elements["objects"]),
                "locations": list(self.composition_elements["locations"])
            },
            "refined_items": self.refined_items,
            "legal_combinations_count": len(self.legal_combinations),
            "legal_combinations_sample": list(self.legal_combinations.values())[:5],
            "currently_visible": self.current_visible_objects,
            "memory_objects": self.memory_objects
        }
        
        self.logger.log_module_output(
            module_name="SolutionSpaceAnalyzer (Module B - 3 Layers)",
            step=step,
            output_data=json.dumps(output_data, indent=2, ensure_ascii=False),
            img_path=dual_img_path if dual_img_path else img_path
        )

    def get_solution_space_dict(self):
        return {
            "capabilities": self.capabilities,
            "legal_combinations": self.legal_combinations,
            "legal_locations": list(self.composition_elements.get("locations", [])),
            "visible_objects": self.current_visible_objects,
            "memory_objects": self.memory_objects,
            "relevant_objects": self.relevant_objects,
            "refined_items": self.refined_items,
            "exploration_context": self.exploration_context,
        }
