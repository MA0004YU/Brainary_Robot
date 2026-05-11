import json
import os
from .llm_client import LLMClient
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
        self.logger = logger
        self.memory_objects = {}
        self.current_visible_objects = []
        self.capabilities = []
        self.legal_combinations = {}
        
        self.composition_elements = {"actions": set(), "items": set(), "locations": set()}
        self.refined_items = {}
        self.relevant_objects = []
        self.relevant_locations = []

    def _detect_composition(self, skill_set: list):
        """1. 组成判断层 (Composition Detection Layer)"""
        self.composition_elements = {"actions": set(), "locations": set(), "objects": set(), "pickable_objects": set()}
        for cap in skill_set:
            cap_clean = cap.replace("navigate to the ", "nav_").replace("navigate to ", "nav_")
            if cap_clean.startswith("nav_"):
                loc = cap_clean[4:].strip()
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

    def _refine_and_filter(self, global_intent: dict):
        """2. 选项细化与筛选层 (Option Refinement & Filtering Layer)"""
        all_locs = list(self.composition_elements["locations"])
        mem_objs = list(self.memory_objects.keys())
        
        filter_user_prompt = SOLUTION_SPACE_FILTER_USER_PROMPT.format(
            global_intent=json.dumps(global_intent, ensure_ascii=False) if global_intent else "{}",
            visible_objects=json.dumps(self.current_visible_objects, ensure_ascii=False),
            memory_objects=json.dumps(mem_objs, ensure_ascii=False),
            all_locations=json.dumps(all_locs, ensure_ascii=False)
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

        self.refined_items = {
            "visible": self.current_visible_objects,
            "in_memory_only": [obj for obj in self.memory_objects.keys() if obj not in self.current_visible_objects],
            "all_locations": all_locs,
            "relevant_objects": self.relevant_objects,
            "relevant_locations": self.relevant_locations
        }

    def _apply_combination_constraints(self, skill_set: list, current_location: str, visited_locations_state: dict, held_object: str):
        """3. 组合约束层 (Combination Constraint Layer)"""
        valid_combinations = {}
        for i, cap in enumerate(skill_set):
            is_valid = False
            cap_clean = cap.replace("navigate to the ", "nav_").replace("navigate to ", "nav_")
            
            if cap_clean.startswith("nav_"):
                loc = cap_clean[4:].strip()
                # 导航：避免原地导航，并且需通过相关性过滤
                if loc != current_location and loc in self.relevant_locations:
                    is_valid = True
                    visited_locs_lower = {k.lower(): v for k, v in visited_locations_state.items()}
                    
                    # 【强迫症探索约束】：
                    current_rec_state = self.receptacle_states.get(current_location.lower(), "")
                    current_visit_state = visited_locs_lower.get(current_location.lower(), "")
                    
                    # 1. 如果这扇门正开着，绝不准离开！必须先关上！
                    if current_rec_state == "open":
                        is_valid = False
                    # 2. 如果这扇门关着，且它的探索状态还是 visited_closed，也不准离开（必须开一次！）
                    elif current_rec_state != "open" and current_visit_state == "visited_closed":
                        is_valid = False
                        
                    # 如果手里没东西 (EXPLORE模式)，不允许前往没有有效记忆物品的已探索地点
                    if not held_object and loc.lower() in visited_locs_lower:
                        has_relevant_item = False
                        for m_obj, m_info in self.memory_objects.items():
                            if m_info.get("location", "").lower() == loc.lower() and m_obj in self.relevant_objects:
                                has_relevant_item = True
                                break
                        if not has_relevant_item:
                            is_valid = False
            else:
                for prefix in ["pick up the ", "open the ", "close the ", "interact with the ", "place at the "]:
                    if cap.startswith(prefix):
                        action_type = prefix.split()[0].strip() # 'pick', 'open', 'close', 'interact', 'place'
                        obj = cap[len(prefix):].strip()
                        
                        if action_type == "pick":
                            if obj in self.relevant_objects:
                                # Pick 动作必须肉眼可见，且必须【双手空闲】才能拿！防止报错！
                                if obj in self.current_visible_objects and not held_object:
                                    is_valid = True
                        elif action_type in ["open", "close", "place", "interact"]:
                            # Open/Close/Place 动作依赖于当前所在物理位置匹配，不受 relevant_objects 限制
                            if obj.lower() in current_location.lower() or current_location.lower() in obj.lower():
                                is_valid = True
                                
                                # 【智能开关防呆】
                                current_rec_state = self.receptacle_states.get(current_location.lower(), "")
                                if action_type == "open" and current_rec_state == "open":
                                    is_valid = False  # 已经开了，不准再开
                                elif action_type == "close" and current_rec_state != "open":
                                    is_valid = False  # 没开，不准关
                                elif action_type == "place" and not held_object:
                                    is_valid = False  # 手里没东西，不准放
                                
                        # 强制无用物品脱手：如果手里拿着跟任务毫无关系的物品，强制允许 place 或 drop 动作
                        if held_object and obj == held_object and obj not in self.relevant_objects:
                            if action_type in ["place", "drop"]:
                                is_valid = True
                                
                        break
            
            if is_valid:
                valid_combinations[i] = cap

        # 9999 拦截站：必须手里拿着替代品才能解锁 9999
        if held_object and held_object in self.relevant_objects:
            valid_combinations[9999] = "communicate with user and propose alternative"
            
        self.legal_combinations = valid_combinations

    def update_capabilities(self, skill_set: list):
        """更新动作能力并提取原子要素"""
        self.capabilities = [str(skill) for skill in skill_set]
        self._detect_composition(self.capabilities)

    def update_observation(self, observation_text: str, current_location: str, visited_locations_state: dict, step: int, global_intent: dict = None, img_path: str = None, dual_img_path: str = None, held_object: str = None, receptacle_states: dict = None):
        """解析环境反馈，提取可视物体，并贯穿三层解空间构造"""
        self.receptacle_states = receptacle_states or {}
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
        
        for obj in visible:
            is_open = False
            for loc, state in visited_locations_state.items():
                if loc in current_location and state == "fully_explored":
                    is_open = True
                    break
                    
            if obj not in self.memory_objects:
                self.memory_objects[obj] = {
                    "location": current_location,
                    "requires_open": is_open
                }
            else:
                if isinstance(self.memory_objects[obj], str):
                    self.memory_objects[obj] = {
                        "location": current_location,
                        "requires_open": is_open
                    }
                else:
                    self.memory_objects[obj]["location"] = current_location
                    self.memory_objects[obj]["requires_open"] = is_open
            
        self._refine_and_filter(global_intent)
        self._apply_combination_constraints(self.capabilities, current_location, visited_locations_state, held_object)
        
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
            "memory_objects": self.memory_objects
        }
