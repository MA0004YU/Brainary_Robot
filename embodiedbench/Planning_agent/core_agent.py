import json
from .llm_client import LLMClient
from .logger_utils import AgentLogger
from .goal_reasoner import GoalReasoner
from .solution_space import SolutionSpaceAnalyzer
from .task_validator import TaskValidator
from .memory_manager import MemoryManager

class IntentReasoningAgent:
    def __init__(self, episode_id=None, model_name="gpt-4o-mini", resume_feedback=None, scene_id=None):
        import os
        import re
        log_dir = None
        session_file = os.path.join(os.path.dirname(__file__), "session_context.md")
        if resume_feedback and os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                content = f.read()
                log_dir_match = re.search(r'- \*\*log_dir\*\*: (.*)', content)
                if log_dir_match:
                    found_dir = log_dir_match.group(1).strip()
                    if found_dir and found_dir != 'None':
                        log_dir = found_dir

        self.logger = AgentLogger(episode_id, log_dir=log_dir)
        self.llm = LLMClient(model_name=model_name)
        
        # Initialize Core Modules
        self.goal_reasoner = GoalReasoner(self.llm, self.logger)
        self.solution_space = SolutionSpaceAnalyzer(self.llm, self.logger)
        self.task_validator = TaskValidator(self.llm, self.logger)
        self.memory_manager = MemoryManager(self.llm, self.logger)
        
        self.global_intent = {}
        self.action_history = []
        self.current_location = "unknown starting location"
        self.held_object = None
        
        self.scene_id = scene_id if scene_id else (episode_id if episode_id else "unknown_scene")
        session_state = self.memory_manager.load_session_state(fallback_scene_id=self.scene_id)
        
        if resume_feedback:
            if session_state:
                self.logger.info("[CoreAgent] Restoring session state from markdown...")
                self.global_intent = session_state.get("global_intent", {})
                self.action_history = session_state.get("action_history", [])
                
                # process feedback to update global_intent and preferences
                new_intent = self.memory_manager.process_feedback(resume_feedback)
                if new_intent:
                    self.global_intent = new_intent
                    
        if session_state:
            self.solution_space.memory_objects = session_state.get("known_objects", {})
            v_locs = session_state.get("visited_locations", {})
            self.visited_locations = v_locs if isinstance(v_locs, dict) else {loc: "visited_closed" for loc in v_locs}
        else:
            self.visited_locations = {}

    def step(self, instruction: str, observation_text: str, skill_set: list, step_idx: int, img_path: str = None, dual_img_path: str = None) -> dict:
        """
        核心调度循环
        :param instruction: 当前关卡的文本指令
        :param observation_text: 当前视角的文字描述/日志
        :param skill_set: 当前允许的 Action_IDs 列表 (元组或字符串形式)
        :param step_idx: 当前环境的步数
        :param img_path: (新增) 第一人称视觉截图路径，供多模态解析
        :param dual_img_path: (新增) 双视角截图路径，供日志记录
        :return: 决定执行的动作详情字典 (dict)
        """
        self.logger.info(f"\n========== STARTING STEP {step_idx} ==========")
        
        # 0. 错误恢复与容错机制 (Error Recovery Mechanism)
        if "Feedback:" in observation_text and "successfully" not in observation_text.lower():
            if len(self.action_history) > 0:
                failed_action = self.action_history.pop()
                self.logger.info(f"[Error Recovery] Last action '{failed_action}' failed. Removed from history.")
                
                # 如果是拾取失败，大概率是视觉误判/幻觉，立刻从记忆中抹除该物品
                if failed_action.lower().startswith("pick "):
                    obj = failed_action.lower().replace("pick up the ", "").replace("pick up ", "").replace("pick the ", "").replace("pick ", "").strip()
                    if obj in self.solution_space.memory_objects:
                        del self.solution_space.memory_objects[obj]
                        self.logger.info(f"[Error Recovery] Deleted hallucinated object '{obj}' from memory.")
        
        # 1. 目标提取 (仅在第一步执行，且未被 resume 恢复时)
        if step_idx == 0 and not self.global_intent:
            self.global_intent = self.goal_reasoner.extract_intent(instruction)
            
        # 提取 visited_locations (升级为 dict)
        visited_locations_state = self.visited_locations.copy()
        if isinstance(visited_locations_state, list):
            visited_locations_state = {loc: "visited_closed" for loc in visited_locations_state}
            
        self.receptacle_states = {}
        for action in self.action_history:
            action_lower = action.lower()
            if "navigate to" in action_lower:
                loc = action_lower.replace("navigate to the ", "").replace("navigate to ", "").strip()
                if loc not in visited_locations_state:
                    visited_locations_state[loc] = "visited_closed"
            elif action_lower.startswith("open "):
                loc = action_lower.replace("open the ", "").replace("open ", "").strip()
                # 寻找匹配的容器位置
                matched_loc = loc
                for k in visited_locations_state.keys():
                    if loc in k or k in loc:
                        matched_loc = k
                        break
                # 一旦执行了 open，该容器探索阶段完成，转为 fully_explored
                visited_locations_state[matched_loc] = "fully_explored"
                self.receptacle_states[matched_loc] = "open"
            elif action_lower.startswith("close "):
                loc = action_lower.replace("close the ", "").replace("close ", "").strip()
                # 寻找匹配的容器位置
                matched_loc = loc
                for k in visited_locations_state.keys():
                    if loc in k or k in loc:
                        matched_loc = k
                        break
                visited_locations_state[matched_loc] = "fully_explored"
                self.receptacle_states[matched_loc] = "closed"
                
        # 智能状态推断：如果当前地点不可打开（表面），直接标记为 fully_explored
        if self.current_location in visited_locations_state and visited_locations_state[self.current_location] == "visited_closed":
            has_open_action = False
            for cap in skill_set:
                if cap.lower().startswith("open ") and (self.current_location.lower() in cap.lower() or cap.lower().replace("open the ", "").strip() in self.current_location.lower()):
                    has_open_action = True
                    break
            if not has_open_action:
                visited_locations_state[self.current_location] = "fully_explored"
                
        # 解析 held_object
        self.held_object = None
        for action in self.action_history:
            action_lower = action.lower()
            if action_lower.startswith("pick "):
                obj = action_lower.replace("pick up the ", "").replace("pick up ", "").replace("pick the ", "").replace("pick ", "").strip()
                self.held_object = obj
            elif "place " in action_lower or "put " in action_lower or "drop " in action_lower:
                self.held_object = None
                
        self.visited_locations = visited_locations_state
        self.logger.info(f"[Visited Locations State] {visited_locations_state}")
        self.logger.info(f"[Inventory] held_object: {self.held_object}")

        # 2. 解空间更新
        self.solution_space.update_capabilities(skill_set)
        self.solution_space.update_observation(
            observation_text=observation_text, 
            current_location=self.current_location,
            visited_locations_state=visited_locations_state,
            step=step_idx, 
            global_intent=self.global_intent,
            img_path=img_path, 
            dual_img_path=dual_img_path,
            held_object=self.held_object,
            receptacle_states=self.receptacle_states
        )
        current_solution_space = self.solution_space.get_solution_space_dict()
        
        # 3. 验证与决策 (Merged C & D)
        validation_result = self.task_validator.validate_action_feasibility(
            global_intent=self.global_intent,
            solution_space=current_solution_space,
            action_history=self.action_history,
            visited_locations=visited_locations_state,
            current_location=self.current_location,
            step=step_idx,
            max_steps=50,
            persistent_memory_text=self.memory_manager.get_full_memory_context()
        )
        
        # 4. 提取动作并更新状态
        action_id = validation_result.get("action_id", 0)
        
        if action_id == 9999:
            # 强制拦截：如果手中没有任何合法的替代品，绝对不允许调用 9999
            if not self.held_object:
                self.logger.info(f"[CoreAgent] Intercepted illegal Action 9999. Agent is not holding any object!")
                action_id = list(current_solution_space.get("legal_combinations", {}).keys())[0] if current_solution_space.get("legal_combinations") else 0
            else:
                msg = validation_result.get("communication_to_user", "I need your assistance.")
                self.logger.info(f"[Communication to User] {msg}")
                
                # 当决定跟用户对话时，保存断点上下文
                self.memory_manager.save_session_state(
                    scene_id=self.scene_id,
                    global_intent=self.global_intent,
                    visited_locations=visited_locations_state,
                    known_objects=self.solution_space.memory_objects,
                    action_history=self.action_history,
                    last_agent_message=msg,
                    log_dir=self.logger.run_dir
                )
                validation_result["stop_and_save"] = True
                return validation_result
            
        if action_id is None:
            self.logger.log_module_output(
                module_name="CoreAgent",
                step=step_idx,
                output_data="Agent decided to stop. No action selected (action_id is None)."
            )
            # Default to 0 or some valid format to avoid crashing, or return a special termination signal.
            # Usually we return a stop action. If no stop action is known, return 0 as dummy.
            validation_result["action_id"] = 0
            return validation_result
        
        action_str = skill_set[action_id] if action_id < len(skill_set) else "UNKNOWN"
        self.action_history.append(action_str)
        
        self.logger.info(f"[Step {step_idx}] CoreAgent Final Decision: Action_ID {action_id} -> {action_str}")
        
        # 写入简版 Markdown 战报
        self.logger.log_step_markdown(
            step=step_idx,
            img_path=dual_img_path if dual_img_path else img_path,
            held_object=self.held_object,
            legal_combos_count=len(current_solution_space.get("legal_combinations", {})),
            visited_locations=visited_locations_state,
            memory_objects=self.solution_space.memory_objects,
            ranked_plans=validation_result.get("filtered_ranked_plans", []),
            selected_action=f"Action {action_id} -> {action_str}"
        )
        
        # Check if navigate action to update current_location
        if "navigate to" in action_str.lower():
            target_loc = action_str.lower().replace("navigate to the", "").replace("navigate to", "").strip()
            self.current_location = target_loc
            
        return validation_result
