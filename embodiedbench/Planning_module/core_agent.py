import json
from .llm_client import LLMClient
from .logger_utils import AgentLogger
from .goal_reasoner import GoalReasoner
from .solution_space import SolutionSpaceAnalyzer
from .task_validator import TaskValidator
from .memory_manager import MemoryManager

class IntentReasoningAgent:
    def __init__(self, episode_id=None, model_name="gpt-4o-mini", scene_id=None, inject_priors=False):
        self.logger = AgentLogger(episode_id)
        self.llm = LLMClient(model_name=model_name)

        self.goal_reasoner = GoalReasoner(self.llm, self.logger)
        self.solution_space = SolutionSpaceAnalyzer(self.llm, self.logger)
        self.task_validator = TaskValidator(self.llm, self.logger)
        self.memory_manager = MemoryManager(self.llm, self.logger)

        self.global_intent = {}
        self.action_history = []
        self.current_location = "unknown starting location"
        self.held_object = None
        self.inject_priors = inject_priors
        self.failed_actions = {}
        self.visited_locations = []

        self.scene_id = scene_id if scene_id else (episode_id if episode_id else "unknown_scene")

    @staticmethod
    def _fallback_action_id(legal_combinations: dict, prefer_place: bool = False) -> int:
        if not legal_combinations:
            return 0
        if prefer_place:
            for aid, desc in legal_combinations.items():
                dl = (desc or "").lower()
                if dl.startswith("place ") or dl.startswith("put ") or dl.startswith("drop "):
                    return aid
        for aid in sorted(k for k in legal_combinations.keys() if k != 9999):
            return aid
        return next(iter(legal_combinations.keys()))

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
        # 环境反馈格式: 失败 → "Last action is invalid..."  成功 → "Last action executed successfully..."
        action_failed = "last action is invalid" in observation_text.lower()
        
        if action_failed:
            if len(self.action_history) > 0:
                failed_action = self.action_history.pop()
                self.logger.info(f"[Error Recovery] Last action '{failed_action}' failed. Removed from history.")
                
                # 记录失败次数
                self.failed_actions[failed_action] = self.failed_actions.get(failed_action, 0) + 1
                
                # 如果是拾取失败，且连续失败达 2 次以上，判定为视觉误判/物理死锁，从记忆和 LTM 中抹除该物品
                if failed_action.lower().startswith("pick ") and self.failed_actions[failed_action] >= 2:
                    obj = failed_action.lower().replace("pick up the ", "").replace("pick up ", "").replace("pick the ", "").replace("pick ", "").strip()
                    if obj in self.solution_space.memory_objects:
                        del self.solution_space.memory_objects[obj]
                    self.memory_manager.ltm.update_state(f"[STATE UPDATE] The object '{obj}' at '{self.current_location}' was a hallucination and DOES NOT EXIST. Please remove it from memory.")
                    self.logger.info(f"[Error Recovery] Sent STATE UPDATE to delete hallucinated object '{obj}'.")
        else:
            # 动作成功执行，重置所有失败计数器（因为环境/物理状态已发生改变）
            self.failed_actions.clear()
            
            # 如果是状态突变动作，写入 LTM 情景记忆 (Episodic Memory)
            if step_idx > 0 and len(self.action_history) > 0:
                last_act = self.action_history[-1].lower()
                state_msg = ""
                
                if last_act.startswith("nav_") or last_act.startswith("navigate "):
                    target_loc = last_act.replace("navigate to the ", "").replace("navigate to ", "").replace("nav_", "").strip()
                    # 判断当前地点是否为无需 open 的表面（如桌子），如果是，到达即视为 fully explored
                    has_open_action = False
                    for cap in skill_set:
                        if cap.lower().startswith("open ") and (self.current_location.lower() in cap.lower() or cap.lower().replace("open the ", "").strip() in self.current_location.lower()):
                            has_open_action = True
                            break
                    if not has_open_action:
                        state_msg = f"[STATE UPDATE] Agent navigated to and fully explored '{target_loc}'."
                    else:
                        state_msg = f"[STATE UPDATE] Agent navigated to '{target_loc}'. It has closed compartments that need to be opened to be explored."
                elif last_act.startswith("open "):
                    target_loc = last_act.replace("open the ", "").replace("open ", "").strip()
                    visible_objs = ", ".join([f"'{obj}'" for obj in self.solution_space.current_visible_objects])
                    if visible_objs:
                        state_msg = f"[STATE UPDATE] Agent fully explored '{target_loc}' by opening it. Inside: {visible_objs}."
                    else:
                        state_msg = f"[STATE UPDATE] Agent fully explored '{target_loc}' by opening it. It is completely empty."
                elif last_act.startswith("close "):
                    target_loc = last_act.replace("close the ", "").replace("close ", "").strip()
                    state_msg = f"[STATE UPDATE] Agent closed '{target_loc}'."
                elif last_act.startswith("pick "):
                    obj = last_act.replace("pick up the ", "").replace("pick up ", "").strip()
                    state_msg = f"[STATE UPDATE] Agent picked up '{obj}' from '{self.current_location}'."
                elif last_act.startswith("place ") or last_act.startswith("put ") or last_act.startswith("drop "):
                    obj = self.held_object if self.held_object else "object"
                    state_msg = f"[STATE UPDATE] Agent placed '{obj}' at '{self.current_location}'."
                    
                if state_msg:
                    self.memory_manager.ltm.update_state(state_msg)
                    self.logger.info(f"[CoreAgent] Sent EPISODIC UPDATE to LTM: {state_msg}")
        
        # 1. 目标提取 (仅在第一步执行)
        if step_idx == 0 and not self.global_intent:
            self.global_intent = self.goal_reasoner.extract_intent(instruction)
            
        # 因为 LTM (GraphRAG) 的全局索引更新有异步延迟，
        # 我们在本地通过 action_history 精准维护一套已探索地点作为 Short-Term Memory 补充。
        # 规则：
        # - 如果执行过 nav_X 或 navigate to X：
        #   - 如果该地点无需开门（如 counter, table, sofa, sink），直接视为 visited_closed
        #   - 如果该地点是个容器，必须在 action_history 中同时包含 open X，才视为 visited_open (即完全探索完)
        visited_locations_state = {}
        
        # 1. 找出所有去过的地点
        navigated_locs = []
        for action in self.action_history:
            action_lower = action.lower()
            if action_lower.startswith("nav_") or "navigate to" in action_lower:
                loc = action_lower.replace("navigate to the ", "").replace("navigate to ", "").replace("nav_", "").strip()
                if loc not in navigated_locs:
                    navigated_locs.append(loc)

        # 2. 判定每个去过的地点是否已经探索完 (open 了，或者根本不是容器)
        # 获取所有可 open 的地点前缀集合
        openable_locs = set()
        for cap in skill_set:
            if cap.lower().startswith("open "):
                openable_locs.add(cap.lower().replace("open the ", "").replace("open ", "").strip())

        # 3. 实时判断状态
        for loc in navigated_locs:
            if loc in openable_locs:
                # 是个容器，检查历史中是否有过 open
                has_opened = False
                for action in self.action_history:
                    if action.lower().startswith("open ") and (loc in action.lower() or action.lower().replace("open the ", "").replace("open ", "").strip() in loc):
                        has_opened = True
                        break
                if has_opened:
                    visited_locations_state[loc] = "visited_open"
            else:
                # 只是普通表面，到达即完全探索完
                visited_locations_state[loc] = "visited_open"

        # 解析 held_object 并推断容器开关状态 (receptacle_states)
        self.held_object = None
        receptacle_states = {}  # {"refrigerator": "open", "cabinet 1": "closed", ...}
        for action in self.action_history:
            action_lower = action.lower()
            if action_lower.startswith("pick "):
                obj = action_lower.replace("pick up the ", "").replace("pick up ", "").replace("pick the ", "").replace("pick ", "").strip()
                self.held_object = obj
            elif "place " in action_lower or "put " in action_lower or "drop " in action_lower:
                self.held_object = None
            elif action_lower.startswith("open "):
                loc = action_lower.replace("open the ", "").replace("open ", "").strip()
                receptacle_states[loc] = "open"
            elif action_lower.startswith("close "):
                loc = action_lower.replace("close the ", "").replace("close ", "").strip()
                receptacle_states[loc] = "closed"
                
        # 在 self.visited_locations 被单步临时字典覆盖之前，先创建一个真正的历史已探索地点快照，
        # 以便在 Step 结尾施加【记忆单调递增约束】合并时使用，防止记忆随着 LTM 注意力稀释而减少！
        historical_explored_snapshot = list(self.visited_locations) if hasattr(self, "visited_locations") and self.visited_locations else []
        
        self.visited_locations = visited_locations_state
        self.logger.info(f"[Visited Locations State] {visited_locations_state}")
        self.logger.info(f"[Inventory] held_object: {self.held_object}")
        self.logger.info(f"[Receptacle States] {receptacle_states}")

        # 2. 解空间更新
        self.solution_space.update_capabilities(skill_set, inject_priors=self.inject_priors)
        # 生成当前黑名单（连续失败 >= 2 次的动作直接被屏蔽）
        blacklisted = [act for act, count in self.failed_actions.items() if count >= 2]
        
        self.solution_space.update_observation(
            observation_text=observation_text, 
            current_location=self.current_location,
            visited_locations_state=visited_locations_state,
            step=step_idx, 
            global_intent=self.global_intent,
            img_path=img_path, 
            dual_img_path=dual_img_path,
            held_object=self.held_object,
            receptacle_states=receptacle_states,
            blacklisted_actions=blacklisted
        )
        # 从 LTM 同步获取最新已探索地点列表，并与本地通过 action_history 推断的已探索地点取并集，
        # 从而完美抵消 GraphRAG 的异步索引延迟。
        # 此外，为了防止大模型在 LTM 长期记忆中对旧节点产生注意力稀释（遗忘），
        # 我们强行引入【记忆单调递增约束】：与上一轮保存的历史已探索列表取并集，确保记忆绝对只增不减！
        ltm_explored = self.solution_space.world_model.get("explored_locations", [])
        local_explored = list(visited_locations_state.keys())
        current_step_explored = list(set(ltm_explored) | set(local_explored))
        
        # 融入历史记忆快照
        historical_explored = historical_explored_snapshot if isinstance(historical_explored_snapshot, list) else list(historical_explored_snapshot)
        merged_explored = list(set(current_step_explored) | set(historical_explored))
        
        self.visited_locations = merged_explored
        self.logger.info(f"[Visited Locations State Updated] {merged_explored} (LTM: {ltm_explored}, Local: {local_explored}, History Snapshot: {historical_explored})")
        
        # 将合并后的已探索地点同步回 solution_space 的 world_model，以确保内部约束（如 _apply_combination_constraints）过滤准确
        self.solution_space.world_model["explored_locations"] = merged_explored

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
            persistent_memory_text=self.memory_manager.get_full_memory_context(),
            held_object=self.held_object,
        )
        
        # 4. 提取动作并更新状态
        action_id = validation_result.get("action_id", 0)
        
        if action_id == 9999:
            # 允许 9999：手持物在 LLM relevant_objects 中，或已探索全部地点且双手为空
            all_locs = list(self.solution_space.composition_elements.get("locations", []))
            all_explored = len(self.visited_locations) >= len(all_locs) if all_locs else False
            relevant_lower = [o.lower() for o in self.solution_space.relevant_objects]
            held_ok = bool(self.held_object and self.held_object.lower() in relevant_lower)
            legal = current_solution_space.get("legal_combinations", {})

            if self.held_object and not held_ok:
                self.logger.info(
                    f"[CoreAgent] Intercepted Action 9999: holding irrelevant '{self.held_object}'"
                )
                action_id = self._fallback_action_id(legal, prefer_place=True)
                validation_result["action_id"] = action_id
            elif not self.held_object and not all_explored:
                self.logger.info(
                    "[CoreAgent] Intercepted Action 9999: not holding a valid alternative "
                    "and exploration is incomplete."
                )
                action_id = self._fallback_action_id(legal)
                validation_result["action_id"] = action_id
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
            visited_locations=self.solution_space.world_model.get("explored_locations", []),
            memory_objects=self.solution_space.memory_objects,
            ranked_plans=validation_result.get("ranked_plans", []),
            selected_action=f"Action {action_id} -> {action_str}"
        )
        
        # Check if navigate action to update current_location
        if "navigate to" in action_str.lower():
            target_loc = action_str.lower().replace("navigate to the", "").replace("navigate to", "").strip()
            self.current_location = target_loc
            
        return validation_result
