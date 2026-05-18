import json
from .llm_client import LLMClient
from .prompt_templates import SOLUTION_RANKING_SYSTEM_PROMPT, SOLUTION_RANKING_USER_PROMPT
from .logger_utils import AgentLogger

def is_action_aligned(plan_desc: str, action_str: str) -> bool:
    plan_desc_lower = plan_desc.lower()
    action_str_lower = action_str.lower()
    
    # 1. 动词类别匹配
    verb_categories = ["pick", "place", "put", "navigate", "nav", "open", "close"]
    matched_verb = None
    for verb in verb_categories:
        if verb in action_str_lower:
            matched_verb = verb
            break
            
    if matched_verb:
        if matched_verb in ["navigate", "nav"]:
            if "nav" not in plan_desc_lower and "go to" not in plan_desc_lower and "move" not in plan_desc_lower and "travel" not in plan_desc_lower and "explore" not in plan_desc_lower:
                return False
        elif matched_verb == "pick":
            if "pick" not in plan_desc_lower and "take" not in plan_desc_lower and "get" not in plan_desc_lower and "grab" not in plan_desc_lower:
                return False
        elif matched_verb in ["open", "close"]:
            if matched_verb not in plan_desc_lower:
                return False
                
    # 2. 关键名词实体匹配
    stopwords = {"the", "a", "an", "to", "in", "at", "of", "up", "on", "from", "with", "navigate", "pick", "place", "put", "open", "close", "nav"}
    action_words = [w for w in action_str_lower.split() if w not in stopwords]
    
    if not action_words:
        return True
        
    has_match = False
    for word in action_words:
        if word.isdigit():
            if word in plan_desc_lower:
                has_match = True
        else:
            if word in plan_desc_lower:
                has_match = True
                
    return has_match

OPEN_PRIORITY_SCORE_BOOST = 40
NAVIGATE_AWAY_PENALTY = 25
VISIBLE_RELEVANT_PICK_SCORE = 88
TARGET_VISIBLE_PICK_SCORE = 95

class SolutionRanker:
    def __init__(self, llm_client: LLMClient, logger: AgentLogger):
        self.llm = llm_client
        self.logger = logger

    def _apply_open_priority_boost(self, plans: list, legal_combinations: dict, exploration_context: dict) -> list:
        """当前位置有待打开容器时，提升 open 方案分数、压低离开现场的 navigate。"""
        pending_loc = (exploration_context or {}).get("pending_open_location")
        pending_aid = (exploration_context or {}).get("pending_open_action_id")
        if not pending_loc:
            return plans

        boosted = []
        for plan in plans:
            plan = dict(plan)
            aid = plan.get("first_action_id")
            action_str = (legal_combinations.get(aid) or "").lower()
            score = plan.get("rank_score", 0)

            if aid == pending_aid or (
                action_str.startswith("open ") and pending_loc in action_str
            ):
                score += OPEN_PRIORITY_SCORE_BOOST
                plan["open_priority_boost"] = True
            elif action_str.startswith("nav_") or "navigate to" in action_str:
                score -= NAVIGATE_AWAY_PENALTY
                plan["navigate_away_penalty"] = True

            plan["rank_score"] = score
            boosted.append(plan)

        boosted.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        self.logger.info(
            f"[Open Priority] Boosted open-at-'{pending_loc}' plans; "
            f"top score now {boosted[0].get('rank_score') if boosted else 'n/a'}"
        )
        return boosted

    @staticmethod
    def _find_pick_action_for_object(obj: str, legal_combinations: dict):
        obj_lower = obj.lower().strip()
        for aid, cap in legal_combinations.items():
            if aid == 9999:
                continue
            cap_lower = (cap or "").lower()
            if not cap_lower.startswith("pick "):
                continue
            picked = (
                cap_lower.replace("pick up the ", "")
                .replace("pick up ", "")
                .replace("pick the ", "")
                .replace("pick ", "")
                .strip()
            )
            if picked == obj_lower or obj_lower in picked or picked in obj_lower:
                return aid
        return None

    def _inject_visible_relevant_pick_plans(
        self,
        plans: list,
        solution_space: dict,
        legal_combinations: dict,
        global_intent: dict,
        held_object: str,
    ) -> list:
        """
        当 relevant_objects ∩ visible 非空时，为每个可见替代品强制插入 pick 方案参与打分。
        不覆盖 Top-1 选择逻辑，仅保证 LLM 漏掉的 pick 能进入候选池。
        """
        if held_object:
            return plans

        relevant = solution_space.get("relevant_objects") or []
        visible = solution_space.get("visible_objects") or []
        visible_lower = {v.lower() for v in visible}
        intersection = [o for o in relevant if o.lower() in visible_lower]
        if not intersection:
            return plans

        plans = [dict(p) for p in plans]
        existing_ids = {p.get("first_action_id") for p in plans}
        target = (global_intent.get("target_object") or "").lower()

        for obj in intersection:
            pick_id = self._find_pick_action_for_object(obj, legal_combinations)
            if pick_id is None:
                self.logger.info(
                    f"[Pick Inject] Visible relevant '{obj}' has no matching pick in legal_combinations"
                )
                continue
            if pick_id in existing_ids:
                continue

            if target and (target in obj.lower() or obj.lower() in target):
                score = TARGET_VISIBLE_PICK_SCORE
            else:
                score = VISIBLE_RELEVANT_PICK_SCORE

            plans.append({
                "plan_description": f"Pick up the {obj} (visible and listed in Relevant Objects).",
                "first_action_id": pick_id,
                "success_probability": "High",
                "step_overhead_estimate": 1,
                "budget_check": "Pass",
                "rank_score": score,
                "injected_visible_relevant_pick": True,
            })
            existing_ids.add(pick_id)
            self.logger.info(
                f"[Pick Inject] Inserted pick plan for '{obj}' -> action {pick_id} (score={score})"
            )

        plans.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        return plans

    def generate_and_rank_solutions(self, global_intent: dict, solution_space: dict, current_location: str, visited_locations: dict, unvisited_locations: list, step: int, remaining_steps: int, held_object: str = None) -> dict:
        """
        方案生成与多维排序机制
        """
        legal_combinations = solution_space.get("legal_combinations", {})
        exploration_context = solution_space.get("exploration_context") or {}
        
        # 将合法的组合格式化，便于 LLM 理解
        combinations_str = "\n".join([f"ID {k}: {v}" for k, v in legal_combinations.items()])

        user_prompt = SOLUTION_RANKING_USER_PROMPT.format(
            global_intent=json.dumps(global_intent, ensure_ascii=False),
            current_location=current_location,
            visited_locations=visited_locations,
            unvisited_locations=unvisited_locations,
            memory_objects=json.dumps(solution_space.get("memory_objects", {}), ensure_ascii=False),
            remaining_steps=remaining_steps,
            legal_combinations=combinations_str,
            held_object=held_object or "None",
            relevant_objects=json.dumps(solution_space.get("relevant_objects", []), ensure_ascii=False),
            exploration_context=json.dumps(exploration_context, ensure_ascii=False),
        )
        
        result_dict = self.llm.generate_json(
            system_prompt=SOLUTION_RANKING_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        if not result_dict:
            result_dict = {
                "plans": [],
                "reasoning": "LLM failed to generate ranking.",
                "selected_action_id": 0
            }

        # 可见且相关的替代品：强制插入 pick 方案（若 legal 中存在对应 pick）
        result_dict["plans"] = self._inject_visible_relevant_pick_plans(
            result_dict.get("plans", []),
            solution_space,
            legal_combinations,
            global_intent,
            held_object,
        )
        
        # Budget Check & Filtering (预算限制截断)
        filtered_plans = []
        for plan in result_dict.get("plans", []):
            estimated_steps = plan.get("step_overhead_estimate", 999)
            if estimated_steps <= remaining_steps:
                plan["budget_check"] = "Pass"
                filtered_plans.append(plan)
            else:
                plan["budget_check"] = "Fail (Exceeds Remaining Steps)"
                
        # 重新根据 rank_score 排序 (降序)
        filtered_plans.sort(key=lambda x: x.get("rank_score", 0), reverse=True)

        # 当前位置有待打开容器：提升 open、压低离开现场的 navigate
        filtered_plans = self._apply_open_priority_boost(
            filtered_plans, legal_combinations, exploration_context
        )
        if result_dict.get("plans"):
            result_dict["plans"] = self._apply_open_priority_boost(
                result_dict["plans"], legal_combinations, exploration_context
            )
        
        # 语义对齐校验与自动纠错机制 (Semantic Alignment & Auto-Healing)
        aligned_filtered_plans = []
        for plan in filtered_plans:
            plan_id = plan.get("first_action_id")
            if plan_id in legal_combinations:
                action_str = legal_combinations[plan_id]
                if is_action_aligned(plan.get("plan_description", ""), action_str):
                    aligned_filtered_plans.append(plan)
                else:
                    self.logger.info(f"[Semantic Alignment] Rejected plan '{plan.get('plan_description')}' with ID {plan_id} ('{action_str}') due to mismatch.")

        selected_id = None
        
        # 1. 优先从完美对齐的过滤计划中选择 Top-1
        if aligned_filtered_plans:
            selected_id = aligned_filtered_plans[0].get("first_action_id")
            self.logger.info(f"[Alignment Success] Selected aligned Plan: {aligned_filtered_plans[0].get('plan_description')} -> Action {selected_id}")
        else:
            # 2. 自动纠错 (Auto-Healing): 遍历合法动作列表，寻找能够对齐 Top-1 方案物理描述的真实 ID
            if filtered_plans:
                top_plan = filtered_plans[0]
                top_plan_desc = top_plan.get("plan_description", "")
                self.logger.info(f"[Auto-Healing] LLM hallucinated ID for plan '{top_plan_desc}'. Scanning legal combinations...")
                
                best_match_id = None
                for action_id, action_str in legal_combinations.items():
                    if is_action_aligned(top_plan_desc, action_str):
                        best_match_id = action_id
                        self.logger.info(f"[Auto-Healing] Fixed to matching action: ID {action_id} -> '{action_str}'")
                        break
                
                if best_match_id is not None:
                    selected_id = best_match_id
                    # 同步更新过滤后的计划列表以保持输出一致
                    top_plan["first_action_id"] = selected_id
                    aligned_filtered_plans.append(top_plan)

        # 3. 终极兜底
        if selected_id is None or selected_id not in legal_combinations:
            selected_id = list(legal_combinations.keys())[0] if legal_combinations else 0
            self.logger.info(f"[Alignment Fallback] No matching action found. Fallback to Action {selected_id}")

        # 4. 待打开容器硬约束：有 pending open 时强制选 open
        pending_aid = exploration_context.get("pending_open_action_id")
        if pending_aid is not None and pending_aid in legal_combinations:
            selected_action = (legal_combinations.get(selected_id) or "").lower()
            if not selected_action.startswith("open "):
                self.logger.info(
                    f"[Open Priority] Overriding action {selected_id} -> {pending_aid} "
                    f"({legal_combinations[pending_aid]})"
                )
                selected_id = pending_aid

        final_output = {
            "all_generated_plans": result_dict.get("plans", []),
            "filtered_ranked_plans": aligned_filtered_plans,
            "reasoning": result_dict.get("reasoning", ""),
            "selected_action_id": selected_id,
            "communication_to_user": result_dict.get("communication_to_user")
        }

        self.logger.log_module_output(
            module_name="SolutionRanker (Solution Generation & Ranking)",
            step=step,
            output_data=json.dumps(final_output, indent=2, ensure_ascii=False)
        )
        
        return final_output
