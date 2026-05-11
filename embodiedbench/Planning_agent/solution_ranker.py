import json
from .llm_client import LLMClient
from .prompt_templates import SOLUTION_RANKING_SYSTEM_PROMPT, SOLUTION_RANKING_USER_PROMPT
from .logger_utils import AgentLogger

class SolutionRanker:
    def __init__(self, llm_client: LLMClient, logger: AgentLogger):
        self.llm = llm_client
        self.logger = logger

    def generate_and_rank_solutions(self, global_intent: dict, solution_space: dict, current_location: str, visited_locations: dict, unvisited_locations: list, step: int, remaining_steps: int) -> dict:
        """
        方案生成与多维排序机制
        """
        legal_combinations = solution_space.get("legal_combinations", {})
        
        # 将合法的组合格式化，便于 LLM 理解
        combinations_str = "\n".join([f"ID {k}: {v}" for k, v in legal_combinations.items()])

        user_prompt = SOLUTION_RANKING_USER_PROMPT.format(
            global_intent=json.dumps(global_intent, ensure_ascii=False),
            current_location=current_location,
            visited_locations=visited_locations,
            unvisited_locations=unvisited_locations,
            memory_objects=json.dumps(solution_space.get("memory_objects", {}), ensure_ascii=False),
            remaining_steps=remaining_steps,
            legal_combinations=combinations_str
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
        
        # 确保选出的动作 ID 确实存在且合法
        selected_id = result_dict.get("selected_action_id", 0)
        
        if filtered_plans and filtered_plans[0].get("first_action_id") in legal_combinations:
            selected_id = filtered_plans[0].get("first_action_id")
        elif selected_id not in legal_combinations:
            # Fallback
            selected_id = list(legal_combinations.keys())[0] if legal_combinations else 0

        final_output = {
            "all_generated_plans": result_dict.get("plans", []),
            "filtered_ranked_plans": filtered_plans,
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
