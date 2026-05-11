import json
from .llm_client import LLMClient
from .logger_utils import AgentLogger
from .solution_ranker import SolutionRanker

class TaskValidator:
    def __init__(self, llm_client: LLMClient, logger: AgentLogger):
        self.llm = llm_client
        self.logger = logger
        self.ranker = SolutionRanker(llm_client, logger)

    def validate_action_feasibility(self, global_intent: dict, solution_space: dict, action_history: list, visited_locations: dict, current_location: str, step: int, max_steps: int = 50, persistent_memory_text: str = "") -> dict:
        """
        作为决策执行器，调用 SolutionRanker 拿到排序后的方案列表，并选择 Top-1 方案的首个动作
        """
        remaining_steps = max_steps - step
        
        all_locations = solution_space.get("refined_items", {}).get("all_locations", [])
        unvisited_locations = [loc for loc in all_locations if loc not in visited_locations]
        
        # 传递给 Ranker 进行多维评估和排序
        ranking_result = self.ranker.generate_and_rank_solutions(
            global_intent=global_intent,
            solution_space=solution_space,
            current_location=current_location,
            visited_locations=visited_locations,
            unvisited_locations=unvisited_locations,
            step=step,
            remaining_steps=remaining_steps
        )
        
        action_id = ranking_result.get("selected_action_id", 0)
        
        result_dict = {
            "action_id": action_id,
            "reasoning": ranking_result.get("reasoning", "No reasoning provided."),
            "communication_to_user": ranking_result.get("communication_to_user"),
            "ranked_plans": ranking_result.get("filtered_ranked_plans", [])
        }
        
        self.logger.info(f"[Step {step}] TaskValidator accepted Top Ranked Action ID: {action_id}")
            
        return result_dict
