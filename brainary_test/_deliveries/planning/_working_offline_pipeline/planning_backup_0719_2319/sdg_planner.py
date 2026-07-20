import json
from .llm_client import LLMClient
from .prompt_templates import SDG_SYSTEM_PROMPT, SDG_USER_PROMPT

class SDGPlanner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_sdg(self, goal_intent: dict, constraints: dict) -> dict:
        user_prompt = SDG_USER_PROMPT.format(
            goal_intent=json.dumps(goal_intent, ensure_ascii=False, indent=2),
            constraints=json.dumps(constraints, ensure_ascii=False, indent=2)
        )
        
        print("SDGPlanner generating abstract State Dependency Graph...")
        
        sdg_data = self.llm.generate_json(
            system_prompt=SDG_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        if sdg_data:
            return sdg_data
        else:
            print("SDG Planner failed to generate SDG.")
            return {}
