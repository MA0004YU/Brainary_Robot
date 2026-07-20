import json
from .llm_client import LLMClient
from .prompt_templates import GOAL_REASONER_SYSTEM_PROMPT, GOAL_REASONER_USER_PROMPT

class GoalReasoner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_intent(self, instruction: str) -> dict:
        user_prompt = GOAL_REASONER_USER_PROMPT.format(instruction=instruction)
        
        print(f"GoalReasoner evaluating instruction: '{instruction}'")
        
        result_dict = self.llm.generate_json(
            system_prompt=GOAL_REASONER_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        if result_dict:
            return result_dict
        else:
            print("GoalReasoner failed to extract intent.")
            return {}
