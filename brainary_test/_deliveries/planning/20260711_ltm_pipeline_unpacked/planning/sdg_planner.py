import json
from .llm_client import LLMClient
from .prompt_templates import SDG_SYSTEM_PROMPT, SDG_USER_PROMPT
from .ltm_client import LTMClient

class SDGPlanner:
    def __init__(self, llm_client: LLMClient, use_ltm: bool = False):
        self.llm = llm_client
        self.use_ltm = use_ltm
        self.ltm_client = LTMClient() if use_ltm else None

    def generate_sdg(self, goal_intent: dict, constraints: dict) -> dict:
        ltm_context = ""
        if self.use_ltm and self.ltm_client:
            try:
                intent_str = goal_intent.get("deep_intent", str(goal_intent))
                ltm_result = self.ltm_client.query(intent_str)
                if ltm_result:
                    ltm_context = f"\nRelevant long-term memory context:\n{ltm_result}\n"
            except Exception as e:
                print(f"[LTMClient] Query exception: {e}")

        user_prompt = SDG_USER_PROMPT.format(
            goal_intent=json.dumps(goal_intent, ensure_ascii=False, indent=2),
            constraints=json.dumps(constraints, ensure_ascii=False, indent=2),
            ltm_context=ltm_context
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
