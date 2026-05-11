import json
from .llm_client import LLMClient
from .prompt_templates import GOAL_REASONER_SYSTEM_PROMPT, GOAL_REASONER_USER_PROMPT
from .logger_utils import AgentLogger

class GoalReasoner:
    def __init__(self, llm_client: LLMClient, logger: AgentLogger):
        self.llm = llm_client
        self.logger = logger
        self.global_intent = {}

    def extract_intent(self, instruction: str) -> dict:
        """
        根据初始指令，使用 5 Whys 提取全局深度意图
        """
        user_prompt = GOAL_REASONER_USER_PROMPT.format(instruction=instruction)
        
        self.logger.info(f"GoalReasoner evaluating instruction: '{instruction}'")
        
        result_dict = self.llm.generate_json(
            system_prompt=GOAL_REASONER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_override="gpt-4o"
        )
        
        if result_dict:
            self.global_intent = result_dict
            self.logger.log_module_output(
                module_name="GoalReasoner (Module A)",
                step=0,
                output_data=json.dumps(self.global_intent, indent=2, ensure_ascii=False)
            )
        else:
            self.logger.error("GoalReasoner failed to extract intent.")
            
        return self.global_intent

    def get_global_intent(self):
        return self.global_intent
