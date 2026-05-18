from .llm_client import LLMClient
from .logger_utils import AgentLogger
from .ltm_client import LTMClient

class MemoryManager:
    def __init__(self, llm_client: LLMClient, logger: AgentLogger, **kwargs):
        self.llm = llm_client
        self.logger = logger
        self.ltm = LTMClient()

    def get_full_memory_context(self) -> str:
        """获取长期记忆的完整上下文，用于传递给验证模块"""
        ltm_memory = self.ltm.query("What are the key facts, failed attempts, and environment rules discovered so far?")
        if ltm_memory:
            return f"=== Long-term Memory ===\n{ltm_memory}\n"
        return "=== Long-term Memory ===\nNo prior knowledge found.\n"

    def save_session_state(self, scene_id: str, global_intent: dict, visited_locations: dict, known_objects: dict, action_history: list, last_agent_message: str, log_dir: str):
        """保存任务检查点至 LTM"""
        state_desc = f"[EPISODE CHECKPOINT] Scene: {scene_id}, Intent: {global_intent}. Last message to user: {last_agent_message}. Visited: {list(visited_locations.keys())}."
        self.ltm.insert(state_desc)
        self.logger.info("[MemoryManager] Session checkpoint saved to LTM.")
