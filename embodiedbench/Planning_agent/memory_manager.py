import os
import re
from .llm_client import LLMClient
from .logger_utils import AgentLogger
from .prompt_templates import MEMORY_UPDATE_SYSTEM_PROMPT, MEMORY_UPDATE_USER_PROMPT

class MemoryManager:
    def __init__(self, llm_client: LLMClient, logger: AgentLogger, 
                 persistent_file: str = "persistent_memory.md",
                 session_file: str = "session_context.md"):
        self.llm = llm_client
        self.logger = logger
        self.persistent_file = os.path.join(os.path.dirname(__file__), persistent_file)
        self.session_file = os.path.join(os.path.dirname(__file__), session_file)
        
        # Ensure files exist
        if not os.path.exists(self.persistent_file):
            with open(self.persistent_file, 'w', encoding='utf-8') as f:
                f.write("# User Persistent Memory\n\n## Preferences\n- 暂无偏好记录。\n\n## Scene Knowledge\n")
                
        if not os.path.exists(self.session_file):
            self.clear_session_context()

    def clear_session_context(self):
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write("# Current Session Context\n\n## State\n- **scene_id**: None\n- **log_dir**: None\n- **global_intent**: {}\n- **action_history**: None\n\n## Dialogue History\n")

    def read_persistent_memory(self) -> str:
        with open(self.persistent_file, 'r', encoding='utf-8') as f:
            return f.read()

    def write_persistent_memory(self, content: str):
        with open(self.persistent_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def read_session_context(self) -> str:
        if not os.path.exists(self.session_file): return ""
        with open(self.session_file, 'r', encoding='utf-8') as f:
            return f.read()

    def write_session_context(self, content: str):
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def get_full_memory_context(self) -> str:
        return f"=== Persistent Memory ===\n{self.read_persistent_memory()}\n\n=== Session Context ===\n{self.read_session_context()}"

    def append_to_dialogue(self, text: str):
        session_text = self.read_session_context()
        if "## Dialogue History" not in session_text:
            session_text += "\n## Dialogue History\n"
        session_text += f"{text}\n"
        self.write_session_context(session_text)

    def process_feedback(self, feedback_text: str) -> dict:
        self.append_to_dialogue(f"- [User]: {feedback_text}")
        
        current_persistent = self.read_persistent_memory()
        current_session = self.read_session_context()
        
        user_prompt = MEMORY_UPDATE_USER_PROMPT.format(
            current_persistent=current_persistent,
            current_session=current_session,
            feedback_text=feedback_text
        )
        
        self.logger.info(f"[MemoryManager] Processing user feedback: {feedback_text}")
        
        result_dict = self.llm.generate_json(
            system_prompt=MEMORY_UPDATE_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        if result_dict:
            new_memory_markdown = result_dict.get("new_memory_markdown", "")
            new_global_intent = result_dict.get("new_global_intent", {})
            if new_memory_markdown:
                self.write_persistent_memory(new_memory_markdown)
                self.logger.info("[MemoryManager] Persistent memory updated.")
            return new_global_intent
        else:
            self.logger.error("[MemoryManager] Failed to process feedback.")
            return {}

    def _update_scene_knowledge(self, scene_id: str, visited_locations: list, known_objects: dict):
        # 暂时禁用场景先验记忆的持久化
        return
        persistent_text = self.read_persistent_memory()
        
        scene_pattern = re.compile(rf'### {re.escape(scene_id)}\n(.*?)(?=\n### |\Z)', re.DOTALL)
        
        v_loc_str = ', '.join(visited_locations) if visited_locations else 'None'
        k_obj_str = ""
        if known_objects:
            for obj, loc in known_objects.items():
                k_obj_str += f"  - {loc}: {obj}\n"
        else:
            k_obj_str = "  - None\n"
            
        new_scene_block = f"### {scene_id}\n- **visited_locations**: {v_loc_str}\n- **known_objects**:\n{k_obj_str}"
        
        if scene_pattern.search(persistent_text):
            persistent_text = scene_pattern.sub(new_scene_block.replace('\\', '\\\\'), persistent_text, count=1)
        else:
            if "## Scene Knowledge" not in persistent_text:
                persistent_text += "\n## Scene Knowledge\n"
            persistent_text += f"\n{new_scene_block}\n"
            
        self.write_persistent_memory(persistent_text)

    def save_session_state(self, scene_id: str, global_intent: dict, visited_locations: list, known_objects: dict, action_history: list, last_agent_message: str, log_dir: str):
        if scene_id and scene_id != "unknown_scene":
            self._update_scene_knowledge(scene_id, visited_locations, known_objects)
        
        session_text = self.read_session_context()
        dialogue_part = ""
        if "## Dialogue History" in session_text:
            dialogue_part = session_text[session_text.find("## Dialogue History"):]
        else:
            dialogue_part = "## Dialogue History\n"
            
        if last_agent_message:
            dialogue_part += f"- [Agent]: {last_agent_message}\n"
            
        state_part = f"""# Current Session Context

## State
- **scene_id**: {scene_id}
- **log_dir**: {log_dir}
- **global_intent**: {global_intent}
- **action_history**: {', '.join(action_history) if action_history else 'None'}

"""
        self.write_session_context(state_part + dialogue_part)
        self.logger.info("[MemoryManager] Session state saved to disk.")

    def load_session_state(self, fallback_scene_id: str = None) -> dict:
        session_text = self.read_session_context()
        persistent_text = self.read_persistent_memory()
        
        scene_id_match = re.search(r'- \*\*scene_id\*\*: (.*)', session_text)
        log_dir_match = re.search(r'- \*\*log_dir\*\*: (.*)', session_text)
        intent_match = re.search(r'- \*\*global_intent\*\*: (.*)', session_text)
        history_match = re.search(r'- \*\*action_history\*\*: (.*)', session_text)
        
        scene_id = scene_id_match.group(1).strip() if scene_id_match else ""
        if (not scene_id or scene_id == "None") and fallback_scene_id:
            scene_id = fallback_scene_id
        log_dir = log_dir_match.group(1).strip() if log_dir_match else ""
        
        try:
            import ast
            global_intent = ast.literal_eval(intent_match.group(1).strip()) if intent_match else {}
        except:
            global_intent = {}
            
        history_str = history_match.group(1).strip() if history_match else ""
        action_history = [x.strip() for x in history_str.split(',')] if history_str and history_str != 'None' else []
        
        visited_locations = []
        known_objects = {}
        
        # 暂时禁用场景先验记忆的加载
        # if scene_id and scene_id != "unknown_scene" and scene_id != "None":
        #     scene_pattern = re.compile(rf'### {re.escape(scene_id)}\n(.*?)(?=\n### |\Z)', re.DOTALL)
        #     scene_match = scene_pattern.search(persistent_text)
        #     if scene_match:
        #         scene_block = scene_match.group(1)
        #         v_loc_match = re.search(r'- \*\*visited_locations\*\*: (.*)', scene_block)
        #         if v_loc_match:
        #             v_str = v_loc_match.group(1).strip()
        #             if v_str and v_str != 'None':
        #                 visited_locations = [x.strip() for x in v_str.split(',')]
        #                 
        #         obj_match = re.search(r'- \*\*known_objects\*\*:\n((?:  - .*\n?)*)', scene_block)
        #         if obj_match:
        #             lines = obj_match.group(1).strip().split('\n')
        #             for line in lines:
        #                 if ':' in line and 'None' not in line:
        #                     loc, obj = line.strip().lstrip('- ').split(':', 1)
        #                     known_objects[obj.strip()] = loc.strip()
                            
        return {
            "scene_id": scene_id,
            "log_dir": log_dir,
            "global_intent": global_intent,
            "visited_locations": visited_locations,
            "known_objects": known_objects,
            "action_history": action_history
        }
