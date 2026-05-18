import requests
import json
import logging
import os

logger = logging.getLogger(__name__)

class LTMClient:
    def __init__(self, base_url=None, timeout=120):
        self.base_url = base_url or os.environ.get("EMBODIED_LTM_URL", "http://127.0.0.1:8000")
        self.timeout = timeout

    def insert(self, query: str) -> bool:
        """基础的插入方法"""
        logger.info(f"[LTMClient] Inserting to LTM:\n{query}")
        try:
            response = requests.post(
                f"{self.base_url}/insert", 
                data={"query": query}, 
                timeout=self.timeout
            )
            if response.status_code == 200:
                logger.info("[LTMClient] Insert success.")
                return True
            else:
                logger.error(f"[LTMClient] Insert failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"[LTMClient] Insert exception: {e}")
            return False

    def query(self, query: str, mode: str = "hybrid") -> str:
        """基础的查询方法"""
        logger.info(f"[LTMClient] Querying LTM:\n{query}")
        try:
            data = {"query": query, "mode": mode, "use_pm": False}
            response = requests.post(
                f"{self.base_url}/query", 
                data=data, 
                timeout=self.timeout
            )
            if response.status_code == 200:
                result = response.json()
                answer = result.get('final_answer', str(result))
                logger.info(f"[LTMClient] Query success. Answer:\n{answer}")
                return answer
            else:
                logger.error(f"[LTMClient] Query failed: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"[LTMClient] Query exception: {e}")
            return ""

    def update_state(self, observation_desc: str) -> bool:
        """用于专门发送状态更新的包装方法"""
        prompt = f"[STATE UPDATE] {observation_desc}. Please update the exact locations and states of the objects in your memory to reflect this exact new state. Remove any old locations that contradict this new state."
        return self.insert(prompt)

    def initialize_environment_priors(self, locations: list, actions: list) -> bool:
        """初始化环境先验知识"""
        if not locations and not actions:
            return False
        loc_str = ", ".join(locations) if locations else "Unknown"
        act_str = ", ".join(actions) if actions else "Unknown"
        prompt = f"The environment contains the following locations: {loc_str}. The agent is capable of the following base actions: {act_str}."
        return self.insert(prompt)
