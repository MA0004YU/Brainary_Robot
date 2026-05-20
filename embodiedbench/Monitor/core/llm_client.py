import os
import requests


class LLMClient:
    """统一 LLM 调用客户端，支持 Qwen / GPT / Grok。"""

    def __init__(self, model: str = "qwen-plus", temperature: float = 0.7, top_p: float = 0.9):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.token = self._get_token()
        self.url = self._get_url()
        self.total_tokens = 0

    def _get_token(self) -> str:
        if "gpt-4" in self.model or "gpt-3.5" in self.model:
            key = os.getenv("OPENAI_API_KEY")
        elif "grok" in self.model:
            key = os.getenv("XAI_API_KEY")
        elif "qwen" in self.model.lower():
            key = os.getenv("DASHSCOPE_API_KEY")
        else:
            key = os.getenv("LOCAL_LLM_API_KEY")
        if not key:
            raise ValueError(f"未找到模型 '{self.model}' 对应的 API Key 环境变量")
        return key

    def _get_url(self) -> str:
        if "gpt-4" in self.model or "gpt-3.5" in self.model:
            return "https://api.openai.com/v1/chat/completions"
        elif "grok" in self.model:
            return "https://api.x.ai/v1/chat/completions"
        elif "qwen" in self.model.lower():
            return "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        else:
            return "https://chat.binghamton.edu/api/chat/completions"

    def call(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(self.url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        usage = result.get("usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        return result["choices"][0]["message"]["content"].strip()
