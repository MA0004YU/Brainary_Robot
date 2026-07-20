import os
import json
import requests
import urllib3

urllib3.disable_warnings()

class LLMClient:
    def __init__(self, model_name="gpt-4o"):
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_zhongzhuan")
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY or API_zhongzhuan is not set.")
        self.base_url = os.environ.get("VLM_BASE_URL", "https://165.154.193.90").rstrip("/")
        self.model_name = model_name

    def generate_json(self, system_prompt: str, user_prompt: str, model_override=None) -> dict:
        """
        调用标准 OpenAI API，强制输出 JSON 结构。
        """
        model = model_override if model_override else self.model_name
        
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                              json=body, timeout=120)
            r.raise_for_status()
            resp = r.json()
            
            content = resp["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"LLM API Call failed: {e}")
            return {}
