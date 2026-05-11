import os
import json
import openai
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class LLMClient:
    def __init__(self, model_name="gpt-5.4-mini"):
        """
        初始化 LLM 客户端。默认使用 gpt-5.4-mini。
        自动从环境变量中读取 OPENAI_API_KEY。
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def generate_response(self, system_prompt: str, user_prompt: str, response_format="text", model_override=None) -> str:
        """
        调用纯文本 LLM 接口
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": model_override if model_override else self.model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API Call failed: {e}")
            return "{}" if response_format == "json_object" else ""

    def generate_json(self, system_prompt: str, user_prompt: str, model_override=None) -> dict:
        """便捷方法：强制输出 JSON 并解析为字典"""
        result_str = self.generate_response(system_prompt, user_prompt, response_format="json_object", model_override=model_override)
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM JSON output: {result_str}")
            return {}

    def generate_vision_json(self, system_prompt: str, user_prompt: str, image_path: str) -> dict:
        """调用 Vision多模态模型解析图片并输出 JSON"""
        base64_image = encode_image(image_path)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        kwargs = {
            "model": "gpt-4o",  # Force high-fidelity model for vision
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            result_str = response.choices[0].message.content
            return json.loads(result_str)
        except Exception as e:
            print(f"Vision API Call failed: {e}")
            return {}
