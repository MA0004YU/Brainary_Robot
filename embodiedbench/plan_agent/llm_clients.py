import json
from typing import Dict

import requests
from openai import OpenAI


class BaseLLMClient:
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict:
        raise NotImplementedError


class OpenAISDKClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return json.loads(content)


class OpenAIHTTPClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        response.raise_for_status()

        if not response.text.strip():
            raise ValueError("Empty response body from LLM service")

        try:
            data = response.json()
        except Exception as e:
            raise ValueError(f"Response is not valid JSON. Raw body: {response.text[:1000]}") from e

        if "choices" not in data:
            raise ValueError(f"Unexpected response format: {data}")

        content = data["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"Model content is not valid JSON: {content}") from e