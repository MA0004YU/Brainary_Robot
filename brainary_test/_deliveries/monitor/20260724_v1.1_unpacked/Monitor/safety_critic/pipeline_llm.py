"""流水线口径的 LLM 客户端（给 SafetyCritic 用）。

SafetyCritic 只依赖 LLM 客户端的两个东西：`.call(prompt: str) -> str` 和 `.total_tokens`。
本类保持这个接口不变（duck-typed 替换 vendored 的 core.llm_client.LLMClient），
但内部复用**本仓库既有的 LLM 约定**，与感知/规划阶段一致，不引入新的环境变量：

  - API key : OPENAI_API_KEY 或 API_zhongzhuan（同 main.py / planning/llm_client.py）
  - base_url: VLM_BASE_URL（默认 https://api.openai.com/v1），走 /chat/completions
  - model   : SAFETY_CRITIC_MODEL 覆盖，默认 gpt-4o（与规划阶段一致）

这样用户不必为 safety_critic 另配 DASHSCOPE/qwen 等 key。
原版 core/llm_client.py 仍保留在包里留档，但默认不使用。
"""

from __future__ import annotations

import os


class PipelineLLMClient:
    """OpenAI 兼容 chat/completions 客户端，接口对齐 vendored LLMClient。"""

    def __init__(self, model: str | None = None, temperature: float = 0.0, top_p: float = 1.0):
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_zhongzhuan")
        if not self.api_key:
            raise ValueError(
                "safety_critic 需要 LLM：未设置 OPENAI_API_KEY / API_zhongzhuan 环境变量"
            )
        self.base_url = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("SAFETY_CRITIC_MODEL", "gpt-4o")
        # 安全裁判要确定性：温度默认 0
        self.temperature = temperature
        self.top_p = top_p
        self.total_tokens = 0

    def call(self, prompt: str) -> str:
        # 延迟 import，保持与本仓库其它阶段一致（不在无 key 时也强制加载）
        import requests
        import urllib3

        urllib3.disable_warnings()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=120,
            verify=False,
        )
        r.raise_for_status()
        result = r.json()
        self.total_tokens += result.get("usage", {}).get("total_tokens", 0)
        return result["choices"][0]["message"]["content"].strip()
