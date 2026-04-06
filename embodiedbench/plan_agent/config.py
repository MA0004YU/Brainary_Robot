import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_mode: str = os.getenv("LLM_MODE", "http")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-5")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://www.openclaudecode.cn/v1")