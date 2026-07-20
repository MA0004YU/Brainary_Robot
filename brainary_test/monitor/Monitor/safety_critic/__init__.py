"""Safety Critic 子模块（vendored + 适配）。

vendored（来自 Robot-Safety-Guardrails/Monitor，见 VENDORED_FROM.md）：
  core/     BaseModule / SafetyContext / Hazard / RiskLevel 等 + 原版 LLMClient（留档）
  modules/m5_safety_critic.py   逐步安全裁判（malicious / not malicious）

本仓库适配（新代码）：
  pipeline_llm.py   流水线口径的 LLM 客户端（.call 接口，复用 OPENAI_API_KEY/VLM_BASE_URL）
  critic_runner.py  planned_actions.json + memory_snapshot -> 逐步 SafetyContext -> 裁决 -> 写文件
  __main__          独立入口（python -m Monitor.safety_critic）
"""
