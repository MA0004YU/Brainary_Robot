# Vendored Safety Critic

本目录的 `core/` 与 `modules/` 是从「Robot-Safety-Guardrails / Monitor」**vendor 复制**进来的
核心最小集（M5 Safety Critic + 其依赖闭包）。

## 来源

| 项 | 值 |
|---|---|
| 源仓库路径 | `/mnt/d/Postdoctoral_Research/Embodied_monitor/Robot-Safety-Guardrails` |
| 源仓库 commit | `caadc54dfcd65e5352cdbbed194aa4f6e5229216` |
| Monitor 子目录 commit | `63d6b5ae8f0eb80aea92ca0d08b808108e7e4626` |
| 复制日期 | 2026-07-19 |

## 复制范围（M5 import 闭包最小集）

- `core/base_module.py`  —— `BaseModule` 抽象基类
- `core/context.py`      —— `SafetyContext` / `Hazard` 数据类
- `core/enums.py`        —— `RiskLevel` / `EpistemicState` / `GateDecision`（context.py 需要全部三个）
- `core/llm_client.py`   —— 原版 `LLMClient`（**留档，默认不使用**，见下）
- `core/__init__.py`     —— 重导出（本仓库补写，对齐源仓库）
- `modules/m5_safety_critic.py` —— `SafetyCritic`（你研发的模块，主体）
- `modules/m2_scene_parser.py`  —— `SceneSafetyParser`（自然语言模式建 `scene_graph_dict`，
  被 critic_runner 复用；protea 模式的 `utils` 依赖在源码里已 try/except，NL 模式无需额外文件）

## 有意省略

- `modules/m1,m3,m4,m6~m10`、`pipeline.py`、`main.py`、`integration/`、`utils/`、`data/`：
  本次只接入 M5（+ M2 建场景图），其余 WP4 模块未纳入。
- 源仓库依赖 `requests`；本仓库 critic 走自研 `pipeline_llm.PipelineLLMClient`（也只用 requests）。

## 与本仓库的适配（新代码，不在 vendor 内）

- `pipeline_llm.py`：流水线口径 LLM 客户端，保持 `.call(prompt)->str` 接口，
  但复用本仓库 `OPENAI_API_KEY / API_zhongzhuan + VLM_BASE_URL`，默认 `gpt-4o`。
  **默认用它替换原版 `core/llm_client.py`**（后者按 model 名选 DASHSCOPE/XAI 等 key，与本仓库约定不一致）。
- `critic_runner.py`：planned_actions.json + memory_snapshot → 逐步 `SafetyContext` → 裁决 → 写文件。
- `__main__.py`：独立入口（`python -m Monitor.safety_critic`，支持 `--dry-run` 离线自检）。

## import 风格

`modules/*.py` 用相对 import（`from ..core.xxx`），vendor 后目录层级保持
`Monitor/safety_critic/{core,modules}/` 不变，故**无需改任何 import**。
外部通过 `from Monitor.safety_critic.xxx import ...`（依赖仓库根在 sys.path）。
