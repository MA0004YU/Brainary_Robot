# Safety Critic 接入文档（逐动作安全裁判 → 感知-记忆流水线）

本文记录把「Safety Critic」（Robot-Safety-Guardrails / Monitor 的 M5 模块）接入本流水线的改动。
它与 SSP 同属 **Monitor 安全监控层**，接入日期 2026-07-19。SSP 部分见
[`SSP_INTEGRATION.md`](SSP_INTEGRATION.md)。

---

## 1. 一句话概览

在流水线「规划」之后，新增**第 5 阶段 Safety Critic**：读规划出的动作序列（plan）+ 记忆快照，
用 LLM 逐个动作判断「malicious / not malicious」，产出一份逐步安全评价报告。

```
感知 → 记忆 → [3/5] SSP 场景风险(回流约束) → [4/5] 规划 → [5/5] Safety Critic 逐动作裁判
```

与 SSP 的分工：
- **SSP**（规则/图推理）：从场景**结构**推候选安全约束（CT-*），不调 LLM。
- **Safety Critic**（LLM 裁判）：对**具体动作序列**逐步判是否恶意/危险，产出裁决 + 理由。

---

## 2. Monitor 目录归并（本次同时做的结构调整）

本次把之前接入的 SSP 一起收进 `Monitor/`，形成统一的安全监控层：

```
Monitor/
├── __init__.py
├── ssp_pkg/         （原根目录 ssp_pkg/，vendored SSP 引擎，移入，未改内容）
├── ssp_adapter/     （原根目录 adapters/，SSP 适配层，移入并改名；内部 import 前缀改为 Monitor.ssp_adapter）
└── safety_critic/   （★本次新增）
    ├── __init__.py
    ├── VENDORED_FROM.md      来源（commit caadc54 / 2026-07-19）
    ├── core/                vendored：base_module / context / enums / llm_client(留档) + __init__
    ├── modules/
    │   ├── m5_safety_critic.py   vendored：SafetyCritic（你研发的模块，主体）
    │   └── m2_scene_parser.py    vendored：SceneSafetyParser（NL 模式建 scene_graph_dict）
    ├── pipeline_llm.py      ★本仓库适配：流水线口径 LLM 客户端（.call 接口）
    ├── critic_runner.py     ★本仓库适配：plan + 记忆 → 逐步 SafetyContext → 裁决 → 写文件
    └── __main__.py          ★独立入口（python -m Monitor.safety_critic，支持 --dry-run）
```

> 归并只是移动 + 改 import 前缀（`adapters.` → `Monitor.ssp_adapter.`），SSP 逻辑与产物不变，
> 已回归验证仍输出 72 activated / 144 候选约束、终端人身风险 0。

---

## 3. 新增 / 改动文件清单

### 新增：vendored Safety Critic（见 `Monitor/safety_critic/VENDORED_FROM.md`）
- `core/{base_module,context,enums,llm_client}.py` + `__init__.py`
- `modules/{m5_safety_critic,m2_scene_parser}.py` + `__init__.py`

### 新增：本仓库适配（工程主体）
- `Monitor/safety_critic/pipeline_llm.py` —— LLM 客户端，复用本仓库 env 约定
- `Monitor/safety_critic/critic_runner.py` —— 核心编排
- `Monitor/safety_critic/__main__.py` —— 独立入口

### 改动：既有文件（只做加法）
- `main.py`：新增 `run_safety_critic()`（第 5 阶段）；`main()` 里 SSP 之后加 `try/except` 调用；
  SSP 相关 import/路径改为 `Monitor.ssp_adapter` / `Monitor/ssp_pkg`；阶段号 `[4/4]`→`[4/5]`+`[5/5]`；
  完成清单加 `safety_critic_review.json`。
- `requirements.txt`：Safety Critic 只用 `requests`（已在依赖内），无新增第三方库。

### 新增：output 产物
- `output/safety_critic_review.json` —— 逐步裁决（每步 decision / risk_level / reason / would_halt_here）+ 汇总

---

## 4. 数据流

```
planned_actions.json  [{id, action, target, depends_on}]
        │  _action_to_text:  grasp X -> "grasp X"；place -> "place the held object into KLT"
        ▼
   plan_steps: List[str]                       memory_snapshot.json
        │                                         working.observation
        │                                    (visible_objects / memory_objects / held_object / ...)
        │                                              │ _extract_scene_fields
        ▼                                              ▼
   SafetyContext(input_mode="natural_language", plan_steps=..., scene_objects_list=..., ...)
        │  SceneSafetyParser(M2).process   → 填 scene_graph_dict（对象→位置文本）
        ▼
   逐步循环 i=0..N-1：设 current_step_index=i → SafetyCritic(M5).process
        │  LLM 判 malicious / not malicious（默认 malicious，除非明确回 not malicious）
        ▼
   CriticRunResult（逐步裁决 + 汇总）──▶ output/safety_critic_review.json
```

### 评价语义（已与用户确认：评完整条 plan）
- 逐步评价**每一个** planned 动作。
- 遇到第一个判 malicious 的动作，记录 `would_halt_here=True` 且在汇总里给 `first_halt_index`
  （表示"按 critic 原始语义，真机在线会在此中断"），**但仍继续评完后续步**，给出全貌。
- `past_actions` 只累加通过（not malicious）的动作，保持 critic 原始语义（历史里不含被拦截动作）。

### 动作格式 → 为何走自然语言裁判
`planned_actions.json` 的 target 是中文名（如「黄色杯子」），不含 PROTEA 的 `<obj> (id)` 形态，
故 critic 的 `_is_protea_action` 判为 false，走**自然语言裁判 prompt**，`input_mode="natural_language"`。

---

## 5. LLM 配置（复用流水线约定）

Safety Critic 需要调 LLM。原版 `core/llm_client.py`（按 model 名选 `DASHSCOPE_API_KEY`/`XAI_API_KEY`
等、URL 硬编码）与本仓库约定不一致，故**默认不使用**，改用 `pipeline_llm.PipelineLLMClient`：

| 项 | 取值 | 与哪一阶段一致 |
|---|---|---|
| API key | `OPENAI_API_KEY` 或 `API_zhongzhuan` | 感知 / 规划 |
| base_url | `VLM_BASE_URL`（默认 `https://api.openai.com/v1`），走 `/chat/completions` | 感知 |
| model | `SAFETY_CRITIC_MODEL`，默认 `gpt-4o` | 规划 |
| temperature | 0（裁判要确定性） | — |

接口保持 `.call(prompt)->str` + `.total_tokens`，与 vendored `SafetyCritic` 期望一致（duck-typed 替换）。

---

## 6. 如何运行 / 验证

### 独立验证（无需 API key —— 离线结构自检）
```bash
conda activate biea_ssp
cd Perception_memory_pileline
python -m Monitor.safety_critic --dry-run
# 用假 LLM 跑通全流程，验证 plan→SafetyContext→逐步裁决→写文件，不发请求
```

### 独立运行（需 API key —— 真实裁判）
```bash
export OPENAI_API_KEY=...     # 或 API_zhongzhuan
python -m Monitor.safety_critic
```

### 端到端（完整 5 阶段）
```bash
export OPENAI_API_KEY=...
python main.py
# [5/5] SafetyCritic 会在 SSP 之后自动跑；失败不影响前 4 阶段（try/except 包裹）
```

### 产物核对
`output/safety_critic_review.json`：
```json
{
  "summary": {"num_steps": 12, "num_malicious": 0, "first_halt_index": null, "overall": "safe", ...},
  "reviews": [
    {"index": 0, "action_id": "T1", "action": "grasp 黄色杯子",
     "decision": "not malicious", "risk_level": "low", "reason": "...", "would_halt_here": false},
    ...
  ]
}
```

---

## 7. 缺口 / 未来工作

1. **只接入了 M5（+ M2 建场景图）**：Monitor 原有 M1/M3/M4/M6~M10（规则库、约束生成、OOD、
   置信度门控、回退、安全记忆）与 `pipeline.py` 编排器未纳入。若要完整 WP4 管线需继续 vendor。
2. **SSP ↔ Critic 尚未联动**：目前两者独立产出。未来可把 SSP 的候选约束（CT-*）作为上下文
   喂给 critic，或把 critic 的裁决回灌给规划触发 replan。
3. **裁决语义偏保守**：critic 原始逻辑是"默认 malicious，除非 LLM 明确回 not malicious"，
   对 LLM 输出格式敏感。真机上线前建议校准 prompt 与解析。
4. **动作串是简化映射**：`grasp X / place ... into KLT` 为自然语言近似；若未来规划输出带坐标/
   PROTEA id，可切到 protea 裁判路径（更精确的状态追踪 + 环境模拟）。
