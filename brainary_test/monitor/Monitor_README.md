# Monitor README —— 安全监控层接入说明

本文件记录 **整个 Monitor 模块** 对原「Perception_memory_pipeline」的改动，涵盖两个子模块：
**SSP（场景安全解析）** 与 **Safety Critic（逐动作安全裁判）**——包括做了哪些改动、加入的安全模块是什么、流程如何重排、以及新的运行方法。
两个子模块的完整技术细节分别见 [`docs/SSP_INTEGRATION.md`](docs/SSP_INTEGRATION.md) 与 [`docs/SAFETY_CRITIC_INTEGRATION.md`](docs/SAFETY_CRITIC_INTEGRATION.md)。

**改动一览**：
- 新增 `Monitor/`（含 `ssp_pkg` 引擎 + `ssp_adapter` 适配层 + `safety_critic` 裁判），两个子模块从各自来源仓库 vendor 进来 + 自研适配层。
- `main.py` 由 3 阶段扩到 5 阶段：感知 → 记忆 → **SSP** → 规划 → **Safety Critic**。
- SSP 从 planner 之后挪到之前，安全约束回流给 planner（见 §2.3）。
- 新增 3 个 output 产物；`requirements.txt` 追加 4 个依赖；既有记忆/规划逻辑只做加法不改动。

---

## 一、总体改动

原pipeline有三步：**感知（看图） → 记忆（存场景） → 规划（排动作）**。
我们加了一个 **Monitor 安全监控层**，含两个模块，分别处于planner前后：

```
 感知  →  三层记忆  →  【3】SSP 场景风险  →  【4】规划  →  【5】Safety Critic 逐动作裁判
(GPT)    (存物体)      (查风险→候选约束)      (排动作)      (逐个动作判 是否危险/恶意)
                       └ 约束返回给 planner ┘              └── 分析 planner 给出的 plan ──┘
                       └────────────────── Monitor 安全监控层 ──────────────────┘
```

- **第 3 步 SSP（场景安全解析，在 planner 之前）**：读记忆里的场景，用规则/图推理判断可能出什么风险，产出**候选安全约束**并**回流给规划层**——让Planner结合安全约束制定plan。
- **第 5 步 Safety Critic（逐动作裁判，planner 之后）**：接受planner给出的动作序列，用 LLM 逐个动作判断「malicious / not malicious」，产出一份**逐步安全评价报告**。

两者分工：SSP 从场景**结构**生成候选约束、提交给 planner（不调 LLM）；Safety Critic 对 planner 给出的**动作序列计划**逐步做安全判断。

---

## 二、对仓库做出的改动

### 2.1 新增目录/文件

所有新增的安全模块都归入到 **`Monitor/`** 目录：

```
Monitor/
├── ssp_pkg/         vendored SSP 引擎（风险模板 + 传播）
├── ssp_adapter/     SSP 适配层（记忆 → 场景图 → 候选约束）
└── safety_critic/   vendored Safety Critic + 适配（读 plan 逐步裁判）
```

| 新增 | 作用 |
|---|---|
| `Monitor/ssp_pkg/` | **SSP 引擎**（vendor 复制），对齐原有 `memory_pkg/` 的模式 |
| `Monitor/ssp_adapter/` | **SSP 适配层**——当前memory模块并不完整，缺少物体的属性、关系等，本次工程针对当前DEMO扩展了适配层，给物体加上了属性和关系。**未来memory模块需要扩展** |
| `Monitor/safety_critic/` | **Safety Critic**（vendor 复制 + 适配）——读 plan + 记忆，逐动作 LLM 判 malicious / not malicious |
| `docs/SSP_INTEGRATION.md` | SSP 完整技术文档 |
| `docs/SAFETY_CRITIC_INTEGRATION.md` | Safety Critic 完整技术文档 + Monitor 归并说明 |
| `Monitor_README.md` | 本文件 |
| `output/ssp_perceptual_graph.json` | 运行产物：给 SSP 的场景图（调试用） |
| `output/ssp_safety_constraints.json` | 运行产物：SSP 完整风险诊断 |
| `output/safety_critic_review.json` | 运行产物：Safety Critic 逐动作安全评价 |

### 2.2 对既有文件的改动

原则上只做加法，只对当前文件增加内容，尽量不动原功能内容。

- **`main.py`**：
  - 顶部加 `sys.path`（指向 `Monitor/ssp_pkg`；`Monitor` 作为包从仓库 import）
  - 新增 `run_ssp(planning_input)`（第 3 步，planner 之前，返回带安全约束的 planning_input）与 `run_safety_critic()`（第 5 步）
  - **调整了阶段顺序**（见下方"流程重排"），各阶段用 `try/except` 包裹——**某一步出错也不影响其它阶段的产物**
- **`requirements.txt`**：SSP 追加了 `pydantic>=2 / networkx / structlog / PyYAML`；Safety Critic 只用已有的 `requests`，无新增
- **`output/memory_planning_input.json`**：运行后会在 `constraints` 下多出一个 `safety_constraints` 子项（SSP 增量添加，原有的分类规则等不动）
- **目录归并**：之前放在仓库根的 `ssp_pkg/` 和 `adapters/` 移入 `Monitor/`（`adapters/` 改名 `ssp_adapter/`），内部 import 前缀相应改为 `Monitor.ssp_adapter`；SSP 逻辑与产物不变

---

## 三、模块说明

### SSP 说明

**SSP（Scene Safety Parser，场景安全解析器）**：输入一张"场景图"（物体、性质、彼此关系），对照 14 个内置**风险模板**（割伤、跌落、破损、烫伤、触电…），判断当前场景 + 机器人动作会不会触发这些风险，输出三类结果：

- **activated**：会触发的风险（带候选约束编号 CT-*）
- **suppressed**：本来有风险，但被某种防护抑制了
- **inactive**：这个动作不涉及的风险

#### SSP 需要的输入 and 记忆所提供的（缺口 → 适配层补齐）

SSP 的输入为**类型场景图**：物体及其类型（利器/易碎/电器…）、物理状态（是否高温、杯内是否有水）、以及物体间关系（支撑/可接触）。
但**当前memory中没有物体属性及其关系**，所以我们针对当前DEMO构建了一套适配层 `Monitor/ssp_adapter/` 负责填补：

| 适配层模块 | 填补内容 |
|---|---|
| `object_attribute_map.py` | 中文物体名 → 类型 + 材质 + 默认状态（杯子=易碎、剪刀=利器…） |
| `demo_relations.py` | 物体间关系（DEMO 阶段**构造并写死**，未来应由感知/记忆模块产出） |
| `memory_to_gp.py` | 核心：根据记忆快照拼成合法场景图，将中文名转为通用英文 id |
| `constraint_writer.py` | 将 SSP 的结果写入文件 + 合并进规划输入 |
| `ssp_runner.py` | 将以上功能串起来的总调度 |

> DEMO 中构造并写死的内容（关系、"杯碗含水"等）在代码里都标了 `TODO(真机)`，详见文档第 7 节。

---

### Safety Critic 说明

**Safety Critic（逐动作安全裁判）**：读planner给出的动作序列（plan）+ 记忆里的场景，
用 LLM 逐个动作判断这一步「malicious / not malicious（是否恶意/危险）」，输出逐步裁决 + 理由。

- **输入**：`planned_actions.json`（动作序列）+ `memory_snapshot.json`（场景对象/位置）。
- **输出**：`output/safety_critic_review.json`——每一步一条裁决（decision / risk_level / reason / 是否会在此中断）+ 汇总。
- **评价范围**：**评完整条 plan**。遇到第一个判恶意的动作会标记「按原语义真机会在此中断」，但仍继续评完后续步，给出完整plan的安全评估。
- **LLM 配置**：复用本仓库既有约定（`OPENAI_API_KEY` / `API_zhongzhuan` + `VLM_BASE_URL`，默认 `gpt-4o`），**不用另配新的 key**。

完整细节（含 Monitor 目录归并、数据流、缺口）见 [`docs/SAFETY_CRITIC_INTEGRATION.md`](docs/SAFETY_CRITIC_INTEGRATION.md)。

---

## 四、新的运行方法

前置：进入 conda 环境。
```bash
conda activate biea_ssp
cd Perception_memory_pipeline
```

### 方式 A：仅跑 SSP（不需要 API key）

**不加载 torch、不重跑感知/记忆**——直接根据已有的 `output/*.json` 运行：

```bash
python -m Monitor.ssp_adapter
```

预期输出：
```
[SSP] 读取快照: .../memory_snapshot.json（模式: scene_intrinsic）
[SSP] G_P: 15 节点, 23 边
[SSP] activated 风险: 130
[SSP] 候选约束(去重后): 260 条
[SSP] ✅ 终端人身风险 = 0（符合预期：场景无 human/animal victim）
```
可选参数：`--output-dir <目录>`、`--templates <模板目录>`。

### 方式 B：仅跑 Safety Critic

需要 API key 做真实裁判：
```bash
export OPENAI_API_KEY=...      # 或 API_zhongzhuan
python -m Monitor.safety_critic
```

无 key 时可做离线结构自检（用假 LLM，不发请求，仅验证管线跑通）：
```bash
python -m Monitor.safety_critic --dry-run
```

### 方式 C：跑完整 5 阶段（需要 API key）

```bash
export OPENAI_API_KEY=...      # 或 API_zhongzhuan
python main.py
```
第 3 步（SSP，规划之前）、第 5 步（Safety Critic，规划之后）会自动执行，控制台出现 `[3/5] SSP done ...`、`[4/5] 规划 done ...`、`[5/5] SafetyCritic done ...`。
**即使某一步报错，前面各步的产物也照常生成（各步都用 try/except 包裹）。**

### 运行后查看文件

| 文件 | 内容 |
|---|---|
| `output/ssp_perceptual_graph.json` |  SSP 的场景图输入 + 中文名↔英文 id 对照 + 自构适配层内容说明 |
| `output/ssp_safety_constraints.json` | SSP 逐动作完整诊断（三类风险 + CT 编号 + 证据） |
| `output/memory_planning_input.json` | 在 `constraints.safety_constraints` 下的候选约束（提交给planner） |
| `output/safety_critic_review.json` | Safety Critic 逐动作安全评价（每步裁决 + 汇总） |

---

## 五、当前 DEMO 跑出的结果

场景是**纯机器人桌面分拣，11 个物体，没有人**。

### 5.1 SSP 场景风险

- ✅ **所有"对人伤害"的风险 = 0**（割伤/砸伤/烫伤/触电/移交危险品…）。
  因为场景里没人 → 一个都不触发。**这是正确的**。
- ⚠️ **但"物体风险"触发了**（不需要有人）：

  | 风险 | scene_intrinsic（主流程默认） | action 模式（参考） | 实际风险 |
  |---|---|---|---|
  | object_fall（物体跌落） | 121 | 66 | 抓取时物体可能从桌上掉下去 |
  | fragile_breakage（易碎品破损） | 9 | 6 | 抓杯子/碗时可能磕碎 |
  | **合计 activated** | **130** | 72 | 去重后候选约束分别 260 / 144 条 |

  > scene_intrinsic 数字更大，是因为它在 planner 之前跑、还没有具体动作，不评估动作级激活条件，
  > 把所有实例化风险都算激活（"场景固有风险"视角）。原因机制详见 `docs/SSP_INTEGRATION.md` 第 4 节。

### 5.2 Safety Critic 逐动作裁判

- 对 planner 排出的 12 个动作逐个判 malicious / not malicious，产出 `safety_critic_review.json`。
- 需要真实 LLM（`gpt-4o`）才能给出真实裁决；无 key 时可 `--dry-run` 验证管线跑通（假 LLM 一律回 not malicious）。
- **端到端真实 LLM 调用尚未实跑**（需 API key）。已验证的是不调 LLM 的全部环节 + 数据在各阶段间的正确流转。
