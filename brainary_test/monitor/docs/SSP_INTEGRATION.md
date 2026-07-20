# SSP 接入文档（场景安全解析器 → 感知-记忆流水线）

本文记录把「场景安全解析器 SSP」接入「感知-记忆流水线」的全部改动、缺口与 DEMO 写死项。
接入日期 2026-07-18。

---

## 1. 一句话概览

SSP 场景安全解析在流水线里**排在 planner 之前**（记忆之后、规划之前）：
读记忆快照 → 经适配层转成 SSP 的强类型 `PerceptualGraph` → **scene_intrinsic 模式**跑场景风险解析
→ 产出候选安全约束（CT-* id），双写到独立诊断文件 + 合并进规划输入 → **回流给 planner**，
让规划在生成动作序列时就带上安全约束。

> 流程顺序（`main.py`）：感知 → 记忆 → **[3/5] SSP** → **[4/5] planner** → **[5/5] safety_critic**。
> SSP 用 scene_intrinsic 模式（不依赖 plan），故能在 planner 之前跑并反向喂约束。
> （runner 仍保留 action_conditioned 模式，用于 planner 之后的逐动作诊断，可用 `--mode` 切换。）

**SSP 边界红线（务必牢记）**：SSP 只输出**候选约束模板 id（CT-*）+ 实体绑定 + 证据**，
**绝不生成最终 LTL 约束、也不做 accept/reject** —— 那是下游 L3 约束层的职责。
所有回写内容都以 `status: "candidate"` 标注，并带 `_boundary` 说明。

---

## 2. 本次新增 / 改动文件清单

> 注：本次接入后，SSP 引擎与适配层都归入 **`Monitor/`** 安全监控层（与 safety_critic 并列）。
> safety_critic 的接入细节见 [`SAFETY_CRITIC_INTEGRATION.md`](SAFETY_CRITIC_INTEGRATION.md)。

### 新增：vendored SSP 引擎（在 Monitor 下）
```
Monitor/ssp_pkg/
├── VENDORED_FROM.md            来源（commit 7501079 / 2026-07-18）+ 复制范围说明
├── ssp/                        核心最小集（import 闭包）
│   ├── __init__.py, parser.py
│   ├── ontology/  entities/relations/risk_events/schema/template_registry + __init__
│   ├── graph/     g_percept/g_reason/lift/g_activated/g_constraint + __init__
│   ├── propagation/ 全部
│   └── query/     activate/predicates/result + __init__
└── configs/re_templates/*.yaml 14 个风险事件(RE)模板
```
省略：`utils/`（空壳）、`bridge/`（空壳 stub）、`ablation/`、`bench_*`、`rule_derived_gt`。
绝对 import `from ssp.xxx`，同名 vendor，无需改任何 import。

### 新增：SSP 适配层（本仓库工程主体，在 Monitor 下）
```
Monitor/ssp_adapter/            （原 adapters/，接入 Monitor 后改名并入）
├── __init__.py
├── object_attribute_map.py   中文物体名/category → EntityType+subtype+StateSchema（+含液覆盖表）
├── demo_relations.py         DEMO 硬编码物体间 L0 关系（near/supports）
├── memory_to_gp.py           核心适配器：memory_snapshot.json → 合法 PerceptualGraph
├── constraint_writer.py      QueryResult → 独立诊断文件 + 合并进 planning_input
├── ssp_runner.py             编排：记忆 → G_P → SSP → 回写（被 __main__ 和 main.py 复用）
└── __main__.py               独立入口（python -m Monitor.ssp_adapter），可脱离 torch/API 单独验证
```

### 新增：文档
```
docs/SSP_INTEGRATION.md        本文件
```

### 改动：既有文件（只做加法，不动记忆/规划逻辑）
- `main.py`：
  - 顶部加 `sys.path.insert(0, str(ROOT / "Monitor" / "ssp_pkg"))`
  - 新增 `run_ssp(planning_input)` 函数（**第 3 阶段，planner 之前**），返回合并了 safety_constraints 的 planning_input
  - `main()` 里记忆之后、规划之前调用 `run_ssp()`，把返回的 dict 喂给 `planner.generate_plan()`（回流）
  - 各阶段 try/except 包裹，失败不影响其它阶段
  - 完成清单里加 `ssp_perceptual_graph.json` / `ssp_safety_constraints.json`
- `requirements.txt`：追加 `pydantic>=2 / networkx / structlog / PyYAML`

### 新增：output 产物
- `output/ssp_perceptual_graph.json` —— 喂给 SSP 的 G_P（调试用，含 id↔中文名映射 + demo_notes）
- `output/ssp_safety_constraints.json` —— 完整诊断（scene_intrinsic 场景风险 + CT-id + 证据）
- `output/memory_planning_input.json` —— 在 `constraints.safety_constraints` 下合并候选约束（增量），planner 读它

---

## 3. 数据流与驱动模式

```
memory_snapshot.json  （planner 之前，不需要 plan）
  working.observation.memory_objects
  （中文名 → {category,location,...}）
        │
        ▼  object_attribute_map (属性映射) + demo_relations (关系注入)
   PerceptualGraph (Node + Edge, 已 validate)
        │
        ▼         SceneSafetyParser.query_risk(g_p)   —— scene_intrinsic（无 action）
   QueryResult (activated / suppressed, 每条带 CT-id + 证据)
        │  constraint_writer
        ├──▶ output/ssp_safety_constraints.json    （完整场景诊断）
        └──▶ output/memory_planning_input.json      （constraints.safety_constraints，聚合去重）
                    │
                    ▼  回流
             planner.generate_plan(planning_input)   —— 规划带上 SSP 安全约束
```

**驱动模式：scene_intrinsic（planner 之前，对齐"SSP 反向喂 planner"的架构）。**
不依赖 plan，直接对场景图 `query_risk(g_p)`（无 action），把所有实例化 factor 视为激活。

- 优点：能在 planner 之前跑，约束真正回流影响本轮规划。
- 代价：scene_intrinsic **不评估** `activation_conditions`，风险条数比 action 模式多
  （当前 DEMO：130 activated / 260 候选约束，vs action 模式 72 / 144）。这是"场景固有风险"视角。
- runner 仍保留 `action_conditioned` 模式（逐个 planned 动作 `grasp→pick`/`place→place`，
  place 的篮子 id 不在图中会 skip），可用 `--mode action_conditioned` 在 planner 之后做逐动作诊断。

---

## 4. 当前 DEMO 的风险结果与解读（重要）

**场景 = 纯机器人桌面分拣，11 个物体，无 human / animal 节点。**

| 风险类别 | 结果 | 原因 |
|---|---|---|
| 10 类**终端人身风险**（割伤/移交/碰撞/夹伤/跌落伤/烧伤/触电/化学/生物污染/窒息） | **0（严格）** | 这些模板的 `vulnerable_targets` 是 `human`/`animal`，场景里没有 → 一个都不实例化 |
| **中间物体风险**（object_fall 物体跌落 / fragile_breakage 易碎品破损） | **有**（逐动作激活） | 这两类的 `vulnerable_targets` 是 `physical_object`，不需要 human |

> **实测数字**：
> - **scene_intrinsic 模式（主流程默认，planner 之前）**：activated 130，去重候选约束 260 条。
> - action_conditioned 模式（planner 之后逐动作，`--mode` 可切）：activated 72（object_fall 66 +
>   fragile_breakage 6），去重候选约束 144 条。
> - 两者 CT-id 都是 4 种：`CT-STABLE-GRASP-ELEVATED`、`CT-CLEAR-PATH-BELOW`、`CT-GENTLE-GRASP`、
>   `CT-CLEAR-PATH-FRAGILE`。终端人身风险恒 = 0。

### 为什么中间风险会被激活（机制）

- `object_fall` 的实例化条件是 `relation_exists(supports, src=hazard, dst=hazard)`（自环）——
  源码里当 src/dst 同角色时退化成"该物体是否参与任一 supports 边"。
- 适配层按**需求 2** 注入的 `table_surface supports 每个物体`，正好让所有 11 个物体都满足
  → object_fall 对每个 (hazard, target) 对实例化。这是主要来源。
- `fragile_breakage` 同理（`near|supports` 自环），3 个易碎容器（黄杯/蓝杯/红碗）互相成对。

### 与"0 风险预期"的关系（如实说明）

用户初始判断"DEMO 必然 0 风险"对**终端人身风险完全成立**，已验证。
但一旦履行需求 1（易碎/属性映射）+ 需求 2（注入 supports/near 关系），**中间物体风险必然被激活**
—— 这不是 bug，是"补齐关系与物理状态"带来的正确副产物，恰好体现接入价值。
经与用户确认：**如实保留全部关系注入**，不为凑 0 而隐藏或删关系。

> 注：主流程用 `scene_intrinsic` 模式（planner 之前、约束回流给 planner），它**不评估**
> `activation_conditions`（模板本意是 `stability in [unstable,tipping]` 才激活，但 DEMO 物体都是
> stable），会把所有实例化 factor 算激活（130 条），属"场景固有风险"视角。若改用 planner 之后的
> `action_conditioned` 模式，则按具体动作精细激活（72 条），更贴合"动作前安全检查"——两种模式都保留。

---

## 5. 本仓库（流水线侧）需补的功能

这些是当前记忆/感知没采集、导致适配层必须"靠映射表/写死"补齐的缺口：

1. **感知 Schema 未采集物理属性**：`perception.json` 的 object 只有
   `name/category/appearance/location/seen_in_views`，没有 SSP 需要的
   `energy / containment / stability / orientation / motion / vulnerability / subtype`。
   → 当前靠 `Monitor/ssp_adapter/object_attribute_map.py` 按类目**推断默认值**（标 `uncertainty=inferred`）。
   真机应让感知直接输出这些物理状态。
2. **记忆未存实体间关系**：`memory_objects` 无 `near/supports/reachable/contains` 等关系；
   感知阶段产出的 `relations[]` 也只是"物体→桌面方位"，且在 `run_memory` 中被丢弃。
   → 当前靠 `Monitor/ssp_adapter/demo_relations.py` **硬编码**。真机应由感知/记忆产出实体间空间关系。
3. **无 human / victim 建模**：感知不识别人、记忆无 human 节点、无 `vulnerability`。
   → 导致所有终端人身风险无法实例化（当前 DEMO 恰好因此 = 0）。真机若有人在场需补 human 节点。
4. **中文名不是稳定 id**：`memory_objects` 用中文名作 key，不能直接当图节点 id。
   → 当前靠 `Monitor/ssp_adapter/memory_to_gp.py` 做「中文名 → 拼音+序号 ASCII id」并保留双向映射。
   真机建议感知/记忆直接分配稳定实例 id。

---

## 6. SSP 侧缺的功能（vendored 包里是空壳或未建）

1. **`frontend/`（真机 WM → G_P）未建**：SSP 目前只有 benchmark 适配器（`bench_adapter.py`）。
   真机世界模型 → PerceptualGraph 的前端未实现 —— 本次由本仓库 `Monitor/ssp_adapter/` 顶替了这一角色。
2. **`bridge/` 是空壳 stub**：`constitution.py`（RE-id ↔ ASIMOV Constitution 映射）、
   `ltl_skeleton.py`（CT-id → LTL/STL 约束骨架生成）都只有 docstring，无实现。
   → 即 **CT-id → 最终 LTL 约束的桥接层尚未实现**。
3. **L3 约束生成 / accept-reject 未实现**：SSP 只到候选 CT-id 为止，
   约束的实例化、裁决（accept/reject）、冲突消解都属于下游 L3，未在 SSP 内。
4. **链式传播**：中间风险（object_fall）的 `downstream_events`（collision/fragile/pinch）
   到终端风险的多跳链式传播，当前 DEMO 因无 human 终端 target 未触发，链路未端到端验证。

---

## 7. DEMO 写死项清单（及"未来由谁产出"）

| 写死项 | 位置 | 内容 | 未来由谁产出 |
|---|---|---|---|
| 物体间关系 | `demo_relations.py` | `table_surface supports 每个物体`；同 location 分组内两两 `near` | **感知/记忆**（几何邻近、支撑面检测） |
| table 平面节点 | `memory_to_gp.py` | 追加一个 `table_surface`（SURFACE）作为 supports 源 | 感知的支撑面/桌面检测 |
| 容器含液 | `object_attribute_map.py` `HAS_LIQUID` + `memory_to_gp.py` | 杯/碗写死"含液"→ 补 substance 水节点 + `contains` 边，`containment=open` | **感知**判断容器内容物 |
| 物体物理状态 | `object_attribute_map.py` `_base_state` | 所有物体默认 `stability=stable/orientation=upright/motion=static/energy=none` | 感知直接输出物理状态 |
| 咖啡机能量 | `object_attribute_map.py` | `energy=none`（**保守，不伪造**通电/发热；subtype 仍 electronic） | 感知/传感器判断通电与温度 |
| 动作类型映射 | `ssp_runner.py` `_ACTION_TYPE_MAP` | `grasp→pick, place→place` | 规划输出对齐 SSP action 词表后可去除 |

所有写死处代码里都留了 `TODO(真机)` 注释。

---

## 8. 属性映射表口径说明（`object_attribute_map.py`）

匹配优先级：**名字关键词（更具体） > category > 兜底 `other`**。

| 物体 / category | EntityType | PhysicalObjectSubtype | 特殊状态 |
|---|---|---|---|
| 杯子 / 碗 | physical_object | fragile_object | containment=open + 含液（补水 substance 节点） |
| 剪刀 / 折叠刀 / 工具 | physical_object | sharp_object | — |
| 咖啡机 / 家电 | physical_object | electronic | energy=none（保守） |
| 香蕉 / 橙子 / 水果 | physical_object | food_item | — |
| 茶盒 / 包装盒 | physical_object | container | — |
| 方块 | physical_object | other | — |
| 抽屉柜 / 家具 | physical_object | furniture_support | — |
| （table 平面，适配层追加） | surface | — | — |
| （容器液体，适配层追加） | substance | — | containment=open |

- 状态字段一律用 `StateSchema(**dict)` 一次性构造（触发 pydantic enum 校验），
  不用 bench_adapter 里 `setattr` 逐字段赋值的写法（那会绕过校验）。
- 所有推断属性标 `uncertainty=inferred`，写死项标 `assumed`，与"观测(observed)"区分。

---

## 9. 如何运行 / 验证

### 独立验证适配层（无需 API key、不碰 torch，吃现成 output/）
```bash
conda activate biea_ssp
cd Perception_memory_pipeline
python -m Monitor.ssp_adapter                       # 默认 scene_intrinsic（planner 之前那步）
# 期望：终端人身风险 = 0；G_P 15 节点/23 边；activated 130 / 候选约束 260；无异常
python -m Monitor.ssp_adapter --mode action_conditioned   # 可选：planner 之后逐动作诊断
```

### 端到端（需 API key，跑完整 5 阶段）
```bash
conda activate biea_ssp
export OPENAI_API_KEY=...   # 或 API_zhongzhuan
python main.py
# 顺序：感知 → 记忆 → [3/5] SSP（出约束回流）→ [4/5] planner（吃约束）→ [5/5] safety_critic
# 各阶段 try/except 互不影响
```

### 产物核对
- `output/ssp_perceptual_graph.json`：喂 SSP 的 G_P + id↔中文名映射 + demo_notes
- `output/ssp_safety_constraints.json`：逐动作完整诊断（activated/suppressed/inactive + CT-id + 证据）
- `output/memory_planning_input.json` → `constraints.safety_constraints`：聚合去重的候选约束


