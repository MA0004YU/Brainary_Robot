# SSP 风险数量分析报告 —— 给 SSP 方法开发者

> 面向对象：SSP（Scene Safety Parser）方法/引擎的开发者。
> 目的：本 DEMO 场景只有 **11 个桌面物体、且全部处于静止稳定状态**，SSP 却激活了
> **130 条风险、输出 260 条候选约束**。本文把当前场景背景、喂给 SSP 的完整输入（实体 /
> 关系 / 动作）、以及数量放大的两个机制原因讲清楚，请开发者判断是否为 SSP 方法本身
> 在「自指风险实例化」与「scene_intrinsic 激活口径」上有待完善之处。
>
> 报告里的所有数字都来自实跑产物 `output/ssp_perceptual_graph.json` 与
> `output/ssp_safety_constraints.json`，非文档估算。

---

## 0. 一句话结论

**130 = 121（object_fall）+ 9（fragile_breakage）。二者都等于 `hazard × target` 的笛卡尔积
（11×11 与 3×3），而「跌落 / 破损」是自指事件（物体自己掉 / 自己碎），target 这一维在语义上
是多余的；再叠加 `scene_intrinsic` 模式不评估 `activation_conditions`（全部 stable 的物体本应
一个都不触发跌落），于是把本应约 14 条的风险放大成 130 条。** 详见 §4。

---

## 1. 场景背景

### 1.1 流水线位置

本仓库是「感知 → 记忆 → **SSP** → 规划 → Safety Critic」五阶段流水线。SSP 处于**规划器之前**，
以 `mode="scene_intrinsic"` 运行：读记忆快照 → 适配层拼出 `PerceptualGraph (G_P)` → SSP 解析 →
输出候选安全约束回流给规划器。此时**还没有 plan、没有具体动作**。

### 1.2 场景内容

纯机器人桌面分拣任务，桌上 **11 个物体，无人、无动物**，全部静置。感知（GPT-5.5）枚举结果：

| # | 物体 | category | 适配层归的 subtype | 备注 |
|---|---|---|---|---|
| 1 | 绿色方块 | 方块 | other | |
| 2 | 黄色杯子 | 杯子 | fragile_object | DEMO 写死「视为含液」 |
| 3 | 蓝色杯子 | 杯子 | fragile_object | DEMO 写死「视为含液」 |
| 4 | 红色碗 | 碗 | fragile_object | DEMO 写死「视为含液」 |
| 5 | 剪刀 | 工具 | sharp_object | |
| 6 | 折叠刀 | 工具 | sharp_object | |
| 7 | 橙子 | 水果 | food_item | |
| 8 | 茶盒 | 包装盒 | container | |
| 9 | 香蕉 | 水果 | food_item | |
| 10 | 咖啡机 | 家电 | electronic | energy 保守置 none |
| 11 | 抽屉柜 | 家具 | furniture_support | |

### 1.3 一个重要前提：属性和关系都是适配层「补」的

当前 memory 模块**不存物体的物理属性，也不存实体间关系**（只有 category / location /
seen_in_views）。SSP 模板要靠「类型 + 物理状态 + 实体间关系」才能实例化风险，所以本 DEMO 由
适配层 `Monitor/ssp_adapter/` 补齐——**这些补齐值都是 DEMO 写死的保守基线**，不是感知真值：

- 物理状态一律为中性静止基线：`stability=stable, orientation=upright, motion=static, energy=none`
  （见 `object_attribute_map.py:_base_state`）。**没有任何物体是 unstable / tipping / hot / powered。**
- 3 个容器（黄杯 / 蓝杯 / 红碗）被写死「视为含液」，各补一个 substance 水节点 + `contains` 边。

---

## 2. 喂给 SSP 的完整输入（G_P）

实跑产物 `output/ssp_perceptual_graph.json`：**15 个节点，23 条边**。

### 2.1 节点（15）

- 1 个 `surface`：`table_surface`（适配层写死追加的桌面支撑平面）
- 11 个 `physical_object`：即 §1.2 的 11 个物体
- 3 个 `substance`：`黄色杯子_liquid / 蓝色杯子_liquid / 红色碗_liquid`（写死含液补的）

### 2.2 边（23）—— 适配层构造，全部 DEMO 写死

关系来源 `Monitor/ssp_adapter/demo_relations.py`，依据物体 location 字段推断，**真机应由感知/记忆产出**：

| 关系 | 条数 | 构造规则 |
|---|---|---|
| `supports` | 11 | `table_surface` supports 每一个桌面物体（写死：桌子支撑全部物体） |
| `contains` | 3 | 3 个含液容器 contains 各自的 liquid 节点 |
| `near` | 9 | 同一 location 分组内两两 near（中间组、中偏左组、左侧组、右上角组） |

`near` 的分组：`[绿色方块,折叠刀]`、`[剪刀,橙子]`、`[黄杯,蓝杯,茶盒,香蕉]`、`[红碗]`、`[咖啡机,抽屉柜]`。

> ⚠️ 注意：**没有注入 `reachable`**，也**没有 human / robot / animal 类 victim 节点**——
> 这是「对人伤害类风险 = 0」的正确来源。

### 2.3 动作（action）—— 空

SSP 在规划器**之前**跑，此刻没有 plan。`ssp_runner._run_scene_intrinsic` 调用
`parser.query_risk(gp.graph)`，**不带 action**，进入 `scene_intrinsic` 分支。这一点对理解 §4.2 很关键。

---

## 3. 具体激活了哪些风险

`output/ssp_safety_constraints.json` 的 `summary`：
`num_activated=130, num_suppressed=0, num_inactive=0, num_entities=15, num_factor_nodes=130, converged=true`。

仓库内置 14 个风险模板（割伤/砸伤/烫伤/触电/跌落/破损/夹伤/泼溅/窒息/化学/生物/火/危险品移交/人跌落），
**只有 2 个被实例化并激活**：

| re_type | 模板 id | 激活数 | = hazard × target | 绑定的候选约束模板 |
|---|---|---|---|---|
| `object_fall` | RE-FALLOBJ-01 | **121** | 11 × 11 | CT-STABLE-GRASP-ELEVATED, CT-CLEAR-PATH-BELOW |
| `fragile_breakage` | RE-FRAG-01 | **9** | 3 × 3 | CT-GENTLE-GRASP, CT-CLEAR-PATH-FRAGILE |
| **合计** | | **130** | | 候选约束 = 130 × 2 = **260** |

其余 12 个模板 0 激活：割伤/夹伤等需要 human/robot victim 或 `reachable`/`near(robot,·)`，本场景没有 → 未实例化。**这部分行为完全正确。**

### 3.1 object_fall 的 121 条构成（逐条核对）

- 11 个 hazard（全部 physical_object），每个都和全部 11 个 physical_object 配成 target：
  - `hazard == target` 的自配对：11 条
  - `hazard != target` 的交叉对：110 条（例如 `FN_RE-FALLOBJ-01_剪刀_咖啡机`：剪刀掉落会砸到咖啡机？）
- 每个 factor node 的 `activation_strength=1.0`、`activation_evidence=["scene_intrinsic"]`、
  `severity(object_fall)=0.6, likelihood=0.76`，`status=candidate`。

### 3.2 fragile_breakage 的 9 条构成

- hazard 收窄为 3 个 `fragile_object`（黄杯 / 蓝杯 / 红碗），target 也是这 3 个 → 3×3 = 9。
- 同样含 3 条自配对 + 6 条交叉对（如「黄杯碎」牵连「红碗」）。

---

## 4. 数量放大的两个机制原因

我们把根因定位到 SSP 引擎的两处，请开发者判断是否为方法待完善点。**注意：这两处都在
`Monitor/ssp_pkg/`（vendored 的 SSP 引擎本体），不在适配层。** 适配层只提供了「桌子 supports
全部物体」这个合理输入。

### 原因 A：lift 阶段对自指风险做了 hazard × target 笛卡尔展开

文件 `ssp_pkg/ssp/graph/lift.py:216-262`，实例化的核心是**双重循环**：

```python
for hazard in hazard_candidates:          # object_fall: 11 个
    for target_node in target_candidates: # object_fall: 又是 11 个
        # 逐条 instantiation_conditions ...
        # 通过则 new FactorNode(id=FN_..._{hazard}_{target})
```

而 `re_fallobj_01.yaml` 的模板设定让这个乘积无法收敛：

```yaml
hazard_source:      {entity_type: physical_object, subtypes: []}   # 11 全中
vulnerable_targets: [{entity_type: physical_object, subtypes: []}] # 11 全中
instantiation_conditions:
  - {op: entity_type_in, args: {field: hazard.type, values: [physical_object]}}  # 只约束 hazard
  - {op: relation_exists, args: {relation: [supports], src: hazard, dst: hazard}} # src=dst=hazard，只约束 hazard
```

**关键观察：两条 instantiation_conditions 全都只引用 hazard，没有任何一条约束 target。**
于是 target 维度完全不被过滤，11 个 hazard × 11 个 target = 121 全部实例化。fragile 同理，
只是 hazard/target filter 收窄到 `fragile_object`（3 个）→ 3×3=9。

但从模板自身的语义看，object_fall / fragile_breakage 是 **event_role: intermediate_outcome 的
自指事件**（「hazard 自己从支撑面掉落 / 自己破碎」），其证据也印证 target 是多余维度：

- `propagation_edges: {from_role: hazard, to_role: hazard}` —— 传播只在 hazard 自身，target 不参与；
- `activation_rules[*].conditions: [{op: participant_is, args: {role: target, matches: hazard}}]`
  —— 激活规则要求 `target == hazard` 才成立，等于承认只有自配对有物理意义。

**换言之：模板在传播和激活上都把 target 绑回 hazard，唯独 lift 的实例化阶段没有绑，
放任 target 自由取遍全场 → 多出的 110 + 6 条交叉对（「A 掉下来砸到 B」）在当前模板语义里
既不传播也不会被动作激活，是「结构性冗余 factor」。**

> 请开发者确认：lift 是否应在「模板未对 target 施加任何 instantiation 约束」或
> 「propagation/activation 把 target 绑定到 hazard」时，只实例化 `target = hazard` 一条，
> 而非做全笛卡尔展开？

### 原因 B：scene_intrinsic 模式不评估 activation_conditions，全部 factor 记为 activated

文件 `ssp_pkg/ssp/parser.py:105-124`，`_build_scene_intrinsic_result`：

```python
for fn_id, factor in g_r.factor_nodes.items():
    event = self._classify_factor(..., activation_strength=1.0,
                                   activation_evidence=["scene_intrinsic"])
    # 只按有无 HARD 抑制边分流 activated / suppressed；无抑制边一律 activated
```

也就是说 scene_intrinsic 下：**只要 factor 被实例化，且没有 HARD 抑制边，就直接算 activated**，
`num_inactive` 恒为 0。它**完全不评估模板里的 `activation_conditions`**：

```yaml
# re_fallobj_01.yaml
activation_conditions:
  - "supports(surface, hazard)"
  - "stability(hazard) in [unstable, tipping]"   # ← 本场景全部 stable，本应过滤掉所有跌落
```

本 DEMO 所有物体 `stability=stable`，若这条 `activation_conditions` 被评估，object_fall 应该
**一条都不激活**。但 scene_intrinsic 路径根本不看它，130 条全部进 activated。

补充：这些 `activation_conditions`（如 `stability in [unstable,tipping]`）在 action 模式下似乎也未被
`activate.py` 使用——`_try_activate_factor` 只评估 `activation_rules[*].conditions`（即
`participant_is target matches hazard`），未消费顶层 `activation_conditions`。请开发者确认
`activation_conditions` 字段的预期消费点。

> 另外，`supports` 边在本场景**不会**触发 object_fall 的 HARD 抑制。模板的 `mitigation_relations`
> 要求抑制边关系是 `neutralized_by`（surface）或 `isolated_by`（locked container），而适配层建的是
> `supports` 边（`lift._find_suppressors` + `_has_grounding_edge` 要求 relation 完全一致）。
> 所以 `num_suppressed=0` —— 「桌子接着」这一事实没有被当成对跌落的抑制。这是否也是预期？

### 两个原因的叠加效应

| 阶段 | 若按模板语义严格执行 | 当前实际 |
|---|---|---|
| lift 实例化 | object_fall 应只按 hazard 实例化 → 11 条；fragile → 3 条 | 笛卡尔积 → 121 / 9 |
| scene_intrinsic 激活 | 评估 `stability in [unstable,tipping]` → **全 stable → 0 条激活** | 不评估 → 全部 activated |
| **合计 activated** | **0（严格动作前）或 ~14（仅计自指实例）** | **130** |

---

## 5. 影响

- **候选约束 260 条**（130×2）回流给规划器，其中约 232 条来自交叉对冗余（原因 A），
  会显著稀释规划器真正该关注的约束信号。
- 数字随场景物体数**平方增长**（N 物体 → object_fall ≈ N²），可扩展性堪忧：
  20 个物体就是 ~400 条 object_fall。

---

## 6. 给开发者的问题清单

1. **lift 笛卡尔积（原因 A）**：对「target 无 instantiation 约束」或「propagation/activation 把
   target 绑回 hazard」的自指风险模板（object_fall / fragile_breakage），lift 是否应只实例化
   `target=hazard`，避免 N² 展开？还是说交叉对（A 砸 B）是有意保留、预期由下游别的机制裁掉？
2. **scene_intrinsic 激活口径（原因 B）**：scene_intrinsic 不评估 `activation_conditions`、把所有
   factor 记为 activated，是否为设计本意（「场景固有风险」全景视角）？若是，规划器侧是否需要另一
   套「按 activation_conditions 过滤」的收窄接口，避免拿到全 stable 却全激活的结果？
3. **`activation_conditions` 字段**：它当前似乎在 scene_intrinsic 和 action 两条路径下都未被消费
   （action 路径只看 `activation_rules`）。这个字段的预期读取点在哪？是否为未完成实现？
4. **supports vs neutralized_by（抑制口径）**：「桌面 supports 物体」是否应被视为对 object_fall 的
   一种抑制/降权？当前因关系名不匹配（模板要 `neutralized_by`）而 `num_suppressed=0`。

---

## 7. 复现方式

```bash
# 不需要 API key、不重跑感知/记忆，直接吃现成 output/*.json 重算 SSP：
python -m Monitor.ssp_adapter
# 预期：G_P 15 节点/23 边，activated 130，候选约束 260
```

关键产物：
- 输入图：`output/ssp_perceptual_graph.json`（15 节点 / 23 边 / id↔中文名对照 / demo_notes 写死项说明）
- 输出诊断：`output/ssp_safety_constraints.json`（`scene_diagnostics.activated_risk_events` 130 条）

相关源码：
- 实例化笛卡尔积：`Monitor/ssp_pkg/ssp/graph/lift.py:216-262`
- scene_intrinsic 分类：`Monitor/ssp_pkg/ssp/parser.py:105-124`
- 动作激活（未消费 activation_conditions）：`Monitor/ssp_pkg/ssp/query/activate.py:32-69`
- 涉及模板：`Monitor/ssp_pkg/configs/re_templates/re_fallobj_01.yaml`、`re_frag_01.yaml`
- 适配层输入（写死关系/属性，供参考，非本问题根因）：
  `Monitor/ssp_adapter/demo_relations.py`、`object_attribute_map.py`


