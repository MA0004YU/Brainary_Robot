# WP4 Safety Brain Monitor

WP4 的运行时安全监控模块。在机器人每执行一步动作后，评估当前环境和动作是否安全，返回放行或拦截决策。

## 功能

Safety Brain Monitor 作为独立的安全审查层，在机器人执行动作的过程中**逐步监控**：

1. 每步动作执行后，接收当前环境状态和已执行动作
2. 解析场景中的可见物体，构建环境上下文
3. 检索相关安全规则（如"不要将金属放入微波炉"）
4. 调用 LLM 判断当前动作是否存在安全风险
5. 根据风险等级输出门控决策（放行 / 拦截）

## 输入与输出

### 输入

通过 `MonitorInterface.check(observation, memory_snapshot)` 接收：

- `observation`：原始环境观测（当前版本未使用，预留给未来 VLM 接入）
- `memory_snapshot`：完整的记忆系统快照，结构如下：

```python
memory_snapshot = {
    "episode_id": str,
    "working": {
        "goal": {
            "instruction": "bring me a cup of water"  # 用户指令
        },
        "observation": {
            "visible_objects": ["knife", "cup", "table"],  # 当前可见物体
            "memory_objects": {"knife": {"location": "counter"}},  # 物体记忆信息
            "text": "A kitchen with a counter and table.",  # 环境文本描述
            "current_location": "kitchen",
            "held_object": "cup"  # 机器人当前持有的物体
        },
        "action_buffer": [
            {"action": "pick up the cup", "success": True, ...}  # 最近执行的动作
        ]
    }
}
```

Monitor 从中提取：
- `working.goal.instruction` → 用户指令
- `working.observation` → 场景上下文
- `working.action_buffer[-1].action` → 当前待评估的动作

### 输出

```python
(ok: bool, alert_message: Optional[str])
```

- `(True, None)` — 动作安全，放行
- `(False, "reason")` — 动作危险，拦截并返回原因

## 与其他模块的交互

```
memory_manip (WP2)                    Monitor (WP4)
┌──────────────────┐                 ┌──────────────────┐
│ Working Memory   │──snapshot()──→  │ SafetyBrainMonitor│
│ - goal           │                 │ - 场景解析 (M2)   │
│ - observation    │                 │ - 规则检索 (M3)   │
│ - action_buffer  │                 │ - 安全判断 (M5)   │
│ - robot_state    │                 │ - 置信度评估 (M7) │
└──────────────────┘                 └────────┬─────────┘
                                              │
                                     (ok, alert_msg)
                                              │
                                              ▼
                                     evaluator 根据结果
                                     记录 milestone 或干预
```

- **memory_manip**：提供 `memory_snapshot`，Monitor 从中读取环境和动作信息
- **evaluator**：在每步动作后调用 `run_monitor()`，接收安全决策
- **Planning_module**：当前无直接交互（未来版本将支持 replan 信号）

## 内部 Pipeline

```
check() 被调用
    │
    ▼
format_adapter: memory_snapshot → SafetyContext
    │
    ▼
M2 SceneSafetyParser: visible_objects → 环境上下文字典
    │
    ▼
M3 RuleLibrary: 加载 20 条安全规则
    │
    ▼
M5 SafetyCritic: LLM 判断动作是否 malicious/not malicious
    │
    ▼
M7 ConfidenceController: risk_level → gate_decision (ALLOW/BLOCK)
    │
    ▼
返回 (ok, alert_message)
```

## 状态管理

Monitor 是**有状态的**：
- 跟踪当前 episode 内的历史动作（用于上下文判断）
- 当 `episode_id` 变化时自动重置状态

## 使用方式

```python
from embodiedbench.Monitor import SafetyBrainMonitor

# 初始化（需要 DASHSCOPE_API_KEY 环境变量）
monitor = SafetyBrainMonitor(model="qwen-plus")

# 在 EmbodiedManipulationMemorySystem 中注册
from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem
mem = EmbodiedManipulationMemorySystem(monitor=monitor)

# 之后 evaluator 每步调用 mem.run_monitor(obs) 即可
```

## 依赖

- `requests`：LLM API 调用
- `DASHSCOPE_API_KEY` 环境变量：qwen-plus 模型访问

## 目录结构

```
Monitor/
├── __init__.py          # 导出 SafetyBrainMonitor
├── monitor.py           # SafetyBrainMonitor 主类（实现 MonitorInterface）
├── format_adapter.py    # memory_snapshot → SafetyContext 转换
├── pipeline.py          # 模块编排器
├── core/
│   ├── context.py       # SafetyContext 数据结构
│   ├── enums.py         # RiskLevel, GateDecision 枚举
│   ├── base_module.py   # 模块基类
│   └── llm_client.py    # LLM 调用客户端
├── modules/
│   ├── m2_scene_parser.py   # 场景解析
│   ├── m3_rule_library.py   # 安全规则库
│   ├── m5_safety_critic.py  # LLM 安全判断
│   └── m7_confidence.py     # 置信度与门控决策
└── data/
    └── rules/default_rules.json  # 20 条默认安全规则
```

## 当前限制

- 每步调用 LLM 判断，延迟约 1-3 秒（正在开发scene safety parser模块，待更新）
- 场景信息依赖 `visible_objects` 列表，尚未接入完整 scene graph
- 安全规则库为静态 20 条，未实现动态检索
- OOD 检测、约束生成等模块尚未实现（返回默认值）
