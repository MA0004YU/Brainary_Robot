# Brainary Robot — EB-Manipulation 记忆系统

本项目基于 [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) 的 **EB-Manipulation（机械臂操控）** 子集，在其评测框架之上增加了两个原创模块：

- **意图推理智能体**（`Planning_agent/`）：具备五层 Why 意图解析、断点交互与长期记忆的 Agent
- **三层记忆系统**（`memory_manip/`）：Working Memory → Episodic Memory → Semantic Memory 的分层架构

架构示意图见：`docs/memory_architecture.png`

---

## 三层记忆架构

```
┌─────────────────────────── Working Memory ────────────────────────────┐
│  ActiveGoal    : 当前指令 + GoalReasoner 五层 Why 意图结构             │
│  Observation   : 可见物体、场景嵌入 (T,P,D)、环境文本、当前位置        │
│  ActionBuffer  : 本 episode 内的 (动作, 结果) 滚动缓冲                │
│  RobotState    : 末端位姿、关节角、力矩（真机填充，仿真时为 None）      │
│  TaskClock     : 全局步数、episode 步数、预算剩余、milestone 标记      │
│  生命周期: 单 episode，边界时清空                                      │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ end_episode() 时固化
                                ▼
┌─────────────────────────── Episodic Memory ───────────────────────────┐
│  EpisodeRecord : task_instruction / success / steps[] / 遇到的物体    │
│  steps[]       : 每步 (obs_summary, action, outcome) 完整轨迹         │
│  检索          : StringRetriever 关键词匹配（接口可换 embedding）       │
│  持久化        : append-only JSONL，进程重启数据不丢                   │
│  生命周期: 跨 episode 持久，最多保留 N 条（可配置）                    │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ 每 N 个 episode 批量泛化
                                ▼
┌─────────────────────────── Semantic Memory ───────────────────────────┐
│  ObjectKB      : 物体 → 可操作性、历史位置频次                         │
│  UserPreference: 用户需求 → 偏好/回避物体（explicit > episodic 权重）  │
│  TaskSchema    : 任务类型 → 成功率统计、常见步骤模板                   │
│  SpatialTopo   : 位置节点图，含邻居关系与物体频次（可对接 SLAM）        │
│  持久化        : 原子写入 JSON（.tmp → os.replace，防断电损坏）         │
│  生命周期: 长期积累，永不清空                                          │
└───────────────────────────────────────────────────────────────────────┘
```

### Planning_agent 数据桥接

`PlanningAgentBridge` 单向读取 Planning_agent 的 MD 文件，同步到记忆层：

| 文件 | 数据方向 | 目标层 |
|------|---------|-------|
| `persistent_memory.md` | 用户偏好、场景知识 → | Semantic Memory |
| `session_context.md` | 当前意图、已探索位置 → | Working Memory |

桥接严格只读，不修改 Planning_agent 的任何文件。

---

## 模块代码对应关系

| 模块 | 文件 | 说明 |
|------|------|------|
| 三层记忆入口 | `memory_manip/agent_memory.py` | `EmbodiedManipulationMemorySystem` |
| Working Memory | `memory_manip/working_memory.py` | 5 个槽位，含真机字段预留 |
| Episodic Memory | `memory_manip/episodic_memory.py` | JSONL 持久化 + StringRetriever |
| Semantic Memory | `memory_manip/semantic_memory.py` | JSON 原子写入，4 个知识库 |
| 配置 | `memory_manip/config.py` | 容量、路径、泛化触发阈值 |
| 接口桩 | `memory_manip/interfaces.py` | Perception / Simulation / Monitor |
| 桥接器 | `memory_manip/bridges.py` | Planning_agent MD → 记忆层 |
| 远程服务 | `memory_manip/longterm_memory.py` | EmbodiedLTM HTTP 客户端 |
| 意图推理 Agent | `Planning_agent/core_agent.py` | IntentReasoningAgent |
| 意图解析 | `Planning_agent/goal_reasoner.py` | 五层 Why 方法论 |
| 记忆管理（Agent侧）| `Planning_agent/memory_manager.py` | MD 文件读写 |

---

## 目录结构

```
Brainary_Robot/
├── docs/
│   └── memory_architecture.png
└── embodiedbench/
    ├── Planning_agent/          # 意图推理智能体
    │   ├── core_agent.py
    │   ├── goal_reasoner.py
    │   ├── memory_manager.py
    │   ├── persistent_memory.md
    │   └── session_context.md
    ├── memory_manip/            # 三层记忆系统
    │   ├── agent_memory.py      # 顶层 facade
    │   ├── working_memory.py
    │   ├── episodic_memory.py
    │   ├── semantic_memory.py
    │   ├── bridges.py
    │   ├── interfaces.py
    │   ├── longterm_memory.py
    │   ├── config.py
    │   └── store/               # 运行时数据（.gitignore 忽略）
    ├── configs/eb-man.yaml
    └── envs/eb_manipulation/    # CoppeliaSim 仿真环境
```

---

## 快速上手

### 安装依赖

EB-Manipulation 仿真依赖 **CoppeliaSim** 与 **PyRep**；仅开发记忆逻辑时无需启动仿真：

```bash
pip install numpy
# 仿真相关依赖见 embodiedbench/envs/eb_manipulation/requirements.txt
```

### 启用记忆系统（评测入口）

在配置字典中设置 `use_architectural_memory: 1`（见 `evaluator/eb_manipulation_evaluator.py`）：

```bash
python run_eb_manipulation_memory.py \
    --model_name gpt-4o \
    --use_architectural_memory 1 \
    --eval_sets base
```

启用后，每个 episode 在 `running/.../results/` 下额外输出 `memory_episode_*.json` 快照，`memory_manip/store/` 下持久保存 `episodes.jsonl` 和 `semantic_kb.json`。

### 独立使用记忆系统

```python
from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem, MemorySystemConfig

mem = EmbodiedManipulationMemorySystem(
    MemorySystemConfig(
        episodic_generalize_every_n=5,   # 每 5 个 episode 泛化一次
        store_dir="/data/robot_memory",  # 自定义持久化路径（真机部署）
    )
)

# Episode 生命周期
mem.reset_episode(scene_id="FloorPlan22")
mem.begin_task("给我拿一杯可乐", intent=goal_reasoner_output)

# 每步更新
mem.update_observation(visible_objects=["cola", "table"], current_location="kitchen")
mem.record_action("pick up the cola", action_id=3, success=True)

# Episode 结束，自动固化到 Episodic Memory，定期泛化到 Semantic Memory
mem.end_episode(success=True)

# 查询接口
mem.query_object("cola")                      # 物体知识
mem.query_user_preference("thirst")           # 用户偏好
mem.query_similar_episodes("拿饮料")           # 相似历史 episode
mem.query_task_schema("fetch me a drink")     # 任务成功率与步骤模板
```

### 对接 Planning_agent 桥接器

```python
# 读取 Planning_agent 已积累的用户偏好和场景知识（只读，不修改 MD 文件）
result = mem.sync_from_planning_agent()
# {"semantic_entries_synced": 12, "working_memory_synced": True, ...}
```

### 真机接入

Working Memory 的 `RobotState` 槽位预留了真机字段，仿真时为 `None`，接入实机时填充即可：

```python
mem.update_robot_state(
    gripper_pose=np.array([x, y, z, roll, pitch, yaw]),
    joint_angles=np.array([q1, q2, q3, q4, q5, q6]),
    force_torque=np.array([Fx, Fy, Fz, Tx, Ty, Tz]),
    gripper_aperture=0.8,
    timestamp=time.time(),
)
```

---

## 接入指南

### Perception

实现 `PerceptionInterface.extract_features`，返回形状 `(T, P, D)` 的 `float32` 数组：

```python
from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem, PerceptionInterface

class MyPerception(PerceptionInterface):
    def extract_features(self, observation, *, meta=None):
        # 返回 (T, P, D) float32 numpy array
        return your_vision_backbone(observation)

mem = EmbodiedManipulationMemorySystem(perception=MyPerception())
mem.run_perception(obs)  # 自动写入 working.observation.scene_embedding
```

### Simulation / Monitor

```python
from embodiedbench.memory_manip import SimulationInterface, MonitorInterface

class MySimulation(SimulationInterface):
    def predict(self, state_summary, action, *, horizon=1):
        return your_world_model(state_summary, action)

class MyMonitor(MonitorInterface):
    def check(self, observation, memory_snapshot):
        ok = your_safety_check(observation)
        return ok, None if ok else "constraint violated"

mem = EmbodiedManipulationMemorySystem(simulation=MySimulation(), monitor=MyMonitor())
```

---

## 环境依赖

| 依赖 | 用途 |
|------|------|
| numpy | Working Memory 特征计算 |
| CoppeliaSim + PyRep | EB-Manipulation 仿真（仅运行仿真时需要） |
| OpenAI API Key | Planning_agent LLM 调用 |
| EmbodiedLTM（可选） | 远程记忆服务，默认 `http://127.0.0.1:8000` |
