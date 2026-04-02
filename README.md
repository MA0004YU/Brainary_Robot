# EB-Manipulation 记忆系统独立包

本目录是从 [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) 抽出的 **EB-Manipulation（机械臂操控）** 子集，并增加与架构图一致的 **工作记忆（Working Memory）** 与 **长时记忆（Long-term Memory）** 参考实现，便于你们在仿真评测之外单独迭代记忆模块，并逐步对接实机。

架构示意（与课题图一致）见：`docs/memory_architecture.png`。

## 架构与代码对应关系

| 图中模块 | 代码位置 | 说明 |
|----------|----------|------|
| Working — Symbolic | `embodiedbench/memory_manip/working_memory.py` → `SymbolicWorkingMemory` | 离散符号：动作向量串、技能名等 |
| Working — Abstract | `AbstractWorkingMemory` | 高层任务/子目标文字摘要 |
| Working — Semantic | `SemanticWorkingMemory` | 当前步语义事实（物体类别、关系、感知元数据等） |
| Working — Spatial / 3D | `Spatial3DWorkingMemory` | 即时位姿与场景嵌入（可与 `(T,P,D)` 感知对齐） |
| Working — 任务/时序（图中日程图标） | `TaskScheduleWorkingMemory` | 步计数、milestone 标记 |
| LTM — Conceptual world model | `longterm_memory.py` → `ConceptualWorldModel` | 实体与概念关系（可接物理/规则库） |
| LTM — Temporal | `TemporalKnowledgeBase` | episode 摘要与时间线 |
| LTM — Spatial | `SpatialKnowledgeGraph` | 持久空间拓扑/路标（可接 SLAM / 地图） |
| 感知接口 Perception | `interfaces.py` → `PerceptionInterface`，默认 `NullPerception` | 约定输出 **`(T, P, D)`** `float32` 特征，由同事接入视觉 backbone |
| 仿真接口 Simulation | `SimulationInterface` / `NullSimulation` | 占位：接工作记忆或世界模型 rollout |
| 监控接口 Monitor | `MonitorInterface` / `NullMonitor` | 占位：安全与进度监控 |
| WM → LTM | `agent_memory.py` → `consolidate_to_long_term` | 参考实现：episode 结束时将摘要写入 LTM |

顶层编排类型：`embodiedbench/memory_manip/agent_memory.py` 中的 `EmbodiedManipulationMemorySystem`。

## 目录结构（核心）

```
eb_manipulation_memory/
├── README.md                          # 本文件（中文）
├── run_eb_manipulation_memory.py      # 运行入口（配置 sys.path）
├── requirements_eb_manipulation_env.txt
├── docs/
│   └── memory_architecture.png
└── embodiedbench/
    ├── main.py                        # 轻量 logger（满足 evaluator / planner 引用）
    ├── configs/eb-man.yaml
    ├── memory_manip/                 # 新增：记忆系统
    ├── envs/eb_manipulation/        # 完整 EB-Man 仿真与任务资源（vlm、amsolver、task_ttms 等）
    ├── planner/                      # ManipPlanner 及远程模型调用
    └── evaluator/                    # 评测循环 + 可选记忆钩子
```

## 如何启用记忆钩子

在配置字典中设置 `use_architectural_memory: 1`（见 `embodiedbench/evaluator/eb_manipulation_evaluator.py`）。  
启用后，每个 episode 会在 `running/.../results/` 下额外写出 `memory_episode_*.json` 快照。

命令行示例：

```bash
cd eb_manipulation_memory
python run_eb_manipulation_memory.py --model_name gpt-4o --use_architectural_memory 1 --eval_sets base
```

## 环境依赖（仿真）

EB-Manipulation 依赖 **CoppeliaSim**、**PyRep** 以及 `requirements_eb_manipulation_env.txt` 所列包；与上游 EmbodiedBench 安装流程一致。若仅开发记忆逻辑，可在不启动仿真的情况下 `import embodiedbench.memory_manip` 做单元实验。

远程 VLM 还需配置各厂商 API Key（与原 `remote_model.py` 相同）。

## 接入指南（英文注释在源码中）

1. **Perception**：实现 `PerceptionInterface.extract_features`，返回形状为 `(T, P, D)` 的 `numpy.ndarray`（`float32`），并在策略循环里调用 `EmbodiedManipulationMemorySystem.on_perception_features` 或 `run_perception`。
2. **Simulation**：实现 `SimulationInterface.predict`，从 `long_term.conceptual` 或 `working` 快照读取状态摘要，返回预测轨迹/接触等。
3. **Monitor**：实现 `MonitorInterface.check`，对当前观测与 `memory.snapshot()` 做约束检查。

构造记忆系统时可注入自定义实现：

```python
from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem, MemorySystemConfig

mem = EmbodiedManipulationMemorySystem(
    MemorySystemConfig(),
    perception=my_perception,
    simulation=my_sim,
    monitor=my_monitor,
)
```

评测入口默认使用 `Null*` 桩。若在配置里传入 `arch_memory_instance=EmbodiedManipulationMemorySystem(..., perception=..., simulation=..., monitor=...)`，则会使用该实例并忽略默认构造。


