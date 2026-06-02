# 各模块对脊脑接口兼容性审查

> 审查日期：2026-06-02  
> 参考接口文档：`embodiedbench/connection/VLM_BRAIN_INTERFACE.md` 第10节输出案例模板

---

## 数据流总览

```
Isaac Sim
   ↓ scene_state.json + image.png + task.txt
[Perception] → 组装 scene_state
[Memory]     → 提供历史上下文
[Planning]   → 生成 skill_blueprint JSON
   ↓ skill_blueprint.json
脊脑 (franka_state_machine_cerebellum)
   ↓ 底层控制
机械臂
```

脊脑的两个核心输入：

1. **scene_state.json** — `{cube_pose, target_pose, ee_pose, gripper_width, robot_joint_pos}`（7D pose 格式：`[x,y,z,qw,qx,qy,qz]`）
2. **skill_blueprint JSON** — 符合接口文档第10节模板的 `execution_graph`

---

## 模块一：Perception ⚠️ 有格式适配问题

**当前输出（`process_scene()` 返回值）：**

```json
{
  "stage": "perception_initialization",
  "timestamp": 1234567890.0,
  "objects": [
    {"name": "cube_1", "type": "primitive_box", "size_whd": [...], "pose": [x,y,z,qw,qx,qy,qz]},
    {"name": "target_1", "type": "primitive_box", "size_whd": [...], "pose": [x,y,z,qw,qx,qy,qz]}
  ]
}
```

**脊脑需要：**

```json
{
  "cube_pose":       [x, y, z, qw, qx, qy, qz],
  "target_pose":     [x, y, z, qw, qx, qy, qz],
  "ee_pose":         [x, y, z, qw, qx, qy, qz],
  "gripper_width":   0.08,
  "robot_joint_pos": [...],
  "step_index":      0
}
```

**具体问题：**

| 问题 | 说明 |
|------|------|
| ❌ 命名不匹配 | 感知输出 `cube_1`/`target_1`（带序号），脊脑需要 `cube_pose`/`target_pose`（固定名） |
| ❌ 缺少 `ee_pose` | 末端执行器位姿来自机器人本体状态，不来自感知相机，感知模块没有这个数据 |
| ❌ 缺少 `gripper_width` | 同上，来自关节传感器 |
| ❌ 缺少 `robot_joint_pos` | 来自关节编码器 |
| ✅ 四元数格式正确 | 感知输出 `[x,y,z,qw,qx,qy,qz]`，脊脑也是同格式，匹配 |

**需要做什么：**

写一个 `scene_state_adapter.py`，把感知输出 + 机器人本体状态合并成脊脑需要的格式：

```python
def build_scene_state(perception_out, robot_state, step_index=0):
    objects = {obj["name"].rsplit("_", 1)[0]: obj["pose"]
               for obj in perception_out["objects"]}
    return {
        "cube_pose":       objects.get("cube"),
        "target_pose":     objects.get("target"),
        "ee_pose":         robot_state["ee_pose"],        # 来自 Isaac Sim
        "gripper_width":   robot_state["gripper_width"],  # 来自 Isaac Sim
        "robot_joint_pos": robot_state["joint_pos"],      # 来自 Isaac Sim
        "step_index":      step_index,
    }
```

---

## 模块二：Memory ⚠️ 内容可用但完全未对接

**现状：** 记忆系统存储了有价值的历史知识，但没有任何接口把它注入到脊脑的输入链路中。

**记忆能提供什么，能不能用：**

| 记忆内容 | 接口 | 能否用于脊脑 |
|----------|------|-------------|
| 任务成功率 | `semantic.query_task_schema(instruction)` | ✅ 可以辅助 VLM 选择保守/激进参数 |
| 物体历史位置 | `semantic.query_object(obj)` → `likely_locations` | ✅ 辅助估计 target pose 初始值 |
| 相似历史 episode | `episodic.query_similar(query)` | ⚠️ 可参考，但历史步骤是 SAPIEN 动作，不是 Blueprint |
| TaskSchema 的 `common_steps` | `semantic.query_task_schema()` → `common_steps` | ❌ 存的是 SAPIEN 的 7D 向量动作，对 Blueprint 生成毫无意义 |

**最大的结构性问题：** `TaskSchema.common_steps` 在历次 episode 中积累的是 SAPIEN 的离散动作字符串（`act[0,3,5,1,2,0,1.0]`），不是 Blueprint 的技能名（`move_above`, `descend` 等）。随着项目切换到新架构，这部分历史数据已经没有直接参考价值。

**需要做什么：**

1. 在 VLM Brain 调用前，先从 memory 提取上下文，补充到 `task.txt` 或 `scene_state.json` 的额外字段：

```python
context = memory.query_task_schema(task_instruction)
# 把 success_rate、likely_locations 等写入给 VLM 的 prompt 上下文
```

2. 未来新 episode 的 `common_steps` 应改为记录 Blueprint 技能序列（`["move_above", "descend", "grasp", "lift", "place"]`），而不是 SAPIEN 动作。

---

## 模块三：Planning（ManipPlanner）❌ 与脊脑完全不兼容

这是**最严重的问题**。

**当前 ManipPlanner 输出：**

```python
# json_to_action() 解析出的是 SAPIEN 7D 离散动作向量
action = [vx, vy, vz, rx, ry, rz, gripper]  # 例如 [3, 5, 2, 1, 0, 2, 1.0]
```

**脊脑需要的：**

```json
{
  "blueprint_id": "...",
  "task": "...",
  "execution_graph": { "start": "p1", "nodes": {...} }
}
```

**具体问题：**

| 问题 | 说明 |
|------|------|
| ❌ System Prompt 完全错误 | 现有 prompt 让 LLM 输出 SAPIEN 的 `executable_plan`（7D 动作列表），不是 Blueprint JSON |
| ❌ 输出解析器错误 | `json_to_action()` 只解析 `executable_plan` 字段，不能解析 `execution_graph` |
| ❌ 动作空间概念不同 | ManipPlanner 用 `VOXEL_SIZE` / `ROTATION_RESOLUTION` 离散化空间，Blueprint 用物理尺寸（米、速度）|

**需要做什么：**

ManipPlanner 在新架构里需要被完全替换或重写。有两条路：

- **路线 A（推荐）**：ManipPlanner 不参与新架构，直接用 jinao 的 `vlm_brain/run_vlm_inference.py` 作为规划器。Brainary_Robot 的 ManipPlanner 只用于旧的 SAPIEN 评测流程。

- **路线 B**：给 ManipPlanner 增加 Blueprint 模式，新写一套 prompt 和 `json_to_blueprint()` 解析器，输入改为 `scene_state + task`，输出 Blueprint JSON。代码改动量很大。

---

## 模块四：Monitor ❌ 未实现

**现状：** 只有 `NullMonitor`，`check()` 永远返回 `(True, "")`。

**脊脑侧的情况：** `condition_evaluator.py` 内部已实现了：

| 条件名 | 状态 |
|--------|------|
| `object_in_gripper` | ✅ 已实现 |
| `object_near_target` | ✅ 已实现 |
| `ee_reached_target` | ✅ 已实现 |
| `timeout` | ✅ 已实现 |
| `collision_detected` | ❌ TODO，传感器未接入，永远返回 False |

所以脊脑自己的条件判断基本可用，但有两个问题：

1. **`collision_detected` 条件未实现**（脊脑侧的 TODO），如果 Blueprint 里有这个条件节点会永远判 False
2. **Brainary_Robot 的 Monitor 没有接收脊脑执行反馈**，无法在高层做"执行失败 → 重新规划"的闭环

**需要做什么：**

- 脊脑侧：补全 `collision_detected` 的力传感器接入（`condition_evaluator.py` 第34行 TODO）
- Brainary_Robot 侧：实现真实的 `MonitorInterface`，读取脊脑的执行日志（`performance_collector.py` 的输出），判断是否需要大脑重新生成 Blueprint

---

## 模块六：Management ❌ 与脊脑完全断路，需协调层

**所在路径**：`embodiedbench/management/`

### 模块结构

| 文件 | 类 | 作用 |
|------|----|------|
| `body_protocol.py` | `RobotWalkBody`（Protocol）| 机器人本体接口：`get_camera_images()` + `set_velocity_command(vx, vy, yaw_rate, period)` |
| `heuristic_policy.py` | `BrightnessAvoidancePolicy` | 启发式避障策略：用摄像头亮度判断障碍，输出速度指令 |
| `planning_loop.py` | `TaskPlanningLoop` | 闭环控制主循环：视觉 → 分析 → 决策 → 速度指令，可配置频率（Hz） |
| `sapien_velocity_body.py` | `SapienVelocityBody` | SAPIEN 移动机器人本体（仿真） |
| `repair_walk_body.py` | `RepairWalkBody` | 真实机器人本体，通过 DDS 接入 `g1_walk_controller`（疑似 Unitree G1 人形机器人）|

### 这个模块是做什么的

Management 是一个**移动底盘导航控制循环**，设计用于会走路的机器人（移动底盘或人形机器人），核心数据流为：

```
摄像头图像 → BrightnessAvoidancePolicy 分析 → (vx, vy, yaw_rate) → RobotWalkBody.set_velocity_command()
```

这和脊脑（`franka_state_machine_cerebellum`）的 Franka 机械臂操作是**完全不同的机器人域**。

### 对脊脑的兼容性问题

| 问题 | 说明 |
|------|------|
| ❌ 输出格式完全不同 | Management 输出速度指令 `(vx, vy, yaw_rate, period)`，脊脑需要 `skill_blueprint` JSON，两者没有任何数据交换接口 |
| ❌ 无 Blueprint 概念 | `TaskPlanningLoop` 没有 `execution_graph`、技能节点、条件节点等概念，不知道如何生成或消费 Blueprint |
| ❌ 与记忆/规划模块完全脱节 | `TaskPlanningLoop` 是独立的闭环，不调用 `EmbodiedManipulationMemorySystem`，不调用 VLM Brain，无法参考历史知识 |
| ❌ SAPIEN 仿真体强依赖 | `SapienVelocityBody` 依赖 `SapienNavigationEngine`，需替换为 Isaac Sim 等效体 |
| ⚠️ 策略过于简单 | `BrightnessAvoidancePolicy` 仅做亮度阈值判断，无法处理真实场景中的语义障碍物 |

### 关键架构缺口

**如果机器人是移动操作臂（移动底盘 + Franka 臂）**：
- Management 负责底盘导航（"走到桌子旁"）
- 脊脑负责机械臂操作（"抓起物体"）
- **当前缺失**：两者之间没有任何协调层，Management 导航到位后无法触发脊脑开始执行 Blueprint

**如果机器人是纯 Franka 臂（无移动底盘）**：
- Management 模块与当前脊脑流程完全无关

**如果机器人是人形机器人（G1）**：
- `RepairWalkBody` 通过 DDS 接 `g1_walk_controller`，是真实机器人接口
- G1 做操作任务时，双臂控制需要对接脊脑的 Blueprint，但当前 Management 只有速度命令接口，没有手臂操作接口

### 负责同学需要做什么

**Step 1：明确机器人构型**（先确认再动代码）

- 纯 Franka 臂 → Management 模块跳过，直接跑脊脑
- 移动底盘 + Franka 臂 → 需要写协调层（Step 2）
- G1 人形 → 需要扩展 `RobotWalkBody` 接口加入手臂控制（Step 3）

**Step 2：写底盘与脊脑的协调层**（移动操作臂场景）

```python
class MobileManipulatorOrchestrator:
    def __init__(self, management_loop: TaskPlanningLoop, memory: EmbodiedManipulationMemorySystem):
        self.nav = management_loop
        self.memory = memory

    def execute_task(self, task_instruction: str):
        # 1. 导航阶段：调用 management 移动到目标位置
        self.nav.navigate_to_target(...)   # 需要新增此方法
        # 2. 操作阶段：生成 Blueprint 并触发脊脑
        context = self.memory.query_vlm_context(task_instruction)
        blueprint = vlm_brain.generate(task_instruction, scene_state, context)
        spine_brain.execute(blueprint)
        # 3. 记录结果
        self.memory.record_blueprint_execution(task_instruction, blueprint_skills, success)
```

**Step 3：替换 SAPIEN 仿真体为 Isaac Sim**

```python
class IsaacVelocityBody:
    """替换 SapienVelocityBody，实现 RobotWalkBody 协议"""

    def start(self) -> None:
        # 初始化 Isaac Sim 移动机器人场景

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        # 从 Isaac Sim 相机传感器读取 head/left/right 图像

    def set_velocity_command(self, vx, vy, yaw_rate, period) -> None:
        # 写入 Isaac Sim ArticulationController 的轮子关节目标速度
        # 注意：Isaac Sim 的轮子速度单位是 rad/s，需要根据轮径换算
```

**Step 4：升级 `BrightnessAvoidancePolicy`**

当前策略只用亮度做障碍判断，在真实场景中不可靠。建议：
- 接入 Perception 模块的 `process_scene()` 输出（语义物体检测）
- 或使用 Isaac Sim 的 `RangeSensorExtension`（激光雷达）做距离判断

---

## 模块五：Simulator ⚠️ SAPIEN 专用，需替换

**现状：**
- `embodiedbench/envs/eb_manipulation/EBManEnv` — SAPIEN/amsolver 环境，`step()` 接受 7D 离散动作
- `simulation/src/simulator/` — SAPIEN 场景构建器和物理检测器

在新的脊脑架构里，Isaac Sim 本身就是仿真器，`franka_state_machine_cerebellum` 直接运行在 Isaac Sim 中，**EBManEnv 不在数据流中**。

唯一需要确认的是：Isaac Sim 能正确提供 `get_scene_state()` 给脊脑的 `cerebellum.get_scene_state()`，返回格式为：

```python
{
    "ee_pose_w":      [...],  # 世界坐标系末端位姿 [x,y,z,qw,qx,qy,qz]
    "cube_pose_w":    [...],
    "target_pose_w":  [...],
    "gripper_width":  float,
}
```

这个接口在 `condition_evaluator.py` 的 `_scene_state()` 里调用，是脊脑内部的，由 Isaac Sim 同学实现。

---

## 总结

| 模块 | 兼容状态 | 核心问题 | 优先级 |
|------|----------|----------|--------|
| **Perception** | ⚠️ 需适配 | 输出格式是 objects 列表，需要 adapter 组装成 scene_state，缺少 ee_pose/gripper_width | P1 |
| **Memory** | ⚠️ 需对接 | 内容可用但未注入 VLM 输入；TaskSchema 历史记录了 SAPIEN 动作，无法直接用 | P2 |
| **Planning** | ❌ 不兼容 | 输出 SAPIEN 7D 动作，prompt/解析器与 Blueprint 完全不同架构，需替换或新写 | P0 |
| **Monitor** | ❌ 未实现 | 只有 NullMonitor；脊脑侧 collision_detected 也有 TODO | P1 |
| **Simulator** | ⚠️ 需替换 | EBManEnv 是 SAPIEN 专用，Isaac Sim 流程中不涉及，由 Isaac Sim 同学负责 | P0（已知） |
| **Management** | ❌ 与脊脑断路 | 速度指令接口与 Blueprint JSON 无交集；无记忆/VLM 对接；缺失底盘与机械臂的协调层 | P1（构型确认后） |

**最关键的两条**：

1. ManipPlanner 输出的是 SAPIEN 离散动作，和脊脑需要的 Blueprint JSON 是两套完全不同的架构。如果 Brainary_Robot 的规划器要参与新架构，需要重写 prompt 和解析器；如果只用 jinao 的 VLM Brain 做规划，ManipPlanner 在新流程里可以完全跳过。**需要先确认规划模块的分工**。

2. Management 模块的优先级取决于机器人构型：纯 Franka 臂可以完全跳过；移动操作臂或 G1 人形机器人则需要写一个协调层把底盘导航和脊脑 Blueprint 执行串联起来。**需要先确认目标机器人平台**。

---

---

# 各模块适配工作清单（详细分工指南）

> 本节是上方审查结论的行动版本。每个模块的负责同学按照本节指南操作。  
> Memory 模块已完成改造（见下文 §Memory 已完成项），其他模块按需调用即可。

---

## Memory 模块已完成的改动（2026-06-02）

> 改动文件：`embodiedbench/memory_manip/agent_memory.py`、`embodiedbench/memory_manip/interfaces.py`

以下接口已可直接调用，无需再改 memory 代码：

| 新增方法 | 作用 |
|---------|------|
| `memory.register_perception(p)` | 感知模块就绪后热插拔，替换 NullPerception |
| `memory.register_monitor(m)` | Monitor 就绪后热插拔，替换 NullMonitor |
| `memory.register_simulation(s)` | Isaac Sim 前向模型就绪后热插拔 |
| `memory.ingest_scene_state(scene_state)` | 把脊脑格式的 scene_state 写入 working memory（robot_state + observation） |
| `memory.export_vlm_context(task, path)` | 把历史知识写成 JSON 文件，供 VLM Brain 读取 |
| `memory.record_blueprint_execution(task, skills, success, scene_state)` | 脊脑执行完毕后回写结果，更新 TaskSchema |
| `memory.query_vlm_context(task)` | 返回结构化历史上下文 dict（可直接注入 prompt） |

`interfaces.py` 中已补全各接口的实现模板注释，包含 `VisionPerception`、`SpineConditionMonitor`、`IsaacForwardModel` 的骨架代码。

---

## 感知（Perception）模块适配工作

**负责同学**：感知模块开发者  
**文件**：`embodiedbench/perception/perception_inference.py`，新建 `embodiedbench/perception/vision_perception.py`

### 与 Memory 连接

**目标**：让感知结果进入 memory 的 working memory（目前两者互不认识）。

**原因**：Memory 中的 `working.observation` 存的是空壳或文字，没有真实感知信息。VLM Brain 最终生成的 Blueprint 质量高度依赖感知输出的准确性；如果感知不写入 memory，历史场景知识就无法积累。

**需要做的事**：

1. 实现 `VisionPerception(PerceptionInterface)`，放在 `embodiedbench/perception/vision_perception.py`：

```python
from embodiedbench.memory_manip.interfaces import PerceptionInterface
import numpy as np

class VisionPerception(PerceptionInterface):
    def extract_features(self, observation, *, meta=None):
        image_paths = observation.get("image_paths", [])
        # 调用现有 perception_inference.py 的检测逻辑
        # 返回 scene_state 格式（供 memory 和 VLM Brain 使用）
        scene_state = self._detect_objects(image_paths)
        # 把 scene_state 存到 meta 供驱动脚本使用
        if meta is not None:
            meta["scene_state"] = scene_state
        # 同时返回特征向量（可先返回 zeros，后续扩展）
        return np.zeros((1, 64, 512), dtype=np.float32)

    def _detect_objects(self, image_paths):
        # 整合 perception_inference.py 的 process_scene() 输出
        # 组装成脊脑格式：
        # {cube_pose, target_pose, ee_pose, gripper_width, robot_joint_pos, step_index}
        ...
```

2. 在评估器或 demo 驱动脚本中注册：

```python
from embodiedbench.perception.vision_perception import VisionPerception
memory.register_perception(VisionPerception())
```

3. 评估器调用 `memory.run_perception(obs)` 时传入 `meta={}` 并取出 `scene_state`：

```python
meta = {}
memory.run_perception({"image_paths": img_path_list, "obs": obs}, meta=meta)
scene_state = meta.get("scene_state")  # 用于后续写文件给 VLM Brain
```

### 与脊脑连接

**目标**：感知模块输出的格式必须与脊脑 `scene_state.json` 完全匹配。

**需要做的事**：

写 `scene_state_adapter.py`（已在 audit 模块一节描述），合并感知输出（cube/target pose）与机器人本体状态（ee_pose/gripper_width/joint_pos）：

```python
# embodiedbench/perception/scene_state_adapter.py
def build_scene_state(perception_out, robot_state, step_index=0):
    objects = {obj["name"].rsplit("_", 1)[0]: obj["pose"]
               for obj in perception_out.get("objects", [])}
    return {
        "cube_pose":       objects.get("cube"),
        "target_pose":     objects.get("target"),
        "ee_pose":         robot_state["ee_pose"],
        "gripper_width":   robot_state["gripper_width"],
        "robot_joint_pos": robot_state["joint_pos"],
        "step_index":      step_index,
    }
```

机器人本体状态（ee_pose、gripper_width、joint_pos）由 **Isaac Sim** 提供，感知模块不负责这部分。

---

## 规划（Planning / VLM Brain）模块适配工作

**负责同学**：规划模块（jinao VLM Brain）开发者  
**文件**：`embodiedbench/connection/ntu_jinao_repo/vlm_brain/run_vlm_inference.py`

### 与 Memory 连接

**目标**：VLM Brain 生成 Blueprint 前，先读取 memory 的历史知识（成功率、推荐技能序列、物体位置先验）。

**原因**：当前 `build_generation_prompt()` 只接受 `task + scene_state`，不知道历史经验。Memory 中积累的 Blueprint 技能序列（`recommended_blueprint_skills`）可以直接作为 prompt 中的参考，提升生成质量。

**需要做的事**：

**方案 A（推荐，文件解耦）**：Memory 把历史上下文写成文件，VLM Brain 读文件。不改架构，最安全。

1. 在驱动脚本（demo 或评估器）中，VLM Brain 调用之前写文件：

```python
# 在调用 run_vlm_inference.py 之前
memory.export_vlm_context(task_instruction, output_dir / "memory_context.json")
```

2. 修改 `run_vlm_inference.py`，新增 `--memory_context` 参数：

```python
parser.add_argument("--memory_context", default=None,
                    help="Path to memory_context.json written by memory.export_vlm_context()")
```

3. 修改 `build_generation_prompt()` 接受并注入 memory_context：

```python
def build_generation_prompt(task, scene_state, memory_context=None):
    memory_section = ""
    if memory_context:
        rate = memory_context.get("task_success_rate")
        skills = memory_context.get("recommended_blueprint_skills", [])
        similar = memory_context.get("similar_episodes", [])
        if rate is not None:
            memory_section += f"\n## Historical performance\nSuccess rate for this task type: {rate:.1%}\n"
        if skills:
            memory_section += f"Previously successful skill sequence: {skills}\n"
        if similar:
            memory_section += "Similar past episodes:\n"
            for ep in similar[:2]:
                memory_section += f"  - {ep['task']} → {'✓' if ep['success'] else '✗'} skills={ep['blueprint_skills']}\n"
    return (
        f"{system_prompt}\n\n"
        f"{memory_section}"
        f"{generation_prompt}\n\n"
        ...
    )
```

4. 在 `main()` 中读取并传入：

```python
memory_context = None
if args.memory_context:
    memory_context = _read_json(Path(args.memory_context))
prompt = build_generation_prompt(args.task, scene_state, memory_context)
```

**方案 B（进程内调用）**：如果 VLM Brain 和 memory 在同一进程中运行：

```python
from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem
memory = EmbodiedManipulationMemorySystem(...)
ctx = memory.query_vlm_context(task)
prompt = build_generation_prompt(task, scene_state, ctx)
```

### 脊脑执行结果回写 Memory

**目标**：脊脑执行完一次 Blueprint 后，把结果（成功/失败、执行的技能序列）写回 memory，让下次规划能参考。

**需要做的事**：在执行完成后调用（由驱动脚本负责，不需要 VLM Brain 内部改动）：

```python
memory.record_blueprint_execution(
    task_instruction=task,
    blueprint_skills=["move_above", "descend", "grasp", "lift", "place", "retreat"],
    success=True,
    scene_state=final_scene_state,  # 最终场景状态（可选）
)
```

---

## Monitor 模块适配工作

**负责同学**：Monitor / 安全监控开发者  
**文件**：`embodiedbench/memory_manip/interfaces.py`（参考模板），新建 `embodiedbench/monitor/spine_monitor.py`

### 与 Memory 连接

**目标**：用脊脑执行结果驱动 memory 的 monitor 钩子，让 memory 能感知执行异常并记录里程碑。

**原因**：当前 `NullMonitor` 永远返回 OK，memory 永远不知道执行过程中发生了碰撞或超时。Monitor 回报 `ok=False` 时，memory 会在 working memory 的 clock 中添加 `monitor_alert` 里程碑，这个里程碑会被 episode 记录下来，供未来分析。

**需要做的事**：

1. 实现 `SpineConditionMonitor`，包装脊脑的 condition_evaluator 结果：

```python
# embodiedbench/monitor/spine_monitor.py
from embodiedbench.memory_manip.interfaces import MonitorInterface

class SpineConditionMonitor(MonitorInterface):
    """读取脊脑 condition_evaluator.py 的执行结果，汇报给 memory。"""

    def check(self, observation, memory_snapshot):
        # observation 由评估器/驱动脚本在每个 env.step() 后传入
        # 格式：{"timeout": bool, "collision_detected": bool, "task_success": bool, ...}
        timeout   = observation.get("timeout", False)
        collision = observation.get("collision_detected", False)
        if timeout:
            return False, "execution_timeout"
        if collision:
            return False, "collision_detected"
        return True, None
```

2. 注册到 memory：

```python
from embodiedbench.monitor.spine_monitor import SpineConditionMonitor
memory.register_monitor(SpineConditionMonitor())
```

3. 评估器在每个 step 后调用（已有此调用，无需新增）：

```python
memory.run_monitor({"obs": obs, "timeout": info.get("timeout"), "collision_detected": info.get("collision_detected")})
```

### 与脊脑连接

**目标**：接收脊脑 `condition_evaluator.py` 的输出，并在上层（大脑）判断是否需要重新规划。

**脊脑侧待补全**：`collision_detected` 条件（`condition_evaluator.py` 第34行 TODO）目前永远返回 False，需力传感器数据接入。这是脊脑同学的任务。

**大脑侧重规划逻辑**（驱动脚本中，非必要步骤）：

```python
ok = memory.run_monitor(execution_result)
if not ok:
    # 监控报警，重新生成 Blueprint
    memory.export_vlm_context(task, output_dir / "memory_context.json")
    # 重新调用 run_vlm_inference.py
```

---

## Simulator（仿真平台）适配工作

**负责同学**：Isaac Sim 集成开发者  
**文件**：新建 `embodiedbench/envs/isaac_sim/isaac_sim_env.py`；`embodiedbench/memory_manip/interfaces.py` 中的 `SimulationInterface`

### 与 Memory 连接

**目标**：Isaac Sim 提供完整的 `scene_state` dict（含 ee_pose、gripper_width、joint_pos），以便 memory 和 VLM Brain 能使用真实的机器人状态。

**原因**：感知只能提供物体 pose（camera 看到的），末端执行器位姿和关节角度必须来自仿真平台。如果这部分缺失，`scene_state.json` 就是不完整的，脊脑无法正常工作。

**需要做的事**：

1. Isaac Sim 场景每步提供如下 dict，传给大脑驱动脚本：

```python
robot_state = {
    "ee_pose":         sim.get_ee_pose(),        # [x,y,z,qw,qx,qy,qz]
    "gripper_width":   sim.get_gripper_width(),  # float, 单位米
    "joint_pos":       sim.get_joint_angles(),   # list[float], 7 DOF
}
```

2. 驱动脚本把感知输出 + robot_state 合并后写入 scene_state.json，再调用 `memory.ingest_scene_state(scene_state)`：

```python
scene_state = build_scene_state(perception_out, robot_state, step_index=step)
memory.ingest_scene_state(scene_state)
# 写文件供脊脑读取
with open(output_dir / "scene_state.json", "w") as f:
    json.dump(scene_state, f)
```

3. **阶段二（可选）**：实现 `IsaacForwardModel(SimulationInterface)` 以支持 memory 的 `propose_simulated_outcome()`：

```python
# embodiedbench/envs/isaac_sim/isaac_forward_model.py
from embodiedbench.memory_manip.interfaces import SimulationInterface

class IsaacForwardModel(SimulationInterface):
    def predict(self, state_summary, action, *, horizon=1):
        # 调用 Isaac Sim physics rollout
        return {"predicted_ee_pose": ..., "predicted_object_pose": ...}

memory.register_simulation(IsaacForwardModel())
```

### Isaac Sim 与脊脑的连接

**脊脑已实现的接口**（`cerebellum.get_scene_state()`）：
- 脊脑内部直接调用 Isaac Sim 场景获取 `{ee_pose_w, cube_pose_w, target_pose_w, gripper_width}`
- 这部分已在脊脑代码中定义，由 Isaac Sim 同学确保 Isaac Sim 场景中的 ArticulationView / RigidPrimView 能正确返回这些值

**Isaac Sim 同学需确认**：
- Franka 末端执行器的 frame 名（通常是 `panda_hand` 或 `right_fingertip_midpoint`）与脊脑 `get_scene_state()` 中的 query 匹配
- 物体（cube、target）的 prim path 与脊脑的 `scene_objects` 配置一致
- 场景重置时（`env.reset()`）确保物体位置正确随机化且返回新的 scene_state

---

## Management 模块适配工作

**负责同学**：移动底盘控制开发者  
**文件**：`embodiedbench/management/`（现有），新建 `embodiedbench/management/orchestrator.py`

### 与 Memory 连接

**目标**：导航完成后把目标位置信息写入 memory 的 spatial_topo，让 memory 积累空间知识。

**原因**：Management 模块导航过程中经过了哪些位置、在哪里找到了目标，这些都是有价值的空间知识。如果不写入 memory，每次导航都是从零开始，无法利用历史经验。

**前提**：先确认机器人构型（纯臂 / 移动底盘+臂 / G1 人形）。纯臂时本节可跳过。

**需要做的事**（移动操作臂场景）：

```python
# 导航结束后，把当前位置写入 memory 的 spatial_topo
memory.semantic.spatial_topo.add_location(
    location="table_area",
    pose=[x, y, z, 0, 0, 0, 1]  # 如果有精确位姿
)
memory.semantic.spatial_topo.record_object_at("table_area", "cube", count=1)
memory.semantic.save()

# 导航路径上如果经过多个位置，可建立边（相邻关系）
memory.semantic.spatial_topo.add_edge("start_pos", "table_area")
```

### 与脊脑连接（移动操作臂构型）

**目标**：Management 导航到位后，自动触发脊脑执行 Blueprint，形成"导航 → 操作"的串行流程。

**需要做的事**：在 `embodiedbench/management/orchestrator.py` 中新建协调层：

```python
class MobileManipulatorOrchestrator:
    """协调底盘导航与机械臂操作两个子系统。"""

    def __init__(self, nav_loop, memory, vlm_brain_runner, spine_brain_client):
        self.nav = nav_loop           # TaskPlanningLoop
        self.memory = memory          # EmbodiedManipulationMemorySystem
        self.vlm = vlm_brain_runner   # 调用 run_vlm_inference.py 的 wrapper
        self.spine = spine_brain_client  # 调用脊脑执行接口

    def execute_task(self, task_instruction, target_location):
        # 阶段1：导航
        nav_success = self.nav.navigate_to(target_location)  # 需在 TaskPlanningLoop 中新增此方法
        if not nav_success:
            return False

        # 阶段2：感知 + memory 查询
        scene_state = self._get_current_scene_state()
        self.memory.ingest_scene_state(scene_state)
        ctx_path = self.memory.export_vlm_context(task_instruction, "tmp/memory_context.json")

        # 阶段3：VLM Brain 生成 Blueprint
        blueprint = self.vlm.generate(task_instruction, scene_state, ctx_path)

        # 阶段4：脊脑执行
        result = self.spine.execute(blueprint)
        executed_skills = result.get("executed_skills", [])
        success = result.get("success", False)

        # 阶段5：结果回写 memory
        self.memory.record_blueprint_execution(
            task_instruction, executed_skills, success, scene_state
        )
        return success
```

`SapienVelocityBody` 替换为 `IsaacVelocityBody`（已在审查 §模块六 中给出骨架）。

---

## 整体数据流（改造完成后）

```
Isaac Sim
    │
    ├─► robot_state (ee_pose, gripper_width, joint_pos)
    │
感知模块 ──► perception_out (cube_pose, target_pose)
    │
    └─► scene_state_adapter.build_scene_state()
              │
              ├─► scene_state.json ──────────────────────────► 脊脑
              │
              ├─► memory.ingest_scene_state(scene_state)
              │         │
              │         └─► working memory 更新 robot_state + observation
              │
              └─► memory.export_vlm_context(task, "memory_context.json")
                        │
                        ▼
               VLM Brain (run_vlm_inference.py)
                   --memory_context memory_context.json
                   --scene_state scene_state.json
                   --task task.txt
                        │
                        ▼
               skill_blueprint.json ────────────────────────► 脊脑执行
                                                                  │
                                          memory.record_blueprint_execution() ◄─────────┘
                                                  │
                                         TaskSchema 更新（下次参考）
```

---

## 改动优先级排序

| 优先级 | 模块 | 具体任务 | 理由 |
|--------|------|---------|------|
| **P0** | Simulator (Isaac Sim) | 提供完整 `robot_state`（ee_pose / gripper_width / joint_pos） | 没有这个，scene_state.json 不完整，脊脑无法运行 |
| **P0** | Perception | 写 `scene_state_adapter.py`，合并感知输出和 robot_state | scene_state.json 的直接来源 |
| **P1** | Planning (VLM Brain) | `build_generation_prompt()` 加 `--memory_context` 参数 | 让 memory 知识进入 Blueprint 生成 |
| **P1** | 驱动脚本（新建） | 串联 memory → export → VLM Brain → 脊脑 → 回写的完整流程 | 当前各模块之间没有调用关系，需要胶水脚本 |
| **P2** | Monitor | 实现 `SpineConditionMonitor`，读取脊脑执行结果 | 有了才能实现"执行失败 → 重新规划"闭环 |
| **P3** | Management | 写 `MobileManipulatorOrchestrator`（仅移动底盘构型） | 纯臂场景可跳过 |
| **P3** | Simulator | 实现 `IsaacForwardModel(SimulationInterface)` | 可选，供 memory 做前向预测用 |
