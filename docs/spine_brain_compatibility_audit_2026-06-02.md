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

**最关键的一条**：ManipPlanner 输出的是 SAPIEN 离散动作，和脊脑需要的 Blueprint JSON 是两套完全不同的架构。如果 Brainary_Robot 的规划器要参与新架构，需要重写 prompt 和解析器；如果只用 jinao 的 VLM Brain 做规划，ManipPlanner 在新流程里可以完全跳过。**需要先确认规划模块的分工**。
