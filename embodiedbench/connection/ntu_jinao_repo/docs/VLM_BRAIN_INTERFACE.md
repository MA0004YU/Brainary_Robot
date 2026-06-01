# VLM/大脑模块接口说明（可替换协议）

本文档面向要“替换现有 VLM 模型”的同学。目标是让你们只做“大脑决策”，就能无缝接到当前技能执行器（小脑）上完成机器人控制。

---

## 1. 一句话接口定义

你们的模块需要完成这件事：

- **输入**：场景信息（`scene_state` + 可选图像）与任务文本（`task`）
- **输出**：一个合法的 `skill_blueprint` JSON（符合本仓库蓝图 schema）

执行器不会接收你们输出的连续控制动作。高频控制由 `franka_state_machine_cerebellum` 内部 IK-Rel 控制器完成。

---

## 2. 端到端数据流（替换点）

当前闭环可以抽象为：

1. 仿真端准备 VLM 输入
   - `01_vlm_inputs/<run_id>/scene_state.json`
   - `01_vlm_inputs/<run_id>/task.txt`
   - `01_vlm_inputs/<run_id>/image.png`（可选但强烈建议）
2. 大脑模块读取输入，生成 `skill_blueprint`（JSON）
3. 执行器读取蓝图并执行（`run_skill_blueprint_executor.py`）
4. 产出执行日志与性能反馈（可再回传大脑迭代）

你们需要替换的是第 2 步，不需要改执行器控制逻辑。

---

## 3. 输入接口（你们会拿到什么）

## 3.1 文件级输入（推荐）

若用统一 run 布局，输入目录为：

- `data/01_vlm_inputs/<run_id>/scene_state.json`
- `data/01_vlm_inputs/<run_id>/task.txt`
- `data/01_vlm_inputs/<run_id>/image.png`（如果仿真启用相机）

其中：

- `task.txt`：自然语言任务，例如 `pick the cube and place it on the target`
- `scene_state.json`：结构化场景状态
- `image.png`：当前场景截图（固定机位）

## 3.2 `scene_state` 字段（执行器实际使用）

执行器内用于 Stage-II 的核心场景字段（来自 `_stage2_scene_state`）：

- `cube_pose`: 7D `[x, y, z, qw, qx, qy, qz]`
- `target_pose`: 7D `[x, y, z, qw, qx, qy, qz]`
- `ee_pose`: 7D `[x, y, z, qw, qx, qy, qz]`
- `gripper_width`: 夹爪宽度
- `robot_joint_pos`: 关节位置数组
- `step_index`: 当前步号

注意：

- 你们也可能看到 `vlm_brain/examples/sample_scene_state.json` 那种更“语义化”的格式（`objects/robot/frame`）。两种都可以支持，但最终产出蓝图时必须引用执行器认可的目标名（见下文）。

---

## 4. 输出接口（你们必须产出什么）

你们必须输出一个 **合法 JSON 对象**，顶层结构：

```json
{
  "blueprint_id": "string",
  "task": "string",
  "execution_graph": {
    "start": "node_id",
    "logic": "sequence | condition | parallel",
    "nodes": {}
  }
}
```

校验器与加载器在这些文件里：

- `vlm_brain/schemas/skill_blueprint_schema.json`
- `source/standalone/franka_state_machine_cerebellum/skill_blueprint_loader.py`
- `source/standalone/franka_state_machine_cerebellum/skill_blueprint_schema.py`

只要你们输出通过上述校验，执行器就能接管并执行。

---

## 5. 蓝图节点类型（必须理解）

`nodes` 中每个节点 `type` 必须是以下之一：

- `skill`：原子技能节点
- `parallel`：并行目标节点（位置+姿态同循环收敛）
- `condition`：条件分支节点
- `terminal`：终止节点（success/failure）

## 5.1 skill 节点

最常见结构：

```json
{
  "type": "skill",
  "skill": "move_above",
  "target": "cube",
  "params": { "...": "..." },
  "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
  "next": "next_node",
  "on_failure": "fallback_node"
}
```

## 5.2 parallel 节点

结构：

```json
{
  "type": "parallel",
  "parallel_mode": "all_success",
  "goals": {
    "position_goal": { "skill": "move_above|reach", "target": "...", "params": {} },
    "orientation_goal": { "skill": "align_orientation", "target": "...", "params": {} }
  },
  "params": { "timeout_steps": 400, "gripper": "keep|open|close" },
  "performance_query": [...],
  "next": "next_node",
  "on_failure": "fallback_node"
}
```

执行语义（非常重要）：

- 位置和姿态在同一个控制循环内同时推进
- 要求“同一时刻”满足收敛，并且连续确认若干步后才判定成功
- 适合“先到目标上方并对齐姿态”的阶段

## 5.3 condition 节点

结构：

```json
{
  "type": "condition",
  "condition": { "name": "object_in_gripper", "...": "..." },
  "if_true": "node_a",
  "if_false": "node_b"
}
```

已支持条件名：

- `object_in_gripper`
- `object_near_target`
- `ee_reached_target`
- `timeout`
- `collision_detected`

## 5.4 terminal 节点

结构：

```json
{ "type": "terminal", "result": "success" }
```

或

```json
{ "type": "terminal", "result": "failure", "failure_reason": "skill_failed" }
```

---

## 6. 目标命名约束（target）

执行器内置支持的目标引用：

- `cube`
- `target`
- `current`

请不要输出其他 target 字符串，除非你同时扩展执行器 `_target_pose()`。

---

## 7. 每个技能的详细说明（重点）

以下是当前执行器支持的 9 个 primitive skills。每个技能都由小脑完成底层控制，你们只需产出参数。

## 7.1 `move_above`

- **作用**：移动到目标（如 cube/target）上方
- **常用阶段**：抓取前、放置前
- **关键参数**：
  - `height_offset`（必需）
  - `xy_offset`（必需）
  - `speed`（必需）
  - `position_tolerance`（必需）
  - `timeout_steps`（必需）
  - `orientation_tolerance`（可选）
  - `orientation_mode` / `keep_top_down` / `fixed_yaw` / `target_rpy`（可选）
- **执行特性**：位置与姿态均参与收敛判断（不仅看位置）

## 7.2 `reach`

- **作用**：到达某个明确位姿/参考点（可带 offset）
- **关键参数**：
  - `speed`、`position_tolerance`、`timeout_steps`（必需）
  - 需要 `target_pose` 或 `target_ref` 二选一
  - 可选 `offset` 与姿态参数组（同上）
- **适用**：精确过渡位姿、非标准抓取入口

## 7.3 `descend`

- **作用**：沿 Z 方向下探接近目标
- **关键参数**：
  - `speed`、`position_tolerance`、`timeout_steps`（必需）
  - `target_height` 或 `relative_z` 至少一个
  - 可选姿态参数组
- **执行特性（关键）**：
  - 下降过程会持续刷新目标 XY（跟踪目标当前位置）
  - 可降低“上方对齐后下降偏移”的风险

## 7.4 `grasp`

- **作用**：闭合夹爪并等待稳定
- **关键参数**：
  - `close_wait_steps`（必需）
  - `check`（必需，通常 `object_in_gripper`）
  - `timeout_steps`（必需）
- **执行特性**：主要是夹爪控制，不做空间移动

## 7.5 `lift`

- **作用**：抓取后向上抬升
- **关键参数**：
  - `lift_height`、`speed`、`position_tolerance`、`timeout_steps`（必需）
  - 可选姿态参数组
- **建议**：抬升后接条件节点确认是否真正抓住

## 7.6 `place`

- **作用**：移动到放置位并开爪释放
- **关键参数**：
  - `place_height`、`release_height`、`open_wait_steps`
  - `position_tolerance`、`target_tolerance`
  - `timeout_steps`（以上均为必需）
  - 可选姿态参数组
- **执行特性**：
  - 先移动到 place pose
  - 再执行 open hold

## 7.7 `retreat`

- **作用**：释放后上抬撤离
- **关键参数**：
  - `retreat_height`、`speed`、`timeout_steps`（必需）
  - `position_tolerance` 可选
  - 可选姿态参数组
- **注意**：`retreat` 的 `position_tolerance` 在 loader 中是可选

## 7.8 `wait`

- **作用**：保持一段时间，可指定夹爪状态
- **关键参数**：
  - `wait_steps`（必需）
  - `gripper`（必需，`open|close|keep`）

## 7.9 `align_orientation`

- **作用**：纯姿态对齐
- **关键参数**：
  - `orientation_mode`、`orientation_tolerance`、`angular_speed`、`timeout_steps`（必需）
  - 可选 `keep_top_down`、`fixed_yaw`、`target_rpy`
- **常见用途**：
  - 在 parallel 节点中作为 `orientation_goal`
  - 单独用于姿态修正阶段

---

## 8. `orientation_mode` 设计建议

常用模式策略建议（请按任务选）：

- 对准物体抓取：`align_yaw_with_target` + `keep_top_down: true`
- 仅保持当前姿态：`keep_current`
- 禁止姿态控制：`none`（仅特殊情况下使用）

如果模型不确定，优先输出稳定保守策略：

- 抓取前并行：`align_yaw_with_target + keep_top_down`
- `descend/lift/retreat`：倾向 `keep_current`

---

## 9. `performance_query` 怎么选

`performance_query` 是“希望采集/评估哪些指标”的声明。可用指标列表见：

- `skill_blueprint_schema.py` 中 `PERFORMANCE_METRICS`

最低建议集合（任何节点都可用）：

- `success`
- `execution_steps`
- `timeout`
- `failure_reason`

对于关键节点建议额外加：

- 位姿误差：`final_ee_position_error`、`final_ee_orientation_error`
- 物体结果：`final_object_position`、`object_target_position_error`、`object_target_xy_error`
- 速度安全：`final_ee_linear_speed`、`average_ee_linear_speed`

---

## 10. 输出案例模板（可直接改字段）

下面给一份“可替换接口模板”（并行抓取 + 并行放置）：

```json
{
  "blueprint_id": "bp_<team>_<timestamp>",
  "task": "<自然语言任务>",
  "execution_graph": {
    "start": "p1",
    "logic": "parallel",
    "nodes": {
      "p1": {
        "type": "parallel",
        "parallel_mode": "all_success",
        "goals": {
          "position_goal": {
            "skill": "move_above",
            "target": "cube",
            "params": {
              "height_offset": 0.12,
              "xy_offset": [0.0, 0.0],
              "speed": 0.08,
              "position_tolerance": 0.02
            }
          },
          "orientation_goal": {
            "skill": "align_orientation",
            "target": "cube",
            "params": {
              "orientation_mode": "align_yaw_with_target",
              "keep_top_down": true,
              "orientation_tolerance": 0.08,
              "angular_speed": 0.08
            }
          }
        },
        "params": {
          "timeout_steps": 400,
          "gripper": "keep"
        },
        "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
        "next": "s2",
        "on_failure": "t_failure"
      },
      "s2": {
        "type": "skill",
        "skill": "descend",
        "target": "cube",
        "params": {
          "target_height": 0.025,
          "speed": 0.035,
          "position_tolerance": 0.02,
          "timeout_steps": 220
        },
        "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
        "next": "s3",
        "on_failure": "t_failure"
      },
      "s3": {
        "type": "skill",
        "skill": "grasp",
        "target": "cube",
        "params": {
          "close_wait_steps": 80,
          "check": "object_in_gripper",
          "timeout_steps": 100
        },
        "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
        "next": "c1",
        "on_failure": "t_failure"
      },
      "c1": {
        "type": "condition",
        "condition": {
          "name": "object_in_gripper",
          "object": "cube",
          "min_lift_delta": 0.03,
          "max_ee_object_distance": 0.08
        },
        "if_true": "s4",
        "if_false": "t_failure_grasp"
      },
      "s4": {
        "type": "skill",
        "skill": "lift",
        "target": "cube",
        "params": {
          "lift_height": 0.18,
          "speed": 0.08,
          "position_tolerance": 0.025,
          "timeout_steps": 260
        },
        "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
        "next": "p2",
        "on_failure": "t_failure"
      },
      "p2": {
        "type": "parallel",
        "parallel_mode": "all_success",
        "goals": {
          "position_goal": {
            "skill": "move_above",
            "target": "target",
            "params": {
              "height_offset": 0.14,
              "xy_offset": [0.0, 0.0],
              "speed": 0.08,
              "position_tolerance": 0.035
            }
          },
          "orientation_goal": {
            "skill": "align_orientation",
            "target": "target",
            "params": {
              "orientation_mode": "align_yaw_with_target",
              "keep_top_down": true,
              "orientation_tolerance": 0.08,
              "angular_speed": 0.08
            }
          }
        },
        "params": {
          "timeout_steps": 350,
          "gripper": "close"
        },
        "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
        "next": "s6",
        "on_failure": "t_failure"
      },
      "s6": {
        "type": "skill",
        "skill": "place",
        "target": "target",
        "params": {
          "place_height": 0.045,
          "release_height": 0.08,
          "open_wait_steps": 60,
          "position_tolerance": 0.035,
          "target_tolerance": 0.05,
          "timeout_steps": 320
        },
        "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
        "next": "s7",
        "on_failure": "t_failure"
      },
      "s7": {
        "type": "skill",
        "skill": "retreat",
        "target": "current",
        "params": {
          "retreat_height": 0.15,
          "speed": 0.08,
          "timeout_steps": 180
        },
        "performance_query": ["success", "execution_steps", "timeout", "failure_reason"],
        "next": "t_success",
        "on_failure": "t_failure"
      },
      "t_success": {
        "type": "terminal",
        "result": "success"
      },
      "t_failure": {
        "type": "terminal",
        "result": "failure",
        "failure_reason": "skill_failed"
      },
      "t_failure_grasp": {
        "type": "terminal",
        "result": "failure",
        "failure_reason": "object_not_in_gripper"
      }
    }
  }
}
```

---

## 11. 你们实现大脑模块时的强约束（Checklist）

上线前请逐项确认：

- 输出必须是单一 JSON 对象，不能混 markdown、解释文字
- 节点图无环，`start` 可达
- 每个分支最终可到达 `terminal`
- `skill/condition/parallel` 名字都在支持集合内
- `target` 仅使用 `cube|target|current`
- 必需参数齐全（见 loader 规则）
- `performance_query` 指标都在允许列表中

建议执行前先走一次本地校验器：

- `vlm_brain/blueprint_validator.py`

---

## 12. 常见失败原因（便于大脑做防御）

常见 `failure_reason` 包括：

- `timeout`
- `orientation_not_converged`
- `position_not_converged`
- `object_not_in_gripper`
- `object_not_near_target`
- `skill_failed`
- `env_done`

建议你们在策略中做这些防御：

- 抓取前使用 parallel 做“位置+姿态”同时收敛
- `descend` 速度不要过高，避免触发抓取错位
- 对 `grasp` 后强制接 `condition` 检查
- `place` 后接 `object_near_target` 检查

---

## 13. 推荐给“可替换大脑”团队的最小实现

最小可用版本（MVP）：

1. 读取 `task + scene_state (+ image)`
2. 直接填充本文件第 10 节模板（仅改参数）
3. 调用 validator 校验
4. 输出 `generated_blueprint.json`

这样就能先打通与技能执行器交互，再逐步替换为真正的 VLM/策略模型。

