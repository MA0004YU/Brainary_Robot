# Planning 模块接入 Memory 指南

**读者**：负责 Planner / Simulation / Project Management 模块的同学  
**版本**：最新（含 streaming 支持 + clear_occlusions）  
**前提**：解压 `brainary_memory_pkg.zip`，设置好 PYTHONPATH（见 `README_FIRST.md`）

---

## 一、你拿到什么

Memory 模块每一步都会输出一个 JSON 文件，**这是你和 memory 之间的唯一接口**：

```json
{
  "task_instruction": "把食品放到篮子一，把杯子放到篮子二，把文具放到篮子三",

  "manipulable_objects": {
    "香蕉":    ["grasp", "place"],
    "大食品盒": ["grasp", "place"],
    "水杯一":  ["grasp", "place"],
    "剪刀":    ["grasp", "place"]
  },

  "available_skills": [
    "move_above", "descend", "grasp", "lift",
    "place", "retreat", "wait", "align_orientation", "reach"
  ],

  "constraints": {
    "category_rules": {
      "食品类": "篮子一",
      "杯具类": "篮子二",
      "工具类": "篮子三"
    },
    "no_category_mixing": true,
    "collision_avoidance": true,
    "occlusions": [
      {
        "blocked":          "香蕉",
        "blocker":          "大食品盒",
        "blocker_on_list":  true,
        "target":           "篮子三"
      }
    ]
  }
}
```

---

## 二、三个字段从哪里来

| 字段 | 内容 | Memory 来源 |
|------|------|-------------|
| `manipulable_objects` | 可操作物体 + 抓取能力 | 语义记忆 ObjectKB（历史积累）；无历史时默认 `["grasp","place"]` |
| `available_skills` | 完整可用 skill 列表 | 脊脑接口静态定义 + 语义记忆 TaskSchema 动态补充 |
| `constraints` | 物理规则 + 遮挡信息 | `category_rules` 由 Planner 注入；`occlusions` 由 Simulation 写入 |

---

## 三、Streaming 工作流（每步循环）

**这个接口是为循环调用设计的，不是一次性的。** 每个控制步骤 Memory 都会刷新 JSON 文件，Planner 读到的永远是当前步的最新状态。

```
┌─────────────────────────────────────────────────────────┐
│  每步循环                                                 │
│                                                         │
│  感知 → 更新 Memory → 写 planning_input.json            │
│                              │                          │
│                              ▼                          │
│                         Planner 读文件 → 规划           │
│                              │                          │
│                              ▼                          │
│                    Simulation 仿真验证                   │
│                    （发现遮挡 → 写回 Memory）             │
│                              │                          │
│                              ▼                          │
│                    PM 优化 → 机械臂执行                  │
│                              │                          │
│                              ▼                          │
│                    记录动作结果 → 下一步                  │
└─────────────────────────────────────────────────────────┘
```

文件写入是**原子操作**（先写 `.tmp` 再 rename），Planner 不会读到写了一半的内容。

---

## 四、各模块调用什么 API

### Planner 模块

```python
# 任务开始时：注入分类规则（只调用一次）
pipeline.set_task_constraints({
    "category_rules": {
        "食品类": "篮子一",
        "杯具类": "篮子二",
        "工具类": "篮子三",
    },
    "no_category_mixing": True,
    "collision_avoidance": True,
})

# 每步：读 JSON 文件做规划
import json
data = json.load(open("planning_input.json", encoding="utf-8"))
objects  = data["manipulable_objects"]   # 可操作物体
skills   = data["available_skills"]      # 可用 skill
rules    = data["constraints"]           # 规则 + 遮挡
```

### Simulation 模块

```python
# 发现遮挡时写入
pipeline.record_occlusion(
    blocked_object="香蕉",
    blocker_object="大食品盒",
    blocker_on_collection_list=True,   # 大食品盒也在收纳清单
    blocker_target_basket="篮子三",
)

# 遮挡解决后清除（支持精确清除或全部清除）
pipeline.clear_occlusions(blocked_object="香蕉")  # 只清"香蕉"的遮挡
# pipeline.clear_occlusions()                      # 清除所有遮挡记录
```

### PM 模块

```python
# 读 constraints，做优化决策
data = json.load(open("planning_input.json", encoding="utf-8"))

for oc in data["constraints"].get("occlusions", []):
    if oc["blocker_on_list"]:
        # 遮挡物也在收纳清单 → 直接放到目标篮子（一步到位）
        print(f"{oc['blocker']} 直接放 {oc['target']}，省去临时搬运")
    else:
        # 遮挡物不在清单 → 临时移开，抓完目标后放回
        print(f"临时移开 {oc['blocker']}")
```

### 记录动作结果（所有模块）

```python
pipeline.record_action(
    action="grasp 大食品盒",
    success=True,
    feedback="gripper closed",
    reasoning="大食品盒在收纳清单，直接放篮子三",
)
```

---

## 五、完整流程示例

```python
import json
from memory_module import PerceptionMemoryPipeline

# ── 初始化（由集成同学完成，Planning 同学接收 pipeline 对象）──
pipeline = PerceptionMemoryPipeline.create(store_dir="memory_store/")
pipeline.session_start()

# ── Episode 开始 ──
pipeline.begin_episode(
    scene_id="sort_001",
    task_instruction="把食品放到篮子一，把杯子放到篮子二，把文具放到篮子三",
)

# Planner：注入任务约束（一次性）
pipeline.set_task_constraints({
    "category_rules": {"食品类": "篮子一", "杯具类": "篮子二", "工具类": "篮子三"},
    "no_category_mixing": True,
    "collision_avoidance": True,
})

# ── 主循环（每步）──
while not task_done:

    # Memory 更新感知（由集成同学的代码调用）
    pipeline.process_perception(["current_frame.png"], current_location="table")

    # Memory 输出 JSON 给 Planning
    pipeline.export_planning_input("planning_input.json")

    # ── Planner 读文件 ──
    data = json.load(open("planning_input.json", encoding="utf-8"))
    # data["manipulable_objects"] → 规划抓取对象
    # data["available_skills"]   → 选择执行 skill
    # data["constraints"]        → 遵守规则

    # ── Simulation 仿真验证 ──
    # 如果发现遮挡：
    pipeline.record_occlusion(
        blocked_object="香蕉",
        blocker_object="大食品盒",
        blocker_on_collection_list=True,
        blocker_target_basket="篮子三",
    )
    # Memory 刷新（遮挡信息现在在 constraints 里）
    pipeline.export_planning_input("planning_input.json")

    # ── PM 优化 ──
    data = json.load(open("planning_input.json", encoding="utf-8"))
    for oc in data["constraints"].get("occlusions", []):
        if oc["blocker_on_list"]:
            print(f"PM: {oc['blocker']} 一步到位放 {oc['target']}")

    # ── 执行动作 ──
    pipeline.record_action("grasp 大食品盒", success=True, feedback="ok")

    # ── 遮挡已解决，清除记录 ──
    pipeline.clear_occlusions(blocked_object="香蕉")

# ── Episode 结束 ──
pipeline.end_episode(
    success=True,
    blueprint_skills=["move_above", "grasp", "lift", "place", "retreat"],
)
pipeline.session_end()
```

---

## 六、API 速查

| 方法 | 调用方 | 说明 |
|------|--------|------|
| `pipeline.set_task_constraints(dict)` | Planner | episode 开始时注入分类规则，一次性 |
| `pipeline.export_planning_input(path)` | 集成/Memory | 每步输出 JSON，覆盖写入，原子操作 |
| `pipeline.record_occlusion(...)` | Simulation | 检测到遮挡时写入，下步 JSON 中可见 |
| `pipeline.clear_occlusions(blocked="")` | Simulation | 遮挡解决后清除；不传参数则清全部 |
| `pipeline.record_action(action, success, feedback)` | 所有模块 | 每次执行后记录结果 |
| `pipeline.get_planning_context()` | 任意 | 返回 PlanningContext 对象（Python API） |

---

## 七、注意事项

- `occlusions` 只增不减，**必须在遮挡解决后调用 `clear_occlusions()`**，否则旧遮挡记录会一直出现在 JSON 里
- `available_skills` 第一次 episode 时只有静态默认列表；随着 episode 积累，TaskSchema 会补充历史出现过的 skill
- `manipulable_objects` 里的 affordance 在前几次 episode 都是默认值 `["grasp","place"]`，随着机器人积累经验会越来越准确
- JSON 文件路径由集成同学统一约定，Planning 模块只需要读同一个路径
