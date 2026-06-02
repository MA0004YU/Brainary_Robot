# Brainary_Robot 可实现 Demo 指南

本文档面向各模块负责同学，列出 5 个从简到难、可独立运行的 demo。每个 demo 只依赖其对应模块，不强求其他模块就绪，方便并行开发和单独展示。

---

## Demo 总览

| # | 名称 | 主要模块 | 复杂度 | 是否需要 LLM | 是否需要 Isaac Sim |
|---|---|---|---|---|---|
| 1 | 物理检测器压力测试 | `detectors.py` + `scene_builder.py` | ★☆☆☆☆ | 否 | 可选 |
| 2 | 机械臂蓝图图执行 | `robot_controller.py` + `scene_builder.py` | ★★☆☆☆ | 否 | 可选 |
| 3 | 三层记忆生命周期 | `memory_manip/` | ★★☆☆☆ | 否（mock）| 否 |
| 4 | 意图规划闭环对话 | `Planning_module/` | ★★★☆☆ | 是 | 否 |
| 5 | 端到端操作全链路 | 全部模块 | ★★★★☆ | 是 | 推荐 |

---

## Demo 1：物理检测器压力测试

### 目标
不依赖机器人、不依赖 LLM，只用 `SceneBuilder` 搭出一张桌子场景，然后用脚本主动施加过载力，验证四种 `PhysicsBoundaryDetectors` 是否能正确触发并返回结构化 Feedback。

### 展示内容
- `SceneBuilder.build_twin_world()` 从 hardcoded JSON 构建 SAPIEN 场景
- `PhysicsBoundaryDetectors` 四个检测器全部触发一遍：
  - `DestructionFeedback`（超过材质屈服力）
  - `TippingFeedback`（物体倾斜角超限）
  - `IntrusionFeedback`（意外碰撞）
  - `DeadlockKinematicFeedback`（关节力矩饱和）

---

### 负责同学（仿真/物理检测组）代码改动

**第一步：准备 hardcoded 场景 JSON**

在 `embodiedbench/simulation/inputs/` 下新建 `demo1_scene.json`：

```json
{
  "objects": [
    {
      "name": "plastic_box_1",
      "size_whd": [0.15, 0.12, 0.10],
      "pose": [0.5, 0.0, 0.06, 0, 0, 0, 1]
    },
    {
      "name": "iron_block_1",
      "size_whd": [0.08, 0.08, 0.08],
      "pose": [0.5, 0.0, 0.18, 0, 0, 0, 1]
    },
    {
      "name": "wooden_rack_1",
      "size_whd": [0.60, 0.40, 0.03],
      "pose": [0.5, 0.0, 0.015, 0, 0, 0, 1]
    }
  ]
}
```

**第二步：新建 demo 脚本 `embodiedbench/simulation/demo1_physics_check.py`**

```python
import json
import time
import numpy as np
import yaml
import sapien.core as sapien

from src.simulator import SceneBuilder, PhysicsBoundaryDetectors
from src.memory import PHYSICS_DICTIONARY

def run_demo1():
    # 1. 加载配置和场景
    with open("config/global_config.yaml") as f:
        config = yaml.safe_load(f)
    with open("inputs/demo1_scene.json") as f:
        scene_json = json.load(f)

    builder = SceneBuilder(config, PHYSICS_DICTIONARY)
    actors = builder.build_twin_world(scene_json)
    detectors = PhysicsBoundaryDetectors(config)
    dt = config["simulator_config"]["time_step"]

    print("=== 场景构建完成，开始物理压力测试 ===")

    # --- 测试 1：TippingFeedback（手动倾斜 iron_block）---
    iron = next(a for a in actors if a.name == "iron_block_1")
    from scipy.spatial.transform import Rotation
    tilt_q = Rotation.from_euler('x', 30, degrees=True).as_quat()  # xyzw
    iron.set_pose(sapien.Pose(iron.get_pose().p, tilt_q))
    builder.scene.step()
    t = time.time()
    fb = detectors.check_stability_and_tipping(builder.scene, actors, t)
    print(f"[TippingFeedback] {'触发 ✓' if fb else '未触发 ✗'}: {fb}")

    # --- 测试 2：DestructionFeedback（在 plastic_box 上堆超重物体）---
    # 动态提高 iron_block 位置让其自由落体砸中 plastic_box
    iron.set_pose(sapien.Pose([0.5, 0.0, 0.6], [0, 0, 0, 1]))
    for _ in range(300):
        builder.scene.step()
        t = time.time()
        fb = detectors.check_stiffness_and_destruction(builder.scene, PHYSICS_DICTIONARY, t)
        if fb:
            print(f"[DestructionFeedback] 触发 ✓: culprit={fb.culprit_actor}, force={fb.max_normal_force_N:.1f}N")
            break
    else:
        print("[DestructionFeedback] 300 步内未触发（可调整 iron_block 高度或 yield_stress_N 阈值）")

    # --- 测试 3：IntrusionFeedback（需要机器人，此处跳过，用日志说明）---
    print("[IntrusionFeedback] 该检测器需配合机器人运动，在 Demo 2/5 中验证")

    print("=== Demo 1 完成 ===")

if __name__ == "__main__":
    run_demo1()
```

**需要关注的代码细节：**

- `detectors.py:56` — `ground` 和 `camera_mount` 被过滤，确认你的场景 JSON 中没有用这些名称命名非地面物体
- `detectors.py:67-69` — 物理检测用的是 impulse/dt 换算成力，不是直接力传感器。如果触发不了，先降低 `detector_thresholds.unplanned_collision_force`（当前默认 5.0 N）
- `scene_builder.py:62-64` — `semantic_category == "environment_structure"` 的物体会被 build 成 kinematic，**不参与碰撞力计算**，所以 `wooden_rack` 不会触发 DestructionFeedback；测试破坏应用 movable_object 类别的物体

---

### Isaac Sim 负责同学操作指南

Demo 1 的 Isaac Sim 版本只需要替换场景构建和接触查询，机器人部分不涉及。

**步骤 1：在 Isaac Sim Python 环境中安装依赖**
```bash
~/.local/share/ov/pkg/isaac_sim-*/python.sh -m pip install pyyaml scipy numpy
```

**步骤 2：用 Isaac Sim 重写 `build_twin_world` 等效逻辑**

```python
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
import numpy as np

world = World(physics_dt=1/500.0, stage_units_in_meters=1.0)
world.scene.add_ground_plane()

# 注意：Isaac Sim 四元数格式是 wxyz，JSON 里存的是 xyzw
# 转换：pose[3:7] = [qx, qy, qz, qw] → Isaac: [qw, qx, qy, qz]
def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])

plastic_box = world.scene.add(DynamicCuboid(
    prim_path="/World/plastic_box_1",
    name="plastic_box_1",
    position=np.array([0.5, 0.0, 0.06]),
    orientation=xyzw_to_wxyz([0, 0, 0, 1]),
    size=np.array([0.15, 0.12, 0.10]),
))
# 对应 PHYSICS_DICTIONARY["plastic_box"]["density_kg_m3"] = 1200
# 质量 = density * volume
plastic_box.set_mass(1200 * 0.15 * 0.12 * 0.10)

world.reset()  # 必须在 reset() 之后才能读取物理状态
```

**步骤 3：接触力读取（替换 `check_stiffness_and_destruction`）**

```python
from omni.isaac.sensor import ContactSensor

# 在对应 prim 下添加 Contact Sensor
contact_sensor = ContactSensor(
    prim_path="/World/plastic_box_1/contact_sensor",
    name="plastic_box_contact",
    min_threshold=0, max_threshold=1e10,
    radius=-1,
)
world.reset()

# 步进并读取
for _ in range(300):
    world.step(render=False)
    reading = contact_sensor.get_current_frame()
    if reading and reading.get("force"):
        total_force = np.linalg.norm(reading["force"])
        if total_force > 68.0:  # plastic_box yield_stress * 0.85
            print(f"[Isaac DestructionFeedback] force={total_force:.1f}N > threshold")
            break
```

**步骤 4：倾角检测（替换 `check_stability_and_tipping`）**

```python
from omni.isaac.core.utils.transformations import get_prim_transform_matrix
from scipy.spatial.transform import Rotation

_, orient_wxyz = iron_block.get_world_pose()
# Isaac 返回 wxyz，转 xyzw 再算欧拉角
q_xyzw = [orient_wxyz[1], orient_wxyz[2], orient_wxyz[3], orient_wxyz[0]]
rot = Rotation.from_quat(q_xyzw)
euler = rot.as_euler('xyz', degrees=True)
tilt = np.sqrt(euler[0]**2 + euler[1]**2)
if tilt > 15.0 * 0.85:
    print(f"[Isaac TippingFeedback] tilt={tilt:.1f}°")
```

---

## Demo 2：机械臂蓝图图执行

### 目标
hardcode 一段 JSON 格式的 `blueprint_json`，直接喂给 `BlueprintGraphEngine.execute_blueprint()`，驱动 Panda 臂完成一次 pick-and-place，并实时打印每个节点的执行结果。不需要 LLM，不需要感知管线。

### 展示内容
- `SceneBuilder` 实例化物理场景
- `BlueprintGraphEngine` 的 DAG 图遍历逻辑（`execute_blueprint`）
- 9 个原子技能中的 5 个：`move_above` → `descend` → `grasp` → `lift` → `place`
- 当遇到运动学不可达时自动走 `on_failure` 分支

---

### 负责同学（机器人控制组）代码改动

**第一步：准备 hardcoded 蓝图 JSON**

在 `embodiedbench/simulation/inputs/` 下新建 `demo2_blueprint.json`：

```json
{
  "task": "pick iron_block_1 and place on wooden_rack_1",
  "execution_graph": {
    "start": "n1",
    "nodes": {
      "n1": {
        "type": "skill",
        "skill": "move_above",
        "target": "iron_block_1",
        "params": {"height_offset": 0.12},
        "next": "n2",
        "on_failure": "fail_node"
      },
      "n2": {
        "type": "skill",
        "skill": "descend",
        "target": "iron_block_1",
        "params": {"target_height": 0.22},
        "next": "n3",
        "on_failure": "fail_node"
      },
      "n3": {
        "type": "skill",
        "skill": "grasp",
        "target": "iron_block_1",
        "params": {"close_wait_steps": 80},
        "next": "n4",
        "on_failure": "fail_node"
      },
      "n4": {
        "type": "condition",
        "condition": {"name": "object_in_gripper"},
        "if_true": "n5",
        "if_false": "retry_grasp"
      },
      "retry_grasp": {
        "type": "skill",
        "skill": "grasp",
        "target": "iron_block_1",
        "params": {"close_wait_steps": 120},
        "next": "n5",
        "on_failure": "fail_node"
      },
      "n5": {
        "type": "skill",
        "skill": "lift",
        "target": "iron_block_1",
        "params": {"lift_height": 0.20},
        "next": "n6",
        "on_failure": "fail_node"
      },
      "n6": {
        "type": "skill",
        "skill": "move_above",
        "target": "wooden_rack_1",
        "params": {"height_offset": 0.08},
        "next": "n7",
        "on_failure": "fail_node"
      },
      "n7": {
        "type": "skill",
        "skill": "place",
        "target": "wooden_rack_1",
        "params": {"open_wait_steps": 60},
        "next": "success_node",
        "on_failure": "fail_node"
      },
      "success_node": {
        "type": "terminal",
        "result": "success"
      },
      "fail_node": {
        "type": "terminal",
        "result": "failure",
        "failure_reason": "DEMO2_SKILL_FAILED"
      }
    }
  }
}
```

**第二步：新建 demo 脚本 `embodiedbench/simulation/demo2_blueprint.py`**

```python
import json
import yaml
from src.simulator import SceneBuilder, RobotController, PhysicsBoundaryDetectors
from src.memory import PHYSICS_DICTIONARY

def run_demo2():
    with open("config/global_config.yaml") as f:
        config = yaml.safe_load(f)
    with open("inputs/demo1_scene.json") as f:
        scene_json = json.load(f)
    with open("inputs/demo2_blueprint.json") as f:
        blueprint = json.load(f)

    # 构建场景
    builder = SceneBuilder(config, PHYSICS_DICTIONARY)
    builder.build_twin_world(scene_json)

    # 初始化控制器（detectors 必须传入）
    detectors = PhysicsBoundaryDetectors(config)
    robot = RobotController(builder.scene, config, detectors, PHYSICS_DICTIONARY)

    print("=== 开始执行蓝图 DAG ===")
    result = robot.execute_blueprint(blueprint)
    print(f"\n=== 执行结果 ===")
    print(f"状态: {result['evaluation_status']}")
    for step in result.get("history", []):
        print(f"  节点 {step['node']}: {step['status']}")

if __name__ == "__main__":
    run_demo2()
```

**需要关注的代码细节：**

- `robot_controller.py:24` — 初始关节角 `[0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]` 是 Panda 的 home 姿态，确保 URDF 里关节限位覆盖这些值
- `robot_controller.py:96` — `move_above` 时末端姿态强制为 `[0, 1, 0, 0]`（Top-Down 朝下），如果场景里物体不在机械臂工作空间正下方，IK 会失败并走 `on_failure`；调整 `demo1_scene.json` 里物体的 x/y 坐标到 `[0.3~0.7, -0.3~0.3]` 范围内
- `robot_controller.py:107` — `planner.plan_qpos_to_pose()` 的 `time_step=0.002` 直接决定轨迹点数，越小轨迹越密，500Hz 步进时更平滑，但也更慢
- `robot_controller.py:45` — `_step_physics_with_probes` 中 `List` 未 import，当前代码有一个 bug：`from typing import List` 需要在文件顶部添加（现在文件里没有这行 import）

**需要修复的 bug（`robot_controller.py:45`）：**

```python
# 文件顶部已有 import，但 List 没有被 import，需要加：
from typing import List, Optional, Dict, Any
```

---

### Isaac Sim 负责同学操作指南

**步骤 1：加载 Panda USD**

```python
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage

world = World(physics_dt=1/500.0)
world.scene.add_ground_plane()

# 使用预转换好的 USD（参考 connect_to_isaac_sim.md 第 6 节做离线转换）
add_reference_to_stage(
    usd_path="assets/robots/panda/panda.usd",
    prim_path="/World/panda"
)
robot = world.scene.add(Robot(prim_path="/World/panda", name="panda"))
world.reset()  # 必须先 reset
```

**步骤 2：初始关节角设置**

```python
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

init_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
robot.set_joint_positions(init_qpos)
world.step(render=False)
```

**步骤 3：关节控制（等效 SAPIEN 的 set_drive_target）**

```python
# 每步喂入目标关节角
from omni.isaac.core.utils.types import ArticulationAction

action = ArticulationAction(joint_positions=target_qpos)
robot.get_articulation_controller().apply_action(action)
world.step(render=False)
```

**步骤 4：末端执行器姿态获取（等效 robot.get_links()[-1].get_pose()）**

```python
# panda_hand 是 Panda 末端 link 名称
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom
import omni.isaac.core.utils.transformations as T

ee_pos, ee_rot_wxyz = robot.get_world_poses(
    joint_indices=None
)
# 或者精确获取 panda_hand link 位姿：
hand_prim_path = "/World/panda/panda_hand"
hand_pos, hand_rot = T.get_prim_pose(hand_prim_path)
# hand_rot 是 wxyz 格式，转 xyzw：
hand_rot_xyzw = [hand_rot[1], hand_rot[2], hand_rot[3], hand_rot[0]]
```

**步骤 5：夹爪控制**

```python
# Panda 夹爪是 joint index 7, 8
# open: 0.04，close: 0.0
gripper_action = ArticulationAction(
    joint_positions=np.array([0.0, 0.0]),   # 关闭夹爪
    joint_indices=np.array([7, 8])
)
robot.get_articulation_controller().apply_action(gripper_action)
for _ in range(80):
    world.step(render=False)
```

**运动规划保留 mplib（不换 cuMotion）：**

```python
import mplib
# mplib 只需要 URDF，不依赖 SAPIEN，在 Isaac Sim Python 中同样可用
planner = mplib.Planner(
    urdf="assets/robots/panda/panda.urdf",
    srdf="assets/robots/panda/panda.srdf",
    move_group="panda_hand",
)
# 规划出轨迹后，逐帧喂给 Isaac Sim 的 articulation controller
result = planner.plan_qpos_to_pose(target_pose, current_qpos, time_step=0.002)
if result["status"] == "Success":
    for qpos_target in result["position"]:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=qpos_target[:7])
        )
        world.step(render=False)
```

---

## Demo 3：三层记忆生命周期

### 目标
完全脱离仿真器，用纯 Python 脚本驱动 `EmbodiedManipulationMemorySystem`，演示 Working → Episodic → Semantic 的数据流动，以及跨 episode 的知识积累效果。

### 展示内容
- `reset_episode` / `begin_task` / `record_action` / `end_episode` 完整生命周期
- 运行 3 个 episode 后，`SemanticMemory` 自动归纳出 TaskSchema 和 ObjectKB
- `query_similar_episodes` 返回相关历史
- `query_task_schema` 返回成功率和步骤模板

---

### 负责同学（记忆系统组）代码改动

**新建 `embodiedbench/demo3_memory_lifecycle.py`（项目根目录）**

```python
"""
Demo 3: 三层记忆生命周期独立演示
运行：python demo3_memory_lifecycle.py
无需仿真器、无需 LLM API
"""
import time
from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem
from embodiedbench.memory_manip.config import MemorySystemConfig

def run_demo3():
    # MemorySystemConfig 可以自定义存储路径，避免污染正式数据
    config = MemorySystemConfig(
        store_dir="demo3_memory_store",
        episodic_max_episodes=100,
        episodic_generalize_every_n=3,   # 每3个 episode 触发语义归纳
    )
    mem = EmbodiedManipulationMemorySystem(config=config)

    # ====== Episode 1：成功的 pick-and-place ======
    print("\n=== Episode 1：搬运铁块到货架（成功）===")
    mem.reset_episode(scene_id="scene_001")
    mem.begin_task(
        instruction="Pick up the iron block and place it on the wooden rack",
        task_variation="single_object_relocation",
        intent={"target_object": "iron_block_1", "destination": "wooden_rack_1"}
    )

    steps_ep1 = [
        ("move_above iron_block_1", True, "IK solved, trajectory planned"),
        ("descend to iron_block_1", True, "end effector lowered"),
        ("grasp iron_block_1",      True, "gripper closed, width=0.03m"),
        ("lift iron_block_1",       True, "object lifted 0.2m"),
        ("move_above wooden_rack_1",True, "IK solved"),
        ("place on wooden_rack_1",  True, "gripper opened, object settled"),
    ]
    for action, success, feedback in steps_ep1:
        mem.record_action(action=action, action_id=action, success=success, feedback=feedback)
        time.sleep(0.01)  # 模拟步进时间

    mem.update_observation(
        visible_objects=["iron_block_1", "wooden_rack_1", "plastic_box_1"],
        current_location="manipulation_zone"
    )
    mem.end_episode(success=True)
    print("Episode 1 结束，已存入情景记忆")

    # ====== Episode 2：失败的 pick（物体过重）======
    print("\n=== Episode 2：搬运重铁块（失败）===")
    mem.reset_episode(scene_id="scene_002")
    mem.begin_task(
        instruction="Pick up the iron block and place it on the wooden rack",
        task_variation="heavy_object",
    )

    steps_ep2 = [
        ("move_above iron_block_2", True,  "IK solved"),
        ("descend to iron_block_2", True,  "lowered"),
        ("grasp iron_block_2",      False, "MECHANICAL_DESTRUCTION: force=95.2N > threshold=68.0N"),
    ]
    for action, success, feedback in steps_ep2:
        mem.record_action(action=action, action_id=action, success=success, feedback=feedback)

    mem.end_episode(success=False)
    print("Episode 2 结束（失败）")

    # ====== Episode 3：带记忆引导的成功重试 ======
    print("\n=== Episode 3：利用历史记忆避开重物（成功）===")
    mem.reset_episode(scene_id="scene_003")
    mem.begin_task(
        instruction="Pick up the plastic box and place it on the wooden rack",
        task_variation="single_object_relocation",
    )

    # 查询历史，演示记忆检索
    similar = mem.query_similar_episodes("pick and place wooden rack", top_k=2)
    print(f"  检索到 {len(similar)} 条相似历史:")
    for ep in similar:
        print(f"    - [{ep['episode_id']}] '{ep['task_instruction']}' success={ep['success']}")

    steps_ep3 = [
        ("move_above plastic_box_1", True, "IK solved"),
        ("descend to plastic_box_1", True, "lowered"),
        ("grasp plastic_box_1",      True, "gripper closed, width=0.04m"),
        ("lift plastic_box_1",       True, "lifted"),
        ("place on wooden_rack_1",   True, "placed"),
    ]
    for action, success, feedback in steps_ep3:
        mem.record_action(action=action, action_id=action, success=success, feedback=feedback)

    mem.end_episode(success=True)
    print("Episode 3 结束")

    # ====== 3 个 episode 后，语义归纳自动触发 ======
    print("\n=== 语义归纳结果 ===")
    schema = mem.query_task_schema("pick and place")
    print(f"  TaskSchema: {schema}")

    obj_info = mem.query_object("iron_block")
    print(f"  ObjectKB iron_block: {obj_info}")

    history = mem.get_object_location_history("iron_block_1")
    print(f"  iron_block_1 历史位置分布: {history}")

    # 完整快照
    snap = mem.snapshot()
    print(f"\n  episodic_summary: {snap['episodic']}")
    print(f"  semantic_summary: {snap['semantic']}")

if __name__ == "__main__":
    run_demo3()
```

**需要关注的代码细节：**

- `agent_memory.py:177` — `episodic_generalize_every_n=3` 是触发 `semantic.generalize_from_episodes()` 的阈值；Demo 里跑 3 个 episode 后第三次 `end_episode` 就会触发归纳，**确保 `semantic_memory.py` 里的 `generalize_from_episodes` 已实现**
- `agent_memory.py:183` — `_ltm_client.insert_memory()` 会尝试访问远程服务；Demo 运行时若无服务器，在 `MemorySystemConfig` 里将 `embodiedltm_base_url=None` 或设为空字符串，`EmbodiedLTMClient` 会 gracefully skip
- 存储路径在 `demo3_memory_store/` 下，运行结束后手动清理；正式数据在 `embodiedbench/memory_manip/store/`

---

## Demo 4：意图规划闭环对话

### 目标
给 `IntentReasoningAgent` 喂入一个 hardcoded 的任务指令和观测文本，驱动其完成 3 步决策循环，展示 GoalReasoner → SolutionSpaceAnalyzer → TaskValidator 的完整流程。

### 展示内容
- `GoalReasoner` 从自然语言指令提取五层 Why 意图
- `SolutionSpaceAnalyzer` 结合可见物体给出合法动作候选集
- `TaskValidator` 综合历史和记忆给出最终决策
- 错误恢复机制：连续 2 次失败后从记忆中抹除幻觉物体

---

### 负责同学（规划组）代码改动

**新建 `embodiedbench/demo4_planning_loop.py`（项目根目录）**

```python
"""
Demo 4: 意图规划闭环对话
运行：python demo4_planning_loop.py
需要 LLM API KEY（在 global_config.yaml 或 env var 中配置）
"""
from embodiedbench.Planning_module.core_agent import IntentReasoningAgent

def build_skill_set():
    """模拟 EmbodiedBench 环境给出的合法动作列表"""
    return [
        "navigate to the dining table",        # 0
        "navigate to the kitchen counter",     # 1
        "navigate to the refrigerator",        # 2
        "open the refrigerator",               # 3
        "close the refrigerator",              # 4
        "pick up the apple",                   # 5
        "pick up the orange",                  # 6
        "pick up the milk carton",             # 7
        "place object on dining table",        # 8
        "place object in refrigerator",        # 9
        "done",                                # 9999 用特殊处理
    ]

def run_demo4():
    agent = IntentReasoningAgent(
        episode_id="demo4_ep001",
        model_name="gpt-4o-mini",      # 换成你有 KEY 的模型
        scene_id="kitchen_scene_001",
        inject_priors=False,
    )

    skill_set = build_skill_set()
    instruction = "Please put the apple on the dining table."

    # ====== Step 0：初始观测 ======
    obs_0 = (
        "You are standing in the living room. "
        "You can see the dining table and the kitchen counter from here. "
        "You are not holding anything."
    )
    print(f"\n=== Step 0 ===\nInstruction: {instruction}\nObservation: {obs_0}\n")
    result_0 = agent.step(instruction, obs_0, skill_set, step_idx=0)
    print(f"Decision: Action {result_0['action_id']} -> {skill_set[result_0['action_id']]}")

    # ====== Step 1：到达 kitchen counter，发现 apple ======
    # 模拟上一步执行了 navigate to the kitchen counter
    obs_1 = (
        "You are now at the kitchen counter. "
        "You can see: a red apple, an orange, and some dishes. "
        "You are not holding anything."
    )
    print(f"\n=== Step 1 ===\nObservation: {obs_1}\n")
    result_1 = agent.step(instruction, obs_1, skill_set, step_idx=1)
    print(f"Decision: Action {result_1['action_id']} -> {skill_set[result_1['action_id']]}")

    # ====== Step 2：模拟拾取失败（错误恢复测试）======
    obs_2 = (
        "Last action is invalid: the apple could not be grasped (physics error). "
        "You are still at the kitchen counter. "
        "You can see: a red apple, an orange."
    )
    print(f"\n=== Step 2（失败恢复）===\nObservation: {obs_2}\n")
    result_2 = agent.step(instruction, obs_2, skill_set, step_idx=2)
    print(f"Decision: Action {result_2['action_id']} -> {skill_set[result_2['action_id']]}")
    print(f"Failed action counter: {agent.failed_actions}")

    # ====== Step 3：持有物体，走向目标位置 ======
    obs_3 = (
        "Last action executed successfully. "
        "You are at the kitchen counter holding the apple. "
        "The dining table is visible."
    )
    print(f"\n=== Step 3 ===\nObservation: {obs_3}\n")
    result_3 = agent.step(instruction, obs_3, skill_set, step_idx=3)
    print(f"Decision: Action {result_3['action_id']} -> {skill_set[result_3['action_id']]}")

if __name__ == "__main__":
    run_demo4()
```

**需要关注的代码细节：**

- `core_agent.py:11` — `LLMClient` 从 `llm_client.py` 读取 API KEY，确认你的 OpenAI / Azure / 其他 LLM 接口配置正确
- `core_agent.py:57-58` — 错误恢复触发条件是 `"last action is invalid" in observation_text.lower()`，构造 obs_2 时必须包含这个字符串
- `core_agent.py:67-72` — 连续失败 2 次会调用 `self.memory_manager.ltm.update_state()`，这会尝试写入 LTM 服务；如果 LTM 未启动，需要在 `ltm_client.py` 里加 try/except 保护，避免 demo 崩溃
- `core_agent.py:239-258` — `action_id == 9999` 的特殊处理逻辑：当 agent 认为任务完成但条件不满足时会被拦截并回退；Demo 不需要触发这个分支，确保 step 0~3 里不要让 agent 走到 done

---

## Demo 5：端到端操作全链路

### 目标
整合所有模块，用 hardcoded 感知 JSON（跳过真实相机）完成一次完整的 TAMP 循环：感知 → 记忆入库 → LLM 规划 → 仿真物理验证 → 故障反思 → 重规划。这是展示项目核心价值的 demo。

### 展示内容
- 全链路数据流：perception_json → SceneBuilder → LLMPlanner → BlueprintGraphEngine → PhysicsBoundaryDetectors → PrincipleExtractor → replan
- 至少触发一次物理熔断并成功重规划
- 记忆系统全程记录，最后打印 snapshot

---

### 负责同学（集成组）代码改动

**第一步：准备 hardcoded 感知 JSON**

在 `embodiedbench/simulation/inputs/` 下新建 `demo5_perception.json`：

```json
{
  "scene_entities": [
    {
      "id": "plastic_box_1",
      "label": "plastic_box",
      "pose": {"pos": [0.50, 0.05, 0.06], "quat": [0, 0, 0, 1]},
      "size": [0.15, 0.12, 0.10]
    },
    {
      "id": "iron_block_1",
      "label": "iron_block",
      "pose": {"pos": [0.50, 0.05, 0.18], "quat": [0, 0, 0, 1]},
      "size": [0.08, 0.08, 0.08]
    },
    {
      "id": "wooden_rack_1",
      "label": "wooden_rack",
      "pose": {"pos": [0.42, -0.20, 0.015], "quat": [0, 0, 0, 1]},
      "size": [0.60, 0.40, 0.03]
    }
  ],
  "objects": [
    {
      "name": "plastic_box_1",
      "size_whd": [0.15, 0.12, 0.10],
      "pose": [0.50, 0.05, 0.06, 0, 0, 0, 1]
    },
    {
      "name": "iron_block_1",
      "size_whd": [0.08, 0.08, 0.08],
      "pose": [0.50, 0.05, 0.18, 0, 0, 0, 1]
    },
    {
      "name": "wooden_rack_1",
      "size_whd": [0.60, 0.40, 0.03],
      "pose": [0.42, -0.20, 0.015, 0, 0, 0, 1]
    }
  ]
}
```

> **注意**：`scene_entities` 供 LLMPlanner 使用（含 label），`objects` 供 SceneBuilder 使用（含 name），两者需保持一致。

**第二步：修改 `simulation/main.py` 支持 demo 模式**

在 `run_end_to_end_tamp_loop()` 函数前添加一个入口函数：

```python
def run_demo5(perception_json_path: str = "inputs/demo5_perception.json"):
    """Demo 5 入口：跳过真实感知，使用 hardcoded perception JSON 运行全链路"""
    import json, yaml, dataclasses, time
    import numpy as np
    from src.simulator import SceneBuilder, RobotController, PhysicsBoundaryDetectors
    from src.memory import WorkingMemory, PrincipleExtractor, PHYSICS_DICTIONARY
    from src.plan import LLMPlanner
    # 引入记忆系统
    from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem
    from embodiedbench.memory_manip.config import MemorySystemConfig

    config_path = "config/global_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(perception_json_path) as f:
        vision_out = json.load(f)

    # --- 记忆系统初始化 ---
    mem_sys = EmbodiedManipulationMemorySystem(
        config=MemorySystemConfig(store_dir="demo5_memory_store")
    )
    mem_sys.reset_episode(scene_id="demo5_scene")
    mem_sys.begin_task(
        instruction="Extract plastic_box_1 from under iron_block_1 and place it on wooden_rack_1",
        task_variation="stacked_object_extraction",
    )

    # --- 拓扑提取 & 工作记忆 ---
    scene_topology = extract_topology_from_perception(vision_out)
    wm = WorkingMemory(task_name="stacked_object_extraction", target_id="plastic_box_1")
    wm.update_scene_topology(scene_topology)

    # --- LLM 规划 ---
    print("=================== LLM 初始规划 ===================")
    planner_llm = LLMPlanner(config)
    global_action_plan = planner_llm.generate_initial_global_plan(wm.get_full_context())
    print(f"初始计划步数: {len(global_action_plan)}")

    # --- 仿真场景构建 ---
    scene_sim = SceneBuilder(config_path, PHYSICS_DICTIONARY)
    scene_sim.build_twin_world(vision_out)  # SceneBuilder 使用 vision_out["objects"]
    detectors = PhysicsBoundaryDetectors(config)
    robot_agent = RobotController(scene_sim.scene, config, detectors, PHYSICS_DICTIONARY)
    extractor = PrincipleExtractor(config)

    # --- 主循环（与原 main.py 一致，但加入记忆记录）---
    success = False
    attempt = 0
    while not success and attempt < 5:
        attempt += 1
        triggered_fb = None
        scene_point_cloud = extract_omnidirectional_point_cloud(scene_sim.scene)

        for primitive in global_action_plan:
            robot_base_pose = robot_agent.robot.get_pose()
            proprioceptive_blueprint = map_to_proprioceptive_frame(primitive, robot_base_pose)
            plan_result = robot_agent.execute_skill_primitive(proprioceptive_blueprint, scene_point_cloud)

            if not plan_result["status"]:
                mem_sys.record_action(str(primitive), action_id=str(primitive), success=False, feedback="KINEMATIC_UNREACHABLE")
                continue

            trajectory = plan_result["trajectory"]
            for qpos_target in trajectory:
                robot_agent.robot.set_drive_target(np.concatenate([
                    qpos_target,
                    [robot_agent.gripper_joints[0].get_drive_target()],
                    [robot_agent.gripper_joints[1].get_drive_target()]
                ]))
                scene_sim.scene.step()
                t = time.time()
                active_actors = [a for a in scene_sim.scene.get_all_actors() if not a.is_kinematic]
                triggered_fb = (
                    detectors.check_stiffness_and_destruction(scene_sim.scene, PHYSICS_DICTIONARY, t) or
                    detectors.check_stability_and_tipping(scene_sim.scene, active_actors, t) or
                    detectors.check_unexpected_collision(scene_sim.scene, "panda", "plastic_box_1", t) or
                    detectors.check_feasibility_and_deadlock(robot_agent.robot, t)
                )
                if triggered_fb:
                    break
            if triggered_fb:
                mem_sys.record_action(str(primitive), action_id=str(primitive), success=False,
                                      feedback=triggered_fb.error_type)
                break
            else:
                mem_sys.record_action(str(primitive), action_id=str(primitive), success=True)

        if triggered_fb:
            print(f"[Demo5] 物理熔断: {triggered_fb.error_type}，开始反思重规划...")
            rule = extractor.extract_scene_based_principle(dataclasses.asdict(triggered_fb), wm.get_full_context())
            wm.mount_failure_payload(triggered_fb, rule)
            global_action_plan = planner_llm.replan_triggered_by_event(wm.get_full_context())
        else:
            success = True
            print("任务成功！")

    mem_sys.end_episode(success=success)
    print("\n=== 记忆快照 ===")
    import json
    print(json.dumps(mem_sys.snapshot(), indent=2, ensure_ascii=False, default=str))
```

**感知 JSON 两套 key 的原因说明：**

- `vision_out["objects"]` 是 `SceneBuilder.build_twin_world()` 期望的格式（`name` + `size_whd` + `pose` 为 7 维列表）
- `vision_out["scene_entities"]` 是 `LLMPlanner` 期望的格式（`id` + `label` + 结构化 `pose`）
- 两者可以共存于同一个 JSON 文件，各取所需

**需要关注的代码细节：**

- `main.py:166` — `robot_agent.execute_skill_primitive()` 是 `main.py` 对 `BlueprintGraphEngine` 的封装调用，但当前 `robot_controller.py` 中类名是 `BlueprintGraphEngine`，而 `__init__.py` 导出为 `RobotController`；确认 `__init__.py` 里的别名映射是正确的
- `main.py:186` — `PHYSICS_DICTIONARY` 从 `src.memory` 导入，它是 `src/memory/physics_dictionary.py` 里的一个 dict，不是类实例；在 demo5 中保持同样的导入路径
- 如果 demo 场景的铁块直接压在塑料箱上，`check_stiffness_and_destruction` 在初始沉降期间就可能触发；可以在 `demo5_perception.json` 里把 `iron_block_1` 的 Z 从 0.18 改到 0.25（不接触），让机器人用 `move_above` 靠近时再触发，效果更明显

---

### Isaac Sim 负责同学操作指南

Demo 5 是最复杂的接入工作，建议按以下顺序推进：

**阶段 A：先跑通 Demo 1 + Demo 2 的 Isaac Sim 版**（确认场景构建和机器人控制没问题）

**阶段 B：替换 `SceneBuilder` 中的 SAPIEN 调用**

在 `scene_builder.py` 中添加一个工厂函数：

```python
def create_scene_builder(backend: str, config, physics_dict):
    if backend == "sapien":
        return SceneBuilder(config, physics_dict)
    elif backend == "isaac_sim":
        from .isaac_scene_builder import IsaacSceneBuilder
        return IsaacSceneBuilder(config, physics_dict)
    raise ValueError(f"Unknown backend: {backend}")
```

新建 `embodiedbench/simulation/src/simulator/isaac_scene_builder.py`，实现与 `SceneBuilder` 相同的接口（`build_twin_world` 返回 actor 列表）。

**阶段 C：替换 `BlueprintGraphEngine` 中的 SAPIEN 调用**

关键替换点（参考 `connect_to_isaac_sim.md` 第 3.2 节）：
1. `self.robot = loader.load(urdf_path)` → 加载 USD + `world.scene.add(Robot(...))`  
2. `self.scene.step()` → `world.step(render=False)`  
3. `self.robot.set_drive_target(qpos)` → `robot.get_articulation_controller().apply_action(...)`  
4. `self.robot.get_links()[-1].get_pose()` → `get_prim_pose("/World/panda/panda_hand")`

**阶段 D：替换 `PhysicsBoundaryDetectors` 中的接触查询**

参考 Demo 1 Isaac Sim 部分的 ContactSensor 用法，统一替换 `scene.get_contacts()` 调用。

**Isaac Sim 下运行完整 Demo 5 的命令：**

```bash
# 在 Isaac Sim 自带 Python 环境下运行
~/.local/share/ov/pkg/isaac_sim-*/python.sh \
    embodiedbench/simulation/main.py \
    --demo5 \
    --perception inputs/demo5_perception.json \
    --backend isaac_sim
```

在 `main.py` 的 `if __name__ == "__main__"` 块里用 `argparse` 接收 `--demo5` 和 `--backend` 参数：

```python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo5", action="store_true")
    parser.add_argument("--perception", default="inputs/demo5_perception.json")
    parser.add_argument("--backend", default="sapien", choices=["sapien", "isaac_sim"])
    args = parser.parse_args()

    if args.demo5:
        run_demo5(perception_json_path=args.perception, backend=args.backend)
    else:
        run_end_to_end_tamp_loop()
```

---

## 附录：各模块负责同学任务清单

### 仿真/物理检测组
- [ ] 修复 `robot_controller.py:45` 的 `from typing import List` 缺失
- [ ] 准备 `inputs/demo1_scene.json`
- [ ] 新建并验证 `demo1_physics_check.py` 全部四个检测器触发

### 机器人控制组
- [ ] 确认 `__init__.py` 中 `RobotController` 是 `BlueprintGraphEngine` 的正确别名
- [ ] 准备 `inputs/demo2_blueprint.json`
- [ ] 新建并验证 `demo2_blueprint.py` 中 pick-and-place 跑通（含 condition 分支）
- [ ] 验证 `on_failure` 分支在 IK 不可达时正确跳转

### 记忆系统组
- [ ] 确认 `semantic_memory.py` 中 `generalize_from_episodes()` 已实现
- [ ] 确认 `MemorySystemConfig` 支持 `embodiedltm_base_url=None` 时 graceful skip
- [ ] 新建并验证 `demo3_memory_lifecycle.py` 全流程（含 snapshot 输出）

### 规划组
- [ ] 确认 `ltm_client.py` 里所有网络请求有 try/except 保护
- [ ] 确认 `LLMClient` 支持通过环境变量或配置文件读取 API KEY
- [ ] 新建并验证 `demo4_planning_loop.py` 全 4 步（含错误恢复）

### 集成组
- [ ] 在 `main.py` 中新增 `run_demo5()` 和 `argparse` 入口
- [ ] 准备 `inputs/demo5_perception.json`（注意双套 key）
- [ ] 确认 `vision_out["objects"]` 与 `vision_out["scene_entities"]` 格式一致
- [ ] 端到端跑通（至少触发一次物理熔断 + 重规划）

### Isaac Sim 接入组
- [ ] 离线完成 `panda.urdf` → `panda.usd` 转换
- [ ] 跑通 Demo 1 Isaac Sim 版（ContactSensor 接触力读取）
- [ ] 跑通 Demo 2 Isaac Sim 版（关节控制 + mplib 规划）
- [ ] 实现 `IsaacSceneBuilder`（与 `SceneBuilder` 接口一致）
- [ ] 实现 Demo 5 Isaac Sim 完整版（通过 `--backend isaac_sim` 参数切换）
