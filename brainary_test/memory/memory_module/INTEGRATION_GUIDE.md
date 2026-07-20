# Memory Module 集成指南

**适用对象**：负责将 Brainary_Robot 接入 Isaac Sim 或真实机械臂（Franka）的同学  
**版本**：2026-06-30  
**联系模块**：`memory_module/`（本包）、`perception_vlm.py`、`embodiedbench/memory_manip/`

---

## 1. 总体数据流

```
┌──────────────────────────────────────────────────────────────────┐
│                    Brainary_Robot 数据流                          │
│                                                                  │
│  RGB图像                                                          │
│     │                                                             │
│     ▼                                                             │
│  QwenVLMPerception.recognize_batch()        ← perception_vlm.py │
│     │  List[RecognitionResult]                                   │
│     ▼                                                             │
│  VisionPerceptionAdapter.perceive()         ← perception_adapter │
│     │  (T,P,D) features + visible_objects                        │
│     ▼                                                             │
│  EmbodiedManipulationMemorySystem            ← agent_memory.py  │
│  ┌────────────────────────────────────┐                          │
│  │ Working Memory (每一个 episode)    │                          │
│  │   ActiveGoal / Observation /       │                          │
│  │   ActionBuffer / RobotState /      │                          │
│  │   TaskClock                        │                          │
│  ├────────────────────────────────────┤                          │
│  │ Episodic Memory (跨 episode)       │                          │
│  │   JSONL 轨迹 + 激活值衰减           │                          │
│  ├────────────────────────────────────┤                          │
│  │ Semantic Memory (长期知识)         │                          │
│  │   ObjectKB / UserPref /            │                          │
│  │   TaskSchema / SpatialTopo         │                          │
│  └────────────────────────────────────┘                          │
│     │  PlanningContext                                            │
│     ▼                                                             │
│  PlanningMemoryInterface                    ← planning_interface │
│     │  →  Planning Module (你的规划模块)                          │
│     │  →  VLM Brain (embodiedbench/connection/)                  │
│     │  →  spine brain / Isaac Sim / Franka                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 环境准备

### 2.1 Python 路径

所有代码都依赖 `E:/Brainary_Robot` 在 Python path 中：

```bash
# 方式 A：设置环境变量（推荐）
export PYTHONPATH="E:/Brainary_Robot:$PYTHONPATH"   # Linux/Mac
$env:PYTHONPATH = "E:\Brainary_Robot;" + $env:PYTHONPATH  # PowerShell

# 方式 B：在脚本开头动态设置
import sys
sys.path.insert(0, "E:/Brainary_Robot")
```

### 2.2 依赖安装

```bash
pip install torch transformers accelerate pillow numpy
pip install qwen-vl-utils          # Qwen VLM 工具库
pip install -r requirements_eb_manipulation_env.txt
```

---

## 3. 最小可运行示例（仿真/测试用）

不需要真实相机或机械臂，只需要 RGB 图像文件：

```python
import sys
sys.path.insert(0, "E:/Brainary_Robot")

from memory_module import PerceptionMemoryPipeline

# 初始化 pipeline（第一次运行会下载 Qwen 模型，约 6GB）
pipeline = PerceptionMemoryPipeline.create(
    store_dir="my_memory_store/",  # 持久化文件存放目录
    use_auxiliary_views=True,
    enable_depth=False,
    embodiedltm_base_url=None,     # 不使用远程 LTM 服务
)

# 一个完整 episode 的流程
pipeline.session_start()
pipeline.begin_episode(
    scene_id="episode_001",
    task_instruction="pick up the red cube and place it on the target",
)

# 每个时间步：感知 → 获取规划上下文 → 执行动作 → 记录结果
results = pipeline.process_perception(
    image_paths=["path/to/rgb_frame_0001.png"],
    candidate_labels=["cube", "target", "table", "gripper"],
    current_location="table",
)

# 获取规划上下文（传给规划模块）
ctx = pipeline.get_planning_context()
print(ctx.to_prompt_text())          # 人类可读文本（可注入 VLM prompt）
print(ctx.visible_objects)           # 当前可见物体列表
print(ctx.recommended_skills)        # 历史最优技能序列

# 执行动作后记录结果
pipeline.record_action("grasp", success=True, feedback="gripper closed at 0.3")

# episode 结束
pipeline.end_episode(
    success=True,
    blueprint_skills=["move_above", "descend", "grasp", "lift", "place", "retreat"],
)

pipeline.session_end()
```

---

## 4. Isaac Sim 集成

### 4.1 架构说明

Isaac Sim 和本 pipeline 运行在**不同进程**中。通信通过文件完成：

```
Isaac Sim 进程                     Python Pipeline 进程
     │                                    │
     │── scene_state.json ──────────────► │ pipeline.process_perception()
     │                                    │ pipeline.get_planning_context()
     │◄── memory_context.json ────────── │ pipeline.export_context_for_vlm()
     │◄── generated_blueprint.json ───── │ BrainSpinePipeline.prepare()
     │                                    │
     │── isaac_output/ ─────────────────► │ pipeline.record_blueprint_execution()
     │   result.json                      │ pipeline.end_episode()
```

### 4.2 Isaac Sim 侧（你需要写的部分）

在 Isaac Sim 内部脚本（`.sh -p` 脚本）中，负责：
1. 获取传感器数据 → 写 `scene_state.json`
2. 等待 `generated_blueprint.json` 出现 → 执行 Blueprint
3. 执行结束后写 `isaac_output/result.json`

**scene_state.json 格式**（必须严格遵守）：

```json
{
  "cube_pose":       [x, y, z, qw, qx, qy, qz],
  "target_pose":     [x, y, z, qw, qx, qy, qz],
  "ee_pose":         [x, y, z, qw, qx, qy, qz],
  "gripper_width":   0.08,
  "robot_joint_pos": [q0, q1, q2, q3, q4, q5, q6],
  "step_index":      0
}
```

> 如果某个 pose 字段 Isaac Sim 无法提供，写 `[null, null, null, null, null, null, null]`  
> pipeline 会将 null 替换为 NaN 并记录到 working memory。

**isaac_output/result.json 格式**：

```json
{
  "success": true,
  "skills_executed": ["move_above", "descend", "grasp", "lift", "place", "retreat"],
  "total_steps": 23,
  "failure_reason": null,
  "final_scene_state": { ... }
}
```

### 4.3 Python Pipeline 侧（使用本模块）

```python
import json
import time
import sys
sys.path.insert(0, "E:/Brainary_Robot")

from pathlib import Path
from memory_module import PerceptionMemoryPipeline
from embodiedbench.connection.brain_pipeline import BrainSpinePipeline

# 初始化
pipeline = PerceptionMemoryPipeline.create(store_dir="memory_store/")
brain_spine = BrainSpinePipeline(vlm_config_path="vlm_config.json")

pipeline.session_start()

for episode_id in range(num_episodes):
    run_dir = Path(f"runs/episode_{episode_id:04d}")
    run_dir.mkdir(parents=True, exist_ok=True)

    task = "pick up the red cube and place it on the target"
    pipeline.begin_episode(scene_id=f"ep_{episode_id}", task_instruction=task)

    # ── 等待 Isaac Sim 写出 scene_state.json ──
    scene_state_path = run_dir / "scene_state.json"
    _wait_for_file(scene_state_path)  # 你实现这个等待函数
    scene_state = json.loads(scene_state_path.read_text())

    # ── 感知（RGB 图由 Isaac Sim 录制）──
    image_path = str(run_dir / "rgb_frame.png")
    pipeline.process_perception(
        image_paths=[image_path],
        candidate_labels=["cube", "target"],
        scene_state=scene_state,   # 同时更新 robot state
    )

    # ── 导出记忆上下文给 VLM Brain ──
    pipeline.export_context_for_vlm(run_dir / "memory_context.json")

    # ── VLM Brain 生成 Blueprint ──
    blueprint_path = brain_spine.prepare(
        task=task,
        scene_state=scene_state,
        output_dir=str(run_dir),
        image_path=image_path,
    )
    # → blueprint_path 指向 run_dir/generated_blueprint.json
    # → 将 blueprint 文件路径告知 Isaac Sim（通过信号文件或 socket）

    # ── 等待 Isaac Sim 执行并写出结果 ──
    result_path = run_dir / "isaac_output" / "result.json"
    _wait_for_file(result_path)
    result = json.loads(result_path.read_text())

    # ── 记录 Blueprint 执行结果 ──
    pipeline.record_blueprint_execution(
        blueprint_skills=result["skills_executed"],
        success=result["success"],
        scene_state=result.get("final_scene_state"),
    )

    # ── 结束 episode ──
    pipeline.end_episode(
        success=result["success"],
        blueprint_skills=result["skills_executed"],
    )

pipeline.session_end()


def _wait_for_file(path: Path, timeout: float = 60.0, interval: float = 0.5):
    """等待文件出现，超时后抛出异常。"""
    elapsed = 0.0
    while not path.exists():
        time.sleep(interval)
        elapsed += interval
        if elapsed > timeout:
            raise TimeoutError(f"Timeout waiting for {path}")
```

### 4.4 Isaac Sim 进程间通信建议

| 通信方式 | 适用场景 |
|---------|---------|
| 文件轮询（如上） | 简单，适合 10Hz 以内的控制频率 |
| Unix Domain Socket | 同机高频通信（>10Hz），需要自己实现协议 |
| Redis / ZMQ | 分布式部署，多进程协作 |
| Isaac Sim Python 脚本直接 import | 如果 Isaac Sim 版本支持，可以直接在仿真内调用 pipeline |

---

## 5. 真实 Franka 机械臂集成

### 5.1 架构差异

与 Isaac Sim 不同，真实 Franka 通常通过 **ROS / Franka Control Interface (FCI)** 提供数据。建议在 ROS node 中调用 pipeline。

```
相机 (RealSense/ZED)                 ROS Node (你写)
     │ /camera/rgb                          │
     ▼                                       │
  图像保存为 PNG ──────────────────────────► │ pipeline.process_perception()
                                             │
Franka FCI                                   │
  /franka_state_controller/robot_state       │
     │ ee_pose, joint_pos, gripper           │
     ▼                                       │
  parse → scene_state dict ───────────────► │ pipeline.process_perception(scene_state=...)
                                             │ 或
                                             │ pipeline.update_robot_state(...)
```

### 5.2 ROS 集成示例

```python
#!/usr/bin/env python3
"""ROS node: 连接相机 + Franka + memory pipeline。"""
import sys
sys.path.insert(0, "E:/Brainary_Robot")  # 或 /path/to/Brainary_Robot

import rospy
import numpy as np
from sensor_msgs.msg import Image
from franka_msgs.msg import FrankaState
from cv_bridge import CvBridge
import cv2
import tempfile

from memory_module import PerceptionMemoryPipeline

bridge = CvBridge()
pipeline = PerceptionMemoryPipeline.create(store_dir="/tmp/franka_memory/")
_latest_image_path = None
_latest_robot_state = {}


def image_callback(msg: Image):
    """保存最新 RGB 帧到临时文件（perception_vlm 需要文件路径）。"""
    global _latest_image_path
    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    # 写到临时文件
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
        _latest_image_path = f.name


def franka_state_callback(msg: FrankaState):
    """解析 Franka 状态 → scene_state dict。"""
    global _latest_robot_state
    # ee_pose: FrankaState.O_T_EE 是 4x4 齐次矩阵，需要转成 [x,y,z,qw,qx,qy,qz]
    T = np.array(msg.O_T_EE).reshape(4, 4)
    ee_pos = T[:3, 3].tolist()
    # 旋转矩阵 → 四元数（简化版，实际用 tf.transformations）
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [qx, qy, qz, qw]
    ee_pose = ee_pos + [quat[3], quat[0], quat[1], quat[2]]  # → [x,y,z,qw,qx,qy,qz]

    _latest_robot_state = {
        "ee_pose": ee_pose,
        "robot_joint_pos": list(msg.q),           # 7 joint angles
        "gripper_width": float(msg.q[6]),         # 或用独立 gripper topic
        "step_index": 0,
        # cube_pose / target_pose 来自 SLAM / ArUco marker
        "cube_pose": [float("nan")] * 7,
        "target_pose": [float("nan")] * 7,
    }


def run_episode(task_instruction: str, episode_id: str):
    pipeline.begin_episode(scene_id=episode_id, task_instruction=task_instruction)

    rate = rospy.Rate(2)  # 2 Hz 控制频率（受 VLM 推理时间限制）
    step = 0
    max_steps = 20

    while not rospy.is_shutdown() and step < max_steps:
        if _latest_image_path is None:
            rospy.sleep(0.1)
            continue

        # 感知 + 更新 memory
        results = pipeline.process_perception(
            image_paths=[_latest_image_path],
            candidate_labels=["cube", "target", "table"],
            current_location="table",
            scene_state=_latest_robot_state or None,
        )

        # 获取规划上下文
        ctx = pipeline.get_planning_context()
        rospy.loginfo(f"Step {step}: visible={ctx.visible_objects}")

        # TODO: 调用你的规划模块生成动作
        action_str = "grasp"  # 示例
        # TODO: 通过 FCI 或 MoveIt 执行动作
        success = True        # 从执行结果获取

        pipeline.record_action(action_str, success=success)
        step += 1
        rate.sleep()

    pipeline.end_episode(success=True)


if __name__ == "__main__":
    rospy.init_node("memory_pipeline_node")
    rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
    rospy.Subscriber("/franka_state_controller/robot_state", FrankaState, franka_state_callback)
    pipeline.session_start()
    run_episode("pick up the cube and place it on the target", "franka_ep_001")
    pipeline.session_end()
```

### 5.3 相机 → 文件路径说明

`QwenVLMPerception` 接收**文件路径**，不接受 numpy array 或 ROS message。
对于实时相机流，有两种方案：

| 方案 | 优点 | 缺点 |
|------|------|------|
| 写临时文件（如上示例） | 简单，无改动 | IO 开销，磁盘写入 |
| 修改 `perception_vlm.py` 支持 PIL Image 输入 | 性能更好 | 需要改感知模块代码 |

**推荐方案**：写临时文件到 `/tmp/` 的 tmpfs，IO 开销极小。

---

## 6. 仅使用规划接口（不运行 VLM）

如果你已有其他感知模块，或在测试阶段不想加载 Qwen 模型，可以绕过 VLM，
直接操作 memory 系统：

```python
import sys
sys.path.insert(0, "E:/Brainary_Robot")

from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
from embodiedbench.memory_manip.config import MemorySystemConfig
from memory_module.planning_interface import PlanningMemoryInterface, PlanningContext

# 不加载 VLM，直接使用 memory 系统
config = MemorySystemConfig(store_dir="memory_store/", embodiedltm_base_url=None)
memory = EmbodiedManipulationMemorySystem(config=config)
planning_iface = PlanningMemoryInterface(memory)

# 手动初始化 episode
memory.reset_episode(scene_id="test_001")
memory.begin_task("pick up the cube and place it on the target")

# 手动推入感知数据（不用 VLM）
memory.update_observation(
    visible_objects=["cube", "target", "table"],
    text="cube is at (0.4, 0.1, 0.05), target is at (0.3, -0.1, 0.02)",
    current_location="table",
)

# 使用规划接口
ctx: PlanningContext = planning_iface.get_planning_context("pick up the cube")
print(ctx.to_prompt_text())

# 记录动作
planning_iface.record_action_result("grasp", success=True)

# 结束 episode
memory.end_episode(success=True)
```

---

## 7. PlanningContext 字段速查

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `task_instruction` | str | 调用参数 | 当前 episode 自然语言任务 |
| `task_type` | str | 语义记忆 | 推断的任务类别（如 pick_and_place） |
| `visible_objects` | List[str] | 工作记忆 | 当前感知可见物体 |
| `scene_description` | str | 工作记忆 | VLM 输出的场景文本 |
| `object_attributes` | Dict | 工作记忆 | 物体颜色/形状/纹理属性 |
| `memory_objects` | Dict | 工作记忆 | 已知但当前不可见的物体 |
| `current_location` | str | 工作记忆 | 机器人当前语义位置 |
| `held_object` | str\|None | 工作记忆 | 当前夹持物体 |
| `task_success_rate` | float\|None | 语义记忆 | 该任务类型历史成功率 |
| `common_steps` | List[str] | 语义记忆 | 最常见动作序列 |
| `recommended_skills` | List[str] | 语义记忆 | Blueprint 最优技能序列 |
| `object_location_priors` | Dict | 语义记忆 | 物体历史高频出现位置 |
| `user_preferences` | Dict | 语义记忆 | 用户偏好（偏好/规避） |
| `similar_episodes` | List[Dict] | 情景记忆 | 相似历史 episode 摘要 |
| `recent_actions` | List[Dict] | 工作记忆 | 本 episode 最近动作历史 |

---

## 8. 记忆持久化说明

所有记忆文件存放在 `store_dir/`（默认 `embodiedbench/memory_manip/store/`）：

```
store_dir/
├── episodes.jsonl          # 情景记忆（Append-only，每行一个 JSON）
├── episodes_meta.json      # 激活值索引（读写，原子写入）
└── semantic_kb.json        # 语义知识库（读写，原子写入）
```

**原子写入**：所有写操作先写 `.tmp` 文件，再原子性 rename，防止断电损坏。

**激活值衰减**：每次 `session_end()` 触发 ACT-R 激活衰减，并裁剪低激活情景。
情景记忆容量上限：`config.episodic_max_episodes = 1000`（可在 config 中修改）。

---

## 9. 常见问题

### Q1: 运行报错 `RuntimeError: perception_vlm.py not importable`

确保 PYTHONPATH 包含 `E:/Brainary_Robot`：
```bash
export PYTHONPATH="E:/Brainary_Robot:$PYTHONPATH"
```

### Q2: Qwen 模型加载太慢

首次运行会从 HuggingFace Hub 下载约 6GB 模型。
使用 HF_ENDPOINT 镜像（国内）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: `process_perception()` 调用时间太长（>5 秒/步）

降低感知分辨率或关闭辅助视图：
```python
pipeline = PerceptionMemoryPipeline.create(
    use_auxiliary_views=False,   # 关闭轮廓/纹理辅助视图
    enable_depth=False,
)
```
或使用 GPU 加速（CUDA 可用时自动启用）。

### Q4: 如何重置记忆（全部清空）

```python
import shutil
shutil.rmtree("my_memory_store/")  # 删除 store_dir
```

### Q5: episode 中途记忆状态如何查看

```python
snap = pipeline.snapshot()
import json
print(json.dumps(snap, ensure_ascii=False, indent=2, default=str))
```

---

## 10. 关键 API 速查

```python
# 创建
pipeline = PerceptionMemoryPipeline.create(store_dir="...", ...)

# Session
pipeline.session_start(sync_planning_module=False)
pipeline.session_end()

# Episode
pipeline.begin_episode(scene_id="ep001", task_instruction="...")
pipeline.end_episode(success=True, blueprint_skills=["grasp", ...])

# 每步
results = pipeline.process_perception(["rgb.png"], scene_state={...})
ctx     = pipeline.get_planning_context()          # → PlanningContext
pipeline.record_action("grasp", success=True)

# 机器人状态（真实机械臂）
pipeline.update_robot_state(gripper_pose=[...], joint_angles=[...])

# spine brain 结果回写
pipeline.record_blueprint_execution(["grasp", "lift"], success=True)

# VLM Brain 导出
pipeline.export_context_for_vlm("run_001/memory_context.json")

# 调试
pipeline.snapshot()                                # 全状态快照
pipeline.get_last_perception_results()            # 最近感知结果
```
