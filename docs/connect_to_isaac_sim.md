# Brainary_Robot × Isaac Sim 接入指南

## 1. 当前架构概览

本项目仿真层基于 **SAPIEN**（v2/v3 PhysX 后端）构建，三个核心文件分别承担物理场景、机器人控制与安全检测职责：

```
embodiedbench/simulation/src/simulator/
├── scene_builder.py      → SceneBuilder       (SAPIEN Engine/Scene/Actor)
├── robot_controller.py   → BlueprintGraphEngine (SAPIEN Articulation + mplib IK)
└── detectors.py          → PhysicsBoundaryDetectors (SAPIEN Contact/Impulse 查询)
```

三层记忆系统（WorkingMemory / EpisodicMemory / SemanticMemory）、LLM 规划器（LLMPlanner）、感知管线（PerceptionPipeline）**均与仿真器无关**，接入 Isaac Sim 时无需改动。

---

## 2. Isaac Sim 技术栈说明

| 层次 | SAPIEN（现状） | Isaac Sim（目标） |
|---|---|---|
| 物理引擎 | PhysX（via SAPIEN Python API） | PhysX 5（via Omniverse / USD API） |
| 场景描述 | Python 动态构建 | USD Stage（`.usd` / `.urdf` 转换） |
| 机器人描述 | URDF 直接加载 | URDF → USD 转换（`urdf_importer`） |
| 关节控制 | `ArticulationJointController.set_drive_target()` | `ArticulationController.apply_action()` |
| 接触查询 | `scene.get_contacts()` → impulse | Contact Sensor / RigidPrim PhysX API |
| 渲染 | SapienRenderer | Omniverse RTX Renderer |
| Python 入口 | `import sapien.core as sapien` | `from omni.isaac.core import World` |
| 运动规划 | `mplib.Planner` | `lula` / `cuMotion` / 外部 mplib |

---

## 3. 需要修改的文件与接口

### 3.1 `scene_builder.py` → `SceneBuilder`

**当前依赖（需替换）：**
```python
import sapien.core as sapien
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
scene = engine.create_scene()
scene.add_ground_plane(...)
actor_builder = scene.create_actor_builder()
actor_builder.add_box_collision(...)
actor_builder.add_box_visual(...)
actor = actor_builder.build(is_kinematic=...)
```

**Isaac Sim 等效写法：**
```python
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial

world = World(stage_units_in_meters=1.0)
world.scene.add_ground_plane(z_position=0.0)

# 动态物体（可操作对象）
obj = world.scene.add(DynamicCuboid(
    prim_path=f"/World/{name}",
    name=name,
    position=np.array([x, y, z]),
    orientation=np.array([qw, qx, qy, qz]),  # Isaac Sim 使用 wxyz
    size=np.array([w, h, d]),
    mass=density * w * h * d,
))

# 静态环境结构（kinematic=True 等效）
env_obj = world.scene.add(FixedCuboid(
    prim_path=f"/World/{name}",
    ...
))

# 设置摩擦系数
physics_mat = PhysicsMaterial(
    prim_path=f"/World/PhysicsMaterial/{name}",
    static_friction=friction,
    dynamic_friction=friction,
    restitution=0.0,
)
```

**关键注意事项：**
- Isaac Sim 四元数格式为 **wxyz**，SAPIEN 为 **xyzw**，需转换：
  `quat_wxyz = [qw, qx, qy, qz]` vs SAPIEN `[qx, qy, qz, qw]`
- `build_twin_world()` 中 `hollow_open_top` 的 5 块薄板拼合逻辑需改用 USD Xform + 多个 child Prim 实现
- 重力沉降（`_execute_gravity_settling()`）改为 `world.step(render=False)` 循环若干步

---

### 3.2 `robot_controller.py` → `BlueprintGraphEngine`

**当前依赖（需替换）：**
```python
# 加载机器人
loader = scene.create_urdf_loader()
robot: sapien.Articulation = loader.load(urdf_path)

# 关节控制
for joint in robot.get_active_joints():
    joint.set_drive_property(stiffness, damping, ...)
robot.set_drive_target(target_qpos)

# 物理步进
scene.step()

# 获取末端执行器姿态
ee_pose = robot.get_links()[-1].get_pose()
```

**Isaac Sim 等效写法：**
```python
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationController

# 方式一：USD 导入（推荐）
add_reference_to_stage(usd_path="panda.usd", prim_path="/World/panda")
robot = world.scene.add(Robot(prim_path="/World/panda", name="panda"))
world.reset()  # 必须在 reset 后才能调用 robot API

# 方式二：URDF 直接导入（需先转换）
from omni.importer.urdf import _urdf
urdf_interface = _urdf.acquire_urdf_interface()
urdf_interface.parse_urdf(urdf_path, dest_path, config)

# 关节控制
action = robot.get_articulation_controller().get_joints_state()
robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=target_qpos)
)

# 物理步进
world.step(render=False)

# 末端执行器姿态
ee_link = robot.get_articulation_controller().get_applied_action()
# 或通过 prim 路径获取
from omni.isaac.core.utils.transformations import get_prim_pose
pos, rot = get_prim_pose("/World/panda/panda_hand")
```

**运动规划（mplib → cuMotion / Lula）：**
```python
# 方案 A：保留 mplib（推荐，改动最小）
# mplib 是纯 Python，与仿真器无关，只需把障碍物点云同步过来即可
import mplib
planner = mplib.Planner(urdf=..., srdf=..., move_group="panda_hand")
# 从 Isaac Sim USD stage 提取障碍物 AABB → 更新 planner.update_point_cloud()

# 方案 B：改用 cuMotion（Isaac Sim 原生，GPU 加速）
from omni.isaac.motion_generation import MotionPolicyController
from isaacsim.robot.motion_generation import CuMotion
```

**`_step_physics_with_probes()` 改写要点：**
- 每次 `world.step(render=False)` 对应 SAPIEN 的一次 `scene.step()`
- 500Hz 物理步长（`time_step=0.002`）在 Isaac Sim 中设置：
  ```python
  from omni.physx.scripts.utils import setPhysXSceneAPIValue
  # 或在 World 初始化时：
  world = World(physics_dt=1/500.0)
  ```

---

### 3.3 `detectors.py` → `PhysicsBoundaryDetectors`

这是改动相对集中的文件，四种检测器都依赖 SAPIEN 的接触查询。

**`check_stiffness_and_destruction()` 核心替换：**
```python
# SAPIEN（现状）
contacts = scene.get_contacts()
for contact in contacts:
    for point in contact.points:
        impulse_norm = np.linalg.norm(point.impulse)
        force = impulse_norm / dt  # 500Hz → dt=0.002

# Isaac Sim 等效
from omni.physx import get_physx_interface
# 方式一：Contact Sensor（USD Schema）
from omni.isaac.sensor import ContactSensor
contact_sensor = ContactSensor(
    prim_path="/World/target_object/contact_sensor",
    name="contact_sensor",
    min_threshold=0, max_threshold=1e10,
    radius=-1,  # 监听所有接触
)
reading = contact_sensor.get_current_frame()
# reading["force"] 即接触力（N），无需手动 impulse/dt 转换

# 方式二：直接 PhysX API（更接近当前逻辑）
from omni.physx.scripts.utils import get_rigid_body_contact_data
contact_data = get_rigid_body_contact_data(prim_path)
# 返回 [(pos, normal, force_magnitude), ...]
```

**`check_stability_and_tipping()` 核心替换：**
```python
# SAPIEN：actor.get_pose().q 转欧拉角
from scipy.spatial.transform import Rotation

# Isaac Sim：
from omni.isaac.core.utils.transformations import get_prim_pose
_, orient_wxyz = get_prim_pose(prim_path)
# orient_wxyz = [w, x, y, z]
rot = Rotation.from_quat([orient_wxyz[1], orient_wxyz[2], orient_wxyz[3], orient_wxyz[0]])
euler = rot.as_euler('xyz', degrees=True)
tilt_angle = np.sqrt(euler[0]**2 + euler[1]**2)
```

**`check_feasibility_and_deadlock()` 核心替换：**
```python
# SAPIEN：robot_articulation.get_qf()（关节力矩）
# Isaac Sim：
joint_state = robot.get_joints_state()
# joint_state.efforts → 关节力矩数组（Nm）
# joint_state.positions → 关节角度（rad）
# 对比 target_qpos 计算 tracking_error
```

---

## 4. 建议的抽象层设计

当前代码直接耦合 SAPIEN，建议在接入 Isaac Sim 前先抽出一层薄接口，避免维护两套逻辑：

```python
# embodiedbench/simulation/src/simulator/sim_backend.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class ContactInfo:
    actor_a: str
    actor_b: str
    force_N: float
    position: np.ndarray

@dataclass
class ActorState:
    name: str
    position: np.ndarray    # (3,) 米
    orientation: np.ndarray # (4,) xyzw 四元数（统一用 xyzw）
    linear_vel: np.ndarray  # (3,)

@dataclass
class RobotState:
    qpos: np.ndarray        # 关节角度 (n_joints,)
    qvel: np.ndarray        # 关节速度
    effort: np.ndarray      # 关节力矩 (Nm)
    ee_position: np.ndarray
    ee_orientation: np.ndarray


class SimBackend(ABC):
    """仿真后端抽象接口，隔离 SAPIEN / Isaac Sim 具体 API。"""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(self, render: bool = False) -> None: ...

    @abstractmethod
    def load_actor(
        self, name: str, size_whd: np.ndarray, pose_xyzw: Tuple,
        density: float, friction: float, is_kinematic: bool
    ) -> None: ...

    @abstractmethod
    def load_robot_urdf(self, urdf_path: str) -> None: ...

    @abstractmethod
    def set_joint_targets(self, qpos: np.ndarray) -> None: ...

    @abstractmethod
    def get_robot_state(self) -> RobotState: ...

    @abstractmethod
    def get_actor_state(self, name: str) -> ActorState: ...

    @abstractmethod
    def get_contacts(self) -> List[ContactInfo]: ...

    @abstractmethod
    def get_all_actor_names(self) -> List[str]: ...
```

然后分别实现：
- `SapienBackend(SimBackend)` — 封装现有 SAPIEN 代码
- `IsaacSimBackend(SimBackend)` — 封装 Isaac Sim API

`SceneBuilder`、`BlueprintGraphEngine`、`PhysicsBoundaryDetectors` 改为依赖 `SimBackend`，而不是直接引用 SAPIEN 对象。

---

## 5. 配置文件改动（`global_config.yaml`）

新增 Isaac Sim 专属字段：

```yaml
# 新增顶层字段
sim_backend: "isaac_sim"   # "sapien" | "isaac_sim"

isaac_sim_config:
  headless: true            # 无头模式（服务器/CI 环境）
  physics_dt: 0.002         # 500Hz，与 SAPIEN 一致
  rendering_dt: 0.1         # 渲染帧率（可低于物理频率）
  usd_stage_path: "/tmp/brainary_scene.usd"
  robot_usd_path: "assets/robots/panda/panda.usd"  # 预转换好的 USD
  nucleus_server: null      # 留空则使用本地资产

# 以下字段保持不变
robot_config:
  urdf_path: "assets/robots/panda/panda.urdf"  # 仍保留，用于 mplib
  max_joint_torque: 87.0
  tracking_error_threshold: 0.08
```

---

## 6. URDF → USD 资产预转换

Isaac Sim 推荐以 USD 格式加载机器人，而非每次运行时解析 URDF。一次性转换：

```python
# 在 Isaac Sim Python 环境中运行（离线转换）
from omni.importer.urdf import _urdf
import carb

config = _urdf.ImportConfig()
config.merge_fixed_joints = False
config.fix_base = True
config.make_default_prim = True
config.self_collision = False
config.distance_scale = 1.0  # URDF 单位是米

urdf_interface = _urdf.acquire_urdf_interface()
result, prim_path = urdf_interface.import_robot(
    dest_path="assets/robots/panda/",   # 输出目录
    filename="assets/robots/panda/panda.urdf",
    import_config=config,
    output_prefix="panda"
)
# 生成 assets/robots/panda/panda.usd
```

转换后将 `panda.usd` 提交到仓库，后续直接使用 USD 加载，跳过 URDF 解析。

---

## 7. Isaac Sim 环境启动方式

Isaac Sim 有两种 Python 执行环境：

### 方式 A：standalone 模式（推荐用于 headless 批量测试）
```bash
# 使用 Isaac Sim 自带 Python 解释器（包含所有 omni.* 依赖）
~/.local/share/ov/pkg/isaac_sim-*/python.sh embodiedbench/simulation/main.py
```

### 方式 B：Extension 模式（UI 调试用）
在 Isaac Sim GUI 中以插件形式加载，不适合批量 benchmark。

### 方式 C：Docker（CI / 云服务器）
```bash
docker run --gpus all -it \
  nvcr.io/nvidia/isaac-sim:4.x.x \
  /isaac-sim/python.sh /workspace/embodiedbench/simulation/main.py
```

**注意**：Isaac Sim 的 Python 环境与系统 Python **完全隔离**，`requirements_eb_manipulation_env.txt` 中的依赖需在 Isaac Sim Python 环境中重新安装：
```bash
~/.local/share/ov/pkg/isaac_sim-*/python.sh -m pip install openai mplib scipy
```

---

## 8. 接口对照速查表

| 功能 | SAPIEN API | Isaac Sim API |
|---|---|---|
| 创建 World | `sapien.Engine()` + `engine.create_scene()` | `World(physics_dt=0.002)` |
| 添加地面 | `scene.add_ground_plane()` | `world.scene.add_ground_plane()` |
| 加载 URDF | `scene.create_urdf_loader().load(path)` | `add_reference_to_stage(usd_path, prim_path)` |
| 创建刚体 | `scene.create_actor_builder().build()` | `world.scene.add(DynamicCuboid(...))` |
| 设置 kinematic | `actor_builder.build(is_kinematic=True)` | `FixedCuboid(...)` 或 USD RigidBody API |
| 物理步进 | `scene.step()` | `world.step(render=False)` |
| 获取接触 | `scene.get_contacts()` → `.points[].impulse` | `ContactSensor` 或 PhysX API |
| 设置关节目标 | `joint.set_drive_target(q)` | `robot.apply_action(ArticulationAction(joint_positions=q))` |
| 获取关节状态 | `articulation.get_qpos()` / `get_qf()` | `robot.get_joints_state().positions / .efforts` |
| 获取 Actor 姿态 | `actor.get_pose()` → `.p`, `.q` (xyzw) | `get_prim_pose(path)` → (pos, quat wxyz) |
| 四元数格式 | xyzw | wxyz（需转换！） |
| 渲染 | `SapienRenderer` + `scene.update_render()` | 自动（RTX 或 Path Tracing） |

---

## 9. 已知风险与注意事项

1. **四元数格式差异**：SAPIEN 全部使用 `[x, y, z, w]`，Isaac Sim 使用 `[w, x, y, z]`。`scene_builder.py` 的 `build_twin_world()` 和 `detectors.py` 的姿态计算中所有四元数操作都需转换。

2. **接触力计算方式不同**：SAPIEN 暴露的是 impulse（脉冲），需要除以 dt 换算成力；Isaac Sim 的 ContactSensor 直接返回力（N），`detectors.py` 中的 `impulse / dt` 逻辑需相应修改。

3. **Isaac Sim 必须先 `world.reset()` 再查询 Robot API**：在 `reset()` 之前调用 `robot.get_joints_state()` 会报错，与 SAPIEN 的即时可用不同。

4. **mplib 在 Isaac Sim Python 中可用**：mplib 是纯 C++/Python 绑定，可在 Isaac Sim 自带的 Python 中安装，无需换规划器（最小改动路径）。

5. **headless 模式下渲染管线仍需启动**：即使 `headless=True`，Isaac Sim 也需要 GPU 渲染上下文（用于 RTX 感知相机）。纯 CPU 机器上无法运行。

6. **感知管线摄像机**：`global_config.yaml` 中的三路相机（front/top/side）在 Isaac Sim 中改用 `omni.isaac.sensor.Camera` 替代 SAPIEN 的 mounted camera，内参设置方式不同但数值可复用。

7. **USD 资产路径**：Isaac Sim 推荐将资产放在 Nucleus 服务器或本地 `omniverse://localhost/` 路径下，直接使用文件系统路径时注意路径格式（Windows 需用 `/` 而非 `\`）。

---

## 10. 建议接入步骤（最小改动路径）

```
Step 1  抽象 SimBackend 接口（见第 4 节）
        - 新建 sim_backend.py
        - 将现有 SAPIEN 代码封装为 SapienBackend(SimBackend)
        - 三个 simulator 文件改为依赖 SimBackend，验证现有功能不回退

Step 2  URDF 转 USD
        - 离线运行 URDF 转换脚本，生成 panda.usd
        - 提交到 assets/robots/panda/

Step 3  实现 IsaacSimBackend(SimBackend)
        - 优先实现 step / load_actor / load_robot_urdf / set_joint_targets / get_contacts
        - 单元测试：加载场景 + 执行一步物理，检查返回值格式

Step 4  接触力单位对齐
        - 将 detectors.py 中 impulse/dt 逻辑统一到 SimBackend.get_contacts()
        - IsaacSimBackend 直接返回力（N），SapienBackend 内部做 impulse/dt

Step 5  四元数统一
        - 在 SimBackend 接口中约定统一使用 xyzw
        - IsaacSimBackend 内部转换 wxyz → xyzw

Step 6  端到端冒烟测试
        - 用 simulation/main.py 的单任务流程跑通一个 pick-and-place demo
        - 对比 SAPIEN 和 Isaac Sim 下的物理检测结果一致性
```
