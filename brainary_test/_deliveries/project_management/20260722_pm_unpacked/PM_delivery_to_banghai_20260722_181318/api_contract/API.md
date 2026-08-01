# 仿真模块统一接口契约 (Interface Contract) —— 给「规划 / 技能模块」开发者

> **这份文档就是接口契约。** 你只需要这一个文件,就能照着设计并实现你模块对外的接口:
> 方法名、参数、目标枚举、返回结构、调用约定全在这里。你**不需要**先跑起仿真——
> 照契约把你的模块写好,拿回本项目 `brainary/` 里,接上 `SimInterface` 就能执行。
>
> 集成时的实际入口:`from sim_interface import SimInterface`(位于 `brainary/sim/`)。

**一句话**:`SimInterface` 是仿真环境的**唯一入口**。你用它做两件事:
**① 感知仿真环境**(读相机 / 物体位姿 / 机器人本体,只读无副作用);
**② 给机器人下技能指令**(阻塞式,跑完返回结构化结果 dict)。

> 你的模块 = 决定"什么时候、对哪个目标、调哪个技能",然后调本 API 执行。只依赖本文件列出的方法。

---

## 0. 30 秒上手

```python
from sim_interface import SimInterface        # brainary/sim/ 在 sys.path 上

sim = SimInterface.launch(headless=True, device="cuda:0")   # 起 brainary_test 场景(约 60~120s)

# —— 感知 ——
views  = sim.get_all_cameras()                 # 5 路 {name: {rgb, depth, intrinsics, camera_pose_world}}
banana = sim.get_object_pose("Prop_011_banana")   # {'position':[x,y,z], 'quat_wxyz':[w,x,y,z]} 世界系
state  = sim.get_robot_state()                 # {joints, tcp_pose, gripper_width, contact_forces}

# —— 能力发现(务必先问,别硬编码名字)——
for s in sim.list_skills():
    print(s["skill"], "可作用于:", s["targets"])

# —— 执行技能(阻塞,跑完返回 dict)——
r1 = sim.grasp("Prop_011_banana")              # 抓香蕉
if r1["ok"]:
    r2 = sim.place("Prop_KLT_3")               # 放进 3 号篮子
print(r1, r2)

sim.close()
```

> `BrainaryAPI`(`brainary/sim/brainary_api.py`)是包了一层的便捷入口(限定到本场景现有物体、带中文标签、
> 非法目标会报错)。`BrainaryAPI.launch()` 内部就是 `SimInterface`,`brainary_api.sim` 可拿到完整 `SimInterface`。
> 两个都行。**本契约以 `SimInterface` 为准。**

---

## 1. 生命周期与运行控制

| 方法 | 说明 |
|---|---|
| `SimInterface.launch(*, headless=True, device="cuda:0", seed=1, enable_force=True, settle=15, scene_registry_path=None) -> SimInterface` | 起场景。`device` 必须 `cuda:0`。默认加载 brainary_test 场景 |
| `sim.close()` | 关闭 |
| `sim.pause()` / `sim.resume()` / `sim.is_paused()` | 暂停 / 恢复(技能运行中也可调) |
| `sim.stop()` / `sim.request_stop()` | 中止当前正在跑的技能 |

---

## 2. 感知 API(只读,无副作用)

**相机**(5 路逻辑名:`front / wrist / left / right / top`):

| 方法 | 返回 |
|---|---|
| `get_rgb(camera)` | `HxWx3 uint8` |
| `get_depth(camera)` | `HxW float32`(米) |
| `get_rgbd(camera)` | `(rgb, depth)` |
| `get_all_cameras(require_depth=True)` | `{name: {rgb, depth, intrinsics{fx,fy,cx,cy,width,height}, camera_pose_world{position,quat_wxyz}}}` |
| `get_camera_intrinsics(camera)` / `get_camera_pose(camera)` | 内参 / 世界外参 |

**机器人本体 + 物体**:

| 方法 | 返回 |
|---|---|
| `get_robot_state()` | `{joints, tcp_pose, gripper_width, contact_forces}`(一次全拿) |
| `get_joint_state()` | `{arm[7], all_pos[9], vel[9]}`(弧度;arm=7 臂关节, all_pos 含 2 夹爪指) |
| `get_tcp_pose()` | `{position[3], quat_wxyz[4]}`(世界系) |
| `get_gripper_width()` | `float`(米) |
| `get_contact_forces(body_only=False)` | `[(link_name, force_N), ...]` |
| `get_object_pose(name)` | `{position[3], quat_wxyz[4]}` 或 `None`(世界系) |
| `list_objects()` | 被跟踪物体名列表 |

---

## 3. 能力发现 —— 有哪些技能、能作用于谁(**先问再调**)

```python
sim.list_skills()   # -> [{"skill": "grasp", "targets": [...], "desc": "..."}, ...]
```

`list_skills()` 返回**本场景当前真正可用**的技能 + 每个技能的可选目标(targets 直接来自控制器真实目标源)。
**当前 brainary_test 场景**(无柜子/门/咖啡机/微波炉等家电)可用技能就两个:

| 技能 skill | 目标来自 | 说明 |
|---|---|---|
| `grasp` | `list_graspable()` | Z-approach 抓一个物体 |
| `place` | `list_place_baskets()` | 把手里的物体放进篮子 |

单独查目标:`list_graspable()` / `list_place_baskets()`。

> **⚠️ 名字必须动态取、精确匹配。** 运行时用 `list_graspable()` / `list_place_baskets()` 拿合法名字,
> **不要硬编码字符串**——换场景(别的 registry)这些会变。
>
> 当前场景**示例**(以 `list_graspable()` 返回为准):
> 可抓物 = `Prop_011_banana`(香蕉)/ `Prop_orange_01`(橙子)/ `Prop_037_scissors`(剪刀)/
> `Prop_003_cracker_box`(饼干盒)/ `Prop_SM_Mug_C1`、`Prop_SM_Mug_D1`(两个杯子);
> 放置篮子 = `Prop_KLT_1` / `Prop_KLT_2` / `Prop_KLT_3`。

---

## 4. 执行技能 API(**阻塞式**:调用后内部逐帧推物理,跑完才返回)

### 当前可用技能

| 方法 | 目标 | 说明 |
|---|---|---|
| `grasp(obj, *, max_steps=8000) -> dict` | ∈ `list_graspable()` | 抓;成功后夹爪**保持闭合不松**(等你显式 place/open_gripper 才松) |
| `place(basket, *, max_steps=8000) -> dict` | ∈ `list_place_baskets()` | 放进篮子;**必须先 grasp 持有物体**;放完自动松爪 |
| `open_gripper(steps=40) -> dict` / `close_gripper(steps=40) -> dict` | — | 只动夹爪 |
| `go_home(home_q=None, *, tol=0.03) -> dict` | — | 手臂回 home 关节位形 |

> **夹爪已改为限力(力控)**:闭合出力上限约 **20N**(而非硬位置控制),手指碰到物体表面即停,不再挤穿。
> 需要更大夹持力时用环境变量 `V1_GRIPPER_EFFORT` 调(启动前设)。

### 返回值(统一结构 dict)—— 判定成功只看 `ok`

所有技能返回一个 dict,**都有**:
- `ok`: bool —— **成功判定**(基于物理事实,不是"发了就算成功")
- `skill` / `target`、`steps`(用了多少步)、`timed_out`(是否超时)、`reason`(失败原因,成功时为空)

技能特有字段:
- `grasp`: `holding`(bool,物体是否在手边)、`gripper_width`。**ok = 夹爪合上且物体在手 且 未超时**。
  - 失败 `reason` 常见:`gripper_empty(合爪后没夹住物体)` / `timed_out(未在步数内够到/合爪)`。
- `place`: `released`(bool,夹爪是否张开放下)、`basket_xy`、`object_in_basket`。**ok = 放开且物体落在篮子附近 且 未超时**。
- `go_home`: `joint_err`。

```python
r = sim.grasp("Prop_037_scissors")
# r 例: {"ok": True, "skill": "grasp", "target": "Prop_037_scissors", "holding": True,
#        "gripper_width": 0.021, "steps": 640, "timed_out": False, "reason": ""}
if not r["ok"]:
    print("抓取失败:", r["reason"])   # 够不到 / 没夹住 都在 reason / holding 里体现
```

**典型分拣循环**(你的规划/技能模块大概长这样):
```python
for obj in sim.list_graspable():
    if not sim.grasp(obj)["ok"]:
        continue                        # 抓失败,跳过(或重试/换姿态)
    basket = choose_basket(obj)         # 你的逻辑:按类别选篮子
    sim.place(basket)
sim.go_home()
```

---

## 5. 你的模块怎么接进来

1. 照本契约(§2 感知 / §3 发现 / §4 执行)把你模块**对外的接口**写好,内部封你自己的决策逻辑
   (选目标 / 选技能 / 顺序 / 重试),**只依赖本文件列出的方法**,不要碰底层 Isaac / 状态机(那些会变)。
2. 交付时把你的模块放进 `brainary/`(比如 `brainary/planning/`);集成处 `from sim_interface import SimInterface`
   拿到 `sim`,在闭环 `run_brainary.py` 的规划阶段调你的模块。
3. 开发阶段**不需要仿真在跑**——照签名写;集成到本项目时再接真仿真验证。

**全局约定**:
- 关节 = 7 维绝对弧度;夹爪对外 `0=开 / 1=闭`;四元数一律 `wxyz`;位姿默认**世界系**、单位**米**。
- 技能**阻塞**:调用返回结果 dict 后才继续;用 `ok` / `reason` 做重试与决策。
- 目标名一律**运行时用 `list_*()` 动态取**,别硬编码。

---

## 附录 A. 暂不可用的技能(本场景已移除家电,**先不要对接**)

> 下面这些技能**代码里有**,但 **brainary_test 当前场景没有柜子 / 抽屉 / 门 / 咖啡机**,
> 所以 `list_skills()` **不会**返回它们,`list_drawers()/list_doors()` 返回空。
> **现在请不要基于它们设计接口。** 等场景加回相应家电、并逐个标定+验证后,会重新在 `list_skills()` 里出现,
> 届时再按同样的"发现→执行→看 ok"模式接入即可。此处仅留档,**已注释**:

```python
# ===== 以下技能在当前 brainary_test 场景不可用,请勿对接 =====
#
# sim.open_drawer(drawer, *, max_steps=15000) -> dict     # 拉开抽屉  (drawer ∈ list_drawers(),现为空)
# sim.close_drawer(drawer, *, max_steps=15000) -> dict    # 推回抽屉
# sim.open_door(door="fridge", *, max_steps=15000) -> dict # 开门     (door ∈ list_doors(),现为空)
# sim.close_door(door="fridge", *, max_steps=15000) -> dict# 关门
# sim.operate_coffee(*, max_steps=15000) -> dict           # 抓咖啡把手绕关节轴旋转
#
# 返回结构与 §4 一致(ok / reason / steps / timed_out + 各自 joint_err 等)。
# 注意:即便日后加回家电,这些技能仍需【逐场景标定把手抓取位姿】且受臂展可达性限制,
#       不是加回来就能直接用——以那时 list_skills() 的实际返回为准。
# ============================================================
```

---

有不清楚的字段,看 `sim_interface.py` 里对应方法的中文 docstring,或直接问场景侧负责人。
