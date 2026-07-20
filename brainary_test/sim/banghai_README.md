# SimInterface —— 仿真环境统一接口

给其他模块(感知/规划/记忆等)用的**唯一入口**:**既能操作机器人(执行技能),又能读取感知数据**。
它就是把你**测试 UI 背后的功能**原样封成 API,别人从最底层调用即可。主体:`sim_interface.py`。

## 底层(位置不变,本接口只调用)
- **技能** = `franka_skill_state_machine/state_machine/skill_test_controller.py : SkillTestController`
  —— 就是 test_mode_ui 里那些按钮背后的东西(Z-approach 抓取 / 放进篮子 / 开关抽屉 / 咖啡把手)。
- **场景** = `franka_v1_skill_lab/scene_interface/session.py : SceneSession`(相机 / 机器人 / 物体)。

> v2 变更:技能后端从早先的 BrainInterface(一套简化门面,目标写死 cube/knife、place 只是 xyz)
> **改为直接封装你的 `SkillTestController`**。这样"抓桌面物体 / 放 3 个篮子 / 咖啡机"都是你 UI 现成的能力,不用"注册"。

## 运行环境
`env_isaaclab`;相机+深度默认开;接触力需 `enable_force=True`(默认)。**必须 cuda:0**(Isaac 渲染只支持 GPU0)。
```python
from sim_interface import SimInterface
sim = SimInterface.launch(headless=True)      # 起仿真
```

## 一、感知(只读,直接读 session/provider/接触力传感器)
| 方法 | 返回 |
|---|---|
| `get_rgb/get_depth/get_rgbd(camera)` | camera ∈ front/wrist/left/right/top |
| `get_camera_intrinsics/pose(camera)` | 内参 / 世界位姿 |
| `get_all_cameras()` | **一次拿全 5 相机** {cam:{rgb,depth,intrinsics,pose}} |
| `get_joint_state()` | {arm[7], all_pos[9], vel[9]} 弧度 |
| `get_tcp_pose()` | {position, quat_wxyz} 任务空间(世界系) |
| `get_gripper_width()` | 夹爪开度(米) |
| `get_contact_forces(body_only)` | [(连杆名, 净接触力N)] ⚠️只有接触力,无关节力矩 |
| `get_robot_state()` | 本体一次全拿 |
| `get_object_pose(name)` / `list_objects()` | 物体真值 |

## 二、能力发现(目标枚举)
`list_graspable()` / `list_place_baskets()`(3篮子) / `list_drawers()`(5抽屉) / `list_doors()` / `list_skills()`

## 三、执行(阻塞式:调用→内部逐帧 step 跑完→返回结果)
| 方法 | 目标 |
|---|---|
| `grasp(obj)` | 物体名(有标定抓取位姿最好;无则对场景刚体用俯视兜底位姿) |
| `place(basket)` | ∈ list_place_baskets()(3 个 KLT 篮子) |
| `open_drawer(d)` / `close_drawer(d)` | ∈ list_drawers()(5 个) |
| `open_door(d)` / `close_door(d)` | ∈ list_doors() |
| `operate_coffee()` | 抓咖啡把手绕关节轴旋转(需 handle_coffee_lever 的标定位姿) |
| `stop()` | 中止当前技能 |
返回 `{ok, steps, timed_out, status, failure_reason}`。非法目标抛 `ValueError`。

## 能力现状(诚实)
| 能力 | 状态 |
|---|---|
| 5 相机 RGB+深度 / 关节 / TCP / 夹爪 | ✅ 完全可用 |
| 力 | ⚠️ 仅**接触力**(每连杆净力),无关节力矩 |
| 抓取 grasp | ✅ 对有标定抓取位姿的物体最稳;任意场景刚体可用**俯视兜底位姿**(成功率视物体而定) |
| 放置 place | ✅ 3 个篮子(读篮子世界位,开环抬起→上方→松爪) |
| 抽屉 | ✅ 5 个(Cabinet 3 + Sektion 2;bottom_drawer 出厂锁定) |
| 门 | ✅ open/close |
| 咖啡机 | ✅ operate_coffee(需先在场景里标定 handle_coffee_lever 抓取位姿) |

> 抓取/放置/咖啡走的是控制器的开环运行器(Z-approach / place / 咖啡把手绕轴),与你 UI 按钮完全同一套代码。
> 无头抓取要"能抓哪些"取决于 grasp_poses.json 里标定过的位姿(或对刚体用兜底俯视位姿)——这不是"注册",是"有没有存过抓取位姿"。

## 最小示例
```bash
conda activate env_isaaclab
./isaaclab.sh -p projects/banghai/example_usage.py --headless   # 打印5相机RGBD+本体+目标枚举,并跑一次开抽屉
```
