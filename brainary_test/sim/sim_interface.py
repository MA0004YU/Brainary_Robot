#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
#  SimInterface —— 仿真环境统一接口(交付给其他模块用)
# ======================================================================================
#  一句话:把【测试 UI 背后的功能】原样封成一个 Python API,别人从最底层调用即可,
#          既能【操作机器人(执行技能)】,又能【读取感知数据(相机/本体/力)】。
#
#  底层直接封装你的 UI 控制器(不是旁边那套简化门面):
#    技能 = franka_skill_state_machine/state_machine/skill_test_controller.py : SkillTestController
#           (就是 test_mode_ui 里那些按钮背后的东西:Z-approach 抓取、放进篮子、开关抽屉、咖啡把手)
#    场景 = franka_v1_skill_lab/scene_interface/session.py : SceneSession(相机/机器人/物体)
#  这些文件位置不变,本接口只调用它们。
#
#  技能是【阻塞式】:调用后本接口内部一帧帧 step 控制器 + 推进物理,直到该技能跑完再返回。
#
#  运行:conda activate env_isaaclab; 你的脚本里 SimInterface.launch(...);必须 cuda:0(Isaac 渲染只支持 GPU0)。
# ======================================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# brainary/sim/sim_interface.py -> IsaacLab/projects (仿真框架 franka_v1_skill_lab 等留在 projects/,这里导入)
_PROJECTS_DIR = Path(__file__).resolve().parents[2] / "projects"
sys.path.insert(0, str(_PROJECTS_DIR))

CAMERAS: Tuple[str, ...] = ("front", "wrist", "left", "right", "top")   # 5 个相机逻辑名
_NON_GRASPABLE = {"cabinet", "sektion_cabinet", "coffee_machine", "microwave", "fridge",
                  "dishwasher", "robot", "ground", "table"}


class SimInterface:
    """仿真统一接口:感知(只读)+ 技能执行(封装 SkillTestController)。"""

    # ==================================================================================
    #  启动 / 关闭
    # ==================================================================================
    @classmethod
    def launch(cls, *, headless: bool = True, device: str = "cuda:0", seed: int = 1,
               enable_force: bool = True, settle: int = 15, scene_registry_path=None,
               _app_launcher=None) -> "SimInterface":
        """启动场景并返回接口。相机+深度默认开;enable_force 挂接触力传感器。device 必须 cuda:0。

        ★ 场景【复刻 test_mode_ui 的默认搭建】:spawn 水果/YCB 道具 + L桌 + 3个KLT篮子,
          默认排除冰箱(microwave)+洗碗机,机器人在支架上。这样 API 控制的场景和你的测试 UI 一致
          (篮子/桌面道具/咖啡机/柜子都在)。道具常量从 test_mode_ui import(不复制、不改 UI)。

        scene_registry_path:None=默认加载 v1_active 最新场景;指定某个保存场景目录的
          scene_v1_registry.json 即加载/摆放那个独立场景的布局(道具照样先 spawn,再由该场景
          的 manifest 把它们摆到保存位置)——等价于 test_mode_ui --scene_registry <path>。
        """
        import re
        from franka_v1_skill_lab.scene import V1_BASE_TASK_ID
        from franka_v1_skill_lab.scene_interface import (ResetMode, SceneConfig, SceneMode, SceneSession)
        from franka_v1_skill_lab.scene_interface.test_mode_ui import (
            DEFAULT_PROP_URLS, ISAAC_PROPS, PROP_ROT_OVERRIDE, _IP)

        def _prop_name(path):
            stem = path.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
            if stem.endswith("_base"):
                stem = stem[:-5]
            return "Prop_" + re.sub(r"[^A-Za-z0-9_]", "_", stem)

        # 复刻 test_mode_ui._run 的 extra_assets(默认路径:全开)
        items = []
        for i, url in enumerate(DEFAULT_PROP_URLS):        # 水果:机器人前方一排
            items.append({"name": _prop_name(url), "usd_path": url,
                          "pos": (0.45, -0.20 + 0.12 * i, 0.0), "scale": (1.0, 1.0, 1.0),
                          "rigid": True, "ground": True})
        for i, (path, _zh, rigid) in enumerate(ISAAC_PROPS):   # YCB 道具:+Y 停车网格
            col, row = i % 6, i // 6
            rot = next((r for k, r in PROP_ROT_OVERRIDE.items() if path.endswith(k)), (1.0, 0.0, 0.0, 0.0))
            items.append({"name": _prop_name(path), "usd_path": path,
                          "pos": (-1.0 + col * 0.45, 1.6 + row * 0.45, 0.0), "rot": rot,
                          "scale": (1.0, 1.0, 1.0), "rigid": bool(rigid), "ground": True})
        items.append({"name": "Prop_L_Desk",               # L 形前台桌
                      "usd_path": "SapienAssetPipeline/usd_assets/CloudProps/L_Desk/L_Desk.usd",
                      "pos": (1.8, 1.8, 0.0), "scale": (0.01, 0.01, 0.01), "rigid": False, "ground": True})
        _KLT = _IP + "Props/KLT_Bin/small_KLT_visual_collision.usd"   # 3 个 KLT 篮子
        _n0 = len(items)
        for j in range(3):
            col, row = (_n0 + j) % 6, (_n0 + j) // 6
            items.append({"name": f"Prop_KLT_{j+1}", "usd_path": _KLT,
                          "pos": (-1.0 + col * 0.45, 1.6 + row * 0.45, 0.0),
                          "scale": (1.0, 1.0, 1.0), "rigid": True, "ground": True})

        # 盒装物体:自带碰撞网格比【可见外形】小 -> 夹爪穿模。用贴合可见外形的【有向盒】碰撞体替换
        # (跟随物体真实朝向,倾斜/旋转也贴合)。饼干盒真实窄边≈7.2cm < 夹爪8cm,所以 margin=0(完全贴合)也能抓。
        _BOX_COLLISION = {"Prop_003_cracker_box": ("visualBox", 0.0)}
        for _it in items:
            if _it["name"] in _BOX_COLLISION:
                _it["collision_approx"], _it["collision_margin"] = _BOX_COLLISION[_it["name"]]

        # ★ 只保留【当前场景 manifest 里实际有的】物体：删减场景(如 brainary_test)里被删掉的
        #   道具不再冒到地上，被删掉的家电(柜子/咖啡机/Sektion/刀)也不再出现。按 manifest 过滤。
        _exclude = {"microwave", "dishwasher"}
        _hidden = ()
        try:
            import json as _json
            from franka_v1_skill_lab.scene.scene_registry import resolve_active_scene
            _info = resolve_active_scene(scene_registry_path)
            _kept = {o.get("name") for o in _json.load(open(_info["manifest"]))["objects"]}
            items = [it for it in items if it["name"] in _kept]   # 只 spawn manifest 里有的道具
            _APPL = {"Cabinet": "cabinet", "CoffeeMachine": "coffee_machine",
                     "SektionCabinet": "sektion_cabinet", "Knife": "knife",
                     "Microwave": "microwave", "Fridge": "microwave", "Dishwasher": "dishwasher"}
            for _mn, _cfgn in _APPL.items():
                if _mn not in _kept:                              # manifest 里没有 -> 删掉的 -> 排除
                    _exclude.add(_cfgn)
            # 契约方块 cube_1/2/3 是 base task 成员(不能 exclude 置 None,obs 会引用)->manifest 里删掉的【移出场景外】隐藏
            _hidden = tuple(f"cube_{i}" for i in (1, 2, 3) if f"Cube_{i}" not in _kept)
            print(f"[sim_interface] scene-filtered by manifest: {len(items)} props kept, "
                  f"exclude={sorted(_exclude)}, hidden_cubes={_hidden}", flush=True)
        except Exception as _exc:
            print(f"[sim_interface] WARNING: manifest filter failed ({_exc}); spawning full default set.", flush=True)

        cfg = SceneConfig(
            mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=device, headless=headless,
            enable_cameras=True, enable_fp=True, enable_collision_monitor=enable_force,
            load_latest_scene=True, scene_registry_path=scene_registry_path,
            replace_microwave_with_fridge=True, add_microwave_stand=False,
            add_robot_stand=True, lock_knife=True, free_microwave_door=False, spawn_init_markers=True,
            refine_handle_collisions=True, apply_saved_camera_offsets=True,
            reset_mode=ResetMode.STATIC, control_hz=50.0, seed=seed,
            extra_assets=tuple(items),
            exclude_members=tuple(sorted(_exclude)),   # 排除 manifest 里没有的家电(含默认冰箱/洗碗机)
            disable_auto_reset=True, hidden_members=_hidden,   # manifest 里删掉的方块移出场景外
        )
        session = SceneSession.launch(cfg, _app_launcher=_app_launcher)
        try:
            from franka_v1_skill_lab.scene_interface import camera_offsets
            camera_offsets.apply_saved_offsets_runtime(session.env)
        except Exception:
            pass
        session.reset(seed=seed)
        try:
            cls._color_baskets(session)   # 三个篮子上色:KLT_1=绿, KLT_2=蓝, KLT_3=原色
        except Exception as _exc:
            print(f"[sim_interface] color baskets failed: {_exc}", flush=True)
        for _ in range(int(settle)):
            session.hold()
        inst = cls(session)
        inst._scene_registry_path = scene_registry_path
        return inst

    @staticmethod
    def _color_baskets(session, colors=None):
        """给三个 KLT 篮子上色:KLT_1=绿, KLT_2=蓝, KLT_3=保持原色。

        KLT 三个共用同一 USD(同材质),所以逐个 prim【去实例化】后绑一个强绑定的 PreviewSurface
        覆盖材质(只覆盖要改色的两个;KLT_3 不动)。绑到根 + 每个子 Mesh,确保原 MDL 材质被盖住。"""
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdShade

        if colors is None:
            colors = {"Prop_KLT_1": (0.05, 0.55, 0.10),    # 绿
                      "Prop_KLT_2": (0.10, 0.20, 0.80)}    # 蓝 ; KLT_3 保持原色不改
        stage = omni.usd.get_context().get_stage()
        env_ns = "/World/envs/env_0"
        done = []
        for name, rgb in colors.items():
            prim_path = f"{env_ns}/{name}"
            root = stage.GetPrimAtPath(prim_path)
            if not root.IsValid():
                continue
            for p in Usd.PrimRange(root):        # 去实例化,否则材质绑定改不动实例代理
                try:
                    if p.IsInstanceable():
                        p.SetInstanceable(False)
                except Exception:
                    pass
            mat_path = f"{prim_path}/ColorMat"
            mat = UsdShade.Material.Define(stage, mat_path)
            sh = UsdShade.Shader.Define(stage, mat_path + "/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            targets = [root] + [p for p in Usd.PrimRange(root) if p.GetTypeName() == "Mesh"]
            for p in targets:
                b = UsdShade.MaterialBindingAPI.Apply(p)
                b.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            done.append(name)
        print(f"[sim_interface] baskets colored: {done} (+ KLT_3 keeps original)", flush=True)

    def __init__(self, session):
        self.session = session
        self.provider = session.provider
        self.env = session.env
        self._cams = {name: getattr(session, f"{name}_cam", None) for name in CAMERAS}

        # 技能:直接实例化你的 UI 控制器(headless,不建窗口)。它内部管好 executor/开环运行器/咖啡等。
        from franka_v1_skill_lab.skill_runtime._legacy import ensure_legacy_on_path
        ensure_legacy_on_path()
        from state_machine.skill_test_controller import SkillTestController
        self._ctrl = SkillTestController(session)

        # 接触力监视器(读机器人各连杆净接触力)
        try:
            from runtime.collision_monitor import CollisionMonitor
            self._monitor = CollisionMonitor(self.provider.scene, env_id=0)
        except Exception:
            self._monitor = None

        self._paused = False                 # 暂停标志:阻塞技能中被置位时,_pump 冻结当前姿态不推进
        self._scene_registry_path = None     # launch 时填(供 go_home 找同目录 robot_home.json)
        # 持久夹爪指令(1.0=开 / -1.0=合):抓到物体后置 -1 并【一直保持】,直到 open_gripper / place
        # 触发开爪 -> 空闲 hold 用它,不会把抓到的东西松掉。
        self._gripper_cmd = 1.0

    def close(self) -> None:
        self.session.close()

    # ==================================================================================
    #  A. 感知 —— 相机(5 路 RGB + 深度),直接读 SceneSession 的相机
    # ==================================================================================
    def _cam(self, camera: str):
        if camera not in self._cams:
            raise KeyError(f"未知相机 {camera!r};可选 {list(CAMERAS)}")
        cam = self._cams[camera]
        if cam is None or not cam.is_available():
            raise RuntimeError(f"相机 {camera!r} 未挂载/不可用")
        return cam

    def get_rgb(self, camera: str):
        """RGB,HxWx3 uint8。camera ∈ CAMERAS。"""
        return self._cam(camera).capture(require_depth=False).get("rgb")

    def get_depth(self, camera: str):
        """深度,HxW float32(米)。"""
        return self._cam(camera).capture(require_depth=True).get("depth")

    def get_rgbd(self, camera: str) -> Tuple[Any, Any]:
        f = self._cam(camera).capture(require_depth=True)
        return f.get("rgb"), f.get("depth")

    def get_camera_intrinsics(self, camera: str) -> dict:
        return self._cam(camera).intrinsics()

    def get_camera_pose(self, camera: str) -> dict:
        """相机世界位姿 {position, quat_wxyz}。"""
        return self._cam(camera).camera_pose_world()

    def get_all_cameras(self, require_depth: bool = True) -> Dict[str, dict]:
        """一次拿全 5 相机:{cam:{rgb, depth, intrinsics, pose}}。"""
        out: Dict[str, dict] = {}
        for name in CAMERAS:
            cam = self._cams.get(name)
            if cam is None or not cam.is_available():
                continue
            f = cam.capture(require_depth=require_depth)
            out[name] = {"rgb": f.get("rgb"), "depth": f.get("depth"),
                         "intrinsics": f.get("intrinsics"), "pose": f.get("camera_pose_world")}
        return out

    # ==================================================================================
    #  A. 感知 —— 机器人本体(关节 / 任务空间 / 夹爪 / 接触力),直接读 provider + 传感器
    # ==================================================================================
    def get_joint_state(self) -> dict:
        """关节:{arm[7], all_pos[9], vel[9]}(弧度)。"""
        st = self.provider.get_state()
        return {
            "arm": [float(v) for v in self.provider.arm_joint_pos(st).detach().cpu().tolist()],
            "all_pos": [float(v) for v in st.robot.joint_pos.detach().cpu().tolist()],
            "vel": [float(v) for v in st.robot.joint_vel.detach().cpu().tolist()],
        }

    def get_tcp_pose(self) -> dict:
        """任务空间:末端 TCP 世界位姿 {position[3], quat_wxyz[4]}。"""
        st = self.provider.get_state()
        return {"position": [float(v) for v in st.robot.tcp_pose.pos_w.detach().cpu().tolist()],
                "quat_wxyz": [float(v) for v in st.robot.tcp_pose.quat_w.detach().cpu().tolist()]}

    def get_gripper_width(self) -> float:
        """夹爪开度(米)。"""
        return float(self.provider.get_state().robot.gripper_width)

    def get_contact_forces(self, body_only: bool = False) -> List[Tuple[str, float]]:
        """力感知:机器人各连杆净接触力 [(连杆名, 力N)](只列超阈值的)。
        ⚠️ 仿真只有接触力,无关节力矩。body_only=True 排除两指。需 enable_force=True。"""
        if self._monitor is None or not getattr(self._monitor, "available", False):
            return []
        return list(self._monitor.contacts(body_only=body_only))

    def get_robot_state(self) -> dict:
        """本体一次全拿:关节 + TCP + 夹爪 + 接触力。"""
        return {"joints": self.get_joint_state(), "tcp_pose": self.get_tcp_pose(),
                "gripper_width": self.get_gripper_width(),
                "contact_forces": self.get_contact_forces(body_only=False)}

    def get_object_pose(self, name: str) -> Optional[dict]:
        """物体世界位姿。先查被跟踪物体(cube/knife),再回退场景成员(spawn 的道具)root 位姿。"""
        st = self.provider.get_state()
        obj = st.objects.get(name)
        if obj is not None:
            return {"position": [float(v) for v in obj.pose.pos_w.detach().cpu().tolist()],
                    "quat_wxyz": [float(v) for v in obj.pose.quat_w.detach().cpu().tolist()]}
        try:
            if name in self.provider.scene.keys():
                a = self.provider.scene[name]
                return {"position": [float(v) for v in a.data.root_pos_w[0].detach().cpu().tolist()],
                        "quat_wxyz": [float(v) for v in a.data.root_quat_w[0].detach().cpu().tolist()]}
        except Exception:
            pass
        return None

    def list_objects(self) -> List[str]:
        return list(self.provider.get_state().objects.keys())

    # ==================================================================================
    #  能力发现 —— 每个技能的可选目标(取自控制器真实的目标源)
    # ==================================================================================
    def list_graspable(self) -> List[str]:
        """可抓目标:已标定抓取位姿的物体(grasp_poses.json,排除把手/篮子)+ 被跟踪物体。

        说明:GUI 带抓取面板时会实时列出所有桌面物体;无头时抓取靠【已保存的抓取位姿】或
        对任意场景刚体用【俯视兜底位姿】(见 grasp())。所以这里列的是"最稳"的一批,不代表全部上限。
        """
        names = set()
        try:
            for k in self._ctrl._load_grasp_poses().keys():
                if k.startswith("handle_") or "KLT" in k or k in _NON_GRASPABLE:
                    continue
                # 只列【实际能解析到抓取 pose】的(其参考系成员在场景里),避免虚报没 spawn 的道具
                try:
                    if self._ctrl._resolve_named_pose(k) is not None:
                        names.add(k)
                except Exception:
                    pass
        except Exception:
            pass
        for n in self.list_objects():
            if n.lower() not in _NON_GRASPABLE:
                names.add(n)
        # 过滤掉【被隐藏移出场景外】的成员(如删掉的 cube_1/2/3,停在 ~100,100):不算可抓,别让规划踩空
        out = []
        for n in sorted(names):
            p = self.get_object_pose(n)
            if p is not None and max(abs(p["position"][0]), abs(p["position"][1])) > 50.0:
                continue
            out.append(n)
        return out

    def list_place_baskets(self) -> List[str]:
        """放置目标:3 个篮子。篮子是静态道具(不是被跟踪物体),这里直接【实时读它在 USD stage 里的世界位姿】,
        能读到的才列出。"""
        from state_machine.skill_test_controller import _BASKET_TARGETS
        out = []
        for _label, member in _BASKET_TARGETS:
            if self._resolve_place_pos(member) is not None:
                out.append(member)
        return out

    def _resolve_place_pos(self, basket: str):
        """篮子放置点(世界系 xyz tensor):先试控制器解析(保存位姿/场景成员),
        再回退【直接从 USD stage 实时读该 prim 的世界平移】——篮子无需注册成"物体"。返回 tensor 或 None。"""
        c = self._ctrl
        try:
            pw = c._basket_world_pos(basket)
            if pw is not None:
                return pw
        except Exception:
            pass
        # 回退:遍历 stage 找名字匹配的 prim,读其世界平移(静态篮子的实时 pose)
        try:
            import torch
            from pxr import Usd, UsdGeom
            stage = self.env.unwrapped.scene.stage
            for prim in stage.Traverse():
                if str(prim.GetPath()).rsplit("/", 1)[-1] == basket:
                    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    t = m.ExtractTranslation()
                    return torch.tensor([float(t[0]), float(t[1]), float(t[2])],
                                        dtype=torch.float32, device=c.device)
        except Exception as exc:
            print(f"[SimInterface] stage read basket {basket!r} failed: {exc}", flush=True)
        return None

    def _scene_members(self) -> set:
        """当前场景真实 spawn 出来的成员名集合(用于判断家电是否存在)。"""
        try:
            return set(str(k) for k in self.provider.scene.keys())
        except Exception:
            return set()

    def list_drawers(self) -> List[str]:
        """抽屉目标——只列【本场景真实存在的柜子成员】的抽屉(柜子被删则返回空)。"""
        from state_machine.skill_test_controller import _DRAWER_TARGETS
        try:
            from runtime.drawer_target_config import member_for
            keys = self._scene_members()
            return [d for d in _DRAWER_TARGETS if member_for(d) in keys]
        except Exception:
            return []

    def list_doors(self) -> List[str]:
        """门目标——只列【本场景真实存在的家电成员】的门(微波炉/冰箱被删则返回空)。"""
        from state_machine.skill_test_controller import _DOOR_TARGETS
        try:
            from isaaclab_tasks.manager_based.manipulation.stack.config.franka.microwave_door_config import DOOR_TARGETS
            keys = self._scene_members()
            return [d for d in _DOOR_TARGETS if DOOR_TARGETS.get(d, {}).get("asset_name") in keys]
        except Exception:
            return []

    def list_skills(self) -> List[dict]:
        """只返回【本场景当前真正可用】的技能(目标非空的才列)。家电被删时抽屉/门/咖啡技能不会出现。"""
        coffee = ["handle_coffee_lever"] if "coffee_machine" in self._scene_members() else []
        skills = [
            {"skill": "grasp", "targets": self.list_graspable(), "desc": "Z-approach 抓取一个物体"},
            {"skill": "place", "targets": self.list_place_baskets(), "desc": "把手里的物体放进篮子"},
            {"skill": "open_drawer", "targets": self.list_drawers(), "desc": "拉开抽屉"},
            {"skill": "close_drawer", "targets": self.list_drawers(), "desc": "推回抽屉"},
            {"skill": "open_door", "targets": self.list_doors(), "desc": "开门"},
            {"skill": "close_door", "targets": self.list_doors(), "desc": "关门"},
            {"skill": "operate_coffee", "targets": coffee, "desc": "抓咖啡把手并绕关节轴旋转"},
        ]
        return [s for s in skills if s["targets"]]

    # ==================================================================================
    #  B. 执行 —— 技能(封装 SkillTestController;阻塞式,跑完返回结果)
    # ==================================================================================
    def grasp(self, obj: str, *, max_steps: int = 8000) -> dict:
        """抓取:obj 为物体名。优先用已标定抓取位姿;无则对场景刚体用俯视兜底位姿。"""
        from state_machine.skill_test_controller import _GraspPoseRunner
        pose_w = self._resolve_grasp_pose(obj)
        if pose_w is None:
            raise ValueError(f"grasp: 无法解析 {obj!r} 的抓取位姿(既无标定位姿,场景里也没有此刚体)")
        c = self._ctrl
        c.executor.reset(); c._pending = None; c._place_runner = None
        c._grasp_runner = _GraspPoseRunner(c.adapter, pose_w, device=c.device,
                                           speed=c._runner_speed, arrival_z=c._arrival_z())
        r = self._pump(max_steps, label=f"grasp:{obj}")
        # 开环抓取器不产生 executor.last_result -> 成功判定用【物理事实:夹爪合上且物体在手边】
        held = self._is_holding(obj)
        r["ok"] = bool(held and not r["timed_out"])
        r["holding"] = held
        r["gripper_width"] = round(self.get_gripper_width(), 4)
        # ★ 抓完【一律保持闭合】(gripper_cmd=-1):空闲 hold 不松爪,抓到的物体不会掉。
        #   即使 held 判定误报 False 也不松(物体真在手里也不会掉);ok/holding 字段照实报判定结果。
        #   要松开:大脑显式调 place()(放进篮子自动松)或 open_gripper()。符合"抓了就一直握着等指令"。
        self._gripper_cmd = -1.0
        return r

    def place(self, basket: str, *, max_steps: int = 8000) -> dict:
        """放置:把当前夹着的物体放进 basket(∈ list_place_baskets())。★ 必须先 grasp 持有物体。"""
        from state_machine.skill_test_controller import _PlacePoseRunner
        c = self._ctrl
        pos = self._resolve_place_pos(basket)
        if pos is None:
            raise ValueError(f"place: 无法解析篮子 {basket!r} 的放置 pose(stage 里找不到该 prim)")
        c.executor.reset(); c._pending = None; c._grasp_runner = None
        c._place_runner = _PlacePoseRunner(c.adapter, pos, device=c.device, speed=c._runner_speed,
                                           carry_z=c._carry_z(), arrival_z=c._arrival_z())
        r = self._pump(max_steps, label=f"place:{basket}")
        # 成功判定:跑完(未超时)且夹爪已松开(把物体放下了)
        released = self.get_gripper_width() > 0.06
        r["ok"] = bool(released and not r["timed_out"])
        r["released"] = released
        r["basket_xy"] = [round(float(pos[0]), 3), round(float(pos[1]), 3)]
        # place 的语义就是放下 -> 之后保持【张开】(这就是"放置里的开爪触发")。
        self._gripper_cmd = 1.0
        return r

    def open_drawer(self, drawer: str, *, max_steps: int = 15000) -> dict:
        return self._drawer(drawer, "open_drawer", max_steps)

    def close_drawer(self, drawer: str, *, max_steps: int = 15000) -> dict:
        return self._drawer(drawer, "close_drawer", max_steps)

    def open_door(self, door: str = "fridge", *, max_steps: int = 15000) -> dict:
        return self._door(door, "open_door", max_steps)

    def close_door(self, door: str = "fridge", *, max_steps: int = 15000) -> dict:
        return self._door(door, "close_door", max_steps)

    def operate_coffee(self, *, max_steps: int = 15000) -> dict:
        """操作咖啡机把手(抓 handle_coffee_lever 并绕实测关节轴旋转)。需该把手的标定抓取位姿。"""
        self._ctrl._on_operate_coffee()
        return self._pump(max_steps, label="operate_coffee")

    def stop(self) -> None:
        """中止当前技能(空闲时调用:置标志并推进 3 帧让控制器清理)。"""
        self._ctrl._want_stop = True
        self._pump(3, label="stop")

    # ---- 运行控制:停止 / 暂停 / 恢复 / 回 home(测试时可随时干预) ----
    def request_stop(self) -> None:
        """请求中止【正在阻塞运行】的技能(轻量,只置标志,不自己推进)。
        GUI 里技能跑起来时点它:底层 _pump 下一帧会清掉 runner 并退出(env.step 会处理 UI 事件)。
        暂停中调用也能生效(会顺带解除冻结再退出)。"""
        self._ctrl._want_stop = True

    def open_gripper(self, *, steps: int = 40) -> dict:
        """张开夹爪(保持手臂当前位形驱动数帧),并把持久夹爪指令置为开。返回 {ok, gripper_width}。"""
        self._gripper_cmd = 1.0
        return self._actuate_gripper(1.0, steps, "open_gripper")

    def close_gripper(self, *, steps: int = 40) -> dict:
        """闭合夹爪(保持手臂当前位形驱动数帧),并把持久夹爪指令置为合。返回 {ok, gripper_width}。"""
        self._gripper_cmd = -1.0
        return self._actuate_gripper(-1.0, steps, "close_gripper")

    def _actuate_gripper(self, cmd: float, steps: int, label: str) -> dict:
        import torch
        with torch.inference_mode():
            for _ in range(int(steps)):
                state = self.provider.get_state()
                self.env.step(self.provider.make_hold_joint_action(state, cmd))
        gw = self.get_gripper_width()
        ok = (gw < 0.075) if cmd < 0 else (gw > 0.06)   # 合:开度小 / 开:开度大
        return {"ok": bool(ok), "skill": label, "gripper_width": round(gw, 4),
                "reason": "" if ok else "夹爪未到目标开度"}

    def hold_action(self, state):
        """空闲保持动作:保持当前关节 + 【持久夹爪指令】(抓到后一直闭合,直到 open/place)。
        GUI 主循环空闲时用它,替代会把物体松掉的 make_hold_joint_action(state, 开)。"""
        return self.provider.make_hold_joint_action(state, self._gripper_cmd)

    def pause(self) -> None:
        """暂停:冻结当前姿态,技能不再推进(之后可 resume 继续,或 request_stop 放弃)。"""
        self._paused = True

    def resume(self) -> None:
        """恢复:解除暂停,技能继续推进。"""
        self._paused = False

    def is_paused(self) -> bool:
        return bool(self._paused)

    def go_home(self, home_q=None, *, max_steps: int = 1500, tol: float = 0.03,
                gripper: float = 1.0) -> dict:
        """把手臂驱动回 home 关节位形(阻塞;可被 pause 冻结 / request_stop 中止)。
        home_q=None 时读场景目录的 robot_home.json;读不到则报错。返回 {ok, reason, joint_err, ...}。"""
        import torch
        if home_q is None:
            home_q = self._default_home_q()
        if home_q is None:
            return {"ok": False, "skill": "go_home", "reason": "no_home_q(未提供且读不到 robot_home.json)"}
        c = self._ctrl
        c.executor.reset(); c._grasp_runner = None; c._place_runner = None
        c._pending = None; c._want_stop = False
        hq = torch.as_tensor(home_q, dtype=torch.float32, device=c.device).reshape(-1)
        steps = 0; timed_out = True; stopped = False
        with torch.inference_mode():
            i = 0
            while i < int(max_steps):
                state = self.provider.get_state()
                if self._paused and not c._want_stop:      # 暂停冻结,不消耗步数
                    self.env.step(self.provider.make_hold_joint_action(state, None))
                    continue
                if c._want_stop:                            # 中途请求停止
                    c._want_stop = False; stopped = True; break
                self.env.step(self.provider.make_joint_action_from_q_des(hq, gripper))
                steps += 1; i += 1
                if float((self.provider.arm_joint_pos(state) - hq).abs().max()) < tol:
                    timed_out = False; break
        err = float((self.provider.arm_joint_pos(self.provider.get_state()) - hq).abs().max())
        ok = (not timed_out) and (not stopped)
        return {"ok": ok, "skill": "go_home", "steps": steps, "timed_out": timed_out,
                "stopped": stopped, "joint_err": round(err, 4),
                "reason": "" if ok else ("stopped(被中止)" if stopped else "timed_out(未在步数内到 home)")}

    def _default_home_q(self):
        """从场景 registry 同目录读 robot_home.json 的 home_q;读不到返回 None。"""
        import json
        from pathlib import Path
        reg = getattr(self, "_scene_registry_path", None)
        if reg:
            p = Path(reg).parent / "robot_home.json"
            try:
                if p.exists():
                    return json.load(open(p)).get("home_q")
            except Exception:
                pass
        return None

    # ==================================================================================
    #  内部:触发 + 逐帧推进到技能结束
    # ==================================================================================
    def _drawer(self, drawer: str, skill: str, max_steps: int) -> dict:
        from runtime.skill_types import SkillType
        self._require(drawer, self.list_drawers(), skill)
        c = self._ctrl
        c.sel_drawer = drawer   # _make_autorun_request 会读它
        st = SkillType.OPEN_DRAWER if skill == "open_drawer" else SkillType.CLOSE_DRAWER
        return self._run_autorun([(st, drawer)], f"{skill}:{drawer}", max_steps)

    def _door(self, door: str, skill: str, max_steps: int) -> dict:
        from runtime.skill_types import SkillType
        self._require(door, self.list_doors(), skill)
        c = self._ctrl
        c.sel_door = door
        st = SkillType.OPEN_DOOR if skill == "open_door" else SkillType.CLOSE_DOOR
        return self._run_autorun([(st, door)], f"{skill}:{door}", max_steps)

    def _run_autorun(self, items, label: str, max_steps: int) -> dict:
        """用控制器【已验证的 autorun 驱动器】跑技能(executor 直连,不走途径点绕行,自带完成/超时判定)。
        设置控制器的 _auto 序列状态 -> 逐帧 controller.step()(内部路由到 _auto_step)-> 读 _auto_results。"""
        import torch
        from state_machine.skill_test_controller import _AUTORUN_SETTLE_STEPS
        c = self._ctrl
        # 复位 autorun 状态(同 __init__)
        c._auto = list(items)
        c._auto_i = 0
        c._auto_started = False
        c._auto_steps = 0
        c._settle_left = _AUTORUN_SETTLE_STEPS
        c._auto_results = []
        c._auto_done_logged = False
        try:
            with torch.inference_mode():
                for _ in range(int(max_steps)):
                    state = self.provider.get_state()
                    action = c.step(self.session)   # 内部 _auto_step 驱动
                    if action is None:
                        action = self.provider.make_hold_joint_action(state, 1.0)
                    self.env.step(action)
                    if c._auto_i >= len(c._auto):
                        break
        finally:
            done = c._auto_i >= len(c._auto)
            results = list(c._auto_results)
            c._auto = None   # 清掉,恢复正常(UI/其它)路径
        if results:
            rr = results[-1]
            return {"ok": bool(rr.get("success")), "label": label, "status": rr.get("status"),
                    "failure_reason": rr.get("reason"), "angle": rr.get("angle"), "timed_out": not done}
        return {"ok": False, "label": label, "timed_out": not done, "failure_reason": "no result"}

    def _is_holding(self, obj: Optional[str] = None) -> bool:
        """物理判定是否夹着东西:夹爪合上(未完全张开)。给了 obj 且能读到其位姿时,再要求它在 TCP 附近。"""
        gw = self.get_gripper_width()
        # 放宽:宽口杯抓住时 gw 接近满开(0.08),薄壁物 gw 很小;只排除"完全张开(空)"和"完全空合(~0)"
        closed = 0.003 < gw < 0.079
        if not closed:
            return False
        if obj is None:
            return True
        op = self.get_object_pose(obj)
        if op is None:
            return True
        tcp = self.get_tcp_pose()["position"]
        d = sum((op["position"][i] - tcp[i]) ** 2 for i in range(3)) ** 0.5
        # 放宽到 0.16:高杯/侧抓时物体中心离 TCP 可达 ~11.7cm(SM_Mug 抓取位姿 pos[0,0.06,0.1]),原 0.12 卡边界会误判
        return d < 0.16

    def _resolve_grasp_pose(self, obj: str):
        """抓取位姿:①控制器的实时/存盘位姿;②兜底=场景刚体位置 + 俯视朝向。"""
        c = self._ctrl
        pw = c._resolve_named_pose(obj)   # 实时面板 或 grasp_poses.json
        if pw is not None:
            return pw
        try:
            import torch
            from runtime.scene_state_provider import PoseState
            scene = self.provider.scene
            if obj in scene.keys():
                pos = scene[obj].data.root_pos_w[0].reshape(3).clone()
                quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=pos.device)  # 俯视(TCP+Z朝下)
                return PoseState(pos, quat)
        except Exception:
            pass
        return None

    def _busy(self) -> bool:
        c = self._ctrl
        return (c._grasp_runner is not None or c._place_runner is not None
                or getattr(c, "_pre_mover", None) is not None
                or getattr(c, "_transit_runner", None) is not None
                or getattr(c, "_coffee_calib", None) is not None
                or c._pending is not None
                or c.executor.active_skill is not None
                or bool(getattr(c, "_drawer_queue", [])))

    def _pump(self, max_steps: int, label: str = "") -> dict:
        """逐帧:controller.step -> env.step,直到技能结束(不忙)或超时。
        暂停(_paused)时:冻结当前姿态、不推进技能、也不消耗步数预算(除非同时请求了 stop)。"""
        import torch
        steps = 0
        timed_out = True
        with torch.inference_mode():
            i = 0
            while i < int(max_steps):
                state = self.provider.get_state()
                # 暂停中(且没请求停止):保持当前关节,不推进技能,不占步数 -> 一直冻结到 resume/stop
                if self._paused and not self._ctrl._want_stop:
                    self.env.step(self.provider.make_hold_joint_action(state, None))
                    continue
                action = self._ctrl.step(self.session)
                if action is None:
                    if not self._busy():
                        timed_out = False
                        break
                    # 技能中途控制器暂时没给动作:保持位形,夹爪按当前开度维持(None),不强制张开松物体
                    action = self.provider.make_hold_joint_action(state, None)
                self.env.step(action)
                steps += 1
                i += 1
                if not self._busy():
                    timed_out = False
                    break
        res = getattr(self._ctrl.executor, "last_result", None)
        ok = (not timed_out)
        if res is not None and hasattr(res, "success"):
            ok = ok and bool(res.success)
        return {"ok": ok, "label": label, "steps": steps, "timed_out": timed_out,
                "status": getattr(getattr(res, "final_status", None), "value", None),
                "failure_reason": getattr(res, "failure_reason", None)}

    @staticmethod
    def _require(value: str, valid: List[str], skill: str) -> None:
        if value not in valid:
            raise ValueError(f"{skill}: 非法目标 {value!r};可选 {valid}")
