#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
#  BrainaryAPI —— brainary_test 场景的【精简】仿真接口(交付给大脑/协作模块用)
# ======================================================================================
#  这是给 brainary_test 这个【删减后的小场景】专门做的一份轻量 API。相比通用的
#  projects/banghai/sim_interface.py:SimInterface(全技能:抓/放/开关抽屉/门/咖啡),本文件
#  只保留这个小场景真正需要的能力:
#
#     执行(阻塞式,跑完返回):
#        grasp(obj)   —— 抓起一个【指定物体】(仅限桌面这几样:香蕉/橘子/两个杯子/小刀/食品盒/肉罐)
#        place(basket)—— 把手里的物体放进【三个篮子之一】(Prop_KLT_1/2/3)
#
#     感知(只读):
#        get_all_cameras()  —— 一次读【5 个视角】的画面(RGB[+深度]+内参+位姿)
#        get_rgb/get_depth/get_rgbd(camera) —— 单个视角
#        get_robot_state()  —— 读【机器人状态】(关节 / 末端 TCP 位姿 / 夹爪开度 / 各连杆接触力)
#        get_object_pose(name) —— 读某物体世界位姿
#
#  场景来自本目录的 scene_v1_registry.json(= layout 编辑器保存的删减场景),
#  底层复用 SimInterface 的全部实现(已验证),只是把【可选目标限定】到本场景现有的物品。
#
#  运行(必须 cuda:0,Isaac 渲染只支持 GPU0):
#     conda activate env_isaaclab
#     # 无头自检(枚举 + 读 5 相机 + 读机器人状态 + 抓一个放进篮子):
#     ./isaaclab.sh -p projects/franka_v1_skill_lab/scene/saved_scenes/brainary_test/brainary_api.py
#     # 你自己的脚本里:
#     #   from brainary_api import BrainaryAPI
#     #   sim = BrainaryAPI.launch(headless=True, device="cuda:0")
# ======================================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- 路径(brainary 独立项目布局)---
#   本文件在 brainary/sim/;sim_interface.py 同目录;场景数据在 brainary/scene/brainary_test/。
#   sim_interface.py 自己会把 IsaacLab/projects 加进 sys.path(导入仿真框架 franka_v1_skill_lab)。
_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent                              # brainary/sim
_SCENE_DIR = _HERE.parents[1] / "scene" / "brainary_test"   # brainary/scene/brainary_test
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

# 本场景的 registry(删减场景);launch 时指给底层加载/摆放这个布局。
SCENE_REGISTRY = str(_SCENE_DIR / "scene_v1_registry.json")


def _load_home_q():
    """读本目录 robot_home.json 的 home_q(7 关节);读不到返回 None(底层再兜底)。"""
    import json
    p = _SCENE_DIR / "robot_home.json"
    try:
        if p.exists():
            return json.load(open(p)).get("home_q")
    except Exception:
        pass
    return None

# ---- 本场景【可抓物体】白名单(桌面现有的这几样) + 中文标签 ----
#   名字取自 scene_v1_latest.json 的 objects 列表;有标定抓取位姿的最稳,无则用俯视兜底位姿。
GRASPABLE_LABELS: Dict[str, str] = {
    "Prop_011_banana": "香蕉",
    "Prop_orange_01": "橘子",
    "Prop_SM_Mug_C1": "杯子C",
    "Prop_SM_Mug_D1": "杯子D",
    "Prop_037_scissors": "小刀/剪刀",
    "Prop_003_cracker_box": "食品盒(饼干盒)",
    "Prop_010_potted_meat_can": "肉罐(食品罐)",
}
# ---- 本场景【放置篮子】(三选一) + 中文标签 ----
BASKET_LABELS: Dict[str, str] = {
    "Prop_KLT_1": "篮子1",
    "Prop_KLT_2": "篮子2",
    "Prop_KLT_3": "篮子3",
}


class BrainaryAPI:
    """brainary_test 小场景的精简接口:只有 抓取 / 放置 / 读5相机 / 读机器人状态。

    底层是 projects/banghai/sim_interface.py:SimInterface(复用其全部实现),本类只做两件事:
      1) 启动时把场景指向 brainary_test 的 registry(加载删减布局);
      2) 把 grasp/place 的可选目标【限定】到本场景现有的物品/篮子,并给中文标签。
    """

    def __init__(self, sim):
        self._sim = sim                       # 底层 SimInterface 实例
        self._last_grasped: Optional[str] = None   # 记录当前手里夹着的物体(供 place 判"是否真落进篮子")
        # 只保留【本场景里确实存在】的可抓物体/篮子(按物理存在过滤,不虚报)
        self._graspable = [o for o in GRASPABLE_LABELS if sim.get_object_pose(o) is not None]
        self._baskets = [b for b in BASKET_LABELS if sim._resolve_place_pos(b) is not None]

    # ==================================================================================
    #  启动 / 关闭 / 复位
    # ==================================================================================
    @classmethod
    def launch(cls, *, headless: bool = True, device: str = "cuda:0", seed: int = 1,
               settle: int = 15, _app_launcher=None) -> "BrainaryAPI":
        """启动 brainary_test 场景并返回接口。device 必须 cuda:0。headless=True 无窗口。"""
        from sim_interface import SimInterface
        sim = SimInterface.launch(
            headless=headless, device=device, seed=seed, enable_force=True, settle=settle,
            scene_registry_path=SCENE_REGISTRY, _app_launcher=_app_launcher,
        )
        return cls(sim)

    @classmethod
    def from_sim(cls, sim) -> "BrainaryAPI":
        """已有一个 SimInterface(比如在某个 GUI 入口里)时,直接包一层限定接口。"""
        return cls(sim)

    @property
    def sim(self):
        """需要用到全量 SimInterface(如更多感知方法)时的逃生口。"""
        return self._sim

    def reset(self, *, seed: int = 1) -> dict:
        """恢复到初始状态:重置整个场景(机器人回初始位形 + 所有物体回保存位置)。"""
        self._sim.resume()                     # 顺带解除可能的暂停
        self._sim.session.reset(seed=seed)
        self._last_grasped = None
        return {"ok": True, "skill": "reset", "reason": ""}

    def go_home(self, **kw) -> dict:
        """只把手臂驱动回 home 关节位形(不动场景物体)。home_q 取本目录 robot_home.json。
        可被 pause 冻结 / stop 中止。返回 {ok, reason, joint_err, ...}。"""
        return self._sim.go_home(home_q=_load_home_q(), **kw)

    # ---- 运行控制:测试时可随时干预正在跑的技能 ----
    def stop(self) -> None:
        """中止当前正在跑的技能(轻量,置标志;GUI 里技能运行中点它即可打断)。"""
        self._sim.request_stop()

    def pause(self) -> None:
        """暂停:冻结当前姿态(技能不再推进,可 resume 继续 / stop 放弃)。"""
        self._sim.pause()

    def resume(self) -> None:
        """恢复:解除暂停,技能继续。"""
        self._sim.resume()

    def is_paused(self) -> bool:
        return self._sim.is_paused()

    # ---- 夹爪:显式开 / 合(抓取后夹爪会一直保持闭合,直到 open_gripper 或 place 触发开爪) ----
    def open_gripper(self, **kw) -> dict:
        """张开夹爪(松开物体)。返回 {ok, gripper_width, ...}。"""
        self._last_grasped = None
        return self._sim.open_gripper(**kw)

    def close_gripper(self, **kw) -> dict:
        """闭合夹爪。返回 {ok, gripper_width, ...}。"""
        return self._sim.close_gripper(**kw)

    def close(self) -> None:
        self._sim.close()

    # ==================================================================================
    #  能力发现(本场景限定)
    # ==================================================================================
    def list_graspable(self) -> List[str]:
        """可抓物体(仅本场景现有的这几样)。"""
        return list(self._graspable)

    def list_baskets(self) -> List[str]:
        """放置篮子(三选一)。"""
        return list(self._baskets)

    def describe_targets(self) -> Dict[str, Dict[str, str]]:
        """带中文标签的可选目标,方便大脑/人看:{'graspable':{名:标签}, 'baskets':{名:标签}}。"""
        return {
            "graspable": {o: GRASPABLE_LABELS.get(o, o) for o in self._graspable},
            "baskets": {b: BASKET_LABELS.get(b, b) for b in self._baskets},
        }

    def list_skills(self) -> List[dict]:
        return [
            {"skill": "grasp", "targets": self.list_graspable(), "desc": "抓起一个指定物体"},
            {"skill": "place", "targets": self.list_baskets(), "desc": "把手里的物体放进指定篮子"},
        ]

    # ==================================================================================
    #  执行:抓取 / 放置(阻塞式,跑完返回结果 dict)
    # ==================================================================================
    #  返回值约定(grasp / place 都是这套):
    #    ok(bool)          —— 该技能是否【真的成功】(物理判定,不是"跑完了"就算成功)
    #    reason(str)       —— 失败原因(ok=True 时为 "");大脑可据此决策/重试
    #    skill/target      —— 回显本次调的技能与目标; label_cn = 目标中文标签
    #    error(仅异常时)   —— 底层抛异常时不外抛,而是 ok=False + error=异常文本(绝不"单向"静默)
    #    其余 steps/timed_out/gripper_width 等 = 底层过程量,便于诊断。
    def grasp(self, obj: str, **kw) -> dict:
        """抓起 obj(必须 ∈ list_graspable())。★有失败检测:ok=夹爪合上且物体在手边(物理事实)。
        返回 {ok, reason, holding, gripper_width, steps, timed_out, skill, target, label_cn}。"""
        if obj not in self._graspable:
            raise ValueError(
                f"grasp: 非法目标 {obj!r};本场景可抓目标只有 {self._graspable}"
                f"(标签: {[GRASPABLE_LABELS.get(o, o) for o in self._graspable]})")
        try:
            r = dict(self._sim.grasp(obj, **kw))
        except Exception as exc:   # 底层异常也给结构化返回,不外抛(避免调用方拿到"单向"崩溃)
            return {"ok": False, "skill": "grasp", "target": obj,
                    "label_cn": GRASPABLE_LABELS.get(obj, obj), "reason": "exception",
                    "error": repr(exc)}
        r["skill"] = "grasp"; r["target"] = obj; r["label_cn"] = GRASPABLE_LABELS.get(obj, obj)
        ok = bool(r.get("ok"))
        if ok:
            r["reason"] = ""
        elif r.get("timed_out"):
            r["reason"] = "timed_out(未在步数内够到/合爪)"
        elif not r.get("holding"):
            r["reason"] = "gripper_empty(合爪后没夹住物体)"
        else:
            r["reason"] = "unknown"
        self._last_grasped = obj if ok else None   # 成功才记为"手里有此物体"
        return r

    def place(self, basket: str, **kw) -> dict:
        """把手里的物体放进 basket(必须 ∈ list_baskets())。★双重失败检测:
        (1) 放置前必须【真的持有物体】(否则=放了个空,ok=False);
        (2) 放完后被抓物体必须【真的落在篮子附近】(否则=没进篮子,ok=False)。
        返回 {ok, reason, was_holding, object, object_in_basket, released, basket_xy, steps, ...}。"""
        if basket not in self._baskets:
            raise ValueError(
                f"place: 非法篮子 {basket!r};本场景只有 {self._baskets}"
                f"(标签: {[BASKET_LABELS.get(b, b) for b in self._baskets]})")
        was_holding = bool(self._sim._is_holding())   # 放置前:手里到底有没有东西
        held_obj = self._last_grasped
        try:
            r = dict(self._sim.place(basket, **kw))
        except Exception as exc:
            return {"ok": False, "skill": "place", "target": basket,
                    "label_cn": BASKET_LABELS.get(basket, basket), "reason": "exception",
                    "error": repr(exc), "was_holding": was_holding, "object": held_obj}
        released = bool(r.get("released"))
        timed_out = bool(r.get("timed_out"))
        # 被抓物体是否真落进这个篮子(有明确手中物体且能读到位姿时才判;读不到则 None=不否决)
        in_basket: Optional[bool] = None
        bxy = r.get("basket_xy")
        if held_obj is not None and bxy is not None:
            op = self._sim.get_object_pose(held_obj)
            if op is not None:
                dx = op["position"][0] - bxy[0]
                dy = op["position"][1] - bxy[1]
                in_basket = (dx * dx + dy * dy) ** 0.5 < 0.25   # KLT 篮口半径量级
        ok = bool(was_holding and released and not timed_out and (in_basket is not False))
        r.update({"skill": "place", "target": basket, "label_cn": BASKET_LABELS.get(basket, basket),
                  "was_holding": was_holding, "object": held_obj, "object_in_basket": in_basket,
                  "ok": ok})
        if ok:
            r["reason"] = ""
            self._last_grasped = None                # 放成功,手里清空
        elif not was_holding:
            r["reason"] = "not_holding(放置前手里没有物体→放了个空)"
        elif timed_out:
            r["reason"] = "timed_out(未在步数内完成)"
        elif not released:
            r["reason"] = "not_released(夹爪始终没松开)"
        elif in_basket is False:
            r["reason"] = "not_in_basket(物体没落进篮子)"
        else:
            r["reason"] = "unknown"
        return r

    def stop(self) -> None:
        """中止当前技能。"""
        self._sim.stop()

    # ==================================================================================
    #  感知:5 个视角画面
    # ==================================================================================
    def cameras(self) -> List[str]:
        """5 个相机的逻辑名: front / wrist / left / right / top。"""
        from sim_interface import CAMERAS
        return list(CAMERAS)

    def get_all_cameras(self, require_depth: bool = True) -> Dict[str, dict]:
        """一次读全部 5 个视角:{cam: {rgb(HxWx3 uint8), depth(HxW 米), intrinsics, pose}}。
        某相机未挂载则跳过(不会报错)。"""
        return self._sim.get_all_cameras(require_depth=require_depth)

    def get_rgb(self, camera: str):
        """单个视角 RGB(HxWx3 uint8)。camera ∈ cameras()。"""
        return self._sim.get_rgb(camera)

    def get_depth(self, camera: str):
        """单个视角深度(HxW float32,米)。"""
        return self._sim.get_depth(camera)

    def get_rgbd(self, camera: str):
        """单个视角 (rgb, depth)。"""
        return self._sim.get_rgbd(camera)

    def get_camera_intrinsics(self, camera: str) -> dict:
        return self._sim.get_camera_intrinsics(camera)

    def get_camera_pose(self, camera: str) -> dict:
        return self._sim.get_camera_pose(camera)

    # ==================================================================================
    #  感知:机器人状态
    # ==================================================================================
    def get_robot_state(self) -> dict:
        """机器人本体一次全拿:关节 + 末端TCP位姿 + 夹爪开度 + 各连杆接触力。
        结构: {joints:{arm[7],all_pos[9],vel[9]}, tcp_pose:{position,quat_wxyz},
               gripper_width, contact_forces:[(连杆名,力N)...]}。
        ⚠️ 仿真只有接触力,无关节力矩/力控。"""
        return self._sim.get_robot_state()

    def get_joint_state(self) -> dict:
        return self._sim.get_joint_state()

    def get_tcp_pose(self) -> dict:
        return self._sim.get_tcp_pose()

    def get_gripper_width(self) -> float:
        return self._sim.get_gripper_width()

    def get_object_pose(self, name: str) -> Optional[dict]:
        """某物体世界位姿 {position[3], quat_wxyz[4]},读不到返回 None。"""
        return self._sim.get_object_pose(name)


# ======================================================================================
#  无头自检:枚举 + 读 5 相机 + 读机器人状态 +(可选)抓一个放进篮子
#  跑: ./isaaclab.sh -p projects/franka_v1_skill_lab/scene/saved_scenes/brainary_test/brainary_api.py
# ======================================================================================
def _selftest() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--do_grasp", action="store_true", help="额外跑一次 grasp+place 冒烟(较慢)")
    args = ap.parse_args()
    args.headless = True
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        sim = BrainaryAPI.launch(headless=True, device=args.device, _app_launcher=app_launcher)

        print("\n== 目标(中文标签) ==", flush=True)
        d = sim.describe_targets()
        print("可抓物体:", d["graspable"], flush=True)
        print("篮子    :", d["baskets"], flush=True)

        print("\n== 5 相机画面 ==", flush=True)
        cams = sim.get_all_cameras(require_depth=True)
        for name in sim.cameras():
            f = cams.get(name)
            if not f:
                print(f"  {name:6s}: (未挂载/不可用)", flush=True)
                continue
            rgb, dep = f.get("rgb"), f.get("depth")
            rs = None if rgb is None else tuple(getattr(rgb, "shape", ()))
            ds = None if dep is None else tuple(getattr(dep, "shape", ()))
            print(f"  {name:6s}: rgb={rs} depth={ds}", flush=True)

        print("\n== 机器人状态 ==", flush=True)
        st = sim.get_robot_state()
        print("  关节 arm[7] :", [round(v, 3) for v in st["joints"]["arm"]], flush=True)
        print("  TCP 位置    :", [round(v, 3) for v in st["tcp_pose"]["position"]], flush=True)
        print("  夹爪开度(m) :", round(st["gripper_width"], 4), flush=True)
        print("  接触力(>阈) :", st["contact_forces"], flush=True)

        if args.do_grasp and sim.list_graspable() and sim.list_baskets():
            obj = sim.list_graspable()[0]
            bkt = sim.list_baskets()[0]
            print(f"\n== 冒烟: grasp({obj}) -> place({bkt}) ==", flush=True)
            rg = sim.grasp(obj)
            print("  grasp:", rg, flush=True)
            if rg.get("ok"):
                rp = sim.place(bkt)
                print("  place:", rp, flush=True)

        print("\n[brainary_api] 自检完成 ✅", flush=True)
        sim.close()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(_selftest())
