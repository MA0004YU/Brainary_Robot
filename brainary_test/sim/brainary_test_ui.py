#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""基于 BrainaryAPI 的【抓取 / 放置】技能测试 GUI(brainary_test 场景)。

一个场景挂 3 个面板,方便你【可视化】排查抓/放到底哪儿出问题(够不到 / 碰撞 / 位姿偏):
  1) Brainary Grasp/Place  —— 选物体→点 Grasp / 选篮子→点 Place(阻塞跑完,视口能看到机器人动),
                              执行完把返回 dict(ok / reason / holding / steps / gripper …)显示出来;
                              还能一键读 TCP 与所选物体的世界位姿,自己对比距离/朝向。
  2) Franka Control        —— 手动点动/IK/关节/夹爪:抓失败后可手动把爪挪到物体上,看是够不到还是撞了。
  3) Scene                 —— 场景复位(Reset),重试。

技能全部走 BrainaryAPI(=底层 SimInterface,已验证),返回值/失败检测都在 dict 里,不"单向"。

运行(GUI,不要 --headless):
    conda activate env_isaaclab
    ./isaaclab.sh -p brainary/sim/brainary_test_ui.py --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 本目录:import brainary_api


class BrainaryGraspPanel:
    """调 BrainaryAPI 的抓/放面板。按钮置 pending,主循环 run_pending() 阻塞执行并显示结果。"""

    def __init__(self, sim):
        import omni.ui as ui
        self.sim = sim
        self._pending = None
        self._graspable = sim.list_graspable()
        self._baskets = sim.list_baskets()
        labels = sim.describe_targets()
        # 下拉显示 "name (中文标签)"
        self._g_show = [f"{o}  ({labels['graspable'].get(o, o)})" for o in self._graspable]
        self._b_show = [f"{b}  ({labels['baskets'].get(b, b)})" for b in self._baskets]
        self._models = {}
        self.window = ui.Window("Brainary Grasp/Place", width=460, height=460)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("== BrainaryAPI 抓取 / 放置测试 ==")
                ui.Label(f"抓取目标 ({len(self._graspable)})")
                self._models["grasp"] = ui.ComboBox(0, *self._g_show).model
                ui.Button("Grasp(抓取)", clicked_fn=lambda: self._set("grasp"))
                ui.Separator()
                ui.Label(f"放置篮子 ({len(self._baskets)})")
                self._models["basket"] = ui.ComboBox(0, *self._b_show).model
                ui.Button("Place(放进篮子)", clicked_fn=lambda: self._set("place"))
                ui.Separator()
                ui.Button("Grasp → Place(连贯:先抓所选物,成功再放所选篮)",
                          clicked_fn=lambda: self._set("grasp_then_place"))
                ui.Separator()
                # ---- 运行控制:技能【运行中】也能点(Pause/Resume/Stop 立即生效) ----
                ui.Label("运行控制(技能跑起来时也能点)")
                with ui.HStack(spacing=6, height=26):
                    ui.Button("⏸ Pause 暂停", clicked_fn=self._pause)
                    ui.Button("▶ Resume 恢复", clicked_fn=self._resume)
                    ui.Button("■ Stop 中止", clicked_fn=self._stop)
                with ui.HStack(spacing=6, height=26):
                    ui.Button("↩ Go Home 回home", clicked_fn=lambda: self._set("go_home"))
                    ui.Button("⟲ Reset 复位场景", clicked_fn=lambda: self._set("reset"))
                with ui.HStack(spacing=6, height=26):
                    ui.Button("✊ Close Gripper 合爪", clicked_fn=lambda: self._set("close_gripper"))
                    ui.Button("✋ Open Gripper 开爪", clicked_fn=lambda: self._set("open_gripper"))
                ui.Separator()
                ui.Button("读 TCP / 所选物体位姿(诊断距离/朝向)", clicked_fn=self._read_poses)
                ui.Separator()
                self._status = ui.Label("status: idle")
                self._result = ui.Label("(还没有结果)", word_wrap=True)

    # ---- 运行控制回调(直接执行,不走 pending:技能阻塞运行中也要能立即打断/冻结) ----
    def _pause(self):
        self.sim.pause()
        self._status.text = "status: PAUSED(已暂停,点 Resume 继续 / Stop 放弃)"

    def _resume(self):
        self.sim.resume()
        self._status.text = "status: resumed(已恢复)"

    def _stop(self):
        self.sim.stop()
        self._status.text = "status: STOP requested(中止请求已发,技能会尽快停下)"

    def _sel(self, key, names):
        m = self._models.get(key)
        if m is None or not names:
            return None
        i = int(m.get_item_value_model().get_value_as_int())
        return names[i] if 0 <= i < len(names) else names[0]

    def _set(self, kind):
        if self._pending is not None:
            return
        self._pending = kind
        self._status.text = f"status: pending {kind}"

    def _read_poses(self):
        try:
            obj = self._sel("grasp", self._graspable)
            tcp = self.sim.get_tcp_pose()["position"]
            op = self.sim.get_object_pose(obj) if obj else None
            if op is None:
                self._result.text = f"TCP={_r3(tcp)}  |  {obj}: 读不到位姿"
                return
            p = op["position"]
            d = sum((p[i] - tcp[i]) ** 2 for i in range(3)) ** 0.5
            dxy = ((p[0] - tcp[0]) ** 2 + (p[1] - tcp[1]) ** 2) ** 0.5
            self._result.text = (f"{obj}:\n  物体 pos={_r3(p)} quat={_r3(op['quat_wxyz'])}\n"
                                 f"  TCP  pos={_r3(tcp)}\n  距离 3D={d:.3f}m 水平={dxy:.3f}m dz={p[2]-tcp[2]:+.3f}m")
            self._status.text = "status: 读位姿 ok"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._result.text = f"读位姿 ERROR: {exc}"

    def has_pending(self) -> bool:
        return self._pending is not None

    def run_pending(self) -> None:
        if self._pending is None:
            return
        kind = self._pending
        self._pending = None
        try:
            if kind == "grasp":
                obj = self._sel("grasp", self._graspable)
                self._status.text = f"status: running grasp({obj}) ..."
                r = self.sim.grasp(obj)
                self._show("grasp", obj, r)
            elif kind == "place":
                bkt = self._sel("basket", self._baskets)
                self._status.text = f"status: running place({bkt}) ..."
                r = self.sim.place(bkt)
                self._show("place", bkt, r)
            elif kind == "grasp_then_place":
                obj = self._sel("grasp", self._graspable)
                bkt = self._sel("basket", self._baskets)
                self._status.text = f"status: running grasp({obj}) ..."
                rg = self.sim.grasp(obj)
                if not rg.get("ok"):
                    self._show("grasp", obj, rg, note="→ 抓取失败,不放置")
                    return
                self._status.text = f"status: running place({bkt}) ..."
                rp = self.sim.place(bkt)
                self._show("place", bkt, rp, note=f"(先抓 {obj} 成功)")
            elif kind == "go_home":
                self._status.text = "status: running go_home() ..."
                r = self.sim.go_home()
                self._show("go_home", "home", r)
            elif kind == "reset":
                self._status.text = "status: running reset() ..."
                r = self.sim.reset()
                self._show("reset", "scene", r)
            elif kind == "open_gripper":
                self._status.text = "status: opening gripper ..."
                r = self.sim.open_gripper()
                self._show("open_gripper", "gripper", r)
            elif kind == "close_gripper":
                self._status.text = "status: closing gripper ..."
                r = self.sim.close_gripper()
                self._show("close_gripper", "gripper", r)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._status.text = f"status: {kind} ERROR"
            self._result.text = f"ERROR: {exc}"

    def _show(self, skill, target, r, note=""):
        ok = r.get("ok")
        self._status.text = f"status: {skill}({target}) -> ok={ok}"
        lines = [f"{skill}({target})  ok={ok}  {note}",
                 f"  原因 reason: {r.get('reason') or '(成功)'}"]
        for k in ("holding", "gripper_width", "was_holding", "object", "object_in_basket",
                  "released", "basket_xy", "joint_err", "stopped", "steps", "timed_out", "error"):
            if k in r and r[k] is not None:
                lines.append(f"  {k}: {r[k]}")
        self._result.text = "\n".join(lines)
        print(f"[brainary_ui] {skill}({target}) -> {r}", flush=True)


def _r3(v):
    return [round(float(x), 3) for x in v]


def main() -> int:
    ap = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True          # 5 相机渲染
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return _run(args, app_launcher, simulation_app)
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        try:
            with open("/tmp/brainary_ui_err.txt", "w") as f:
                f.write(tb)
                f.flush()
        except Exception:
            pass
        print("[brainary_ui] FATAL:\n" + tb, flush=True)
        raise
    finally:
        simulation_app.close()


def _run(args, app_launcher, simulation_app) -> int:
    import torch
    from brainary_api import BrainaryAPI

    print("[brainary_ui] 正在建场景(从 brainary_test registry 加载 + spawn 道具),约需 60~120s,"
          "请耐心等到 'ready' 和 3 个窗口,不要提前 Ctrl-C ...", flush=True)
    sim = BrainaryAPI.launch(headless=False, device=args.device, _app_launcher=app_launcher)
    print("[brainary_ui] 场景就绪,正在挂面板 ...", flush=True)

    si = sim.sim                                    # 底层 SimInterface
    ctrl = si._ctrl
    session, env, provider = si.session, si.env, si.provider

    # 相机偏移贴回(所见即所得)
    try:
        from franka_v1_skill_lab.scene_interface import camera_offsets
        camera_offsets.apply_saved_offsets_runtime(env)
    except Exception as exc:
        print(f"[brainary_ui] camera offsets: {exc}", flush=True)

    # 面板 import 必须在 launch() 之后(legacy 路径已由 SceneSession.launch 挂好)
    from franka_v1_skill_lab.scene_interface.franka_control_panel import FrankaControlPanel
    from franka_v1_skill_lab.scene_interface.test_mode_ui import _ScenePanel

    grasp_panel = BrainaryGraspPanel(sim)           # 1) 抓/放测试(BrainaryAPI)
    franka = FrankaControlPanel(env, provider, ctrl.adapter)   # 2) Franka 手动控制
    scene_panel = _ScenePanel()                     # 3) 场景复位
    # 4) 夹取pose编辑面板:选物体/篮子 -> ±X/Y/Z + ±Roll/Pitch/Yaw 调 -> "Move robot to this pose"
    #    验证 -> "Save grasp poses" 存进 grasp_poses.json(技能下次抓取直接用这个新pose)。
    from franka_v1_skill_lab.scene_interface.grasp_pose_panel import GraspPosePanel
    pose_panel = GraspPosePanel(env, provider, headless=False, franka_panel=franka)
    # 5) 资产位姿面板:动态场景里实时移动资产(选物体 -> ±X/±Y/±Z/±Yaw 点动,物理运行中钉在目标位)。
    from franka_v1_skill_lab.scene_interface.asset_pose_panel import AssetPosePanel
    asset_panel = AssetPosePanel(env, provider)
    # 6) 可视化面板:勾 "Show colliders" 就叠加显示碰撞体(绿/紫 overlay),排查碰撞体大小/穿模。
    from franka_v1_skill_lab.scene_interface.test_mode_ui import _VizPanel
    viz_panel = _VizPanel(session, show_colliders=True)

    print("[brainary_ui] ready: Brainary Grasp/Place + Franka Control + Scene + Grasp Pose + Asset Pose + Viz", flush=True)

    while simulation_app.is_running():
        with torch.inference_mode():
            # 抓/放面板有 pending -> 阻塞跑完这个技能(内部自己 env.step,视口会动)
            if grasp_panel.has_pending():
                grasp_panel.run_pending()
                continue

            state = provider.get_state()
            if scene_panel.pop_reset():
                session.reset(seed=args.seed if hasattr(args, "seed") else 1)
                scene_panel.set_status("reset done")
            if scene_panel.pop_save():
                # 把当前场景(含用 Asset Pose 面板移动过的资产)存回【本场景 brainary_test】目录:
                # 写它自己的 registry,下次进来 apply_latest_scene 就用新位置(所见即所得)。
                try:
                    from pathlib import Path as _Path
                    from franka_v1_skill_lab.scene_interface import scene_saver, camera_offsets
                    import omni.usd as _ou
                    _reg = getattr(si, "_scene_registry_path", None)
                    _adir = str(_Path(_reg).resolve().parent) if _reg else None
                    _stage = _ou.get_context().get_stage()
                    _sensors = None
                    try:
                        _sensors = camera_offsets.offsets_to_sensors(
                            camera_offsets.current_offsets_from_stage(_stage))
                    except Exception:
                        pass
                    _info = scene_saver.save_current_stage_as_latest(
                        env=env, active_dir=_adir, registry_path=_reg,
                        note="Saved by brainary_test_ui.", sensors=_sensors)
                    scene_panel.set_status(f"saved -> {_Path(_info['usd']).name} ({len(_info['objects'])} objs)")
                    print(f"[brainary_ui] 场景已保存到 {_adir}", flush=True)
                except Exception as _exc:
                    import traceback
                    traceback.print_exc()
                    scene_panel.set_status(f"save FAILED: {_exc}")

            intent = franka.compute(state)
            if intent is not None:
                si._gripper_cmd = float(intent.gripper)   # 手动点动的夹爪指令也更新持久状态
                actions = provider.make_joint_action_from_q_des(intent.q_des, intent.gripper)
            else:
                # 空闲保持:用【持久夹爪指令】(抓到后一直闭合,不会把物体松掉),不再用会张开的 current_gripper()
                actions = si.hold_action(state)
            env.step(actions)
            franka.update_status()
            pose_panel.apply()          # 画抓取pose坐标轴marker(选中的更大)
            pose_panel.update_status()
            asset_panel.apply()         # 把被点动移动的资产每帧钉在目标位(物理运行中也能移动)
            asset_panel.update_status()

    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
