#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""One-click brain-loop TEST GUI with live per-module panels.

Load it, pick options, click "Run Brain Test": the whole pipeline runs
(Sim -> Perception -> Memory -> SSP -> Planning -> Execution -> Monitor).
The plan is then AUTO-EXECUTED on the robot action-by-action (grasp/place).

UI:
  * Top overall bar: current module + current action (i/N) + running summary.
  * One scrolling log area PER module (labeled), newest line at the bottom,
    scroll up to see history.
  * Speed control (fixes objects being flung on the carry-to-basket move).
  * A manual Grasp/Place panel for single-object diagnosis.

Run (GUI, NOT --headless; needs API_zhongzhuan for GPT/planning/monitor, else falls back):
    conda activate env_isaaclab
    ./isaaclab.sh -p brainary/sim/brainary_brain_ui.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

_SIM = Path(__file__).resolve().parent
_BRAINARY = _SIM.parent
sys.path.insert(0, str(_SIM))
sys.path.insert(0, str(_BRAINARY))

# Slow the skills down so carrying an object to the basket doesn't fling/drop it.
# (BrainaryAPI reads SKILL_TEST_SPEED at launch; 5.0 flung objects, 1.5 still dropped scissors -> 1.0.)
os.environ.setdefault("SKILL_TEST_SPEED", "1.0")
# 抓/放执行器的速度是【另一个】变量:skill_test_controller 读 SKILL_TEST_RUNNER_SPEED,默认 3.0
# (会把 max_joint_step 顶到 0.5rad,偏快)。以前这里只设了 SKILL_TEST_SPEED,手动面板就一直跑 3.0
# 而不是 UI 上选的速度。这里一起设成 1.0,和 Speed 下拉的默认档(Normal 1.0)对齐。
os.environ.setdefault("SKILL_TEST_RUNNER_SPEED", "1.0")

# The real project modules (each maps to a real owner). SSP belongs to Monitor;
# execution belongs to Project Management. Do NOT invent extra modules.
_MODULES = ["IsaacSim", "Perception", "Memory", "Planning", "Monitor", "Simulation", "Project Management"]
# 终端里用中文模块名(omni.ui 面板渲染不了中文,所以只在终端中文化;面板仍是英文)
_MODULE_CN = {
    "IsaacSim": "仿真", "Perception": "感知", "Memory": "记忆", "Planning": "规划",
    "Monitor": "监控", "Simulation": "物理沙盒", "Project Management": "执行",
}
# 阶段标题的中文说明(终端横幅用)
_STEP_CN = {
    "capture scene": "抓取场景快照(5相机 + 物体/机器人状态)",
    "perceive": "视觉感知(识别物体与关系)",
    "build memory": "写入记忆并导出规划输入",
    "SSP safety constraints": "SSP 安全约束生成",
    "plan": "任务规划(生成动作序列)",
    "go home": "机械臂回初始位姿",
    "SafetyCritic judge": "安全评审(逐动作裁判)",
}


# 终端里不出现 gpt/chatgpt 字样:后端名统一换成中文说法。
_BACKEND_CN = {
    "gpt": "视觉大模型", "chatgpt": "视觉大模型", "auto": "自动", "mock": "仿真真值",
    "sim-gt-mock": "仿真真值", "sim-gt": "仿真真值",
}


# 终端里不暴露具体语言模型名(gpt-5.5 / claude-* / qwen-* …),一律显示为「大模型」。
# 各模块自己写的 JSON(plan.json / safety_critic_review.json 等)里可能带模型名,所以在
# 【打印那一刻】统一过一遍;文件内容本身不动(排查时仍可打开原文件看到真实模型)。
_MODEL_PAT = re.compile(
    r"(?i)\b(?:chat)?(?:gpt|claude|gemini|qwen|llama|mistral|deepseek)[\w.]*(?:-[\w.]+)*")


def _mask_model(text: str) -> str:
    return _MODEL_PAT.sub("大模型", str(text))


def _cn_backend(s) -> str:
    """把后端标识换成中文;含 gpt/chatgpt 的一律显示为「视觉大模型」(可带地址后缀)。"""
    raw = str(s or "?").strip()
    low = raw.lower()
    for k, v in _BACKEND_CN.items():
        if low == k:
            return v
    if "gpt" in low:                      # 形如 "gpt (chatgpt/scene_describer:5599)"
        tail = ""
        if ":" in raw:                    # 保留端口信息,便于排查,但不带 gpt 字样
            tail = "(" + raw.split(":")[-1].rstrip(")") + " 服务)"
        return "视觉大模型" + tail
    return raw


def _dump_json(module: str, path, title: str, max_lines: int = 60) -> None:
    """把某个模块产出的 JSON 原样打到终端(中文不转义)。太长就截断并提示完整文件路径。

    用途:监控的安全约束、规划的 plan、记忆的 planning_input 这些结构化产物,逐字段翻译成中文
    成本高且容易过时,直接把模块自己的输出摊开最实在。
    """
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        return
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception as exc:
        _term(module, f"{title}: 读取失败 {exc}")
        return
    lines = _mask_model(text).splitlines()
    _term(module, f"── {title}  ({p.name}, {len(lines)} 行) ──")
    for ln in lines[:max_lines]:
        print(f"      {ln}", flush=True)
    if len(lines) > max_lines:
        print(f"      ... 省略 {len(lines) - max_lines} 行,完整内容见 {p}", flush=True)


def _term(module: str, msg: str) -> None:
    """把一条模块日志打到终端(中文模块名前缀)。面板照旧,不受影响。"""
    print(_mask_model(f"  [{_MODULE_CN.get(module, module)}] {msg}").rstrip(), flush=True)


def _banner(text: str, ch: str = "=") -> None:
    print("\n" + ch * 78 + f"\n{_mask_model(text)}\n" + ch * 78, flush=True)


_SPEED_OPTS = [("Very slow (0.5)", 0.5), ("Slow (0.75)", 0.75), ("Normal (1.0)", 1.0),
               ("Fast (2.0)", 2.0), ("Fastest (5.0)", 5.0)]
# 规划方案存档:正常跑一次后落盘,"Replay saved plan" 直接复用、跳过感知/记忆/SSP/规划(不再推理)。
_SAVED_PLAN = _BRAINARY / "output" / "saved_plan.json"


def _apply_speed_to(sim, speed: float) -> float:
    """把速度档同时落到【技能全局 scale】和【抓/放执行器 + IK 每步限幅】上。

    ★ 必须在每次执行技能前调用。以前只有 BrainTestPanel 走这一步,手动 Grasp/Place 面板直接
    调 sim.grasp() -> 一直用 SKILL_TEST_RUNNER_SPEED 的默认值(3.0),UI 上选的速度对它无效。
    """
    os.environ["SKILL_TEST_SPEED"] = str(speed)
    os.environ["SKILL_TEST_RUNNER_SPEED"] = str(speed)
    try:
        from base_skill import set_speed_scale
        set_speed_scale(speed)
    except Exception as exc:
        print(f"[brain_ui] set_speed_scale: {exc}", flush=True)
    try:
        ctrl = getattr(getattr(sim, "_sim", None), "_ctrl", None)
        if ctrl is not None:
            ctrl._runner_speed = max(0.5, float(speed))
            ctrl._apply_speed_to_adapter()
    except Exception as exc:
        print(f"[brain_ui] apply runner speed: {exc}", flush=True)
    return speed


class _ModuleWindow:
    """A STANDALONE draggable window showing one module's scrolling log.
    Newest line at the bottom; scroll up for history. Drag/resize each window freely."""

    def __init__(self, ui, name: str, dock_target: str = "Console", w: int = 420, h: int = 260):
        self.name = name
        self.window = ui.Window(f"[ {name} ]", width=w, height=h)
        # Dock this window into the same panel group as Content/Console -> becomes a switchable TAB.
        try:
            self.window.deferred_dock_in(dock_target)
            self.window.dock_tab_bar_visible = True
        except Exception:
            pass
        with self.window.frame:
            self.frame = ui.ScrollingFrame()
            with self.frame:
                self.label = ui.Label("", word_wrap=True)
        self.lines: list[str] = []

    def log(self, msg: str, echo: bool = True):
        if echo:                      # 同一条内容镜像到终端,带中文模块名前缀
            _term(self.name, msg)
        self.lines.append(msg)
        if len(self.lines) > 300:
            self.lines = self.lines[-300:]
        self.label.text = "\n".join(self.lines)
        try:
            self.frame.scroll_y = 1.0e7   # clamp to bottom -> show newest
        except Exception:
            pass


class _Step:
    def __init__(self, module: str, title: str, fn):
        self.module = module
        self.title = title
        self.fn = fn


class BrainTestPanel:
    """Run the whole pipeline + auto-execute the plan, with live per-module panels."""

    def __init__(self, sim, simulation_app):
        import omni.ui as ui
        import run_brainary as rb
        self.ui = ui
        self.sim = sim
        self.app = simulation_app
        self.rb = rb
        self._q: deque = deque()
        self._active = None
        self._ctx = {}
        self._logs: dict[str, _ModuleWindow] = {}
        self._act_i = 0
        self._act_n = 0
        self._holding = None
        self._graspable = list(sim.list_graspable())
        self._baskets = list(sim.list_baskets())
        try:
            self._aliases = json.loads(
                (_BRAINARY / "project_management" / "object_aliases.json").read_text(encoding="utf-8"))
        except Exception:
            self._aliases = {}

        # ---- Control + OVERALL window (its own draggable window) ----
        self.window = ui.Window("Brainary Brain Test - Control", width=440, height=270)
        try:
            self.window.position_x = 20.0; self.window.position_y = 20.0
        except Exception:
            pass
        with self.window.frame:
            with ui.VStack(spacing=5, height=0):
                ui.Label("== Brain loop control (Sim->Perception->Memory->SSP->Planning->Exec->Monitor) ==")
                with ui.HStack(spacing=8, height=24):
                    ui.Label("Perception:", width=76)
                    self._perc = ui.ComboBox(0, "auto", "mock", "gpt").model
                    ui.Label("Speed:", width=44)
                    self._spd = ui.ComboBox(2, *[o[0] for o in _SPEED_OPTS]).model
                    self._exec = ui.CheckBox(width=18).model
                    self._exec.set_value(True)
                    ui.Label("Execute on robot", width=120)
                ui.Button(">> Run Brain Test", height=32, clicked_fn=self._start)
                ui.Button("Replay saved plan (skip inference)", height=28, clicked_fn=self._start_replay)
                ui.Separator()
                ui.Label("OVERALL", height=16)
                self._ov_stage = ui.Label("  stage: -")
                self._ov_action = ui.Label("  action: -")
                self._ov_summary = ui.Label("  summary: (not run)", word_wrap=True)
        # ---- one window PER module, all docked as switchable TABS next to Content/Console ----
        for m in _MODULES:
            self._logs[m] = _ModuleWindow(ui, m, dock_target="Console")

    def selected_speed(self) -> float:
        """当前 Speed 下拉选的档位(手动面板也读它,保证【一个速度控件管全部】)。"""
        return _SPEED_OPTS[int(self._spd.get_item_value_model().get_value_as_int())][1]

    def _apply_speed(self) -> float:
        return _apply_speed_to(self.sim, self.selected_speed())

    # -------------------------------------------------- main-loop hooks
    def has_pending(self) -> bool:
        return bool(self._q) or self._active is not None

    def run_pending(self) -> None:
        if self._active is not None:
            step = self._active
            self._active = None
            t0 = time.time()
            try:
                res = step.fn()
                if res:
                    self._logs[step.module].log(f"  done: {res}", echo=False)
                    _term(step.module, f"✅ 完成 ({time.time()-t0:.1f}s): {res}")
                else:
                    _term(step.module, f"✅ 完成 ({time.time()-t0:.1f}s)")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._logs[step.module].log(f"  FAILED: {str(exc)[:120]}", echo=False)
                _term(step.module, f"❌ 失败 ({time.time()-t0:.1f}s): {str(exc)[:200]}")
            if not self._q:
                self._finish()
            return
        if self._q:
            step = self._q.popleft()
            self._ov_stage.text = f"  stage: {step.module} - {step.title}"
            self._step_i = getattr(self, "_step_i", 0) + 1
            cn = _STEP_CN.get(step.title, step.title)
            mod_cn = _MODULE_CN.get(step.module, step.module)
            _banner(f"【{self._step_i}/{self._step_i + len(self._q)}】{mod_cn} · {cn}", "-")
            self._logs[step.module].log(f"> {step.title} ...", echo=False)
            self._active = step
            try:
                self.app.update(); self.app.update()
            except Exception:
                pass

    # -------------------------------------------------- Run button
    def _start(self):
        if self.has_pending():
            return
        # apply speed live (also fixes flinging)
        self._apply_speed()
        self._mode = ["auto", "mock", "gpt"][int(self._perc.get_item_value_model().get_value_as_int())]
        self._do_exec = bool(self._exec.get_value_as_bool())
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = _BRAINARY / "output" / f"ui_{ts}"
        self._dirs = {k: run_dir / k for k in ("sim", "perception", "memory", "planning",
                                               "project_management", "monitor")}
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self._ctx = {}
        self._act_i = 0; self._act_n = 0
        self._step_i = 0
        self._t_run0 = time.time()
        self._holding = None
        for m in _MODULES:
            self._logs[m].lines = []; self._logs[m].label.text = ""
        self._ov_action.text = "  action: -"
        self._ov_summary.text = "  summary: running..."
        _banner(f"大脑闭环启动  感知模式={_cn_backend(self._mode)}  执行={'开' if self._do_exec else '关(只规划)'}  "
                f"速度={_SPEED_OPTS[int(self._spd.get_item_value_model().get_value_as_int())][0]}\n"
                f"输出目录: {run_dir}")
        self._logs["Simulation"].log("physics-sandbox validator: runs after planning "
                                     "(auto-skips if service :5600 is down)", echo=False)
        _term("Simulation", "物理沙盒校验已接入:规划之后自动预演;服务(:5600)没起就跳过")
        self._q = deque([
            _Step("IsaacSim", "capture scene", self._st_sim),
            _Step("Perception", "perceive", self._st_perc),
            _Step("Memory", "build memory", self._st_mem),
            _Step("Monitor", "SSP safety constraints", self._st_ssp),   # SSP is part of Monitor
            _Step("Planning", "plan", self._st_plan),                   # appends PM exec + Monitor(critic)
        ])

    # -------------------------------------------------- Replay saved plan (skip inference)
    def _start_replay(self):
        """复用上次存档的规划方案(_SAVED_PLAN):跳过 感知/记忆/SSP/规划,先回 home 再逐动作直接执行,
        方便反复测试技能时不必每次等 GPT 重新推理。"""
        if self.has_pending():
            return
        if not _SAVED_PLAN.is_file():
            self._ov_summary.text = "  summary: no saved plan yet - run 'Run Brain Test' once first"
            return
        try:
            pa = json.loads(_SAVED_PLAN.read_text(encoding="utf-8"))
        except Exception as exc:
            self._ov_summary.text = f"  summary: load saved plan failed: {exc}"
            return
        self._apply_speed()
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = _BRAINARY / "output" / f"replay_{ts}"
        self._dirs = {k: run_dir / k for k in ("sim", "perception", "memory", "planning",
                                               "project_management", "monitor")}
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self._ctx = {"plan_actions": pa}
        self._holding = None
        self._act_i = 0
        self._act_n = len(pa)
        for m in _MODULES:
            self._logs[m].lines = []
            self._logs[m].label.text = ""
        self._ov_action.text = "  action: -"
        self._ov_summary.text = f"  summary: REPLAY saved plan ({len(pa)} actions, no inference)"
        bk = {"Prop_KLT_1": "KLT_1(blue)", "Prop_KLT_2": "KLT_2(purple)", "Prop_KLT_3": "KLT_3(green)"}
        self._logs["Project Management"].log(
            f"REPLAY {len(pa)} saved actions from {_SAVED_PLAN.name} (skip perception/memory/SSP/planning)")
        for s in pa:
            self._logs["Project Management"].log(f"    {s.get('action')} {bk.get(s.get('target'), s.get('target'))}")

        def _gohome():
            fn = getattr(self.sim, "go_home", None)
            if fn:
                fn()
                return "home"
            return "go_home unavailable - skipped"

        q = deque()
        q.append(_Step("Project Management", "go home", _gohome))
        for idx, s in enumerate(pa):
            q.append(_Step("Project Management", f"{s.get('action')} {s.get('target')}",
                           (lambda a=s, i=idx: self._exec_action(a, i))))
        self._q = q

    # -------------------------------------------------- stages
    def _st_sim(self):
        import numpy as np
        from PIL import Image
        d = self._dirs["sim"]; (d / "rgb").mkdir(exist_ok=True); (d / "depth").mkdir(exist_ok=True)
        cams = self.sim.get_all_cameras() if hasattr(self.sim, "get_all_cameras") else {}
        vf = {}
        for name, fr in (cams or {}).items():
            if fr.get("rgb") is not None:
                p = d / "rgb" / f"{name}.png"
                Image.fromarray(np.asarray(fr["rgb"])[..., :3].astype("uint8")).save(p); vf[name] = str(p)
            if fr.get("depth") is not None:
                np.save(d / "depth" / f"{name}.npy", np.asarray(fr["depth"]))
        state = {"task": self.rb._TASK_DEFAULT, "graspable": self.sim.list_graspable(),
                 "baskets": self.sim.list_baskets(), "objects": {}, "robot": {}}
        for o in state["graspable"] + state["baskets"]:
            op = self.sim.get_object_pose(o)
            if op is not None:
                state["objects"][o] = op
        try:
            state["robot"] = self.sim.get_robot_state()
        except Exception:
            pass
        (d / "scene_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        self._ctx["state"] = state; self._ctx["view_files"] = vf
        self._logs["IsaacSim"].log(f"  graspable: {', '.join(state['graspable'])}", echo=False)
        _term("IsaacSim", f"相机视角 {len(vf)} 路: {', '.join(vf) if vf else '(无)'}")
        _term("IsaacSim", f"可抓物体 {len(state['graspable'])} 个: {', '.join(state['graspable'])}")
        _term("IsaacSim", f"篮子 {len(state['baskets'])} 个: {', '.join(state['baskets'])}")
        return f"{len(vf)} views, {len(state['objects'])} object poses"

    def _st_perc(self):
        p = self.rb.stage_perception(self._mode, self._dirs["perception"], self.sim,
                                     self._ctx["state"], self._ctx["view_files"], "http://127.0.0.1:5599")
        self._ctx["perception"] = p
        _term("Perception", f"后端={_cn_backend(p.get('perception_backend'))}  "
                            f"识别到 {len(p.get('objects', []))} 个物体")
        if p.get("scene_summary"):
            _term("Perception", f"场景概述: {p['scene_summary']}")
        for o in p.get("objects", []):
            self._logs["Perception"].log(f"    {o.get('name')}  [{o.get('category')}]", echo=False)
            # 详细结果:id / 类别 / 外观 / 位置 / 状态(GPT 中文填写)
            _term("Perception", f"  · {o.get('name')}  [{o.get('category')}]  "
                                f"外观={o.get('appearance') or '-'}  位置={o.get('location') or '-'}  "
                                f"状态={o.get('state') or '-'}")
        for r in p.get("relations", [])[:12]:
            _term("Perception", f"  关系: {r.get('subject')} --{r.get('predicate')}--> {r.get('object')}")
        _dump_json("Perception", self._dirs["perception"] / "perception.json", "感知模块完整输出", 80)
        self._ov_summary.text = f"  summary: perception [{p.get('perception_backend')}] {len(p.get('objects',[]))} objects"
        return f"[{p.get('perception_backend','?')}] {len(p.get('objects',[]))} objects"

    def _st_mem(self):
        pi = self.rb.stage_memory(self._dirs["memory"], self._ctx["perception"], self.rb._TASK_DEFAULT)
        self._ctx["planning_input"] = pi
        _term("Memory", f"可操作物体 {len(pi.get('manipulable_objects', []))} 个")
        _dump_json("Memory", self._dirs["memory"] / "planning_input.json", "记忆模块导出的规划输入", 80)
        _dump_json("Memory", self._dirs["memory"] / "memory_snapshot.json", "记忆快照", 40)
        return f"{len(pi.get('manipulable_objects', []))} manipulable objects"

    def _st_ssp(self):
        pi = self.rb.stage_ssp(self._dirs["monitor"], self._dirs["memory"] / "memory_snapshot.json",
                               self._dirs["memory"] / "planning_input.json", self._ctx["planning_input"])
        self._ctx["planning_input"] = pi
        sc = (pi.get("constraints", {}) or {}).get("safety_constraints", {})
        n = len((sc or {}).get("candidate_constraints", [])) if isinstance(sc, dict) else 0
        _term("Monitor", f"生成候选安全约束 {n} 条")
        if isinstance(sc, dict) and sc:
            for ln in json.dumps(sc, ensure_ascii=False, indent=2).splitlines()[:60]:
                print(f"      {ln}", flush=True)
        for _f in ("ssp_constraints.json", "safety_constraints.json"):
            _dump_json("Monitor", self._dirs["monitor"] / _f, "SSP 安全约束完整输出", 60)
        return f"{n} candidate safety constraints"

    def _st_plan(self):
        plan = self.rb.stage_planning(self._dirs["planning"], self._ctx["planning_input"], self.rb._TASK_DEFAULT)
        bk = {"Prop_KLT_1": "KLT_1(blue)", "Prop_KLT_2": "KLT_2(purple)", "Prop_KLT_3": "KLT_3(green)"}
        pa = json.loads((self._dirs["planning"] / "planned_actions.json").read_text(encoding="utf-8"))
        self._ctx["plan_actions"] = pa
        self._ctx["plan"] = plan          # 沙盒校验要用 plan["plan"](带 depends_on 的 DAG)
        try:                                    # 存档方案,供 "Replay saved plan" 复用(跳过推理)
            _SAVED_PLAN.parent.mkdir(parents=True, exist_ok=True)
            _SAVED_PLAN.write_text(json.dumps(pa, ensure_ascii=False, indent=2), encoding="utf-8")
            self._logs["Planning"].log(f"  saved plan -> {_SAVED_PLAN.name} (reuse via 'Replay saved plan')",
                                       echo=False)
            _term("Planning", f"方案已存档 -> {_SAVED_PLAN.name}(可用 'Replay saved plan' 复用,跳过推理)")
        except Exception as exc:
            self._logs["Planning"].log(f"  save plan failed: {exc}", echo=False)
            _term("Planning", f"方案存档失败: {exc}")
        _term("Planning", f"后端={_cn_backend(plan.get('planning_backend'))}  共 {len(pa)} 步:")
        for i, s in enumerate(pa):
            if s.get("action") == "grasp":
                self._logs["Planning"].log(f"    grasp {s.get('target')}", echo=False)
                _term("Planning", f"  {i+1:2d}. 抓取 {s.get('target')}")
            else:
                self._logs["Planning"].log(f"       -> place into {bk.get(s.get('target'), s.get('target'))}",
                                           echo=False)
                _term("Planning", f"  {i+1:2d}. 放入 {bk.get(s.get('target'), s.get('target'))}")
        _dump_json("Planning", self._dirs["planning"] / "plan.json", "规划模块完整输出", 80)
        _dump_json("Planning", self._dirs["planning"] / "planned_actions.json", "动作序列", 60)
        # ---- 规划之后先插【物理沙盒校验】,它跑完再由它排执行 + 监控 ----
        # 顺序必须是 规划 -> 校验(可能 replan) -> 执行,所以执行步骤不能在这里就建好:
        # 万一校验判失败触发重规划,动作序列会变,提前建好的步骤就成了旧的。
        self._q.appendleft(_Step("Simulation", "physics sandbox verify", self._st_verify))
        self._ov_summary.text = f"  summary: plan [{plan.get('planning_backend')}] {plan.get('num_steps')} steps"
        return f"[{plan.get('planning_backend','?')}] {plan.get('num_steps')} steps"

    def _queue_execution(self):
        """把【当前】planned_actions.json 排成执行步骤 + 末尾监控。校验(及可能的重规划)之后才调用。"""
        pa = json.loads((self._dirs["planning"] / "planned_actions.json").read_text(encoding="utf-8"))
        self._ctx["plan_actions"] = pa
        follow = deque()
        if self._do_exec:
            self._act_i = 0
            self._act_n = len(pa)
            for idx, s in enumerate(pa):
                follow.append(_Step("Project Management", f"{s.get('action')} {s.get('target')}",
                                    (lambda a=s, i=idx: self._exec_action(a, i))))
        else:
            self._logs["Project Management"].log("  (execute disabled - plan only)", echo=False)
            _term("Project Management", "执行开关关闭,只规划不动机器人")
        follow.append(_Step("Monitor", "SafetyCritic judge", self._st_mon))
        self._q.extendleft(reversed(follow))   # keep order

    def _st_verify(self):
        """物理沙盒校验(Simulation 模块):把 plan + 相机数据发给 :5600 的沙盒服务预演。

        服务没起 -> 自动跳过、按放行处理(和 run_brainary --verify 完全一样的降级逻辑),
        不会挡住整条流程。判失败且给了反思提示 -> 注入 physics_reflection 重规划一次。
        """
        plan_dag = (self._ctx.get("plan") or {}).get("plan", []) or self._ctx.get("plan_actions", [])
        res = None
        try:
            res = self.rb.stage_verify(self._dirs["planning"], plan_dag, self.sim)
        except Exception as exc:
            _term("Simulation", f"校验异常,按放行处理: {exc}")
        if res is None:
            _term("Simulation", "沙盒服务未运行 -> 跳过物理校验,计划按原样放行")
            _term("Simulation", "(要启用:在 brainary_sim 环境跑 simulation/serve.py,监听 :5600)")
            self._queue_execution()
            return "skipped (service down)"

        ok = bool(res.get("success"))
        _term("Simulation", f"沙盒预演结果: {'通过' if ok else '不通过'} | {res.get('message', '')}")
        _dump_json("Simulation", self._dirs["planning"] / "physics_verify.json", "沙盒校验完整输出", 60)
        if ok:
            self._queue_execution()
            return "verified ok"

        prompt = res.get("llm_reflection_prompt")
        if not prompt:
            _term("Simulation", "判为不通过但没给反思提示,无法重规划,按原计划继续")
            self._queue_execution()
            return "failed (no reflection)"

        before = [f"{a.get('action')} {a.get('target')}" for a in self._ctx.get("plan_actions", [])]
        _term("Simulation", "把反思提示注入规划输入,重新规划一次 ...")
        self._ctx["planning_input"] = {**self._ctx["planning_input"], "physics_reflection": prompt}
        plan = self.rb.stage_planning(self._dirs["planning"], self._ctx["planning_input"], self.rb._TASK_DEFAULT)
        self._ctx["plan"] = plan
        after = [f"{a.get('action')} {a.get('target')}"
                 for a in json.loads((self._dirs["planning"] / "planned_actions.json").read_text(encoding="utf-8"))]
        if after == before:
            # 注意:当前 TaskPlanner.generate_plan 只读 task_instruction/constraints/
            # manipulable_objects/available_skills 四个键,不读 physics_reflection ->
            # 重规划输入实际没变,顺序自然也不会变。这里如实报出来,别让人以为"重排过了"。
            _term("Simulation", "⚠ 重规划后动作序列与之前完全相同(反思未产生影响)")
        else:
            _term("Simulation", f"重规划后序列已变化({len(before)} 步 -> {len(after)} 步):")
            for i, x in enumerate(after, 1):
                _term("Simulation", f"  {i:2d}. {x}")
        self._queue_execution()
        return f"replanned ({'no change' if after == before else 'reordered'})"

    def _resolve(self, name, kind):
        """Resolve a perception/plan target name to a real sim id (grasp->graspable, place->basket)."""
        if not name:
            return None
        valid = self._graspable if kind == "grasp" else self._baskets
        if name in valid:                     # already a sim id
            return name
        a = self._aliases.get(name)           # alias table (Chinese + English)
        if a and a in valid:
            return a
        low = str(name).lower()               # fuzzy: substring either way (catches scissors, KLT_*)
        for v in valid:
            vl = v.lower()
            if low in vl or vl in low:
                return v
        return None

    def _exec_action(self, action: dict, idx: int):
        """Execute ONE plan action on the robot. grasp only fires on a resolvable object;
        place only fires if we are actually holding something (skip empty place)."""
        self._act_i = idx + 1
        act = action.get("action"); raw = action.get("target")
        tgt = self._resolve(raw, act)
        self._ov_action.text = f"  action: {self._act_i}/{self._act_n}  {act} {raw}"
        tag = f"    [{self._act_i}/{self._act_n}]"
        rec = self._ctx.setdefault("exec_results", [])          # 供 _finish 打终端总结

        if act == "grasp":
            if tgt is None:
                self._holding = None
                self._logs["Project Management"].log(f"{tag} grasp {raw}: cannot map to a sim object -> SKIP",
                                                     echo=False)
                _term("Project Management", f"{tag} ⏭ 抓取 {raw}:名字对不上场景里任何物体,跳过")
                rec.append({"action": "抓取", "target": raw, "ok": False, "skipped": True, "why": "名字无法解析"})
                return f"grasp {raw}: unresolved"
            r = dict(self.sim.grasp(tgt))
            self._holding = tgt if r.get("ok") else None
            self._logs["Project Management"].log(f"{tag} grasp {tgt} -> ok={r.get('ok')} {r.get('reason') or ''}",
                                                 echo=False)
            _term("Project Management",
                  f"{tag} {'✅' if r.get('ok') else '❌'} 抓取 {tgt}:{'成功' if r.get('ok') else '失败'}"
                  f"  夹爪开度={r.get('gripper_width')}  步数={r.get('steps')}"
                  + (f"  原因={r.get('reason')}" if r.get("reason") else ""))
            rec.append({"action": "抓取", "target": tgt, "ok": bool(r.get("ok")), "why": r.get("reason")})
            return f"grasp {tgt} ok={r.get('ok')}"

        if act == "place":
            if not self._holding:             # nothing grasped -> do NOT move to basket
                self._logs["Project Management"].log(f"{tag} place {raw}: not holding anything (grasp failed) -> SKIP",
                                                     echo=False)
                _term("Project Management", f"{tag} ⏭ 放置 {raw}:手里没东西(上一步抓取失败),跳过")
                rec.append({"action": "放置", "target": raw, "ok": False, "skipped": True, "why": "空爪"})
                return f"place {raw}: skipped (empty gripper)"
            if tgt is None:
                self._logs["Project Management"].log(f"{tag} place {raw}: not a valid basket -> SKIP", echo=False)
                _term("Project Management", f"{tag} ⏭ 放置 {raw}:不是有效篮子,跳过")
                rec.append({"action": "放置", "target": raw, "ok": False, "skipped": True, "why": "篮子无法解析"})
                return f"place {raw}: unresolved"
            r = dict(self.sim.place(tgt))
            self._holding = None
            self._logs["Project Management"].log(f"{tag} place {tgt} -> ok={r.get('ok')} {r.get('reason') or ''}",
                                                 echo=False)
            _term("Project Management",
                  f"{tag} {'✅' if r.get('ok') else '❌'} 放置 {tgt}:{'成功' if r.get('ok') else '失败'}"
                  f"  落入篮子={r.get('object_in_basket')}  步数={r.get('steps')}"
                  + (f"  原因={r.get('reason')}" if r.get("reason") else ""))
            rec.append({"action": "放置", "target": tgt, "ok": bool(r.get("ok")), "why": r.get("reason")})
            return f"place {tgt} ok={r.get('ok')}"

        return f"{act}: skipped"

    def _st_mon(self):
        pm_actions = self._dirs["project_management"] / "pm_planned_actions.json"
        actions = pm_actions if pm_actions.exists() else self._dirs["planning"] / "planned_actions.json"
        s = self.rb.stage_monitor(self._dirs["monitor"], self._dirs["memory"] / "memory_snapshot.json",
                                  actions, self.rb._LLM_MODEL)
        self._ctx["safety"] = s
        if not s:
            self._logs["Monitor"].log("  skipped (no API_zhongzhuan)", echo=False)
            _term("Monitor", "跳过(未配置 API 密钥)")
            return "skipped (no key)"
        for r in (json.loads((self._dirs["monitor"] / "safety_critic_review.json").read_text(encoding="utf-8"))
                  .get("reviews", []) if (self._dirs["monitor"] / "safety_critic_review.json").exists() else []):
            self._logs["Monitor"].log(f"    [{r.get('index')}] {r.get('action')} -> {r.get('decision')}", echo=False)
            _term("Monitor", f"  [{r.get('index')}] {r.get('action')} -> 裁定={r.get('decision')}"
                             + (f"  理由={r.get('reason')}" if r.get("reason") else ""))
        _dump_json("Monitor", self._dirs["monitor"] / "safety_critic_review.json", "安全评审完整输出", 80)
        return f"{s['num_steps']} steps, overall={s['overall']}, malicious={s['num_malicious']}"

    def _finish(self):
        self._ov_action.text = "  action: (done)"
        saf = self._ctx.get("safety")
        parts = []
        p = self._ctx.get("perception", {})
        if p:
            parts.append(f"perception {len(p.get('objects',[]))}")
        pa = self._ctx.get("plan_actions")
        if pa is not None:
            parts.append(f"plan {len(pa)} steps")
        if saf:
            parts.append(f"safety={saf['overall']}({saf['num_malicious']} risky)")
        self._ov_summary.text = "  summary: DONE - " + " | ".join(parts) if parts else "  summary: DONE"
        self._ov_stage.text = "  stage: (finished)"

        # ---- 终端总结:各模块产出一览 + 执行逐条结果 ----
        lines = [f"大脑闭环结束  总耗时 {time.time() - getattr(self, '_t_run0', time.time()):.1f}s"]
        if p:
            names = [str(o.get("name")) for o in p.get("objects", [])]
            lines.append(f"  感知: 识别 {len(names)} 个物体 -> {', '.join(names) if names else '(无)'}")
        if pa is not None:
            lines.append(f"  规划: {len(pa)} 步动作")
        res = self._ctx.get("exec_results") or []
        if res:
            ok_n = sum(1 for r in res if r.get("ok"))
            lines.append(f"  执行: {ok_n}/{len(res)} 成功")
            for r in res:
                mark = "✅" if r.get("ok") else ("⏭" if r.get("skipped") else "❌")
                extra = f"  ({r['why']})" if r.get("why") else ""
                lines.append(f"     {mark} {r.get('action')} {r.get('target')}{extra}")
        if saf:
            lines.append(f"  监控: 安全裁定={saf['overall']},风险动作 {saf['num_malicious']} 个")
        _banner("\n".join(lines))


# ============================================================ manual grasp/place panel (English)
class GraspPanel:
    def __init__(self, sim, speed_fn=None):
        import omni.ui as ui
        self.sim = sim
        # 取当前速度档的回调(默认接 BrainTestPanel 的 Speed 下拉)。手动抓/放前必须应用速度,
        # 否则跑的是 SKILL_TEST_RUNNER_SPEED 默认档而不是 UI 上选的那档。
        self._speed_fn = speed_fn
        self._pending = None
        self._graspable = sim.list_graspable()
        self._baskets = sim.list_baskets()
        self._m = {}
        self.window = ui.Window("Grasp / Place (manual)", width=430, height=300)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("== Manual grasp / place ==")
                ui.Label(f"Grasp target ({len(self._graspable)})")
                self._m["grasp"] = ui.ComboBox(0, *self._graspable).model
                ui.Button("Grasp", clicked_fn=lambda: self._set("grasp"))
                ui.Label(f"Place basket ({len(self._baskets)})")
                self._m["basket"] = ui.ComboBox(0, *self._baskets).model
                ui.Button("Place", clicked_fn=lambda: self._set("place"))
                with ui.HStack(spacing=6, height=26):
                    ui.Button("Go Home", clicked_fn=lambda: self._set("go_home"))
                    ui.Button("Reset scene", clicked_fn=lambda: self._set("reset"))
                self._status = ui.Label("status: idle")
                self._result = ui.Label("(no result yet)", word_wrap=True)

    def _sel(self, key, names):
        m = self._m.get(key)
        i = int(m.get_item_value_model().get_value_as_int()) if m else 0
        return names[i] if names and 0 <= i < len(names) else (names[0] if names else None)

    def _set(self, kind):
        if self._pending is None:
            self._pending = kind; self._status.text = f"status: pending {kind}"

    def has_pending(self) -> bool:
        return self._pending is not None

    def run_pending(self) -> None:
        kind = self._pending; self._pending = None
        if kind in ("grasp", "place"):          # 和 BrainTestPanel 一样,执行前先把速度落下去
            try:
                spd = _apply_speed_to(self.sim, self._speed_fn() if self._speed_fn else 1.0)
                print(f"[brain_ui] manual {kind}: speed={spd}", flush=True)
            except Exception as exc:
                print(f"[brain_ui] manual {kind}: apply speed failed: {exc}", flush=True)
        try:
            if kind == "grasp":
                o = self._sel("grasp", self._graspable); self._show("grasp", o, self.sim.grasp(o))
            elif kind == "place":
                b = self._sel("basket", self._baskets); self._show("place", b, self.sim.place(b))
            elif kind == "go_home":
                self._show("go_home", "home", self.sim.go_home())
            elif kind == "reset":
                self._show("reset", "scene", self.sim.reset())
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._status.text = f"{kind} ERROR"; self._result.text = f"ERROR: {exc}"

    def _show(self, skill, target, r):
        self._status.text = f"status: {skill}({target}) -> ok={r.get('ok')}"
        lines = [f"{skill}({target}) ok={r.get('ok')}", f"  reason: {r.get('reason') or '(ok)'}"]
        for k in ("holding", "gripper_width", "object_in_basket", "steps"):
            if r.get(k) is not None:
                lines.append(f"  {k}: {r[k]}")
        self._result.text = "\n".join(lines)
        print(f"[brain_ui] {skill}({target}) -> {r}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return _run(args, app_launcher, simulation_app)
    except BaseException:
        import traceback
        Path("/tmp/brainary_brain_ui_err.txt").write_text(traceback.format_exc())
        print("[brain_ui] FATAL:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        simulation_app.close()


def _run(args, app_launcher, simulation_app) -> int:
    import torch
    from brainary_api import BrainaryAPI

    print("[brain_ui] Building scene (~60-120s); wait for 'ready' and the windows ...", flush=True)
    sim = BrainaryAPI.launch(headless=False, device=args.device, _app_launcher=app_launcher)
    si = sim.sim
    session, env, provider = si.session, si.env, si.provider
    try:
        from franka_v1_skill_lab.scene_interface import camera_offsets
        camera_offsets.apply_saved_offsets_runtime(env)
    except Exception as exc:
        print(f"[brain_ui] camera offsets: {exc}", flush=True)

    brain = BrainTestPanel(sim, simulation_app)
    grasp = GraspPanel(sim, speed_fn=brain.selected_speed)   # 手动面板复用同一个 Speed 下拉
    print("[brain_ui] ready: Brainary Brain Test + Grasp/Place", flush=True)

    while simulation_app.is_running():
        with torch.inference_mode():
            if brain.has_pending():
                brain.run_pending(); continue
            if grasp.has_pending():
                grasp.run_pending(); continue
            env.step(si.hold_action(provider.get_state()))

    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
