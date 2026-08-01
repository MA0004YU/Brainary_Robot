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

# The real project modules (each maps to a real owner). SSP belongs to Monitor;
# execution belongs to Project Management. Do NOT invent extra modules.
_MODULES = ["IsaacSim", "Perception", "Memory", "Planning", "Monitor", "Simulation", "Project Management"]
_SPEED_OPTS = [("Very slow (0.5)", 0.5), ("Slow (0.75)", 0.75), ("Normal (1.0)", 1.0),
               ("Fast (2.0)", 2.0), ("Fastest (5.0)", 5.0)]


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

    def log(self, msg: str):
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
                ui.Separator()
                ui.Label("OVERALL", height=16)
                self._ov_stage = ui.Label("  stage: -")
                self._ov_action = ui.Label("  action: -")
                self._ov_summary = ui.Label("  summary: (not run)", word_wrap=True)
        # ---- one window PER module, all docked as switchable TABS next to Content/Console ----
        for m in _MODULES:
            self._logs[m] = _ModuleWindow(ui, m, dock_target="Console")

    # -------------------------------------------------- main-loop hooks
    def has_pending(self) -> bool:
        return bool(self._q) or self._active is not None

    def run_pending(self) -> None:
        if self._active is not None:
            step = self._active
            self._active = None
            try:
                res = step.fn()
                if res:
                    self._logs[step.module].log(f"  done: {res}")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._logs[step.module].log(f"  FAILED: {str(exc)[:120]}")
            if not self._q:
                self._finish()
            return
        if self._q:
            step = self._q.popleft()
            self._ov_stage.text = f"  stage: {step.module} - {step.title}"
            self._logs[step.module].log(f"> {step.title} ...")
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
        try:
            from base_skill import set_speed_scale
            set_speed_scale(_SPEED_OPTS[int(self._spd.get_item_value_model().get_value_as_int())][1])
        except Exception as exc:
            print(f"[brain_ui] set_speed_scale: {exc}", flush=True)
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
        self._holding = None
        for m in _MODULES:
            self._logs[m].lines = []; self._logs[m].label.text = ""
        self._ov_action.text = "  action: -"
        self._ov_summary.text = "  summary: running..."
        self._logs["Simulation"].log("optional physics-sandbox validator - not run in this loop "
                                     "(use run_brainary --verify)")
        self._q = deque([
            _Step("IsaacSim", "capture scene", self._st_sim),
            _Step("Perception", "perceive", self._st_perc),
            _Step("Memory", "build memory", self._st_mem),
            _Step("Monitor", "SSP safety constraints", self._st_ssp),   # SSP is part of Monitor
            _Step("Planning", "plan", self._st_plan),                   # appends PM exec + Monitor(critic)
        ])

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
        self._logs["IsaacSim"].log(f"  graspable: {', '.join(state['graspable'])}")
        return f"{len(vf)} views, {len(state['objects'])} object poses"

    def _st_perc(self):
        p = self.rb.stage_perception(self._mode, self._dirs["perception"], self.sim,
                                     self._ctx["state"], self._ctx["view_files"], "http://127.0.0.1:5599")
        self._ctx["perception"] = p
        for o in p.get("objects", []):
            self._logs["Perception"].log(f"    {o.get('name')}  [{o.get('category')}]")
        self._ov_summary.text = f"  summary: perception [{p.get('perception_backend')}] {len(p.get('objects',[]))} objects"
        return f"[{p.get('perception_backend','?')}] {len(p.get('objects',[]))} objects"

    def _st_mem(self):
        pi = self.rb.stage_memory(self._dirs["memory"], self._ctx["perception"], self.rb._TASK_DEFAULT)
        self._ctx["planning_input"] = pi
        return f"{len(pi.get('manipulable_objects', []))} manipulable objects"

    def _st_ssp(self):
        pi = self.rb.stage_ssp(self._dirs["monitor"], self._dirs["memory"] / "memory_snapshot.json",
                               self._dirs["memory"] / "planning_input.json", self._ctx["planning_input"])
        self._ctx["planning_input"] = pi
        sc = (pi.get("constraints", {}) or {}).get("safety_constraints", {})
        n = len((sc or {}).get("candidate_constraints", [])) if isinstance(sc, dict) else 0
        return f"{n} candidate safety constraints"

    def _st_plan(self):
        plan = self.rb.stage_planning(self._dirs["planning"], self._ctx["planning_input"], self.rb._TASK_DEFAULT)
        bk = {"Prop_KLT_1": "KLT_1(blue)", "Prop_KLT_2": "KLT_2(purple)", "Prop_KLT_3": "KLT_3(green)"}
        pa = json.loads((self._dirs["planning"] / "planned_actions.json").read_text(encoding="utf-8"))
        self._ctx["plan_actions"] = pa
        for s in pa:
            if s.get("action") == "grasp":
                self._logs["Planning"].log(f"    grasp {s.get('target')}")
            else:
                self._logs["Planning"].log(f"       -> place into {bk.get(s.get('target'), s.get('target'))}")
        # ---- append Execution (one step per action) + Monitor ----
        follow = deque()
        if self._do_exec:
            self._act_n = len(pa)
            for idx, s in enumerate(pa):
                follow.append(_Step("Project Management", f"{s.get('action')} {s.get('target')}",
                                    (lambda a=s, i=idx: self._exec_action(a, i))))
        else:
            self._logs["Project Management"].log("  (execute disabled - plan only)")
        follow.append(_Step("Monitor", "SafetyCritic judge", self._st_mon))
        self._q.extendleft(reversed(follow))   # keep order
        self._ov_summary.text = f"  summary: plan [{plan.get('planning_backend')}] {plan.get('num_steps')} steps"
        return f"[{plan.get('planning_backend','?')}] {plan.get('num_steps')} steps"

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

        if act == "grasp":
            if tgt is None:
                self._holding = None
                self._logs["Project Management"].log(f"{tag} grasp {raw}: cannot map to a sim object -> SKIP")
                return f"grasp {raw}: unresolved"
            r = dict(self.sim.grasp(tgt))
            self._holding = tgt if r.get("ok") else None
            self._logs["Project Management"].log(f"{tag} grasp {tgt} -> ok={r.get('ok')} {r.get('reason') or ''}")
            return f"grasp {tgt} ok={r.get('ok')}"

        if act == "place":
            if not self._holding:             # nothing grasped -> do NOT move to basket
                self._logs["Project Management"].log(f"{tag} place {raw}: not holding anything (grasp failed) -> SKIP")
                return f"place {raw}: skipped (empty gripper)"
            if tgt is None:
                self._logs["Project Management"].log(f"{tag} place {raw}: not a valid basket -> SKIP")
                return f"place {raw}: unresolved"
            r = dict(self.sim.place(tgt))
            self._holding = None
            self._logs["Project Management"].log(f"{tag} place {tgt} -> ok={r.get('ok')} {r.get('reason') or ''}")
            return f"place {tgt} ok={r.get('ok')}"

        return f"{act}: skipped"

    def _st_mon(self):
        pm_actions = self._dirs["project_management"] / "pm_planned_actions.json"
        actions = pm_actions if pm_actions.exists() else self._dirs["planning"] / "planned_actions.json"
        s = self.rb.stage_monitor(self._dirs["monitor"], self._dirs["memory"] / "memory_snapshot.json",
                                  actions, self.rb._LLM_MODEL)
        self._ctx["safety"] = s
        if not s:
            self._logs["Monitor"].log("  skipped (no API_zhongzhuan)")
            return "skipped (no key)"
        for r in (json.loads((self._dirs["monitor"] / "safety_critic_review.json").read_text(encoding="utf-8"))
                  .get("reviews", []) if (self._dirs["monitor"] / "safety_critic_review.json").exists() else []):
            self._logs["Monitor"].log(f"    [{r.get('index')}] {r.get('action')} -> {r.get('decision')}")
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


# ============================================================ manual grasp/place panel (English)
class GraspPanel:
    def __init__(self, sim):
        import omni.ui as ui
        self.sim = sim
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
    grasp = GraspPanel(sim)
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
