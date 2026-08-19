"""忠实复现 brainary_brain_ui.py 的【执行序列】(不跑感知/规划),headless。
用 saved_plan.json 里的完整 10 步计划,按 _exec_action 的逻辑逐步 grasp/place,
定位到底是哪一步在 "杯子上方不动了"。

运行:
  conda activate env_isaaclab
  ./isaaclab.sh -p brainary/sim/test_brain_exec.py --headless
"""
import argparse, json, os, sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_BRAINARY = _DIR.parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_BRAINARY / "project_management"))

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--speed", type=float, default=1.0)   # brain UI 默认 1.0
ap.add_argument("--plan", default=str(_BRAINARY / "output" / "saved_plan.json"))
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from brainary_api import BrainaryAPI


def _resolve(name, kind, graspable, baskets, aliases):
    if not name:
        return None
    valid = graspable if kind == "grasp" else baskets
    if name in valid:
        return name
    a = aliases.get(name)
    if a and a in valid:
        return a
    low = str(name).lower()
    for v in valid:
        vl = v.lower()
        if low in vl or vl in low:
            return v
    return None


def main():
    sim = BrainaryAPI.launch(headless=True, device=getattr(args, "device", "cuda:0"),
                             _app_launcher=app_launcher)
    si = sim.sim

    # --- 完全照 _apply_speed 设速度 ---
    os.environ["SKILL_TEST_SPEED"] = str(args.speed)
    os.environ["SKILL_TEST_RUNNER_SPEED"] = str(args.speed)
    ctrl = getattr(si, "_ctrl", None)
    if ctrl is not None:
        ctrl._runner_speed = max(0.5, float(args.speed))
        ctrl._apply_speed_to_adapter()
    print(f"[brain-exec] speed={args.speed} runner_speed={getattr(ctrl,'_runner_speed',None)} "
          f"max_joint_step={getattr(ctrl.adapter,'max_joint_step',None)}", flush=True)

    graspable = list(sim.list_graspable())
    baskets = list(sim.list_baskets())
    aliases = json.loads((_BRAINARY / "project_management" / "object_aliases.json").read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    print(f"[brain-exec] graspable={graspable}", flush=True)
    print(f"[brain-exec] baskets={baskets}", flush=True)
    print(f"[brain-exec] plan={len(plan)} actions", flush=True)

    with torch.inference_mode():                      # ← brain UI 主循环一样的上下文
        # 照 _start_replay:先回 home
        try:
            sim.go_home()
        except Exception as e:
            print(f"[brain-exec] go_home failed: {e}", flush=True)

        holding = None
        results = []
        for i, action in enumerate(plan):
            act = action.get("action"); raw = action.get("target")
            tgt = _resolve(raw, act, graspable, baskets, aliases)
            tag = f"[{i+1}/{len(plan)}]"
            print(f"\n===== {tag} {act} {raw!r} -> resolve={tgt} =====", flush=True)
            if act == "grasp":
                if tgt is None:
                    print(f"{tag} grasp {raw}: UNRESOLVED -> skip", flush=True); holding = None; continue
                r = dict(sim.grasp(tgt))
                holding = tgt if r.get("ok") else None
                print(f"{tag} grasp {tgt} -> ok={r.get('ok')} reason={r.get('reason')} "
                      f"steps={r.get('steps')} timed_out={r.get('timed_out')}", flush=True)
                results.append((tag, "grasp", tgt, r.get("ok"), r.get("reason")))
            elif act == "place":
                if not holding:
                    print(f"{tag} place {raw}: NOT HOLDING -> skip", flush=True)
                    results.append((tag, "place", raw, "skip", "empty")); continue
                if tgt is None:
                    print(f"{tag} place {raw}: UNRESOLVED -> skip", flush=True); continue
                r = dict(sim.place(tgt))
                holding = None
                print(f"{tag} place {tgt} -> ok={r.get('ok')} reason={r.get('reason')} "
                      f"in_basket={r.get('object_in_basket')} steps={r.get('steps')}", flush=True)
                results.append((tag, "place", tgt, r.get("ok"), r.get("reason")))

        print("\n========== SUMMARY ==========", flush=True)
        for tag, kind, t, ok, why in results:
            print(f"  {tag} {kind:5} {t:22} ok={ok} {why or ''}", flush=True)

    sim.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
