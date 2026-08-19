"""逐个测试【所有可抓物体】的 grasp,复现 brain UI 手动 Grasp 面板的调用路径,headless。

brain UI 的 GraspPanel 默认选中的是 list_graspable()[0](= 香蕉),用户点 "Grasp" 就走
BrainaryAPI.grasp -> SimInterface.grasp -> _GraspPoseRunner。本脚本用同样的路径把 6 个
物体各抓一遍,打印每个物体走到哪个 phase、失败原因,定位"UI 抓不起来"到底是哪些物体。

运行:
  conda activate env_isaaclab
  ./isaaclab.sh -p brainary/sim/test_grasp_all.py --headless --enable_cameras [--speed 1.0]
"""
import argparse, sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--speed", type=float, default=1.0)      # brain UI 默认档
ap.add_argument("--only", default="")                    # 逗号分隔,只测这些
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from brainary_api import BrainaryAPI  # noqa: E402


def main():
    # 相机分辨率压到最低:本测试只关心手臂/夹爪,不看图像。默认 640x480 x6 路渲染会把
    # 每步拖到 ~1 step/s(整轮几十分钟);160x120 快数倍,控制逻辑完全不变。
    sim = BrainaryAPI.launch(headless=True, device=getattr(args, "device", "cuda:0"),
                             camera_resolution=(160, 120), _app_launcher=app_launcher)
    si = sim.sim
    ctrl = getattr(si, "_ctrl", None)
    if ctrl is not None:
        ctrl._runner_speed = max(0.5, float(args.speed))
        ctrl._apply_speed_to_adapter()
    objs = list(sim.list_graspable())
    if args.only:
        want = [o.strip() for o in args.only.split(",") if o.strip()]
        objs = [o for o in objs if o in want]
    print(f"[grasp-all] speed={args.speed} max_joint_step={getattr(ctrl.adapter,'max_joint_step',None)}", flush=True)
    print(f"[grasp-all] objects={objs}", flush=True)

    rows = []
    with torch.inference_mode():
        for i, obj in enumerate(objs):
            print(f"\n===== [{i+1}/{len(objs)}] grasp {obj} =====", flush=True)
            try:
                sim.go_home()
            except Exception as e:
                print(f"  go_home failed: {e}", flush=True)
            try:
                r = dict(sim.grasp(obj))
            except Exception as e:
                rows.append((obj, "EXC", str(e))); print(f"  EXC {e}", flush=True); continue
            print(f"  -> ok={r.get('ok')} holding={r.get('holding')} gw={r.get('gripper_width')} "
                  f"steps={r.get('steps')} timed_out={r.get('timed_out')}", flush=True)
            rows.append((obj, r.get("ok"), f"gw={r.get('gripper_width')} steps={r.get('steps')}"))
            try:                                   # 放回去/张爪,免得下一个物体带着上一个跑
                si.open_gripper()
            except Exception:
                pass

    print("\n========== SUMMARY ==========", flush=True)
    for o, ok, extra in rows:
        print(f"  {o:28} ok={ok}  {extra}", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
