"""橘子"抓起来但搬运中掉落"的对照实验,headless。

球体是平行夹爪最难持的形状:只有夹在【赤道】附近两个接触法线才水平对顶(力封闭),而且全靠
摩擦,夹持力不够一加速就滑脱。

★ 第一轮实测已推翻"标定点夹在球顶"的猜测:z偏移=0 时 held=False(夹爪全开 0.08,啥也没夹到),
  说明橘子 root pivot 不在球心而在底部,标定的 +0.035 其实已经在赤道附近、位置是对的。
  所以本轮固定用标定位置,主要对照【夹持力】,外加一个略低的位置做交叉验证。

本脚本对几组 (z偏移, 夹持力) 组合各跑一次 grasp -> place,看:
    grasp_ok    抓起来没有
    gw          闭合开度(越接近球径 6.3cm 说明夹在越粗的位置)
    in_basket   ★ 真正的判据:搬到篮子里时东西还在不在(掉了就 False)

★ 不改 grasp_poses.json —— 偏移在内存里临时合成,你的标定文件原封不动。

运行:
  conda activate env_isaaclab
  ./isaaclab.sh -p brainary/sim/test_orange_grip.py --headless --enable_cameras
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--obj", default="Prop_orange_01")
ap.add_argument("--basket", default="Prop_KLT_3")
ap.add_argument("--speed", type=float, default=1.0)
# "z偏移(米):夹持力(N)" 列表。0.035=标定位置(实测≈赤道), 20N=现状默认力
ap.add_argument("--combos", default="0.035:20,0.035:40,0.035:60,0.025:40")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from brainary_api import BrainaryAPI  # noqa: E402


def main():
    sim = BrainaryAPI.launch(headless=True, device=getattr(args, "device", "cuda:0"),
                             camera_resolution=(160, 120), _app_launcher=app_launcher)
    si = sim.sim
    c = si._ctrl
    c._runner_speed = max(0.5, float(args.speed))
    c._apply_speed_to_adapter()

    from runtime.scene_state_provider import PoseState
    from state_machine.skill_test_controller import _GraspPoseRunner

    base = si._resolve_grasp_pose(args.obj)          # 标定 pose(含 +0.035 偏移)
    base_pos = base.pos_w.reshape(3).clone()
    obj_z = float(si.get_object_pose(args.obj)["position"][2])
    print(f"[grip] {args.obj} root_z={obj_z:.4f}  标定抓取点 z={float(base_pos[2]):.4f} "
          f"(高出球心 {float(base_pos[2]) - obj_z:+.4f} m)", flush=True)

    # 量橘子真实包围盒:root pivot 在球心还是球底,直接决定 z 偏移该取多少(别再靠猜)
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        _stage = omni.usd.get_context().get_stage()
        _prim = None
        for _pp in (f"/World/envs/env_0/{args.obj}", f"/World/{args.obj}"):
            if _stage.GetPrimAtPath(_pp).IsValid():
                _prim = _stage.GetPrimAtPath(_pp); break
        if _prim is not None:
            _bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            _r = _bb.ComputeWorldBound(_prim).ComputeAlignedRange()
            _lo, _hi = _r.GetMin(), _r.GetMax()
            _cz = (_lo[2] + _hi[2]) / 2.0
            print(f"[grip] 包围盒 z=[{_lo[2]:.4f}, {_hi[2]:.4f}] 高={_hi[2]-_lo[2]:.4f}m "
                  f"几何中心z={_cz:.4f}  -> root pivot 在中心上方 {obj_z - _cz:+.4f}m", flush=True)
            print(f"[grip] 直径 x={_hi[0]-_lo[0]:.4f} y={_hi[1]-_lo[1]:.4f}m; "
                  f"赤道对应的 z偏移 ≈ {_cz - obj_z:+.4f}m", flush=True)
    except Exception as _e:
        print(f"[grip] 包围盒测量失败: {_e}", flush=True)

    combos = []
    for item in args.combos.split(","):
        z, _, eff = item.strip().partition(":")
        combos.append((float(z), float(eff or 20.0)))

    rows = []
    with torch.inference_mode():
        for zoff, eff in combos:
            print(f"\n===== z偏移={zoff:+.3f}m  夹持力={eff:.0f}N =====", flush=True)
            sim.reset()                               # 每组从干净场景开始(橘子回原位)
            try:
                si.go_home()
            except Exception as e:
                print(f"  go_home: {e}", flush=True)

            # 夹持力:直接改 actuator 的 effort 上限(运行时生效,不用重启场景)
            try:
                robot = si.provider.scene["robot"]
                ids, _ = robot.find_joints("panda_finger_joint.*")
                lim = torch.full((robot.num_instances, len(ids)), float(eff), device=robot.device)
                robot.write_joint_effort_limit_to_sim(lim, joint_ids=ids)
                print(f"  夹持力 -> {eff:.0f}N", flush=True)
            except Exception as e:
                print(f"  ⚠ 设置夹持力失败({e}),用启动时的默认值", flush=True)

            # 用【当前】橘子位置 + 指定 z 偏移合成抓取 pose(不动标定文件)
            pos = si.get_object_pose(args.obj)["position"]
            p = torch.tensor([float(pos[0]), float(pos[1]), float(pos[2]) + zoff],
                             dtype=torch.float32, device=c.device)
            pose = PoseState(p, base.quat_w.reshape(4).clone())

            c.executor.reset(); c._pending = None; c._place_runner = None
            c._grasp_runner = _GraspPoseRunner(c.adapter, pose, device=c.device,
                                               speed=c._runner_speed, arrival_z=c._arrival_z())
            r = si._pump(8000, label=f"grasp:{args.obj}")
            held = si._is_holding(args.obj)
            gw = si.get_gripper_width()
            si._gripper_cmd = -1.0
            print(f"  抓取: held={held} gw={gw:.4f} steps={r['steps']}", flush=True)

            in_basket, dist, fz = None, None, None
            if held:
                sim._last_grasped = args.obj          # 绕过了 BrainaryAPI.grasp,手动补上落篮判定要的字段
                rp = dict(sim.place(args.basket))     # ★ 搬运 + 放置:掉了这里就露馅
                in_basket = rp.get("object_in_basket")
                op = si.get_object_pose(args.obj)     # 独立复测:落点离篮心多远、多高
                bxy = rp.get("basket_xy")
                if op is not None and bxy is not None:
                    fx, fy, fz = (float(v) for v in op["position"])
                    dist = ((fx - bxy[0]) ** 2 + (fy - bxy[1]) ** 2) ** 0.5
                print(f"  放置: ok={rp.get('ok')} in_basket={in_basket} steps={rp.get('steps')} "
                      f"离篮心={'-' if dist is None else round(dist,3)}m 落点z={'-' if fz is None else round(fz,3)}",
                      flush=True)
            rows.append((zoff, eff, held, round(gw, 4), in_basket, dist, fz))

    print("\n========== 对照结果 ==========", flush=True)
    print(f"  {'z偏移':>8} {'夹持力':>6} {'抓起':>5} {'开度':>7} {'落篮':>6} {'离篮心m':>8} {'落点z':>7}", flush=True)
    for z, e, h, gw, ib, d, fz in rows:
        print(f"  {z:+8.3f} {e:5.0f}N {str(h):>5} {gw:7.4f} {str(ib):>6} "
              f"{'-' if d is None else format(d,'8.3f')} {'-' if fz is None else format(fz,'7.3f')}", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
