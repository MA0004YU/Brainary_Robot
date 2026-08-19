"""诊断 approach 相位 8cm 停滞的根因,headless。

假设(待证/证伪):`_GraspPoseRunner._line_setpoint` 把胡萝卜锚在【当前 TCP】上(c 投影 + lead),
所以只要手臂有稳态跟踪滞后(PD 在重力下的 droop),命令点就跟着滞后 -> 目标永远超不过手臂,
droop ≈ lead 时前进量恰好为 0 -> 固定距离停滞。官方 IK-Abs 命令的是【绝对目标位姿】,没有这个模式。

判据(逐帧打印):
  d_goal   TCP 到抓取点距离      —— 停滞时不再下降
  d_cmd    TCP 到本帧命令点距离  —— 若≈lead 恒定,说明命令点在跟着 TCP 跑
  droop    |q_des - q_actual|    —— 稳态非零 = PD 追不上,存在滞后
  dTCP     本帧 TCP 实际位移      —— 停滞时 ≈ 0

运行:
  conda activate env_isaaclab
  ./isaaclab.sh -p brainary/sim/diag_approach_stall.py --headless --enable_cameras [--obj Prop_011_banana]
"""
import argparse, sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--obj", default="Prop_011_banana")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--max_steps", type=int, default=900)
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

    from state_machine.skill_test_controller import _GraspPoseRunner

    pose_w = si._resolve_grasp_pose(args.obj)
    print(f"[diag] obj={args.obj} grasp_pos={pose_w.pos_w.reshape(3).tolist()}", flush=True)

    with torch.inference_mode():
        si.go_home()
        c.executor.reset(); c._pending = None; c._place_runner = None
        runner = _GraspPoseRunner(c.adapter, pose_w, device=c.device,
                                  speed=c._runner_speed, arrival_z=c._arrival_z())
        c._grasp_runner = runner
        print(f"[diag] lead={runner.lead:.4f}m standoff_pos={runner.above_pos.tolist()} "
              f"max_joint_step={c.adapter.max_joint_step}", flush=True)
        # 目标离机器人底座多远(判是否真在臂展边界) + 接近轴方向(判侧抓/俯抓)
        base = c.adapter.robot.data.root_pos_w[0].reshape(3).to(c.device)
        reach = float(torch.linalg.norm(runner.grasp_pos - base))
        print(f"[diag] base={base.tolist()} reach_to_goal={reach:.3f}m "
              f"approach_dir={runner.approach_dir.tolist()}", flush=True)
        lo, hi = c.adapter._joint_lower, c.adapter._joint_upper

        prev_tcp = None
        for i in range(int(args.max_steps)):
            state = si.provider.get_state()
            tcp = state.robot.tcp_pose.pos_w.reshape(3).to(c.device)
            phase_before = runner.phase
            q_act = c.adapter.robot.data.joint_pos[0, c.adapter._joint_ids].clone()

            action = c.step(si.session)
            if action is None:
                print(f"[diag] step {i}: controller returned None (phase={runner.phase})", flush=True)
                break

            # 从 action 反解出本帧命令的 q_des(raw*scale+offset),量化 PD 滞后
            layout = si.provider._resolve_joint_action()
            raw_arm = action[0, layout["arm_start"]: layout["arm_start"] + layout["arm_dim"]]
            q_des = raw_arm * layout["scale"] + layout["offset"]
            droop = float(torch.linalg.norm(q_des - q_act))

            d_goal = float(torch.linalg.norm(tcp - runner.grasp_pos))
            d_move = 0.0 if prev_tcp is None else float(torch.linalg.norm(tcp - prev_tcp))
            prev_tcp = tcp.clone()

            if phase_before == "approach" and i % 5 == 0:
                # 复算本帧胡萝卜命令点(与 runner 内部同一公式)
                cmd = runner._line_setpoint(tcp, runner.above_pos, runner.grasp_pos)
                d_cmd = float(torch.linalg.norm(tcp - cmd))
                # 有没有关节顶到限位(真臂展边界的标志)
                near = [f"j{k+1}" for k in range(q_act.numel())
                        if min(float(q_act[k] - lo[k]), float(hi[k] - q_act[k])) < 0.05]
                # 接触力:哪个连杆被挡住(停滞若因手臂压在桌面/支架上,这里会有非零力)
                con = ""
                try:
                    hot = [(n, round(float(f), 1)) for n, f in si.get_contact_forces() if float(f) > 1.0]
                    con = f" contact={hot}"
                except Exception as exc:
                    con = f" contact_err={type(exc).__name__}: {exc}"
                print(f"[diag] t={i:4d} phase=approach d_goal={d_goal*100:6.2f}cm "
                      f"d_cmd={d_cmd*100:5.2f}cm droop={droop:.4f}rad dTCP={d_move*1000:5.2f}mm "
                      f"at_limit={near}{con}", flush=True)
            elif phase_before != runner.phase:
                print(f"[diag] t={i:4d} phase {phase_before} -> {runner.phase} "
                      f"(d_goal={d_goal*100:.2f}cm)", flush=True)

            # 一旦 panda_hand 出现明显接触力,把【手附近的场景刚体】按距离列出来 -> 指认挡路的是谁
            if phase_before == "approach" and not getattr(main, "_dumped", False):
                try:
                    hot = [(n, float(f)) for n, f in si.get_contact_forces() if float(f) > 5.0]
                except Exception:
                    hot = []
                if hot:
                    main._dumped = True
                    hand_i = c.adapter._body_idx
                    hand_p = c.adapter.robot.data.body_pos_w[0, hand_i].reshape(3)
                    print(f"[diag] ---- 接触发生: {hot}; panda_hand@{hand_p.tolist()} ----", flush=True)
                    scene = si.provider.scene
                    near = []
                    for key in scene.keys():
                        try:
                            p = scene[key].data.root_pos_w[0].reshape(3)
                        except Exception:
                            continue
                        d = float(torch.linalg.norm(p - hand_p))
                        if d < 0.40:
                            near.append((d, key, [round(float(x), 3) for x in p]))
                    for d, key, p in sorted(near):
                        print(f"[diag]   d={d*100:6.1f}cm  {key:28} pos={p}", flush=True)

            si.env.step(action)
            if runner.done:
                print(f"[diag] runner done at t={i} failed={runner.failed}", flush=True)
                break

    sim.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
