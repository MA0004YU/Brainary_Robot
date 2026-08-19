#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线重放:用某次 run 存下的【4 个静态相机 rgb/depth + camera_params + plan】直接喂沙盒
verify_local_plan —— 不需 Isaac、不走 HTTP。用来查看 DINO 【实际识别的物体名 / 尺寸 / 位置】+ DAG 能否模拟,
方便按真实识别结果给 physics_dictionary / 建模逻辑打补丁。

在【brainary_sim】环境跑(和 serve.py 同环境):
    conda activate brainary_sim
    export GSA_PATH=/home1/banghai/Documents/IsaacLab/brainary/simulation/src/perception/third_party/Grounded-Segment-Anything
    export CUDA_VISIBLE_DEVICES=1
    python simulation/test_verify_offline.py brainary/output/<时间戳>

前提:该 run 的 sim/ 目录里要有 camera_params.json(由更新后的 stage_sim 生成;老的 run 没有,需重抓一次)。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ORIG_CWD = Path.cwd()               # 记住启动目录(chdir 前),用于解析用户传入的 run_dir
_DIR = Path(__file__).resolve().parent
os.chdir(_DIR)                       # 同 serve.py:让 panda.urdf 等相对资产可被找到
sys.path.insert(0, str(_DIR))
import verify_client as vc           # 复用完全一致的 外参转 base 系 + 名字桥接 逻辑


def _load_static_views(sim_dir: Path):
    cp_path = sim_dir / "camera_params.json"
    if not cp_path.is_file():
        raise FileNotFoundError(
            f"{cp_path} 不存在。请用更新后的 stage_sim 重抓一次(run_brainary.py 会顺带存相机内外参)。")
    cp = json.loads(cp_path.read_text(encoding="utf-8"))
    cams = cp.get("cameras", {})
    base = cp.get("robot_base_pose_world")
    T_wb = vc._inv_rigid(vc._T4x4(base)) if base else np.eye(4)   # world->base
    if base is None:
        print("[offline] ⚠ 无 robot_base_pose_world -> 外参按世界系(坐标系可能不对齐)", flush=True)

    rgb_views, depth_views, K, T = {}, {}, {}, {}
    for name in vc._STATIC_CAMS:                                  # 只用 front/left/right/top
        rp, dp = sim_dir / "rgb" / f"{name}.png", sim_dir / "depth" / f"{name}.npy"
        if not (rp.is_file() and dp.is_file() and name in cams):
            print(f"[offline] 跳过 {name}(缺 rgb/depth/params)", flush=True)
            continue
        rgb_views[name] = np.asarray(Image.open(rp).convert("RGB"))
        # 沙盒 cg_wrapper 里 depth/1000(它假设【毫米】);Isaac 存的是【米】-> 这里 ×1000 转毫米,
        # 使沙盒 /1000 后还原成正确的米。否则物体被缩成亚毫米、被体积过滤器全删 -> 空清单。
        depth_views[name] = np.load(dp).astype(np.float32) * 1000.0
        K[name] = np.asarray(vc._K3x3(cams[name]["intrinsics"]), dtype=float)
        T[name] = vc.cam_to_base(cams[name]["pose_world"], T_wb)   # OpenGL->OpenCV->base 系
    if not rgb_views:
        raise RuntimeError("没有可用的静态相机帧")
    print(f"[offline] 载入 {len(rgb_views)} 个静态相机(外参已转 base 系): {list(rgb_views)}", flush=True)
    return rgb_views, depth_views, K, T


def _load_plan(run: Path):
    for cand in (run / "planning" / "plan.json", run / "planning" / "planned_actions.json"):
        if cand.is_file():
            d = json.loads(cand.read_text(encoding="utf-8"))
            steps = d.get("plan", d) if isinstance(d, dict) else d
            return list(steps)
    raise FileNotFoundError("找不到 planning/plan.json 或 planned_actions.json")


def main():
    if len(sys.argv) < 2:
        print("用法: python test_verify_offline.py <run_dir>   (含 sim/ 与 planning/)")
        return
    arg = Path(sys.argv[1])
    run = (arg if arg.is_absolute() else _ORIG_CWD / arg).resolve()   # 相对路径按启动目录解析(非 chdir 后的)
    rgb_views, depth_views, K, T = _load_static_views(run / "sim")

    plan = _load_plan(run)
    bridged, changes = vc._bridge_plan_dag(plan)                 # plan target -> 沙盒 DINO label
    if changes:
        print("[offline] 名字桥接: " + ", ".join(f"{i}:{o}->{n}" for i, o, n in changes), flush=True)

    print("[offline] 构造 PhysicalValidator 并预演(会打印 DINO 实体清单 + DAG 执行)...", flush=True)
    from engine_interface import PhysicalValidator
    validator = PhysicalValidator(config_path=str(_DIR / "config" / "global_config.yaml"))
    res = validator.verify_local_plan(rgb_views=rgb_views, depth_views=depth_views,
                                      dynamic_intrinsics=K, dynamic_extrinsics=T, plan_dag=bridged)
    res = {k: v for k, v in dict(res).items() if k != "physics_raw_data"}
    print("\n===================== verify 结果 =====================", flush=True)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
