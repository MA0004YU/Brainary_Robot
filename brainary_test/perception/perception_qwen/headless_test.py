#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""无头端到端测试:启动场景(headless) -> 抓 front/wrist 两路 RGB -> 存图 + 取仿真 GT 物体清单 ->
POST 到 Qwen server /recognize -> 打印 + 落盘结果。

需要 Qwen server 已在跑(默认 :5601)。GPU 上会和已加载的 Qwen 模型共用显存。

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/perception_qwen/headless_test.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]   # projects/
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402
from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode  # noqa: E402

# 机器人前方一排水果道具(和 test_mode_ui 默认一致),给相机里放点可识别的东西。
DEFAULT_PROP_URLS = [
    "SapienAssetPipeline/usd_assets/SimReadyProps/lemon_01/lemon_01_base.usd",
    "SapienAssetPipeline/usd_assets/SimReadyProps/orange_01/orange_01_base.usd",
    "SapienAssetPipeline/usd_assets/SimReadyProps/pomegranate01/pomegranate01_base.usd",
    "SapienAssetPipeline/usd_assets/SimReadyProps/orange_02/orange_02_base.usd",
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Headless perception end-to-end test (Qwen).")
    ap.add_argument("--qwen_addr", default="http://127.0.0.1:5601")
    ap.add_argument("--out_dir", default="logs/perception_qwen_test")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--settle", type=int, default=15, help="hold steps after reset so cameras render")
    return ap


def main() -> int:
    ap = build_arg_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    rc = _run(args, app_launcher, simulation_app)
    simulation_app.close()
    return rc


def _run(args, app_launcher, simulation_app) -> int:
    import numpy as np
    from PIL import Image

    from franka_v1_skill_lab.scene_interface import SceneSession

    items = []
    for i, url in enumerate(DEFAULT_PROP_URLS):
        items.append({
            "name": f"Prop_fruit_{i}", "usd_path": url,
            "pos": (0.45, -0.20 + 0.12 * i, 0.0),
            "scale": (1.0, 1.0, 1.0), "rigid": True, "ground": True,
        })

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=args.task if hasattr(args, "task") else V1_BASE_TASK_ID,
        device=args.device, headless=True,
        enable_cameras=True, enable_fp=True,
        free_microwave_door=False, load_latest_scene=True,
        add_microwave_stand=False, replace_microwave_with_fridge=True,
        lock_knife=True, enable_collision_monitor=False,
        spawn_init_markers=False, refine_handle_collisions=True,
        apply_saved_camera_offsets=True, control_hz=50.0, reset_mode=ResetMode.STATIC,
        seed=args.seed, extra_assets=tuple(items), add_robot_stand=True,
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)

    # 把保存的相机局部位姿设回(所见即所得,和 test_mode_ui 一致)
    try:
        from franka_v1_skill_lab.scene_interface import camera_offsets

        camera_offsets.apply_saved_offsets_runtime(session.env)
    except Exception as exc:  # pragma: no cover
        print(f"[test] WARNING: apply saved camera offsets failed: {exc}", flush=True)

    session.reset(seed=args.seed)
    for _ in range(int(args.settle)):
        session.hold()

    from franka_v1_skill_lab.perception_qwen.qwen_client import build_qwen_payload, recognize
    from franka_v1_skill_lab.scene_describer.describer_client import _cap_rgb, collect_state

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    front = _cap_rgb(session.front_cam)
    wrist = _cap_rgb(session.wrist_cam)
    if front is None or wrist is None:
        print("[test] ERROR: cameras returned no RGB", flush=True)
        session.close()
        return 1
    Image.fromarray(np.asarray(front)[..., :3].astype("uint8")).save(out / "front.png")
    Image.fromarray(np.asarray(wrist)[..., :3].astype("uint8")).save(out / "wrist.png")

    gt = collect_state(session)
    (out / "gt_state.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[test] saved front.png / wrist.png / gt_state.json -> {out}", flush=True)
    print(f"[test] GT objects in scene: {list(gt.get('objects', {}).keys())}", flush=True)
    print(f"[test] GT articulations: {json.dumps(gt.get('articulations', {}), ensure_ascii=False)}", flush=True)

    print("[test] calling Qwen /recognize ...", flush=True)
    payload = build_qwen_payload(session)
    res = recognize(payload, addr=args.qwen_addr)
    (out / "qwen_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=================== QWEN RESULT ===================", flush=True)
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    print("==================================================", flush=True)

    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
