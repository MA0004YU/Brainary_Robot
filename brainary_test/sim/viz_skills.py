#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技能执行【可视化】测试 —— 用和 UI 完全相同的 BrainaryAPI 调用控制机器人抓/放,
并在 抓取前 / 抓到后 / 放进篮子后 各渲染一张画面(front + top),拼成故事板。

产物:output/viz_<ts>/frames/*.png  +  一份 viz_manifest.json(每步成败 + 帧路径)。

运行(IsaacLab 根目录,先 conda activate env_isaaclab):
    ./isaaclab.sh -p brainary/sim/viz_skills.py
    ./isaaclab.sh -p brainary/sim/viz_skills.py --only Prop_037_scissors
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_SIM = Path(__file__).resolve().parent
_BRAINARY = _SIM.parent
sys.path.insert(0, str(_SIM)); sys.path.insert(0, str(_BRAINARY))


def main() -> int:
    ap = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--only", default=None)
    ap.add_argument("--max_steps", type=int, default=8000)
    args, _ = ap.parse_known_args()
    args.enable_cameras = True
    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    from PIL import Image
    import torch
    import run_brainary as rb
    from brainary_api import BrainaryAPI

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = _BRAINARY / "output" / f"viz_{ts}"
    fdir = outdir / "frames"; fdir.mkdir(parents=True, exist_ok=True)

    print("[viz] 启动 Isaac ...", flush=True)
    sim = BrainaryAPI.launch(headless=True, device=args.device, seed=1, _app_launcher=app_launcher)
    graspable = sim.list_graspable(); baskets = sim.list_baskets()
    if args.only:
        graspable = [o for o in graspable if o == args.only]

    def snap(tag: str):
        """抓 front + top 两张,存盘,返回相对路径。"""
        paths = {}
        for cam in ("front", "top"):
            try:
                rgb = sim.get_rgb(cam)
                if rgb is not None:
                    p = fdir / f"{tag}_{cam}.png"
                    Image.fromarray(np.asarray(rgb)[..., :3].astype("uint8")).save(p)
                    paths[cam] = f"frames/{p.name}"
            except Exception as exc:
                print(f"[viz]   snap {cam} 失败: {exc}", flush=True)
        return paths

    steps = []
    with torch.inference_mode():
        snap("00_scene")
        for i, obj in enumerate(graspable, 1):
            cat = rb._cat_of(obj); basket = rb._SORT_RULES.get(cat, baskets[0] if baskets else None)
            tagp = f"{i:02d}_{obj}"
            print(f"\n[viz] === {i}/{len(graspable)} {obj} -> {basket} ===", flush=True)
            fr = {"object": obj, "basket": basket, "frames": {}}
            fr["frames"]["before"] = snap(f"{tagp}_1before")
            g = dict(sim.grasp(obj, max_steps=args.max_steps))
            fr["grasp_ok"] = bool(g.get("ok")); fr["grasp_reason"] = g.get("reason")
            print(f"[viz]   grasp ok={g.get('ok')} {g.get('reason') or ''}", flush=True)
            fr["frames"]["after_grasp"] = snap(f"{tagp}_2grasp")
            if g.get("ok"):
                p = dict(sim.place(basket, max_steps=args.max_steps))
                fr["place_ok"] = bool(p.get("ok")); fr["place_reason"] = p.get("reason")
                fr["in_basket"] = p.get("object_in_basket")
                print(f"[viz]   place ok={p.get('ok')} in_basket={p.get('object_in_basket')}", flush=True)
                fr["frames"]["after_place"] = snap(f"{tagp}_3place")
            try: sim.go_home()
            except Exception: pass
            steps.append(fr)

    done = sum(1 for s in steps if s.get("grasp_ok") and s.get("place_ok"))
    (outdir / "viz_manifest.json").write_text(json.dumps(
        {"ts": ts, "n": len(steps), "done": done, "steps": steps}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[viz] 完成 · 完整成功 {done}/{len(steps)} · 帧+manifest -> {outdir}", flush=True)

    sim.close(); simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
