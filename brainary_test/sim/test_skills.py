#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技能层执行测试 —— 只测规划出的 grasp/place 动作能否在 Isaac 里真控制机器人完成。
不跑感知/记忆/规划/监控:直接对场景里每个可抓物体执行 grasp→place(按类别进篮子),
失败也继续(不 stop_on_failure),逐个打印结果 + 汇总成功率,写 skill_test_result.json。

运行(IsaacLab 根目录,先 conda activate env_isaaclab):
    ./isaaclab.sh -p brainary/sim/test_skills.py
可选:
    ./isaaclab.sh -p brainary/sim/test_skills.py --only Prop_011_banana   # 只测某个物体
    ./isaaclab.sh -p brainary/sim/test_skills.py --max_steps 6000
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_SIM = Path(__file__).resolve().parent          # brainary/sim
_BRAINARY = _SIM.parent                          # brainary
sys.path.insert(0, str(_SIM))
sys.path.insert(0, str(_BRAINARY))


def main() -> int:
    ap = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--only", default=None, help="只测这个物体 id")
    ap.add_argument("--max_steps", type=int, default=8000, help="每个 grasp/place 的最大步数")
    args, _ = ap.parse_known_args()
    args.enable_cameras = True
    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import run_brainary as rb                    # 复用 _cat_of / _SORT_RULES
    from brainary_api import BrainaryAPI

    print("[skill-test] 启动 Isaac 场景 ...", flush=True)
    sim = BrainaryAPI.launch(headless=True, device=args.device, seed=1, _app_launcher=app_launcher)
    graspable = sim.list_graspable()
    baskets = sim.list_baskets()
    if args.only:
        graspable = [o for o in graspable if o == args.only]
    print(f"[skill-test] 可抓物体({len(graspable)}): {graspable}", flush=True)
    print(f"[skill-test] 篮子: {baskets}", flush=True)

    results = []
    with torch.inference_mode():
        for i, obj in enumerate(graspable, 1):
            cat = rb._cat_of(obj)
            basket = rb._SORT_RULES.get(cat, baskets[0] if baskets else None)
            print(f"\n[skill-test] === {i}/{len(graspable)}  grasp {obj}  (类别={cat} -> {basket}) ===", flush=True)
            t = time.time()
            g = dict(sim.grasp(obj, max_steps=args.max_steps))
            print(f"[skill-test]   grasp: ok={g.get('ok')} reason={g.get('reason')} steps={g.get('steps')}", flush=True)
            p = {}
            if g.get("ok"):
                p = dict(sim.place(basket, max_steps=args.max_steps))
                print(f"[skill-test]   place: ok={p.get('ok')} reason={p.get('reason')} "
                      f"in_basket={p.get('object_in_basket')}", flush=True)
            # 每个物体后回 home,复位机械臂再测下一个
            try:
                sim.go_home()
            except Exception as exc:
                print(f"[skill-test]   go_home 异常: {exc}", flush=True)
            results.append({
                "object": obj, "category": cat, "basket": basket,
                "grasp_ok": bool(g.get("ok")), "grasp_reason": g.get("reason"),
                "grasp_steps": g.get("steps"),
                "place_ok": bool(p.get("ok")), "place_reason": p.get("reason"),
                "object_in_basket": p.get("object_in_basket"),
                "sec": round(time.time() - t, 1),
            })

    # 汇总
    n = len(results)
    grasp_ok = sum(1 for r in results if r["grasp_ok"])
    done = sum(1 for r in results if r["grasp_ok"] and r["place_ok"])
    print("\n[skill-test] ================= 技能执行汇总 =================", flush=True)
    for r in results:
        mark = "✅" if (r["grasp_ok"] and r["place_ok"]) else ("🟡抓到没放进" if r["grasp_ok"] else "❌抓取失败")
        print(f"[skill-test]  {mark}  {r['object']:<26} grasp={r['grasp_ok']} place={r['place_ok']} "
              f"| {r['grasp_reason'] or r['place_reason'] or ''}", flush=True)
    print(f"[skill-test]  抓取成功 {grasp_ok}/{n} | 完整放进篮子 {done}/{n}", flush=True)

    out = _BRAINARY / "output" / f"skill_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": {"n": n, "grasp_ok": grasp_ok, "done": done},
                               "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[skill-test]  结果写入: {out}", flush=True)

    sim.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
