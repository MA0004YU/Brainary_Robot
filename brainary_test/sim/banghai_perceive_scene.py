#!/usr/bin/env python3
"""加载感知(Qwen :5601),输入当前场景的 5 视角,报识别结果。"""
import sys, argparse, base64, io, json, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS = ["cube","banana","orange","lemon","pomegranate","mug","cup","bowl","box","can",
          "bottle","scissors","clamp","knife","basket","KLT bin","cabinet","drawer",
          "coffee machine","robot arm","table"]

def _b64(rgb):
    import numpy as np; from PIL import Image
    b=io.BytesIO(); Image.fromarray(np.asarray(rgb)[...,:3].astype("uint8")).save(b,"PNG")
    return base64.b64encode(b.getvalue()).decode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="http://127.0.0.1:5601")
    ap.add_argument("--out", default="logs/perceive_scene")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap); a = ap.parse_args()
    a.headless = True; a.enable_cameras = True
    app = AppLauncher(a); simapp = app.app
    import numpy as np
    from PIL import Image
    from sim_interface import SimInterface, CAMERAS
    sim = SimInterface.launch(headless=True, device=a.device, _app_launcher=app)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cams = sim.get_all_cameras(require_depth=False)
    images = []
    for name in CAMERAS:
        v = cams.get(name)
        if v is None or v["rgb"] is None: continue
        Image.fromarray(np.asarray(v["rgb"])[...,:3].astype("uint8")).save(out/f"{name}.png")
        images.append({"name": name, "b64": _b64(v["rgb"])})
    print(f"[perceive] captured views: {[i['name'] for i in images]}", flush=True)

    payload = {"images": images, "candidate_labels": LABELS, "save_memory": False}
    req = urllib.request.Request(a.addr.rstrip("/")+"/recognize",
        data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    print("[perceive] calling Qwen /recognize (5 views) ...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
    except Exception as exc:
        print("[perceive] ERROR:", exc, flush=True); sim.close(); simapp.close(); return 1

    print("\n================= 5 视角识别结果 =================", flush=True)
    for res in resp.get("results", []):
        print(f"\n[{res['name']}] primary_label={res['primary_label']} conf={res['confidence']}", flush=True)
        print(f"  scene: {res.get('scene','')}", flush=True)
        for o in res.get("objects", []):
            print(f"    - {o.get('name')} ({o.get('confidence')}): {o.get('evidence','')}", flush=True)
    (out/"result.json").write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[perceive] latency={resp.get('latency_s')}s  saved -> {out}", flush=True)
    sim.close(); simapp.close(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
