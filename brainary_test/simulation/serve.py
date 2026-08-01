#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulation physics-sandbox HTTP service (runs in the ISOLATED brainary_sim env, port 5600).

Wraps PhysicalValidator.verify_local_plan so the Isaac-side pipeline (env_isaaclab) can call it
over HTTP WITHOUT importing sapien -> total env isolation, zero impact on the other modules,
no driver changes (same shared kernel driver; only userspace Python packages differ, per-env).

Launch (in a SEPARATE terminal, brainary_sim env):
    conda activate brainary_sim
    export CUDA_HOME=/home1/banghai/miniconda3/envs/brainary_sim
    export GSA_PATH=/home1/banghai/Documents/IsaacLab/brainary/simulation/src/perception/third_party/Grounded-Segment-Anything
    # optional: pick a GPU that Isaac is NOT using, e.g. the 2nd card
    export CUDA_VISIBLE_DEVICES=1
    cd /home1/banghai/Documents/IsaacLab/brainary/simulation
    python serve.py                      # listens on 0.0.0.0:5600

Endpoints:
    GET  /health  -> {"ok": true, "ready": <models loaded>}
    POST /verify  -> body {rgb, depth, intrinsics, extrinsics, plan_dag}; returns
                     {success, llm_reflection_prompt?, message, cost_sec?}
"""
import base64
import io
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
# 该 simulation 模块所有资产/配置(panda.urdf、srdf、config、inputs/ 调试图)都用【相对 CWD】的路径
# (robot_controller.py 里 os.path.abspath(相对路径))。无论从哪个目录启动 serve.py,都强制把工作目录
# 切到本模块根,否则从 IsaacLab 根跑会拼出 IsaacLab/assets/... 找不到 -> FileNotFoundError: panda.urdf。
os.chdir(_DIR)

_VALIDATOR = None


def _get_validator():
    global _VALIDATOR
    if _VALIDATOR is None:
        import torch  # noqa: F401  (ensure torch/CUDA loaded first)
        from engine_interface import PhysicalValidator
        _VALIDATOR = PhysicalValidator(config_path=str(_DIR / "config" / "global_config.yaml"))
    return _VALIDATOR


def _decode(payload):
    import numpy as np
    from PIL import Image
    rgb = {k: np.asarray(Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB"))
           for k, v in (payload.get("rgb") or {}).items()}
    depth = {k: np.load(io.BytesIO(base64.b64decode(v)))
             for k, v in (payload.get("depth") or {}).items()}
    K = {k: np.asarray(v, dtype=float) for k, v in (payload.get("intrinsics") or {}).items()}
    T = {k: np.asarray(v, dtype=float) for k, v in (payload.get("extrinsics") or {}).items()}
    return rgb, depth, K, T, payload.get("plan_dag") or []


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send(200, {"ok": True, "ready": _VALIDATOR is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/verify":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            rgb, depth, K, T, plan = _decode(payload)
            print(f"[sim-serve] /verify: {len(rgb)} views, {len(plan)} plan nodes -> running sandbox ...",
                  flush=True)
            res = _get_validator().verify_local_plan(
                rgb_views=rgb, depth_views=depth,
                dynamic_extrinsics=T, dynamic_intrinsics=K, plan_dag=plan)
            res = {k: v for k, v in dict(res).items() if k != "physics_raw_data"}
            print(f"[sim-serve] /verify -> success={res.get('success')}", flush=True)
            self._send(200, res)
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"success": False, "error": str(exc)})

    def log_message(self, *_a):
        pass


def main():
    port = int(os.environ.get("SIM_VERIFY_PORT", "5600"))
    print("[sim-serve] loading PhysicalValidator (GroundingDINO + SAM + SAPIEN) ...", flush=True)
    _get_validator()
    print(f"[sim-serve] READY on 0.0.0.0:{port}  (POST /verify, GET /health)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
