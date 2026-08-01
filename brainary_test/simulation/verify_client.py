#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Client for the simulation physics-sandbox HTTP service.

Runs on the ISAAC side (env_isaaclab) -- imports ONLY numpy/PIL/requests, never sapien.
Gathers camera data from BrainaryAPI, POSTs it + the plan to the sim service (default :5600),
returns the verify result {success, llm_reflection_prompt?, message, ...}.
"""
from __future__ import annotations

import base64
import io

import numpy as np


def _b64png(rgb) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb)[..., :3].astype("uint8")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64npy(arr) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr).astype("float32"))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _K3x3(intr):
    if not isinstance(intr, dict):
        a = np.asarray(intr, dtype=float)
        if a.shape == (3, 3):
            return a.tolist()
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _T4x4(pose):
    if not isinstance(pose, dict):
        a = np.asarray(pose, dtype=float)
        if a.shape == (4, 4):
            return a.tolist()
    p = np.asarray(pose["position"], dtype=float)
    q = pose.get("quat_wxyz") or pose.get("quat")
    w, x, y, z = (float(v) for v in q)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T.tolist()


def gather_payload(sim) -> dict:
    """From BrainaryAPI.get_all_cameras() -> {rgb, depth, intrinsics(3x3), extrinsics(4x4)} (base64/lists)."""
    cams = sim.get_all_cameras(require_depth=True)
    rgb, depth, K, T = {}, {}, {}, {}
    for name, fr in (cams or {}).items():
        if fr.get("rgb") is None or fr.get("depth") is None:
            continue
        rgb[name] = _b64png(fr["rgb"])
        depth[name] = _b64npy(fr["depth"])
        K[name] = _K3x3(fr.get("intrinsics"))
        T[name] = _T4x4(fr.get("pose"))
    if not rgb:
        raise RuntimeError("gather_payload: no camera frames with depth")
    return {"rgb": rgb, "depth": depth, "intrinsics": K, "extrinsics": T}


def service_up(addr: str = "http://127.0.0.1:5600") -> bool:
    import requests
    try:
        return requests.get(addr.rstrip("/") + "/health", timeout=3).status_code == 200
    except Exception:
        return False


def verify_via_service(sim, plan_dag, addr: str = "http://127.0.0.1:5600") -> dict:
    """POST plan + camera data to the sim service; return its verify result dict."""
    import requests
    payload = gather_payload(sim)
    payload["plan_dag"] = list(plan_dag)
    r = requests.post(addr.rstrip("/") + "/verify", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()
