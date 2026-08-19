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
import json
import os

import numpy as np

# ============================================================ 名字桥接(plan target -> 沙盒实体 label)
# 物理沙盒有自己独立的一套感知(GroundingDINO + physics_dictionary),把检测到的物体建成 actor,名字是
# 【DINO label 下划线化 + _序号】,如 "yellow_mug_1" / "green_basket_1"。沙盒匹配 plan 节点 target 用
# 【子串匹配】:clean_target(=target.lower().replace(" ","_")) in actor.name。
# 但规划器给的 target 是 scene_describer 的友好名(blue_small_cup / red_package / orange_ball)或分拣
# 篮子 id(Prop_KLT_X)——都不是 DINO label,子串对不上 -> 沙盒判 TARGET_NOT_FOUND。
# 这里在【发送前】把每个 target 改写成对应的 DINO label(下划线),让子串匹配命中。
#
# Prop_ID -> 沙盒 DINO label(必须与 physics_dictionary.py 的 key 下划线化后一致)。
_PROP_TO_LABEL = {
    "Prop_SM_Mug_C1": "yellow_mug",
    "Prop_SM_Mug_D1": "blue_mug",
    "Prop_011_banana": "banana",
    "Prop_orange_01": "orange",
    "Prop_037_scissors": "scissors",
    "Prop_003_cracker_box": "cracker_box",
    "Prop_010_potted_meat_can": "meat_can",
    "Prop_KLT_1": "green_basket",
    "Prop_KLT_2": "blue_basket",
    "Prop_KLT_3": "purple_basket",
}
# 已是 DINO label 的直接放行(下划线形式)。
_KNOWN_LABELS = set(_PROP_TO_LABEL.values())


def _load_aliases() -> dict:
    """读 PM 的 object_aliases.json(友好名/中文名 -> Prop_ID),小写化 key 便于大小写无关匹配。缺失则空。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                     "project_management", "object_aliases.json")
    try:
        raw = json.loads(open(os.path.normpath(p), encoding="utf-8").read())
        return {str(k).strip().lower(): str(v) for k, v in raw.items()}
    except Exception:
        return {}


_ALIAS = _load_aliases()


def _to_sandbox_label(target: str) -> str:
    """把 plan 的 target 名映射成沙盒 DINO label(下划线);映射不到就原样返回(让沙盒自行子串匹配/报错)。"""
    if not target:
        return target
    t = str(target).strip()
    tl = t.lower()
    tu = tl.replace(" ", "_")
    if t in _PROP_TO_LABEL:                       # place 目标 Prop_KLT_X / 规划器直接给 Prop_ID
        return _PROP_TO_LABEL[t]
    if tu in _KNOWN_LABELS:                        # 本就是 DINO label(yellow_mug / scissors …)
        return tu
    prop = _ALIAS.get(tl)                          # 友好名/中文名 -> Prop_ID -> label
    if prop and prop in _PROP_TO_LABEL:
        return _PROP_TO_LABEL[prop]
    return t                                       # 兜底:原样(至少不比现在差)


def _bridge_plan_dag(plan_dag) -> tuple[list, list]:
    """返回(改写后的 plan_dag 深拷贝, [(id, 原 target, 新 target) …] 供日志)。只动 target,不碰依赖/结构。"""
    out, changes = [], []
    for node in (plan_dag or []):
        n = dict(node)
        old = n.get("target")
        new = _to_sandbox_label(old)
        if new != old:
            changes.append((n.get("id"), old, new))
        n["target"] = new
        out.append(n)
    return out, changes


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


# 只用 4 个静态方向相机做多视角融合(排除随臂移动、外参动态、视角刁钻的 wrist)。
_STATIC_CAMS = ("front", "left", "right", "top")


def _inv_rigid(T):
    """4x4 刚体变换求逆: [R|t] -> [R^T | -R^T t]。"""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# Isaac 相机 pose_world 用【ROS body 约定】(实测:光轴=局部 +X、Y 左、Z 上;非 OpenGL 的 -Z)。
# cg_wrapper back-project 用 OpenCV(X 右、Y 下、Z=depth 前)。要把 OpenCV 点云表达到相机局部系,需右乘
# M(opencv->body): opencv +Z(前)->局部 +X, opencv +X(右)->局部 -Y, opencv +Y(下)->局部 -Z。
# 否则点云投到错误方向 -> 物体整体偏移几米。M 为正交旋转(det=+1)。
_CV2CAM = np.array([[0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])


def cam_to_base(pose_world, T_wb):
    """相机 pose_world(ROS body)-> OpenCV 点云对齐 -> base 系 4x4(cam->base 外参),供沙盒 back-project 用。"""
    T_cw = np.asarray(_T4x4(pose_world), dtype=float)          # camera(body)-> world
    return np.asarray(T_wb, dtype=float) @ (T_cw @ _CV2CAM)    # world->base @ (cam->world @ opencv->cam)


def gather_payload(sim) -> dict:
    """From get_all_cameras() -> {rgb, depth, intrinsics(3x3), extrinsics(4x4)}(base64/lists)。

    ★ 外参转到【机械臂 base 系】:沙盒里机械臂加载在原点,但 Isaac 世界原点【不在】base(base 约在
      (2.x,2.x,~0.9))。若直接用世界系外参,物体会被重建到离机械臂几米远处(全 unreachable)。
      这里对每个相机外参左乘 inv(base->world),使重建出的物体都相对机械臂 base,沙盒机械臂(原点)自动一致,
      运动规划(mplib,默认在 base 系)也天然对齐。base 位姿= articulation root(panda_link0)的世界位姿。
    ★ 只用 4 个静态方向相机(front/left/right/top),排除 wrist。"""
    cams = sim.get_all_cameras(require_depth=True)
    # base(articulation root)世界位姿 -> base->world 4x4 -> 求逆得 world->base
    T_wb = np.eye(4)
    try:
        base = sim.get_object_pose("robot")
        if base:
            T_wb = _inv_rigid(_T4x4(base))
    except Exception:
        print("[verify_client] 取机械臂 base 位姿失败 -> 外参按世界系发送(坐标系可能不对齐!)", flush=True)
    rgb, depth, K, T = {}, {}, {}, {}
    for name, fr in (cams or {}).items():
        if name not in _STATIC_CAMS:                       # 排除 wrist 等非静态方向相机
            continue
        if fr.get("rgb") is None or fr.get("depth") is None:
            continue
        rgb[name] = _b64png(fr["rgb"])
        # 沙盒 cg_wrapper 假设深度是【毫米】(内部 /1000);Isaac 给的是【米】-> ×1000 转毫米,
        # 否则物体被缩成亚毫米、被体积过滤器删光 -> 空实体清单。
        depth[name] = _b64npy(np.asarray(fr["depth"], dtype=np.float32) * 1000.0)
        K[name] = _K3x3(fr.get("intrinsics"))
        T[name] = cam_to_base(fr.get("pose"), T_wb).tolist()   # OpenGL->OpenCV->base 系
    if not rgb:
        raise RuntimeError("gather_payload: no static camera frames with depth")
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
    bridged, changes = _bridge_plan_dag(plan_dag)      # plan target -> 沙盒 DINO label(名字桥接)
    if changes:
        print("[verify_client] 名字桥接: " + ", ".join(f"{i}:{o}->{n}" for i, o, n in changes), flush=True)
    payload["plan_dag"] = bridged
    r = requests.post(addr.rstrip("/") + "/verify", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()
