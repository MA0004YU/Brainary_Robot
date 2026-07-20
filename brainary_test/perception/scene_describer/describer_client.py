# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""isaac 进程内客户端:从 SceneSession 抓两路 RGB + 场景状态,base64 打包,POST 到 VLM server。

零第三方依赖(只 stdlib urllib + PIL,PIL 在 isaac env 已装,与 eval 一致)。不创建 SkillExecutor,
不动控制 -- 纯只读取数,可与正在跑的 SkillTestController/BrainInterface 共存。

用法:
    from franka_v1_skill_lab.scene_describer.describer_client import build_payload, describe
    payload = build_payload(session)            # 必须在仿真主线程调(相机/物理读取非线程安全)
    result  = describe(payload, addr="http://127.0.0.1:5599")   # HTTP,可放后台线程
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request

_ARTICULATIONS = ("cabinet", "microwave", "coffee_machine", "dishwasher")


def _b64_png(rgb) -> str:
    """numpy HxWx3 uint8 -> base64 PNG(与 eval_pi05 的 _b64_png 一致)。"""
    import numpy as np
    from PIL import Image

    arr = np.asarray(rgb)[..., :3].astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cap_rgb(cam):
    if cam is None:
        return None
    try:
        return cam.capture(require_depth=False).get("rgb")
    except Exception:
        return None


def _flist(t):
    return [float(v) for v in t.detach().cpu().tolist()]


def collect_state(session) -> dict:
    """从 session/provider 抓数值状态(机器人/物体/把手/家电关节)。纯只读。"""
    provider = session.provider
    st = provider.get_state()
    state: dict = {"robot": {}, "objects": {}, "handles": {}, "articulations": {}}

    # robot
    try:
        state["robot"] = {
            "ee_pose_world": {
                "pos": _flist(st.robot.tcp_pose.pos_w),
                "quat_wxyz": _flist(st.robot.tcp_pose.quat_w),
            },
            "gripper_width_m": float(st.robot.gripper_width),
            "arm_joints_rad": _flist(provider.arm_joint_pos(st)),
        }
    except Exception as exc:
        state["robot"] = {"error": str(exc)}

    # tracked objects (cube_1/2/3, knife, ...)
    try:
        for name in st.objects:
            obj = st.objects[name]
            state["objects"][name] = {
                "pos": _flist(obj.pose.pos_w),
                "quat_wxyz": _flist(obj.pose.quat_w),
            }
    except Exception as exc:
        state["objects"] = {"error": str(exc)}

    # handles (door/drawer/lever) in robot-base frame
    try:
        raw = session.handle_poses_in_base()
        for name, h in raw.items():
            state["handles"][name] = {
                "position_base": h.get("position"),
                "quat_wxyz_base": h.get("quat_wxyz"),
                "calibrated": h.get("calibrated"),
            }
    except Exception as exc:
        state["handles"] = {"error": str(exc)}

    # appliance joints (drawer travel m / door angle rad)
    try:
        scene = provider.scene
        keys = set(scene.keys())
        for asset_name in _ARTICULATIONS:
            if asset_name not in keys:
                continue
            asset = scene[asset_name]
            try:
                jnames = list(asset.joint_names)
                jpos = _flist(asset.data.joint_pos[0])
                state["articulations"][asset_name] = {
                    n: round(p, 4) for n, p in zip(jnames, jpos)
                }
            except Exception as exc:
                state["articulations"][asset_name] = {"error": str(exc)}
    except Exception as exc:
        state["articulations"] = {"error": str(exc)}

    return state


def build_payload(session) -> dict:
    """抓两路 RGB + 状态,组装 POST body。务必在仿真主线程调用。"""
    front = _cap_rgb(session.front_cam)
    wrist = _cap_rgb(session.wrist_cam)
    if front is None or wrist is None:
        raise RuntimeError("front/wrist RGB unavailable (cameras off?). launch with cameras enabled.")
    return {
        "rgb_front_b64": _b64_png(front),
        "rgb_wrist_b64": _b64_png(wrist),
        "state": collect_state(session),
    }


def describe(payload: dict, addr: str = "http://127.0.0.1:5599", timeout: float = 300.0) -> dict:
    """POST payload 到 VLM server 的 /describe,返回解析后的 dict。HTTP 可放后台线程。"""
    url = addr.rstrip("/") + "/describe"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
