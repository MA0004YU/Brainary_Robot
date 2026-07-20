# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""isaac 进程内客户端:抓两路 RGB -> b64 -> POST 到 Qwen server /recognize -> 解析。

零第三方依赖(stdlib urllib + PIL)。复用 scene_describer 已写好的相机抓帧/编码助手,避免重复。
build_qwen_payload 必须在仿真主线程调(相机读取非线程安全);recognize 的 HTTP 可放后台线程。
"""

from __future__ import annotations

import json
import urllib.request

from franka_v1_skill_lab.scene_describer.describer_client import _b64_png, _cap_rgb

# 场景里可能出现的物体类别词表(给 Qwen 当 candidate_labels;传 None/[] 则开放式识别)。
# 覆盖抓取目标 + 默认加载的 YCB/道具大类。可在 UI/调用处覆盖。
DEFAULT_LABELS = [
    "cube", "knife", "mug", "bowl", "banana", "bottle", "can", "box",
    "scissors", "clamp", "pitcher", "beaker", "fridge", "cabinet", "drawer",
]


def build_qwen_payload(session, labels=None, save_memory: bool = True) -> dict:
    """抓 front+wrist 两路 RGB,组装 /recognize 的 POST body。务必主线程调用。"""
    front = _cap_rgb(session.front_cam)
    wrist = _cap_rgb(session.wrist_cam)
    if front is None or wrist is None:
        raise RuntimeError("front/wrist RGB unavailable (cameras off?).")
    return {
        "images": [
            {"name": "front", "b64": _b64_png(front)},
            {"name": "wrist", "b64": _b64_png(wrist)},
        ],
        "candidate_labels": DEFAULT_LABELS if labels is None else list(labels),
        "save_memory": bool(save_memory),
    }


def recognize(payload: dict, addr: str = "http://127.0.0.1:5601", timeout: float = 300.0) -> dict:
    """POST 到 Qwen server /recognize,返回解析后的 dict。HTTP 可放后台线程。"""
    url = addr.rstrip("/") + "/recognize"
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
