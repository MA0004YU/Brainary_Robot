#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PhysicalValidator 接入适配器 —— 把 Brainary 的 plan + 相机数据喂给物理沙盒校验。

契约(来自 simulation 作者):
  在大模型每次生成 plan 后,把 plan_dag + 多视角 RGB/深度/相机内外参 传给
  `PhysicalValidator.verify_local_plan(...)`;引擎即时重构局部物理沙盒预演:
    - success == True  -> 直接把 plan 下发下游(PM/执行);
    - success == False -> 把返回的 llm_reflection_prompt 原样喂回 planning agent 做 replan。

⚠️ 运行前提(见 INTEGRATION.md / WEIGHTS_DOWNLOAD.md):
  1) 必须在 simulation 的【独立 CUDA 环境】里跑(sapien/pytorch3d/nvdiffrast/GroundingDINO/SAM,
     见 requirements.txt)——不能装进 env_isaaclab,会拖坏 Isaac。
  2) 需先下载权重到 weights/cg_weights/(groundingdino_swint_ogc.pth + sam_vit_h_4b8939.pth)。
  3) 需设环境变量 GSA_PATH 指向 Grounded-Segment-Anything。
  因此本适配器在 env_isaaclab 里 import 会失败(sapien 缺)——这是预期,请在 sim 环境用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# ------------------------------------------------------------------ 相机数据采集
def _intrinsics_to_K(intr: Any) -> np.ndarray:
    """相机内参 -> 3x3 K。兼容:已是 3x3 数组 / {fx,fy,cx,cy} dict。"""
    arr = np.asarray(intr, dtype=float) if not isinstance(intr, dict) else None
    if arr is not None and arr.shape == (3, 3):
        return arr
    if isinstance(intr, dict):
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr["cx"]), float(intr["cy"])
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    raise ValueError(f"无法解析相机内参格式: {type(intr)} {intr}")


def _pose_to_T(pose: Any) -> np.ndarray:
    """相机位姿 -> 4x4 T_cam_to_world。兼容:已是 4x4 数组 / {position,quat_wxyz} dict。"""
    arr = np.asarray(pose, dtype=float) if not isinstance(pose, dict) else None
    if arr is not None and arr.shape == (4, 4):
        return arr
    if isinstance(pose, dict) and "position" in pose:
        from scipy.spatial.transform import Rotation as R  # sim 环境自带 scipy
        p = np.asarray(pose["position"], dtype=float)
        q = pose.get("quat_wxyz") or pose.get("quat")
        qw, qx, qy, qz = q
        T = np.eye(4)
        T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = p
        return T
    raise ValueError(f"无法解析相机位姿格式: {type(pose)} {pose}")


def gather_camera_inputs(sim: Any) -> Tuple[dict, dict, dict, dict]:
    """从 BrainaryAPI/SimInterface 一次性取多视角 RGB/深度/内参/外参。

    依赖 `sim.get_all_cameras(require_depth=True)` 返回 {cam:{rgb,depth,intrinsics,pose}}。
    返回 (rgb_views, depth_views, dynamic_intrinsics(3x3), dynamic_extrinsics(4x4)),
    键都是相机名,直接喂 verify_local_plan。
    """
    cams = sim.get_all_cameras(require_depth=True)
    rgb_views, depth_views, K, T = {}, {}, {}, {}
    for name, fr in (cams or {}).items():
        if fr.get("rgb") is None or fr.get("depth") is None:
            continue
        rgb_views[name] = np.asarray(fr["rgb"])[..., :3]
        depth_views[name] = np.asarray(fr["depth"])
        K[name] = _intrinsics_to_K(fr.get("intrinsics"))
        T[name] = _pose_to_T(fr.get("pose"))
    if not rgb_views:
        raise RuntimeError("gather_camera_inputs: 没取到任何带深度的相机帧")
    return rgb_views, depth_views, K, T


# ------------------------------------------------------------------ 校验 + replan 环
def verify_plan(validator: Any, sim: Any, plan_dag: List[Dict[str, Any]]) -> Dict[str, Any]:
    """采集当前相机数据 -> 调 verify_local_plan。返回其结果 dict(含 success / llm_reflection_prompt)。"""
    rgb, depth, K, T = gather_camera_inputs(sim)
    return validator.verify_local_plan(
        rgb_views=rgb, depth_views=depth,
        dynamic_extrinsics=T, dynamic_intrinsics=K,
        plan_dag=plan_dag,
    )


def verify_then_replan(validator, sim, planner, planning_input, task, max_replans: int = 2):
    """契约主循环:规划 -> 物理校验 -> 通过则返回 plan;不通过则把 llm_reflection_prompt
    喂回 planner replan,最多重规划 max_replans 次。

    planner: 需支持 `generate_plan(planning_input) -> [{id,action,target,depends_on,...}]`;
             replan 时本函数把上一轮的 reflection 注入 planning_input['physics_reflection'],
             由 planner 的 prompt 消费(planner 侧需读取该字段——见 INTEGRATION.md)。
    返回 (plan, verify_result)。verify_result['success'] 为最终是否通过。
    """
    plan = planner.generate_plan(planning_input)
    for attempt in range(max_replans + 1):
        res = verify_plan(validator, sim, plan)
        if res.get("success"):
            return plan, res
        reflection = res.get("llm_reflection_prompt") or res.get("message", "")
        print(f"[verify] 第{attempt+1}次校验未过,注入反馈 replan:\n{str(reflection)[:300]}", flush=True)
        planning_input = dict(planning_input)
        planning_input["physics_reflection"] = reflection
        plan = planner.generate_plan(planning_input)
    # 用尽次数仍未过:返回最后一版 plan + 最后的校验结果(由调用方决定是否放行)
    return plan, verify_plan(validator, sim, plan)
