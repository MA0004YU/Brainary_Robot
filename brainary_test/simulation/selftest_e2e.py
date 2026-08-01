#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全链路自检：SAPIEN 离屏渲染一张合成 RGB-D -> PhysicalValidator.verify_local_plan。

不依赖 Isaac、不依赖真机相机，装完环境后用它确认整条链路(GDINO+SAM 感知 -> 早融合建孪生世界
-> DAG 物理预演 -> 结构化裁决)是通的。

用法:
    conda activate brainary_sim
    cd simulation
    export GSA_PATH=$PWD/src/perception/third_party/Grounded-Segment-Anything
    python selftest_e2e.py

预期: 打印实体清单(3 个刚体) + 最终裁决 `PASS` / `success = True`，不应出现 Segmentation fault。

把 `HALF` 调到 0.035(70mm 积木)可以走到另一条分支：夹爪 80mm 极限下下探会压坏积木，
返回 `REPLAN_REQUIRED` + `llm_reflection_prompt`——契约里喂回规划 Agent 反思的那条路。
"""
import json
import os
import sys

import numpy as np

W, H = 640, 480
HALF = 0.022        # 积木半边长 -> 44mm。夹爪极限 80mm，太大(>65mm)下探时会压坏物体
# 三个积木的标签必须落在 PHYSICS_DICTIONARY 里，否则 _clean_label 会把它们全丢掉
CUBES = [
    ("red_cube", [0.9, 0.05, 0.05], [0.45, -0.12, HALF]),
    ("blue_cube", [0.05, 0.1, 0.9], [0.45, 0.10, HALF]),
    ("green_cube", [0.05, 0.8, 0.1], [0.62, -0.01, HALF]),
]


def render_synthetic_view():
    """离屏渲染一张三色积木的 RGB-D，返回 (rgb, depth_mm, K, T_cam2world)。"""
    import sapien

    scene = sapien.Scene()
    scene.set_timestep(1 / 500.0)
    scene.add_ground(0.0)
    scene.set_ambient_light([0.6, 0.6, 0.6])
    scene.add_directional_light([0.2, 0.3, -1], [1.2, 1.2, 1.2], shadow=False)
    scene.add_point_light([0.4, 0.0, 0.8], [8, 8, 8])

    for name, color, pos in CUBES:
        b = scene.create_actor_builder()
        b.add_box_collision(half_size=[HALF] * 3)
        b.add_box_visual(half_size=[HALF] * 3, material=color)
        b.build(name=name).set_pose(sapien.Pose(p=pos))

    cam = scene.add_camera(name="head", width=W, height=H, fovy=np.deg2rad(50), near=0.05, far=10.0)

    # SAPIEN 相机系: +x 前, +y 左, +z 上 —— 用 look-at 摆一个俯视机位
    eye, target = np.array([-0.15, 0.0, 0.55]), np.array([0.52, 0.0, 0.035])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    left /= np.linalg.norm(left)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([fwd, left, np.cross(fwd, left)], axis=1)
    mat[:3, 3] = eye
    cam.set_pose(sapien.Pose(mat))

    for _ in range(60):
        scene.step()
    scene.update_render()
    cam.take_picture()

    rgba = cam.get_picture("Color")
    pos_cam = cam.get_picture("Position")          # OpenGL 相机系，z 朝后为负
    rgb = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
    depth_m = -pos_cam[..., 2]
    depth_m[pos_cam[..., 3] == 0] = 0.0            # 无效像素
    depth_mm = (depth_m * 1000.0).astype(np.float32)   # ⚠️ cg_wrapper 按毫米解析

    K = cam.get_intrinsic_matrix()
    gl2cv = np.diag([1.0, -1.0, -1.0, 1.0])        # OpenGL -> OpenCV(x 右 y 下 z 前)
    return rgb, depth_mm, K, cam.get_model_matrix() @ gl2cv


def main():
    if "GSA_PATH" not in os.environ:
        sys.exit("🚨 先设 GSA_PATH 指向 Grounded-Segment-Anything(见 SETUP_ENV.md 第 5 节)")

    print("🎬 [自检] 离屏渲染合成 RGB-D 视角...", flush=True)
    rgb, depth_mm, K, T = render_synthetic_view()
    print(f"   rgb={rgb.shape} 有效深度像素={int((depth_mm > 0).sum())}", flush=True)

    from engine_interface import PhysicalValidator

    plan_dag = [
        {"id": "n1", "action": "grasp", "target": "red_cube_1", "depends_on": [], "parameters": {}},
        {"id": "n2", "action": "place", "target": "blue_cube_1", "depends_on": ["n1"], "parameters": {}},
    ]

    validator = PhysicalValidator("config/global_config.yaml")
    res = validator.verify_local_plan(
        rgb_views={"head": rgb},
        depth_views={"head": depth_mm},
        dynamic_extrinsics={"head": T},
        dynamic_intrinsics={"head": K},
        plan_dag=plan_dag,
    )

    print("\n================ 自检结果 ================")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print("\n✅ 全链路跑通(没有段错误)。success =", res.get("success"))


if __name__ == "__main__":
    main()
