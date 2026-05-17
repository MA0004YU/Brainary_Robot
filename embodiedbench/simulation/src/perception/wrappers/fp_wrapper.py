# simulation/src/perception/wrappers/fp_wrapper.py

import sys
import os
import tempfile
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List
from scipy.spatial.transform import Rotation as R

FP_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "FoundationPose")
if FP_ROOT not in sys.path:
    sys.path.insert(0, FP_ROOT)

# 导入 FP 原生网络架构
from estim.pose_refiner import PoseRefiner


class FoundationPoseWrapper:
    def __init__(self, assets_dir: str):
        self.assets_dir = Path(assets_dir)

        # 定位权重路径
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        refiner_model_path = project_root / "weights" / "fp_weights" / "2023-10-28-18-33-37" / "model_best.pth"
        scorer_model_path = project_root / "weights" / "fp_weights" / "2024-01-11-20-02-45" / "model_best.pth"

        if not refiner_model_path.exists() or not scorer_model_path.exists():
            raise FileNotFoundError(
                f"[FP_Wrapper ERROR] 缺失 FoundationPose 官方标准权重文件！\n"
                f"请核对并确保以下路径真实存在：\n"
                f"1. Refiner 路径: {refiner_model_path}\n"
                f"2. Scorer 路径: {scorer_model_path}"
            )

        print("[FP_Wrapper] 正在加载 FoundationPose 双核神经网络 (Refiner + Scorer)...")
        # 双核路径下发，读取 PyTorch 权重至 GPU
        self.refiner = PoseRefiner(
            model_path=str(refiner_model_path),
            scorer_path=str(scorer_model_path)
        )

    def _dynamically_scale_mesh(self, label: str, size_whd: List[float]) -> str:
        """内存几何操作"""
        base_mesh_path = self.assets_dir / "unit_cube.obj"
        if not base_mesh_path.exists():
            raise FileNotFoundError("缺失拓扑底片 assets/meshes/unit_cube.obj！")

        scaled_mesh = trimesh.load(base_mesh_path)
        scale_matrix = np.diag([size_whd[0], size_whd[1], size_whd[2], 1.0])
        scaled_mesh.apply_transform(scale_matrix)

        temp_dir = tempfile.gettempdir()
        temp_mesh_path = os.path.join(temp_dir, f"temp_{label}_scaled.obj")
        scaled_mesh.export(temp_mesh_path)
        return temp_mesh_path

    def refine_pose(self, rgb_image: np.ndarray, depth_image: np.ndarray, intrinsics: dict,
                    object_label: str, size_whd: List[float], initial_pose: np.ndarray) -> Dict:

        mesh_path = self._dynamically_scale_mesh(object_label, size_whd)

        # 组装针孔相机内参矩阵 K
        K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ])

        # 将张量强行打入 FP 神经网络进行 Iterative Refinement
        # 如果 GPU 显存不足，此处将OOM 崩溃
        refined_pose_4x4 = self.refiner.predict(rgb_image, depth_image, mesh_path, initial_pose, K)

        # 提取 3x3 旋转矩阵，并在 SciPy 框架下转成 SAPIEN 标准四元数
        rotation_matrix = refined_pose_4x4[0:3, 0:3]
        quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]

        # 清除临时大尺寸 Mesh，严控 IO 污染
        if os.path.exists(mesh_path):
            os.remove(mesh_path)

        return {
            "pos": refined_pose_4x4[0:3, 3].tolist(),
            "quat": quat_wxyz
        }
