import os
import sys
import torch
import trimesh
import numpy as np
from pathlib import Path
from torch.amp import autocast

# 🚀 绝对路径暴力锁定，彻底无视 cwd 工作目录干扰，完美挂载 FoundationPose
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "third_party", "FoundationPose"))

if FP_ROOT not in sys.path:
    sys.path.insert(0, FP_ROOT)

try:
    from estimater import FoundationPose
    from utils.mesh import Mesh
except ImportError as e:
    print(f"🚨 [FP_Wrapper ERROR] 无法导入 FoundationPose 模块: {e}")
    FoundationPose = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class FoundationPoseWrapper:
    def __init__(self, assets_dir="assets/meshes/"):
        print("[FP_Wrapper] 正在挂载 FoundationPose 评估器与渲染引擎...")
        self.assets_dir = PROJECT_ROOT / assets_dir
        self.base_mesh_path = self.assets_dir / "unit_cube.obj"

        weights_dir = PROJECT_ROOT / "weights" / "fp_weights"
        refiner_ckpt = weights_dir / "2023-10-28-18-33-37" / "model_best.pth"
        scorer_ckpt = weights_dir / "2024-01-11-20-02-45" / "model_best.pth"

        if FoundationPose is not None:
            self.estimator = FoundationPose(
                scorer_weight_dir=str(scorer_ckpt.parent),
                refiner_weight_dir=str(refiner_ckpt.parent)
            )
        else:
            self.estimator = None

    def _create_dynamic_mesh(self, size_whd: list):
        trimesh_obj = trimesh.load(str(self.base_mesh_path))
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = size_whd[0]
        scale_matrix[1, 1] = size_whd[1]
        scale_matrix[2, 2] = size_whd[2]
        trimesh_obj.apply_transform(scale_matrix)
        return Mesh(trimesh_obj.vertices, trimesh_obj.faces)

    # 🚀 核心修改：直接接收 K 矩阵 (np.ndarray)
    def refine_pose(self, rgb_image: np.ndarray, depth_image: np.ndarray, K: np.ndarray,
                    object_label: str, size_whd: list, initial_pose: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            return initial_pose

        print(f"  [FP_Wrapper] 正在对 {object_label} 进行 6DoF 亚毫米级位姿精炼...")
        dynamic_mesh = self._create_dynamic_mesh(size_whd)

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.float16):
                # 🚀 完美闭环：将真实的物理 K 矩阵直接喂入底层的位姿神经网路
                refined_pose_4x4 = self.estimator.track_one(
                    rgb=rgb_image,
                    depth=depth_image,
                    K=K,
                    mesh=dynamic_mesh,
                    pose_init=initial_pose
                )

        torch.cuda.empty_cache()
        return refined_pose_4x4
