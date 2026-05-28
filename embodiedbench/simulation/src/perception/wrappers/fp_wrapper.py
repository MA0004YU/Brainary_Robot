# simulation/src/perception/wrappers/fp_wrapper.py

import os
import sys
import torch
import trimesh
import numpy as np
from pathlib import Path
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FP_ROOT = str(PROJECT_ROOT / "third_party" / "FoundationPose")
if FP_ROOT not in sys.path:
    sys.path.insert(0, FP_ROOT)

try:
    from foundationpose.estimater import FoundationPose
    from foundationpose.utils.mesh import Mesh
except ImportError as e:
    print(f"🚨 [FP_Wrapper ERROR] 无法导入 FoundationPose，请检查编译与路径: {e}")
    FoundationPose = None


class FoundationPoseWrapper:
    def __init__(self, assets_dir="assets/meshes/"):
        print("[FP_Wrapper] 正在挂载 FoundationPose 评估器与渲染引擎...")
        self.assets_dir = PROJECT_ROOT / assets_dir
        self.base_mesh_path = self.assets_dir / "unit_cube.obj"
        
        if not self.base_mesh_path.exists():
            raise FileNotFoundError(f"找不到基础网格文件: {self.base_mesh_path}，请先运行生成脚本。")

        # 映射本地权重 
        weights_dir = PROJECT_ROOT / "weights" / "fp_weights"
        refiner_ckpt = weights_dir / "2023-10-28-18-33-37" / "model_best.pth"
        scorer_ckpt = weights_dir / "2024-01-11-20-02-45" / "model_best.pth"

        if FoundationPose is not None:
            # 初始化 FP 的核心模型 (自动载入 GPU)
            self.estimator = FoundationPose(
                scorer_weight_dir=str(scorer_ckpt.parent),
                refiner_weight_dir=str(refiner_ckpt.parent)
            )
        else:
            self.estimator = None

    def _create_dynamic_mesh(self, size_whd: list) -> Mesh:
        """根据 CG 传来的长宽高，动态将 1m^3 的单位正方体拉伸为真实物理大小的网格"""
        trimesh_obj = trimesh.load(str(self.base_mesh_path))
        # 施加非均匀缩放矩阵
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = size_whd[0]
        scale_matrix[1, 1] = size_whd[1]
        scale_matrix[2, 2] = size_whd[2]
        trimesh_obj.apply_transform(scale_matrix)
        
        # 封装为 FoundationPose 认识的 Mesh 对象
        return Mesh(trimesh_obj.vertices, trimesh_obj.faces)

    def refine_pose(self, rgb_image: np.ndarray, depth_image: np.ndarray, intrinsics: dict, 
                    object_label: str, size_whd: list, initial_pose: np.ndarray) -> np.ndarray:
        """
        核心 6DoF 精炼器：以 CG 的粗糙位姿为起点，执行 Render-and-Compare
        """
        if self.estimator is None:
            return initial_pose

        print(f"  [FP_Wrapper] 正在对 {object_label} 进行 6DoF 亚毫米级位姿精炼...")

        # 1. 动态生成与画面中真实物体等大的 3D 刚体网格
        dynamic_mesh = self._create_dynamic_mesh(size_whd)

        # 2. 转换相机内参为 FP 需要的 Numpy 3x3 矩阵
        K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ])

        # 开启混合精度与无梯度推理
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.float16):
                # 调用模型基准跟踪/估计接口
                # 传入 RGBD, 动态网格，以及 CG 提供给我们的起始姿态 (initial_pose)
                refined_pose_4x4 = self.estimator.track_one(
                    rgb=rgb_image,
                    depth=depth_image,
                    K=K,
                    mesh=dynamic_mesh,
                    pose_init=initial_pose
                )

        # 清除渲染器在该物体上残留的显存切片
        torch.cuda.empty_cache()
        return refined_pose_4x4
