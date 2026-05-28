# simulation/src/perception/wrappers/cg_wrapper.py

import os
import sys
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Dict

# 挂载 Grounded-SAM 环境变量
# 必须在所有 import 发生之前，将 GSA 源码路径强行打入 Python 内存字典
# 否则 CG 底层调用 GroundingDINO 时会直接报 ModuleNotFoundError 崩溃
gsa_path = os.environ.get("GSA_PATH")
if gsa_path:
    if gsa_path not in sys.path:
        sys.path.insert(0, gsa_path)
        print(f"[CG_Wrapper] 成功挂载 Grounded-SAM 底层引擎: {gsa_path}")
else:
    print("🚨 [FATAL WARNING] 未检测到 GSA_PATH 环境变量！请确保已执行: export GSA_PATH=/path/to/Grounded-Segment-Anything")

# 挂载 ConceptGraphs 源码路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CG_ROOT = str(PROJECT_ROOT / "third_party" / "concept-graphs") # 注意：官方 repo 默认叫 concept-graphs
if CG_ROOT not in sys.path:
    sys.path.insert(0, CG_ROOT)

# 导入原生的 ConceptGraphs 推理管线
try:
    from conceptgraph.pipeline import CGPipeline
except ImportError as e:
    print(f"🚨 [CG_Wrapper ERROR] 无法导入 ConceptGraphs，请检查依赖是否装全: {e}")
    CGPipeline = None


class ConceptGraphsWrapper:
    def __init__(self):
        print("[CG_Wrapper] 正在挂载真实 ConceptGraphs 权重与 SAM 模型...")
        
        # 锁死权重物理路径
        # 不能只传一个 yaml，必须显式地把模型路径传进去，
        weights_dir = PROJECT_ROOT / "weights" / "cg_weights"
        sam_ckpt_path = weights_dir / "sam_vit_h_4b8939.pth"
        dino_ckpt_path = weights_dir / "groundingdino_swint_ogc.pth"
        
        if not sam_ckpt_path.exists() or not dino_ckpt_path.exists():
            print(f"🚨 [FATAL WARNING] 找不到视觉大模型权重！期待路径:\n{sam_ckpt_path}\n{dino_ckpt_path}")

        if CGPipeline is not None:
            self.pipeline = CGPipeline(
                config_path=str(PROJECT_ROOT / "third_party" / "concept-graphs" / "conceptgraph" / "dataset" / "dataconfigs" / "replica" / "replica.yaml"),
                sam_checkpoint=str(sam_ckpt_path),
                dino_checkpoint=str(dino_ckpt_path)
            )
        else:
            self.pipeline = None

    def extract_coarse_obb(self, rgb_image: np.ndarray, depth_image: np.ndarray, intrinsics: dict) -> List[Dict]:
        """推理管线：直接消化 RGB-D 并在 Open3D 中解算包围盒"""
        if self.pipeline is None:
            print("[CG_Wrapper] 视觉管线未初始化，返回空几何体！")
            return []

        # 将 Numpy 张量打入底层模型，获取真实的语义掩码与点云切分
        with torch.no_grad(): # 加上 no_grad 防止显存爆炸
            raw_instances = self.pipeline.process(rgb_image, depth_image, intrinsics)

        results = []
        for instance in raw_instances:
            label = instance["label"]
            pcd = instance["pcd"]  # 接收 open3d.geometry.PointCloud

            # 如果某次分割失败点云为空，跳过
            if not pcd.has_points() or len(pcd.points) < 10:
                continue

            # 极其优美的 3D 去噪算法
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            clean_pcd = pcd.select_by_index(ind)

            if not clean_pcd.has_points():
                continue

            # OBB 几何计算
            obb = clean_pcd.get_oriented_bounding_box()
            size_whd = obb.extent.tolist()

            # 抽取 4x4 位姿矩阵
            coarse_pose_4x4 = np.eye(4)
            coarse_pose_4x4[0:3, 0:3] = obb.R
            coarse_pose_4x4[0:3, 3] = obb.center

            results.append({
                "label": label,
                "size_whd": size_whd,
                "coarse_pose": coarse_pose_4x4
            })

        return results
