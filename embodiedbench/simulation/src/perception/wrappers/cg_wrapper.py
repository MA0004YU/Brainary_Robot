# simulation/src/perception/wrappers/cg_wrapper.py


import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Dict

CG_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "ConceptGraphs")
if CG_ROOT not in sys.path:
    sys.path.insert(0, CG_ROOT)

# 导入原生的 ConceptGraphs 推理管线
from conceptgraph.pipeline import CGPipeline


class ConceptGraphsWrapper:
    def __init__(self):
        print("[CG_Wrapper] 正在挂载真实 ConceptGraphs 权重与 SAM 模型...")
        # 初始化，强依赖本地 third_party/ConceptGraphs/config
        self.pipeline = CGPipeline(config_path="third_party/ConceptGraphs/config/cg_config.yaml")

    def extract_coarse_obb(self, rgb_image: np.ndarray, depth_image: np.ndarray, intrinsics: dict) -> List[Dict]:
        """毫无保留的真实推理管线：直接消化 RGB-D 并在 Open3D 中解算包围盒"""

        # 将 Numpy 张量打入底层模型，获取真实的语义掩码与点云切分
        raw_instances = self.pipeline.process(rgb_image, depth_image, intrinsics)

        results = []
        for instance in raw_instances:
            label = instance["label"]
            pcd = instance["pcd"]  # 接收open3d.geometry.PointCloud

            # 如果某次分割失败点云为空，跳过
            if not pcd.has_points() or len(pcd.points) < 10:
                continue

            # 去噪
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
