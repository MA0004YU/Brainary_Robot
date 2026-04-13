import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import List, Dict
from schemas.data_types import SceneObject, Pose

class NavigationPerception:
    def __init__(self):
        # 真实环境中，这里初始化你的 2D 视觉大模型 (如 YOLO, Grounding DINO, SAM)
        pass

    def _pixels_to_world_point_cloud(self, u_coords: np.ndarray, v_coords: np.ndarray, 
                                     depths: np.ndarray, intrinsic: np.ndarray, 
                                     extrinsic: np.ndarray) -> np.ndarray:
        """
        核心计算 1：将 2D 像素点阵逆投影为 3D 世界坐标系下的点云
        """
        # 提取相机内参
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        # 1. 计算相机坐标系下的 3D 坐标 (X_c, Y_c, Z_c)
        Z_c = depths
        X_c = (u_coords - cx) * Z_c / fx
        Y_c = (v_coords - cy) * Z_c / fy

        # 组装为相机坐标系下的齐次坐标形状 [N, 4] -> [[X_c, Y_c, Z_c, 1], ...]
        ones = np.ones_like(Z_c)
        points_c = np.stack([X_c, Y_c, Z_c, ones], axis=1)

        # 2. 转换到世界坐标系 (X_w, Y_w, Z_w)
        # P_world = Extrinsic * P_camera
        points_w = (extrinsic @ points_c.T).T
        
        # 返回前 3 列 [N, 3]
        return points_w[:, :3]

    def _calculate_obb(self, points_3d: np.ndarray) -> tuple:
        """
        核心计算 2：利用 Open3D 对点云计算有向包围盒 (OBB)
        """
        # 将 numpy 数组转为 Open3D 格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        # 降采样去噪 (可选，提升计算速度)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # 统计学离群点去除，防止噪点撑大包围盒
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        # 计算 OBB
        obb = pcd.get_oriented_bounding_box()
        
        # 提取中心点 (x, y, z)
        center = obb.center
        
        # 提取旋转矩阵并转换为四元数 (w, x, y, z)
        rotation_matrix = obb.R
        quaternion_scipy = R.from_matrix(rotation_matrix).as_quat() # 返回 [x, y, z, w]
        quaternion = (quaternion_scipy[3], quaternion_scipy[0], quaternion_scipy[1], quaternion_scipy[2])
        
        # 提取长宽高 (L, W, H)
        dimensions = tuple(obb.extent)

        return center, quaternion, dimensions

    def process_image_to_objects(self, rgb: np.ndarray, depth: np.ndarray, 
                                 intrinsic: np.ndarray, extrinsic: np.ndarray) -> List[SceneObject]:
        """
        真实感知接口：输入图像流和相机参数，输出 SAPIEN 可直接使用的实体对象
        """
        # 1. 模拟 2D 目标检测模型输出 (真实情况此处调用 self.detector)
        # 格式：[u_min, v_min, u_max, v_max]
        detected_instances = [
            {"label": "empty_cardboard_box", "bbox": [100, 150, 200, 250]}, 
            {"label": "plastic_curtain", "bbox": [300, 100, 500, 400]}
        ]

        found_objects = []

        for inst in detected_instances:
            u1, v1, u2, v2 = inst["bbox"]
            
            # 生成 BBox 内的所有像素网格坐标 (u, v)
            u_grid, v_grid = np.meshgrid(np.arange(u1, u2), np.arange(v1, v2))
            u_coords = u_grid.flatten()
            v_coords = v_grid.flatten()
            
            # 提取对应的深度值
            depths = depth[v1:v2, u1:u2].flatten()
            
            # 过滤掉深度无效的点 (比如深度为 0 或 NaN)
            valid_mask = (depths > 0) & (~np.isnan(depths))
            u_valid = u_coords[valid_mask]
            v_valid = v_coords[valid_mask]
            depths_valid = depths[valid_mask]

            if len(depths_valid) < 50: # 点太少，忽略
                continue

            # --- 调用底层数学计算 ---
            # 1. 投影到 3D 世界
            points_3d = self._pixels_to_world_point_cloud(u_valid, v_valid, depths_valid, intrinsic, extrinsic)
            
            # 2. 计算 OBB 包围盒
            center, quaternion, dimensions = self._calculate_obb(points_3d)

            # 3. 封装为标准契约格式
            found_objects.append(SceneObject(
                object_id=f"{inst['label']}_{v1}",
                semantic_label=inst["label"],
                dimensions=dimensions,
                current_pose=Pose(
                    position=tuple(center),
                    quaternion=quaternion
                )
            ))
        
        return found_objects
