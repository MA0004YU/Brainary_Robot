# simulation/src/perception/pipeline.py

import yaml
import time
import torch
import numpy as np
from typing import Dict, Any, List
from scipy.spatial.transform import Rotation as R

from .wrappers.cg_wrapper import ConceptGraphsWrapper
from .wrappers.fp_wrapper import FoundationPoseWrapper


class PerceptionPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.intrinsics = self.config['camera_config']['intrinsics']
        self.quat_format = self.config['camera_config'].get('quaternion_format', 'xyzw').strip().lower()

        # 检查四元数格式契约，避免因通道不一致导致的逆反旋转
        if self.quat_format not in ["xyzw", "wxyz"]:
            raise ValueError(
                f"[Perception ERROR] 配置文件中的 quaternion_format 必须明确声明为 'xyzw' 或 'wxyz'，当前输入为: {self.quat_format}")

        # 强契约解析前、上、侧三个相机的 4x4 齐次外参矩阵
        self.cam_extrinsics = {
            "front": self._build_se3(self.config['camera_config']['tf_front_to_world']),
            "top": self._build_se3(self.config['camera_config']['tf_top_to_world']),
            "side": self._build_se3(self.config['camera_config']['tf_side_to_world'])
        }

        self.cg_wrapper = ConceptGraphsWrapper()
        self.fp_wrapper = FoundationPoseWrapper(assets_dir="assets/meshes/")

    def _build_se3(self, tf_config: dict) -> np.ndarray:
        """从配置中严格根据指定格式解析并转换为 4x4 齐次变换矩阵"""
        T = np.eye(4)
        T[0:3, 3] = tf_config['translation']

        raw_quat = tf_config['quaternion']

        # 根据契约进行分支条件格式化对齐
        if self.quat_format == "xyzw":
            # 格式正确，直接喂给 SciPy 转化为 3x3 旋转矩阵
            quat_xyzw = np.array(raw_quat)
        elif self.quat_format == "wxyz":
            # 进行切片变换，将首位的 w 提取出来拼到最后，对齐为 SciPy 的 xyzw 标准
            quat_xyzw = np.array([raw_quat[1], raw_quat[2], raw_quat[3], raw_quat[0]])

        T[0:3, 0:3] = R.from_quat(quat_xyzw).as_matrix()
        return T

    def _apply_transform(self, local_pose_4x4: np.ndarray, T_cam_to_world: np.ndarray) -> np.ndarray:
        """将相机局部标定坐标系下的 4x4 位姿严格映射至机械臂世界坐标系"""
        return T_cam_to_world @ local_pose_4x4

    def _spatial_nms_3d(self, raw_entities: List[Dict], distance_threshold: float = 0.05) -> List[Dict]:
        """
        对比不同视角在世界系下的绝对坐标，重合度高则加权融合，残缺遮挡则予以互补
        """
        fused_entities = []

        for item in raw_entities:
            pos_a = item["pose_world"][0:3, 3]
            label_a = item["label"]

            match_found = False
            for fused in fused_entities:
                if fused["label"] == label_a:
                    pos_b = fused["pose_world"][0:3, 3]
                    # 若绝对欧氏空间距离小于 5cm，则断定为不同视角捕捉到的同一个物理目标
                    if np.linalg.norm(pos_a - pos_b) < distance_threshold:
                        # 进行质心位姿齐次平滑均值融合
                        fused["pose_world"][0:3, 3] = (fused["pose_world"][0:3, 3] + pos_a) / 2.0
                        # 尺寸长宽高消融反光误差，取多目观测平均值
                        fused["size"] = ((np.array(fused["size"]) + np.array(item["size"])) / 2.0).tolist()
                        match_found = True
                        break

            if not match_found:
                fused_entities.append(item)

        return fused_entities

    def process_scene(self, rgb_views: Dict[str, np.ndarray], depth_views: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """多目并行感知总线"""
        all_detected_world_entities = []

        for view_name in ["front", "top", "side"]:
            if view_name not in rgb_views or rgb_views[view_name] is None:
                continue

            rgb = rgb_views[view_name]
            depth = depth_views[view_name]
            T_cam_to_world = self.cam_extrinsics[view_name]

            # 1. 独立解算当前相机的 OBB 方块边界
            cg_results = self.cg_wrapper.extract_coarse_obb(rgb, depth, self.intrinsics)
            torch.cuda.empty_cache()

            # 2. 独立调用 FoundationPose 双核权重进行 6DoF 精炼对齐
            for idx, obj in enumerate(cg_results):
                label = obj["label"]
                size_whd = obj["size_whd"]

                refined_local_pose = self.fp_wrapper.refine_pose(
                    rgb_image=rgb, depth_image=depth, intrinsics=self.intrinsics,
                    object_label=label, size_whd=size_whd, initial_pose=obj["coarse_pose"]
                )

                # 3. 乘以外参齐次变换矩阵，将局部 4x4 位姿统一合并到机械臂世界坐标系下
                pose_world = self._apply_transform(refined_local_pose, T_cam_to_world)

                all_detected_world_entities.append({
                    "label": label,
                    "size": size_whd,
                    "pose_world": pose_world
                })

            torch.cuda.empty_cache()

        # 4. 执行空间 NMS 融合去重与盲区遮挡互补
        final_world_entities = self._spatial_nms_3d(all_detected_world_entities)

        # 5. 统一转换并输出为类脑 VLM 协议与 SceneBuilder 兼容的标准格式
        objects = []
        for idx, entity in enumerate(final_world_entities):
            T_w = entity["pose_world"]
            quat_xyzw = R.from_matrix(T_w[0:3, 0:3]).as_quat()
            
            # 提取 3D 坐标
            x, y, z = T_w[0:3, 3].tolist()
            # 提取四元数并显式重组为 VLM 协议与 SAPIEN 均要求的 [qw, qx, qy, qz]
            qw = float(quat_xyzw[3])
            qx, qy, qz = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])
    
            objects.append({
                "name": f"{entity['label']}_{idx + 1}",  # 契约对齐：id -> name
                "type": "primitive_box",
                "size_whd": entity["size"],              # 契约对齐：size -> size_whd
                "pose": [x, y, z, qw, qx, qy, qz]        # 契约对齐：嵌套字典 -> 7D 扁平数组
            })
    
        return {
            "stage": "perception_initialization",
            "timestamp": time.time(),
            "objects": objects                           # 契约对齐：scene_entities -> objects
        }
