import yaml
import time
import numpy as np
import open3d as o3d
from typing import Dict, Any, List
from scipy.spatial.transform import Rotation as R

from .wrappers.cg_wrapper import ConceptGraphsWrapper
from src.memory.physics_dictionary import PHYSICS_DICTIONARY


class PerceptionPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.cg_wrapper = ConceptGraphsWrapper()

        self.perception_cfg = self.config.get("perception", {})
        self.voxel_size = self.perception_cfg.get("voxel_size", 0.005)
        self.dbscan_eps_multiplier = self.perception_cfg.get("dbscan_eps_multiplier", 4.0)
        self.min_physical_dim = self.perception_cfg.get("min_physical_dim_m", 0.01)
        self.min_physical_vol = self.perception_cfg.get("min_physical_vol_m3", 1e-6)
        self.min_cluster_points = self.perception_cfg.get("min_cluster_points", 50)

    def _clean_label(self, raw_label: str) -> str:
        valid_labels = list(PHYSICS_DICTIONARY.keys())
        raw_label = raw_label.lower()
        for v in valid_labels:
            if v in raw_label:
                return v
        return None

    def _early_fusion_3d(self, grouped_points: Dict[str, Dict[str, List[np.ndarray]]]) -> List[Dict]:
        raw_results = []

        for label, data in grouped_points.items():
            pts_list = data["pts"]
            scores_list = data["scores"]
            if not pts_list: continue

            all_pts = np.vstack(pts_list)
            all_scores = np.vstack(scores_list)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            # 巧妙利用色彩通道透传语义得分：将 DINO Score 挂载到 R 通道传入 C++ 底层
            pcd.colors = o3d.utility.Vector3dVector(all_scores)

            # 阶段一：几何合法性校验与聚类
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            clean_pcd = pcd.select_by_index(ind)

            eps = self.voxel_size * self.dbscan_eps_multiplier
            labels = np.array(
                clean_pcd.cluster_dbscan(eps=eps, min_points=self.min_cluster_points, print_progress=False))

            if len(labels) == 0 or labels.max() < 0:
                continue

            unique_cluster_ids = set(labels)
            for cluster_idx in unique_cluster_ids:
                if cluster_idx < 0: continue

                cluster_pts_idx = np.where(labels == cluster_idx)[0]
                pt_count = len(cluster_pts_idx)

                # 严守底线：点数不达标的微小碎屑和飞点直接剔除，防范假阳性噪点
                if pt_count < self.min_cluster_points:
                    continue

                true_object_pcd = clean_pcd.select_by_index(cluster_pts_idx)
                final_aabb = true_object_pcd.get_axis_aligned_bounding_box()
                extent = final_aabb.get_extent()

                if np.min(extent) < self.min_physical_dim:
                    continue

                volume = extent[0] * extent[1] * extent[2]
                if volume < self.min_physical_vol:
                    continue

                size = extent.tolist()
                final_pose = np.eye(4)
                final_pose[0:3, 3] = final_aabb.get_center()

                # 常识防线：剔除桌面以下的基底建筑
                if final_pose[2, 3] < 0.0:
                    continue

                # 从 R 通道提取该 3D 刚体簇的平均语义置信度
                mean_score = np.mean(np.asarray(true_object_pcd.colors)[:, 0])

                raw_results.append({
                    "label": label,
                    "size": size,
                    "pose_world": final_pose,
                    "point_count": pt_count,
                    "mean_score": float(mean_score)
                })

        # =====================================================================
        # 阶段二：🚀 期望/关联仲裁 (Semantic-Aware NMS)
        # 完全基于大模型置信度进行降维打击，杜绝体积大点数多导致的“漏报”灾难
        # =====================================================================
        raw_results.sort(key=lambda x: x["point_count"], reverse=True)
        kept_results = []

        for res in raw_results:
            is_duplicate = False
            for i, kept in enumerate(kept_results):
                dist = np.linalg.norm(res["pose_world"][:3, 3] - kept["pose_world"][:3, 3])
                if dist < 0.04:  # 发生物理坐标重叠
                    is_duplicate = True

                    # 绝杀：不比点数，比拼语义纯度 (Score)
                    if res["mean_score"] > kept["mean_score"]:
                        print(
                            f"🧹 [语义仲裁] 坐标重叠! 语义得分较量 ({res['mean_score']:.3f} > {kept['mean_score']:.3f}) -> 保留高置信目标 [{res['label']}](点:{res['point_count']}) 💥 剔除幻觉 [{kept['label']}](点:{kept['point_count']})")
                        kept_results[i] = res
                    else:
                        print(
                            f"🧹 [语义仲裁] 坐标重叠! 语义得分较量 ({kept['mean_score']:.3f} > {res['mean_score']:.3f}) -> 保留高置信目标 [{kept['label']}](点:{kept['point_count']}) 💥 剔除幻觉 [{res['label']}](点:{res['point_count']})")
                    break

            if not is_duplicate:
                kept_results.append(res)

        return kept_results

    def process_scene(self, rgb_views: Dict[str, np.ndarray], depth_views: Dict[str, np.ndarray],
                      dynamic_extrinsics: Dict[str, np.ndarray],
                      dynamic_intrinsics: Dict[str, np.ndarray],
                      text_prompt: str = None) -> Dict[str, Any]:

        print(f"[PerceptionPipeline] 🔄 启动多视角 3D 早融合 (Early Fusion) 架构...")
        grouped_points_world = {}

        for view_name, rgb in rgb_views.items():
            if view_name not in dynamic_extrinsics or view_name not in dynamic_intrinsics or rgb is None:
                continue

            depth = depth_views.get(view_name)
            T_cam_to_world = dynamic_extrinsics[view_name]
            K_matrix = dynamic_intrinsics[view_name]

            cg_kwargs = {"text_prompt": text_prompt} if text_prompt else {}
            cg_results = self.cg_wrapper.extract_raw_point_clouds(rgb, depth, K_matrix, **cg_kwargs)

            for obj in cg_results:
                label = obj["label"]
                clean_label = self._clean_label(label)
                if not clean_label:
                    continue

                pts_cam = obj["points_cam"]
                if len(pts_cam) == 0:
                    continue

                # 💡 提取 2D VLM 的原始语义置信度得分 (Score/Logit)
                raw_score = obj.get("score", obj.get("logit", 0.5))
                if isinstance(raw_score, (list, np.ndarray)):
                    raw_score = np.mean(raw_score)
                semantic_score = float(raw_score)

                # 构建一个与点云等长的 Nx3 矩阵，把 Score 塞进第 0 列 (模拟 R 通道)
                score_array = np.zeros((pts_cam.shape[0], 3))
                score_array[:, 0] = semantic_score

                pts_cam_h = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
                pts_world = (T_cam_to_world @ pts_cam_h.T).T[:, :3]

                if clean_label not in grouped_points_world:
                    grouped_points_world[clean_label] = {"pts": [], "scores": []}

                grouped_points_world[clean_label]["pts"].append(pts_world)
                grouped_points_world[clean_label]["scores"].append(score_array)

        final_world_entities = self._early_fusion_3d(grouped_points_world)

        grouped_by_label = {}
        for entity in final_world_entities:
            lbl = entity["label"]
            if lbl not in grouped_by_label:
                grouped_by_label[lbl] = []
            grouped_by_label[lbl].append(entity)

        # 空间绝对排序：完美对接后续规划
        scene_entities = []
        for lbl, entities in grouped_by_label.items():
            entities.sort(key=lambda e: (
                -round(e["pose_world"][1, 3], 1),
                round(e["pose_world"][0, 3], 1)
            ))

            for idx, entity in enumerate(entities):
                unique_name = f"{lbl.replace(' ', '_')}_{idx + 1}"

                T_w = entity["pose_world"]
                quat_xyzw = R.from_matrix(T_w[0:3, 0:3]).as_quat()
                quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]

                scene_entities.append({
                    "id": unique_name,
                    "name": unique_name,
                    "label": lbl,
                    "type": "primitive_box",
                    "size": entity["size"],
                    "pose": {
                        "pos": T_w[0:3, 3].tolist(),
                        "quat": quat_wxyz
                    },
                    "point_count": entity["point_count"],
                    "semantic_score": entity["mean_score"]
                })

        print("\n" + "=" * 105)
        print("📊 [全景量化 X 光机] PerceptionPipeline 最终输出给沙盒的实体清单:")
        print(
            f"{'实体 ID (Entity)':<16} | {'中心坐标 (X, Y, Z)':<22} | {'物理尺寸 (W, H, D)':<22} | {'体积 (m³)':<10} | {'致密度':<8} | {'语义置信度'}")
        print("-" * 105)
        for ent in scene_entities:
            pos = ent['pose']['pos']
            size = ent['size']
            vol = size[0] * size[1] * size[2]
            pos_str = f"{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"
            size_str = f"{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}"
            print(
                f"{ent['id']:<16} | {pos_str:<22} | {size_str:<22} | {vol:.6f}   | {ent['point_count']:<8} | {ent['semantic_score']:.3f}")
        print("=" * 105 + "\n")

        print(f"[PerceptionPipeline] ✅ 早融合完成！空间绝对排序与 [语义-几何双重校验防漏报] 机制已生效。")
        return {
            "stage": "perception_initialization",
            "timestamp": time.time(),
            "objects": scene_entities
        }
