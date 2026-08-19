import os
import sys
import gc
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Dict

# ----------------- 1. 挂载 Grounded-SAM 依赖 -----------------
gsa_path = os.environ.get("GSA_PATH")
if not gsa_path:
    raise RuntimeError("🚨 必须设置 GSA_PATH 环境变量指向 Grounded-Segment-Anything 目录")

sys.path.insert(0, gsa_path)
sys.path.insert(0, os.path.join(gsa_path, "GroundingDINO"))
sys.path.insert(0, os.path.join(gsa_path, "segment_anything"))

try:
    from groundingdino.util.inference import load_model as load_dino, predict as predict_dino
    import groundingdino.datasets.transforms as T
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    print(f"🚨 [CG_Wrapper ERROR] 无法从 GSA 导入模型，请检查环境: {e}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class ConceptGraphsWrapper:
    def __init__(self):
        print("[CG_Wrapper] 正在挂载原生 GroundingDINO + SAM 双核大模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        weights_dir = PROJECT_ROOT / "weights" / "cg_weights"
        sam_ckpt = weights_dir / "sam_vit_h_4b8939.pth"
        dino_ckpt = weights_dir / "groundingdino_swint_ogc.pth"
        dino_config = Path(gsa_path) / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"

        self.dino_model = load_dino(str(dino_config), str(dino_ckpt)).to(self.device)
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt)).to(self.device)
        self.sam_predictor = SamPredictor(sam)

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _get_grounding_output(self, rgb_image: np.ndarray, text_prompt: str):
        from PIL import Image
        pil_img = Image.fromarray(rgb_image)
        image_transformed, _ = self.transform(pil_img, None)

        boxes, logits, phrases = predict_dino(
            model=self.dino_model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=0.25,
            text_threshold=0.25
        )
        # 🚀 修复一：绝不能丢掉 logits，必须返回给外层！
        return boxes, logits, phrases

    def extract_raw_point_clouds(self, rgb_image: np.ndarray, depth_image: np.ndarray, K: np.ndarray,
                                 text_prompt: str = "object .") -> List[Dict]:
        results = []
        # 🚀 接收透传出来的 logits
        boxes, logits, phrases = self._get_grounding_output(rgb_image, text_prompt)
        if len(boxes) == 0:
            return results

        H, W, _ = rgb_image.shape
        boxes = boxes * torch.Tensor([W, H, W, H])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        self.sam_predictor.set_image(rgb_image)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, rgb_image.shape[:2]).to(self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False
        )
        masks = masks[:, 0, :, :].cpu().numpy()

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        for i, mask in enumerate(masks):
            label = phrases[i]
            # 🚀 提取当前目标的语义置信度
            score = logits[i].item()

            if not np.any(mask): continue

            v_indices, u_indices = np.where(mask)
            obj_depths = depth_image[mask].astype(np.float64) / 1000.0
            # 过滤无效深度(inf/nan/0/超量程):mask 边缘常含背景/无返回像素,back-project 后变成无穷远
            # 杂点,污染点云 -> 融合后碎成大量幽灵簇。只保留有限、正、合理量程内的深度。
            valid = np.isfinite(obj_depths) & (obj_depths > 0.05) & (obj_depths < 5.0)
            obj_depths = obj_depths[valid]
            u_indices = u_indices[valid]
            v_indices = v_indices[valid]
            if obj_depths.size < 50:
                continue

            z = obj_depths
            x = (u_indices - cx) * z / fx
            y = (v_indices - cy) * z / fy
            points_3d = np.stack([x, y, z], axis=-1)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            if len(pcd.points) < 50: continue

            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            clean_pcd = pcd.select_by_index(ind)
            if len(clean_pcd.points) < 10: continue

            results.append({
                "label": label,
                "points_cam": np.asarray(clean_pcd.points),
                # 🚀 修复二：将 score 挂载到字典中，交给 pipeline 仲裁使用！
                "score": score
            })

        self.sam_predictor.reset_image()
        gc.collect()
        torch.cuda.empty_cache()

        return results
