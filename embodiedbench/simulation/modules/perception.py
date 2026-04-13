from typing import List
from schemas.data_types import SceneObject, Pose

class NavigationPerception:
    def get_scene_context(self) -> List[SceneObject]:
        print("[Perception] 扫描前方环境...")
        return [
            # 1. 软体门帘 (is_passable 将在属性阶段被 LLM 赋予 True)
            SceneObject(
                object_id="curtain_01",
                semantic_label="plastic_curtain",
                dimensions=(1.5, 0.05, 2.0),
                current_pose=Pose(position=(0.0, 1.0, 1.0)) # 位于正前方 Y=1 处
            ),
            # 2. 空纸箱 (右侧)
            SceneObject(
                object_id="box_empty",
                semantic_label="empty_cardboard_box",
                dimensions=(0.4, 0.4, 0.4),
                current_pose=Pose(position=(0.3, 2.0, 0.2)) # 位于 Y=2, 偏右
            ),
            # 3. 满纸箱 (左侧)
            SceneObject(
                object_id="box_full",
                semantic_label="heavy_sealed_box",
                dimensions=(0.4, 0.4, 0.4),
                current_pose=Pose(position=(-0.3, 2.0, 0.2)) # 位于 Y=2, 偏左
            )
        ]
