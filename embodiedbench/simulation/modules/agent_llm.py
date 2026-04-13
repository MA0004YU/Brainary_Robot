import json
import os
from openai import OpenAI
from typing import List, Optional
from schemas.data_types import SceneObject, NavigationAction, PhysicalProperty, Pose

class NavigationAgentLLM:
    def __init__(self, api_key: str = "YOUR_OPENAI_API_KEY_HERE"):
        # 你只需要在这里替换你的 API Key
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o" # 推荐使用具备强推理能力的模型

    def assign_physics_and_plan(self, objects: List[SceneObject], robot_pos: tuple) -> NavigationAction:
        """
        一次性完成：1. 物理常识赋予 2. 路径规划决策
        """
        # 将当前感知到的场景转换为 LLM 可理解的描述
        scene_desc = []
        for obj in objects:
            scene_desc.append({
                "id": obj.object_id,
                "label": obj.semantic_label,
                "size": obj.dimensions,
                "pos": obj.current_pose.position
            })

        prompt = f"""
        你是一个具身智能导航专家。
        当前环境中有以下物体 (JSON格式): {json.dumps(scene_desc)}
        机器人当前位置: {robot_pos}
        
        任务要求:
        1. 识别物体属性: 判断哪些是门帘(is_passable=true)，哪些是可推挤的轻质物体，哪些是重物。
        2. 做出导航决策: 如果前方是门帘或空箱子，可以直接穿过或推开；如果是重物，必须绕行。
        3. 给出指令: 输出 action_type ("move_forward" 或 "push"), 2D方向向量 direction (dx, dy), 以及推力 apply_force (0-100N)。

        请严格以 JSON 格式输出，不要包含任何额外文字说明。
        示例: {{"reasoning": "...", "action": {{"action_type": "push", "direction": [0.5, 1.0], "apply_force": 70.0}}, "physics_update": {{"id": "...", "is_passable": true, "mass": 0.5}}}}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a helpful Embodied AI planner."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )

        res_data = json.loads(response.choices[0].message.content)
        
        # 更新物体的物理属性（反哺回 SceneObject）
        # 此处省略具体更新逻辑，直接生成 Action
        action_data = res_data["action"]
        return NavigationAction(
            action_type=action_data["action_type"],
            direction=tuple(action_data["direction"]),
            apply_force=float(action_data["apply_force"]),
            duration=2.0
        )
