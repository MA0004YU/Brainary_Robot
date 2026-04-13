import json
from openai import OpenAI
from typing import List
from schemas.data_types import SceneObject, NavigationAction, PhysicalProperty

class NavigationAgentLLM:
    def __init__(self, api_key: str = "YOUR_OPENAI_API_KEY_HERE"):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o" # 使用 GPT-4o 保证空间推理的准确性

    def assign_physics_and_plan(self, objects: List[SceneObject], robot_pos: tuple) -> NavigationAction:
        """
        完整逻辑：
        1. 组装当前环境状态给 LLM
        2. 获取 LLM 的推理 JSON
        3. 解析 JSON 并更新内存中 objects 的质量和穿透属性 (补全的部分)
        4. 返回最终的物理运动指令
        """
        # 1. 构建环境描述
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
        机器人最大推力: 100N。
        
        任务要求:
        1. 推断物理属性: 判断哪些是门帘(is_passable=true)，哪些是轻质物体(mass<5.0)，哪些是重物(mass>20.0)。
        2. 做出导航决策: 如果前方是门帘或轻质空箱子，可以直接穿过或推开；如果是重物，必须绕行。
        3. 输出指令: 严格输出 JSON 格式。

        必须遵守的 JSON 输出格式示例:
        {{
            "reasoning": "前方有两个箱子，根据标签推断 box_empty 很轻，可以推开...",
            "physics_update": [
                {{"id": "plastic_curtain_100", "is_passable": true, "mass": 0.1}},
                {{"id": "empty_cardboard_box_300", "is_passable": false, "mass": 0.5}}
            ],
            "action": {{
                "action_type": "push",
                "direction": [0.5, 1.0], 
                "apply_force": 80.0
            }}
        }}
        """

        # 2. 调用 API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful Embodied AI planner. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.2 # 降低温度，保证物理参数生成的稳定性
        )

        res_data = json.loads(response.choices[0].message.content)
        print(f"[Agent LLM] 思考过程: {res_data.get('reasoning')}")

        # ==========================================
        # 3. 核心补全：将 LLM 推断的物理属性注回内存
        # ==========================================
        physics_updates = res_data.get("physics_update", [])
        # 建立一个以 id 为 key 的字典方便快速查找
        update_dict = {item["id"]: item for item in physics_updates}

        for obj in objects:
            if obj.object_id in update_dict:
                update_info = update_dict[obj.object_id]
                # 创建并赋予真实的物理属性
                obj.physics = PhysicalProperty(
                    mass=update_info.get("mass", 1.0),
                    is_passable=update_info.get("is_passable", False),
                    friction_static=0.5,
                    friction_dynamic=0.5
                )
            else:
                # 兜底：如果 LLM 漏了某个物体，给个默认沉重的属性防止乱飞
                obj.physics = PhysicalProperty(mass=10.0, is_passable=False)
                
        print("[Agent LLM] 物理属性赋予完毕。")

        # 4. 提取 Action
        action_data = res_data["action"]
        
        # 向量归一化安全校验，防止推力因为方向向量过大而失真
        dir_x, dir_y = action_data["direction"]
        norm = (dir_x**2 + dir_y**2)**0.5
        if norm > 0:
            dir_x, dir_y = dir_x / norm, dir_y / norm
        else:
            dir_x, dir_y = 1.0, 0.0

        return NavigationAction(
            action_type=action_data["action_type"],
            direction=(dir_x, dir_y),
            apply_force=float(action_data["apply_force"]),
            duration=3.0
        )
