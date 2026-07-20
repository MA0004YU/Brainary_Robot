# simulation/src/plan/llm_planner.py

import json
from openai import OpenAI
from typing import Dict, Any, List

class LLMPlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["llm_config"]
        self.client = OpenAI(api_key=self.config["api_key"])
        self.model = self.config.get("model_name", "gpt-4-turbo")
        self.sys_prompt = (
            "你是一个具身智能的宏观任务与运动规划器 (TAMP Planner)。\n"
            "你的任务是将高层目标拆解为机械臂可执行的序列，你必须且只能输出一个合法的 JSON 数组，数组中的每个对象代表一个执行步骤。\n\n"

            "【基础字段要求】\n"
            "每个动作节点必须包含三个基础字段：'step' (执行序号), 'action' (技能名称), 'target' (交互目标ID)。\n\n"

            "【技能库 API 及强参数规范】 (严禁虚构其他技能)：\n"
            "1. 'approach': 移动至目标物体上方准备抓取。仅需基础字段。\n"
            "2. 'pick_and_place': 抓取目标并放置到安全区域。\n"
            "   - 强制附加字段: 'placement_offset' (格式: [x, y, z] 的浮点数数组)。\n"
            "   - 约束: 表示相对于目标原位置的平移量(米)。严禁输出绝对世界坐标！例如 [0.3, 0.0, 0.0] 表示向 X 轴正方向平移 30cm 放置。\n"
            "3. 'pull_straight': 夹持目标并沿指定方向进行笛卡尔直线平移拖拽。\n"
            "   - 强制附加字段: 'direction' (格式: [x, y, z] 的浮点数数组)。\n"
            "   - 约束: 必须是数学意义上的归一化方向向量 (模长为1)！例如 [0.0, -1.0, 0.0] 表示纯向 Y 轴负方向拉拽。\n"
            "4. 'retract': 释放物体并安全撤回机械臂。仅需基础字段。\n\n"

            "【绝对物理红线】\n"
            "当你收到由底层物理引擎返回的【重规划剪枝指令】时，该指令具有最高优先级。你必须绝对服从，彻底在搜索空间中砍掉导致死锁或碰撞的分支，调整你的 action 选择或 direction 向量参数，绕开物理死胡同！"
        )

    def _call(self, prompt: str) -> List[Dict[str, Any]]:
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": prompt}],
                temperature=self.config.get("temperature_plan", 0.1),
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            # 适配不同的 JSON 返回外壳
            return data.get("plan", data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        except Exception as e:
            print(f"[Planner] LLM API 请求失败: {e}")
            return []

    def generate_initial_global_plan(self, wm_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._call(f"目标:{wm_context['task_context']['target_object_id']}\n拓扑:{json.dumps(wm_context['dynamic_scene_graph'])}")

    def replan_triggered_by_event(self, wm_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._call(f"最新报错:{json.dumps(wm_context['active_cognitive_payload']['latest_white_box_feedback'])}\n原则库:{json.dumps(wm_context['active_cognitive_payload']['retrieved_principles'])}\n拓扑:{json.dumps(wm_context['dynamic_scene_graph'])}")
