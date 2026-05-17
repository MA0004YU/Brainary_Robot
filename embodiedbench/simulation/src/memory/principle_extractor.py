# simulation/src/memory/principle_extractor.py

import json
from openai import OpenAI


class PrincipleExtractor:
    def __init__(self, global_config: dict):
        self.config = global_config
        cfg_llm = self.config["llm_config"]

        # 初始化
        self.client = OpenAI(api_key=cfg_llm["api_key"])
        self.model_name = cfg_llm["model_name"]
        self.temperature = cfg_llm["temperature_reflection"]

        # 规定反思大模型范式输出抽象的因果链
        self.sys_prompt = (
            "你是一个具身智能物理常识反思专家。你的任务是分析底层的物理违规数据，"
            "并结合宏观任务上下文，提炼出一条具有高度概括性的【三段式场景剪枝原则】。\n\n"
            "【强制输出格式】\n"
            "你的输出必须且只能是一个纯文本字符串，格式严格满足：\n"
            "[前置场景条件] + [无效动作模式] -> [重规划剪枝指令]\n\n"
            "【黄金示例】\n"
            "示例 1：[正上方存在重铁块压迫] + [向水平方向强行拉拽目标] -> [必须先将上方重铁块搬离，严禁直接水平拉拽]\n"
            "示例 2：[目标件处于密闭受限窄槽内] + [采用宽口夹爪直接下潜抓取] -> [必须改用垂直窄边接触连杆或先执行抽屉式拉出，禁止原位强行合拢]"
        )

    def extract_scene_based_principle(self, failure_payload: dict, wm_context: dict) -> str:
        """
        【纯内存常识反思中枢】
        调用大模型执行跨模态认知升维，直接返回原则字符串，绝不留存硬盘痕迹
        """
        print("因果链反思...")

        user_prompt = (
            f"当前任务最高目标：\n{json.dumps(wm_context['task_context'], indent=2)}\n\n"
            f"视觉解算出的场景初始空间拓扑干涉：\n{json.dumps(wm_context['scene_topology'], indent=2)}\n\n"
            f"底层传感器探针熔断时的瞬时微观物理惨状 (F=I/dt, 电机死锁扭矩, 倾角违规)：\n{json.dumps(failure_payload, indent=2)}\n\n"
            "请根据上述物理惨状与常识，直接吐出符合格式要求的反思剪枝原则字符串，严禁携带任何前言、后记或 Markdown 标记。"
        )

        try:
            response = self.client.chat.comeline_clean_call = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            rule_str = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[PrincipleExtractor ERROR] 反思模型调用溃败: {e}")
            rule_str = "[遭遇未知物理边界违规] + [执行原开环轨迹动作] -> [必须熔断当前搜索树分支，强制变更技能原语参数]"

        return rule_str
