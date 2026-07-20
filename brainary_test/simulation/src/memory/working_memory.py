# simulation/src/memory/working_memory.py

import time
from typing import Dict, Any, List


class WorkingMemory:
    def __init__(self, task_name: str, target_id: str):
        """
        初始化结构化的短期工作记忆空间
        """
        self.memory: Dict[str, Any] = {
            "task_context": {
                "objective_task": task_name,
                "target_object_id": target_id,
                "success_condition": f"Safe extraction of {target_id} without system boundary violation",
                "attempt_count": 1  # 当前交互迭代轮次计数
            },
            "dynamic_scene_graph": {
                "timestamp": time.time(),
                "objects_topology": []  # 实时从 SAPIEN 反向解算出来的支撑与约束拓扑对
            },
            "action_history": [],  # 动作执行长序列历史归档
            "active_cognitive_payload": {
                "latest_white_box_feedback": None,  # 最后一轮触发中断的强类型反馈字典
                "retrieved_principles": []  # 核心场景剪枝原则库（三段式）
            }
        }

    def update_scene_topology(self, topology_data: List[Dict[str, Any]]):
        """更新仿真环境沉降后的最新 Scene Graph 物理拓扑"""
        self.memory["dynamic_scene_graph"]["timestamp"] = time.time()
        self.memory["dynamic_scene_graph"]["objects_topology"] = topology_data

    def append_action_history(self, skill: Dict[str, Any], status: str):
        """记录开环盲飞序列中每一步语义技能基元的执行状态"""
        self.memory["action_history"].append({
            "skill_primitive": skill,
            "execution_status": status,
            "timestamp": time.time()
        })

    def mount_failure_payload(self, feedback_dataclass, principle_text: str):
        """
        核心数据流咬合点：当物理探针强行熔断时，
        实时将 dataclass 转化为标准字典，并与新生成的剪枝指令一同挂载至中央记忆区
        """
        import dataclasses
        self.memory["active_cognitive_payload"]["latest_white_box_feedback"] = dataclasses.asdict(feedback_dataclass)
        self.memory["active_cognitive_payload"]["retrieved_principles"].append(principle_text)

        # 迭代轮次自增，准备驱动下一次 LLM 被动唤醒重规划
        self.memory["task_context"]["attempt_count"] += 1

    def get_full_context(self) -> Dict[str, Any]:
        """为顶层 Planner 和因果 Extractor 提供全量上下文快照"""
        return self.memory
