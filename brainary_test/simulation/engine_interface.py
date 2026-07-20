import os
from typing import Dict, Any, List
import numpy as np


from src.api.simulation_service import SimulationSandboxAPI


class PhysicalValidator:
    """
    提供给上层 PM / LLM Agent 的局部动态沙盒验证接口
    """

    def __init__(self, config_path: str = "config/global_config.yaml"):
        # 确保环境变量配了 Grounded-SAM
        if "GSA_PATH" not in os.environ:
            raise EnvironmentError("🚨 请在环境中设置 GSA_PATH 环境变量指向 Grounded-Segment-Anything！")

        print("🚀 [PhysicalValidator] 正在初始化局部沙盒生成器... (加载多模态与物理引擎)")
        self.sandbox = SimulationSandboxAPI(config_path)
        print("✅ [PhysicalValidator] 初始化完毕，随时准备接收视觉输入与 Plan。")

    def verify_local_plan(
            self,
            rgb_views: Dict[str, np.ndarray],
            depth_views: Dict[str, np.ndarray],
            dynamic_extrinsics: Dict[str, np.ndarray],
            dynamic_intrinsics: Dict[str, np.ndarray],
            plan_dag: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        【核心即时推演接口】
        传入当前作业区域的多视角图像与 LLM 计划，当场重构局部沙盒并验证，返回物理痛觉反馈。
        """
        print(f"👁️‍🗨️ [PhysicalValidator] 接收到新的视觉数据与 Plan(节点数:{len(plan_dag)})，开始重构与预演...")

        # 完整的全量闭环：每次都重新提取早融合点云 -> 建沙盒 -> 跑拓扑 -> 销毁沙盒
        report = self.sandbox.evaluate_vlm_blueprint(
            rgb_views=rgb_views,
            depth_views=depth_views,
            dynamic_extrinsics=dynamic_extrinsics,
            dynamic_intrinsics=dynamic_intrinsics,
            plan_dag=plan_dag
        )

        status = report.get("evaluation_status", "UNKNOWN_ERROR")

        if status == "PASS":
            return {
                "success": True,
                "optimized_plan": report.get("optimized_plan"),
                "message": "✅ 局部沙盒验证通过，无任何干涉与倾覆，可以下发真机执行。",
                "cost_sec": report.get("total_evaluation_cost_sec")
            }
        elif status == "REPLAN_REQUIRED":
            return {
                "success": False,
                "failed_node_id": report.get("failed_node"),
                "llm_reflection_prompt": report.get("vlm_reflection_prompt"),
                "physics_raw_data": report.get("physics_feedback"),
                "message": "❌ 动作被物理法则拒绝，请将 llm_reflection_prompt 喂给 Agent 进行反思重规划。",
                "cost_sec": report.get("total_evaluation_cost_sec")
            }
        else:
            return {
                "success": False,
                "message": f"🚨 发生底层严重异常: {report.get('error_details', 'Unknown')}"
            }
