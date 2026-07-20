import yaml
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

from src.perception.pipeline import PerceptionPipeline
from src.simulator import SceneBuilder, DAGSimulationEngine, PhysicsBoundaryDetectors
from src.memory.physics_dictionary import PHYSICS_DICTIONARY, get_dino_prior_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class FeedbackTranslator:
    """物理语义翻译器：将底层的 Dataclass 转化为 LLM 可读的富文本因果物理报告"""

    @staticmethod
    def generate_rich_prompt(node: Dict[str, Any], feedback_obj) -> str:
        if not hasattr(feedback_obj, "error_type"):
            return f"❌ [执行失败] 节点 {node.get('id')} 执行物理动作时受阻。底层原音: {feedback_obj}"

        action = node.get("action", "unknown")
        target = node.get("target", "unknown")
        node_id = node.get("id", "unknown")
        error_type = feedback_obj.error_type
        params = node.get("parameters", {})

        prompt = (f"【物理沙盒阻断勘验报告】\n"
                  f"❌ [执行失败] 动作节点 {node_id} (指令: {action} '{target}') 触发了物理底座保护机制。\n"
                  f"⚙️ 物理引擎反馈类型: {error_type}\n"
                  f"📊 现场量化数据与因果诊断如下：\n")

        # 🚀 全局注入遥测数据：如果有空间对齐和物理宽度数据，优先展示，帮助VLM反思
        offset_x = getattr(feedback_obj, "tcp_offset_x_mm", None)
        offset_y = getattr(feedback_obj, "tcp_offset_y_mm", None)
        actual_w = getattr(feedback_obj, "actual_width_mm", None)
        target_w = getattr(feedback_obj, "target_width_mm", None)

        if offset_x is not None and offset_y is not None:
            prompt += f"- 🔍 空间对齐诊断: 夹爪 TCP 中心偏离物体几何中心 (X轴偏差: {offset_x:.2f} mm, Y轴偏差: {offset_y:.2f} mm)。\n"
        if actual_w is not None and target_w is not None:
            prompt += f"- 🧠 物理联想诊断: 物体真实物理宽度为 {actual_w:.2f} mm，你的指令设定抓取宽度为 {target_w:.2f} mm。\n"

        # ---------------------------------------------------------------------
        # 探针〇：目标实体缺失 (TARGET_NOT_FOUND)
        # ---------------------------------------------------------------------
        if error_type == "TARGET_NOT_FOUND":
            missing = getattr(feedback_obj, "collided_pair", [target, ""])[0]
            prompt += f"- 缺失实体: [{missing}]\n"
            prompt += (f"💡 [因果分析与修正建议]:\n"
                       f"沙盒孪生宇宙中未找到目标 '{missing}'。\n"
                       f"这说明视觉大模型未能在场景中识别到该物体，可能因为视觉特征与文本语义不匹配（例如把平坦的方块叫作 tray）。"
                       f"建议尝试更直白的几何视觉描述，如 'grey box'。")

        # ---------------------------------------------------------------------
        # 探针一：计划外碰撞 (UNEXPECTED_COLLISION)
        # ---------------------------------------------------------------------
        elif error_type == "UNEXPECTED_COLLISION":
            actor0, actor1 = getattr(feedback_obj, "collided_pair", ("unknown", "unknown"))
            raw_xyz = getattr(feedback_obj, "collision_xyz", None)

            prompt += f"- 碰撞实体: [{actor0}] 与 [{actor1}] 发生了计划外的刚体几何干涉。\n"

            # ✅ 修复空值异常：区分是“事后真实碰撞坐标”还是“事前规划预判”
            if raw_xyz is not None:
                xyz = np.round(raw_xyz, 3)
                prompt += f"- 绝对坐标: 世界坐标系碰撞点发生于 (X: {xyz[0]}, Y: {xyz[1]}, Z: {xyz[2]})\n"
            else:
                prompt += f"- 绝对坐标: 预判拦截 (轨迹规划阶段发现必将发生深度穿模)。\n"

            if "finger" in actor0 or "finger" in actor1:
                prompt += (f"💡 [因果分析与修正建议]:\n"
                           f"夹爪末端即将或已经直接撞击表面。请利用上方的【空间对齐诊断】数据修正你的抓取/放置坐标。\n")
            elif action == "place":
                prompt += f"💡 [因果分析与修正建议]: 负载在移动中发生扫碰，或目标位置 (drop_height) 存在物理干涉。建议调整拓扑顺序，或修正安全高度。\n"
            else:
                prompt += f"💡 [因果分析与修正建议]: 机械臂避障净空不足。建议重新审视场景布局或提高悬停高度。\n"

        # ---------------------------------------------------------------------
        # 探针二：刚度极限与结构破坏 (MECHANICAL_DESTRUCTION)
        # ---------------------------------------------------------------------
        elif error_type == "MECHANICAL_DESTRUCTION":
            culprit = getattr(feedback_obj, "culprit_actor", "unknown")

            # ✅ 修复空值异常：安全提取预估力或真实碰撞力
            p_force = getattr(feedback_obj, "predicted_force_N", None)
            m_force = getattr(feedback_obj, "max_normal_force_N", None)
            actual_force = p_force if p_force is not None else (m_force if m_force is not None else 0.0)
            force = round(actual_force, 2)

            limit = round(getattr(feedback_obj, "safety_threshold_N", 0.0), 2)

            prompt += f"- 破坏现场: 预期瞬时法向挤压力达到 {force}N，超出了物体的安全屈服极限 ({limit}N)。\n"
            prompt += f"- 施力源头: [{culprit}]\n"
            prompt += (f"💡 [因果分析与修正建议]:\n"
                       f"典型的【夹具/碰撞碎裂】现象！你的设定宽度过小或发生了剧烈的运动学碰撞，导致物理引擎预估产生了毁灭性的挤压力。\n"
                       f"必须确保 `target_width` 贴近目标物体的实际物理厚度，严禁强行闭合。如果是路径碰撞，请提高悬停高度。")

        # ---------------------------------------------------------------------
        # 探针三：抓空与滑脱 (KINEMATIC_SLIP)
        # ---------------------------------------------------------------------
        elif error_type == "KINEMATIC_SLIP":
            predicted_force = round(getattr(feedback_obj, "predicted_force_N", 0.0), 2)
            min_force = round(getattr(feedback_obj, "min_required_grip_force_N", 0.0), 2)
            prompt += f"- 滑脱现场: 预估产生的夹紧力 ({predicted_force}N) 小于克服目标重力所需的最小静摩擦力 ({min_force}N)。\n"
            prompt += (f"💡 [因果分析与修正建议]:\n"
                       f"典型的【抓空/滑脱】现象。挤压力不足以产生足够的静摩擦力。\n"
                       f"建议在 parameters 中结合物体的真实物理宽度，适当调小 `target_width` 以产生贴合阻力。")

        # ---------------------------------------------------------------------
        # 探针四：重心倾覆与拓扑失稳 (TOPOLOGICAL_TIPPING)
        # ---------------------------------------------------------------------
        elif error_type == "TOPOLOGICAL_TIPPING":
            culprit = getattr(feedback_obj, "culprit_actor", "unknown")
            tilt = round(getattr(feedback_obj, "tilt_angle_deg", 0.0), 1)
            limit = round(getattr(feedback_obj, "max_allowed_angle", 0.0), 1)
            prompt += f"- 坍塌源头: 实体 [{culprit}] 发生了 {tilt}° 的倾覆滑落 (系统允许上限: {limit}°)。\n"
            prompt += f"💡 [因果分析与修正建议]: 目标未能形成稳定支撑。避免“大物叠小物”，优化放置策略。"

        # ---------------------------------------------------------------------
        # 探针五：运动学死锁与动力学超载 (KINEMATIC_DYNAMIC_DEADLOCK)
        # ---------------------------------------------------------------------
        elif error_type == "KINEMATIC_DYNAMIC_DEADLOCK":
            j_id = getattr(feedback_obj, "saturated_joint_id", "Unknown")
            torque = round(getattr(feedback_obj, "current_torque_Nm", 0.0), 2)
            prompt += f"- 饱和关节: Joint_{j_id} | 瞬时扭矩飙升至 {torque}Nm。\n"
            prompt += f"💡 [因果分析与修正建议]: 机械臂由于奇异点或距离过远导致逆运动学无解或动力学超载。建议调整工作域或姿态。"

        # =====================================================================
        # 🚀 核心升级：注入物理底座的黑匣子全景时序记录仪 (Blackbox Telemetry)
        # =====================================================================
        blackbox_data = getattr(feedback_obj, "blackbox_log", "")
        if blackbox_data:
            prompt += f"\n\n📦 [物理底座黑匣子飞行记录仪 (Blackbox Telemetry)]:\n"
            prompt += f"以下是物理引擎在发生错误前后的连续时间切片或全景状态快照，包含三维坐标、绝对倾角及实时受力拓扑，请仔细分析轨迹与受力变化：\n"
            prompt += f"```text\n{blackbox_data}\n```\n"

        prompt += "\n👉 [指令要求] 请 Planning Agent 严格依据上述空间与受力量化数据，并结合【黑匣子飞行记录仪】重构物理认知模型，查明真正的干涉原因，并输出修正参数或前置依赖后的全新 DAG 计划图。"
        return prompt


class SimulationSandboxAPI:
    def __init__(self, config_path: str = "config/global_config.yaml"):
        self.config_path = str(PROJECT_ROOT / config_path)
        with open(self.config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)

        # 🚀 动态加载并挂载剥离出去的机器人本体配置
        robot_cfg_path = PROJECT_ROOT / "config/robot_panda.yaml"
        if robot_cfg_path.exists():
            with open(robot_cfg_path, 'r') as f:
                self.config_dict.update(yaml.safe_load(f))

        print("[Simulation_API] 📥 正在全量冷启动物理沙盒考场...")
        self.perception = PerceptionPipeline(self.config_path)
        print("[Simulation_API] 🎉 沙盒微服务已就绪。")

    def evaluate_vlm_blueprint(
            self,
            rgb_views: Dict[str, np.ndarray],
            depth_views: Dict[str, np.ndarray],
            dynamic_extrinsics: Dict[str, np.ndarray],
            dynamic_intrinsics: Dict[str, np.ndarray],
            plan_dag: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        start_eval_time = time.time()

        dino_prompt = get_dino_prior_prompt()
        print(f"[Simulation_API] 🧠 注入物理先验咒语: '{dino_prompt}'")

        print("\n[Simulation_API] >>> Step 1: 启动早融合 3D 架构提取毫米级空间几何刚体...")
        vision_out = self.perception.process_scene(
            rgb_views=rgb_views,
            depth_views=depth_views,
            dynamic_extrinsics=dynamic_extrinsics,
            dynamic_intrinsics=dynamic_intrinsics,
            text_prompt=dino_prompt
        )

        print("[Simulation_API] >>> Step 2: 正在将视觉包围盒打入 SAPIEN 构建孪生宇宙...")
        simulator = SceneBuilder(self.config_dict, PHYSICS_DICTIONARY)
        try:
            simulator.build_twin_world(vision_out)
            simulator.export_debug_snapshot("inputs/twin_world_debug.png")
        except Exception as e:
            return {"evaluation_status": "SCENE_BUILD_FAILED", "error_details": str(e), "timestamp": time.time()}

        print("[Simulation_API] >>> Step 3: 正在给虚拟小脑挂载四大物理痛觉探针...")
        detectors = PhysicsBoundaryDetectors(self.config_dict)

        graph_engine = DAGSimulationEngine(
            scene=simulator.scene,
            config=self.config_dict,
            detectors=detectors,
            physics_dict=PHYSICS_DICTIONARY
        )

        print("[Simulation_API] >>> Step 4: DAG拓扑预演器接管，启动动作切片验证...")
        execution_report = graph_engine.execute_dag_plan(plan_dag)

        if execution_report.get("evaluation_status") == "REPLAN_REQUIRED":
            failed_node = {
                "id": execution_report.get("failed_node"),
                "action": execution_report.get("failed_action"),
                "target": execution_report.get("failed_target"),
                "parameters": next((node.get("parameters", {}) for node in plan_dag if
                                    node["id"] == execution_report.get("failed_node")), {})
            }
            raw_feedback = execution_report.get("physics_feedback")

            # 🚀 调用增强的翻译器
            execution_report["vlm_reflection_prompt"] = FeedbackTranslator.generate_rich_prompt(failed_node,
                                                                                                raw_feedback)

            if hasattr(raw_feedback, "to_dict"):
                execution_report["physics_feedback"] = raw_feedback.to_dict()
            elif hasattr(raw_feedback, "__dataclass_fields__"):
                execution_report["physics_feedback"] = asdict(raw_feedback)
            else:
                execution_report["physics_feedback"] = str(raw_feedback)

        simulator.scene = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        execution_report["total_evaluation_cost_sec"] = time.time() - start_eval_time
        print(f"[Simulation_API] >>> 🏁 评测结束。最终沙盒裁决状态: {execution_report['evaluation_status']}\n")

        return execution_report
