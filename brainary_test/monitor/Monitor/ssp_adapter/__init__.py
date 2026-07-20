"""SSP 接入适配层。

把「感知-记忆流水线」的记忆快照，转成「场景安全解析器 SSP」的 PerceptualGraph 输入，
跑 SSP 后把候选安全约束（CT-id）回写。设计成可独立运行（吃现成 output/*.json，不碰 torch）。

模块：
  object_attribute_map  中文物体名/category -> EntityType + subtype + StateSchema
  demo_relations        DEMO 硬编码物体间 L0 关系（真机应由感知/记忆产出）
  memory_to_gp          核心适配器：memory_snapshot.json -> PerceptualGraph
  constraint_writer     QueryResult -> 独立诊断文件 + 合并进 planning_input
  ssp_runner            编排：记忆 -> G_P -> SSP -> 回写
  __main__              独立入口（python -m Monitor.ssp_adapter）
"""
