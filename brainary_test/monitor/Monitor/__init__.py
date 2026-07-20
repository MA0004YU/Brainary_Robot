"""Monitor —— 感知-记忆流水线的安全监控层。

汇集两个安全相关子模块，都在「记忆/规划之后」运行、都可独立验证：

  ssp_pkg/        vendored「场景安全解析器 SSP」引擎（风险模板 + 传播）
  ssp_adapter/    SSP 适配层：记忆快照 -> PerceptualGraph -> 候选安全约束（CT-*）
  safety_critic/  vendored「Safety Critic」+ 适配：读 plan 逐步判 malicious / not malicious

流水线中的阶段编号：
  [4/5] SSP           场景风险解析（产出候选约束，给规划参考）
  [5/5] SafetyCritic  逐动作安全裁判（对规划出的 plan 做安全评价）
"""
