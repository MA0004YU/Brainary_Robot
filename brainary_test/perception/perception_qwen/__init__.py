# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""perception_qwen -- 把本地 Qwen2.5-VL-3B 识别器(perception_vlm.py)接进仿真测试。

与 scene_describer(GPT-5.5 API)平行的第二个感知后端:
  * perception_vlm.py          外部开发的识别器(原样保留,不改):QwenVLMPerception + 记忆/复核/辅助视图。
  * qwen_perception_server.py  独立 venv 的 HTTP /recognize,启动时常驻加载一次 3B 模型(占 GPU)。
  * qwen_client.py             isaac 端零依赖客户端:抓两路 RGB -> b64 -> POST -> 解析。

UI 里通过 describe 面板的后端选择切换 GPT / Qwen(单卡共用:GPT 走 API 不占显存,Qwen 本地占显存)。
"""
