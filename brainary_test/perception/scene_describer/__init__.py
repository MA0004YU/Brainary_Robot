# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""scene_describer -- 把两路相机 RGB + 场景状态喂给 GPT(Responses API),拿回结构化场景描述。

两块解耦:
  * vlm_describe_server.py  在独立 venv 起 HTTP /describe(持有 openai SDK + API key),
    收 {rgb_front_b64, rgb_wrist_b64, state{...}} -> 调 gpt-5.5 视觉 -> 返回
    {scene_summary, objects[], relations[]}(JSON Schema 强约束)。
  * describer_client.py     isaac 进程内(零新依赖,只 stdlib urllib):从 SceneSession 抓两图+状态、
    base64 打包、POST 到 server、解析返回。
  * describe_panel.py       omni.ui 按钮面板,点一下描述当前帧(HTTP 跑在后台线程,不卡仿真)。
"""
