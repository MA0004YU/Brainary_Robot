#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
#  planning_bridge —— planning 模块的"三样"实时文件通道
# ======================================================================================
#  planning 同学只需要那三样(物体 / 可执行动作 / 物理约束),要它们实时写在【一个文件】里随时读。
#  这个 bridge 就干这件事:
#     记忆侧(有 pipe 的进程)每步调 publish(pipe)  -> 把三样【原子覆盖】写到 PLANNING_INPUT_PATH
#     planner 侧(你的进程)每步调 read()          -> 读到最新的三样(dict)
#
#  底层就是记忆 7.2 的 PerceptionMemoryPipeline.export_planning_input()(已实现,原子写 .tmp->replace)。
#  文件格式(只含三样):
#     {
#       "task_instruction": "...",
#       "manipulable_objects": {"香蕉": ["grasp","place"], ...},          # ①物体 + 可操作能力
#       "available_skills":    ["move_above","descend","grasp",...],       # ②可执行动作
#       "constraints": {                                                    # ③物理规则与约束
#          "category_rules": {...}, "no_category_mixing": true,
#          "collision_avoidance": true, "occlusions": [ {...} ] }
#     }
# ======================================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# 约定的实时文件路径(记忆写、planner 读,同一个)。可按需改。
PLANNING_INPUT_PATH = "logs/planning/planning_input.json"


def publish(pipe, path: str = PLANNING_INPUT_PATH) -> Dict[str, Any]:
    """【记忆侧】每步调:把当前的三样(物体/动作/约束)原子写到 path。

    pipe: PerceptionMemoryPipeline 实例(已 begin_episode)。返回写入的 dict。
    """
    return pipe.export_planning_input(path)


def read(path: str = PLANNING_INPUT_PATH) -> Dict[str, Any]:
    """【planner 侧】每步调:读最新的三样。返回 {task_instruction, manipulable_objects,
    available_skills, constraints};文件还没生成时返回空壳(不抛异常)。"""
    p = Path(path)
    if not p.is_file():
        return {"task_instruction": "", "manipulable_objects": {}, "available_skills": [], "constraints": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # 极小概率读到写一半(已用原子 replace 规避);读失败就返回空壳,下一帧再读。
        return {"task_instruction": "", "manipulable_objects": {}, "available_skills": [], "constraints": {}}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else PLANNING_INPUT_PATH
    d = read(path)
    print(f"[planning_bridge] read {path}:")
    print(json.dumps(d, ensure_ascii=False, indent=2))
