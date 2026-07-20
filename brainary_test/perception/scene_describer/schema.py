# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GPT 输出的 JSON Schema(结构化场景描述) + 文本提示拼装。纯 python,无第三方依赖。"""

from __future__ import annotations

import json

# Responses API text.format=json_schema 的强约束 schema(strict 要求每个 object 全字段 required
# 且 additionalProperties=False)。物品描述 + 物品间关系两张表。
SCENE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scene_summary": {
            "type": "string",
            "description": "One or two sentences summarizing the whole tabletop scene.",
        },
        "objects": {
            "type": "array",
            "description": "Every distinct object you can identify in the two views.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Use the sim object name (e.g. cube_1, knife, fridge, "
                                       "middle_drawer) when the object maps to one in the provided "
                                       "state; otherwise a short descriptive name.",
                    },
                    "category": {"type": "string", "description": "e.g. cube, knife, appliance, drawer."},
                    "appearance": {"type": "string", "description": "Color / shape / material from the images."},
                    "location": {"type": "string", "description": "Where it sits in the scene, in words."},
                    "state": {
                        "type": "string",
                        "description": "Open/closed, grasped, upright/tipped, empty/occupied, etc. "
                                       "Ground this in the numeric state when available.",
                    },
                },
                "required": ["name", "category", "appearance", "location", "state"],
            },
        },
        "relations": {
            "type": "array",
            "description": "Pairwise spatial / state relations between objects.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {
                        "type": "string",
                        "description": "e.g. on, in, inside, left_of, right_of, in_front_of, behind, "
                                       "near, on_top_of, held_by, supports.",
                    },
                    "object": {"type": "string"},
                    "description": {"type": "string", "description": "Natural-language version of the relation."},
                },
                "required": ["subject", "predicate", "object", "description"],
            },
        },
    },
    "required": ["scene_summary", "objects", "relations"],
}

_SYSTEM = (
    "You are a robotics scene-understanding model for a Franka Panda tabletop manipulation setup. "
    "You receive two camera views and ground-truth numeric state from the simulator. "
    "Report exactly what is present and how objects relate, grounded in BOTH the images and the "
    "numeric state. Prefer the provided sim object names. Do not invent objects that are neither "
    "visible nor in the state. Return ONLY the structured JSON requested."
)

# 语言指令:控制自然语言字段(scene_summary/appearance/location/state/description)的语言。
# name 永远保留 sim 名(cube_1/middle_drawer…)以便和状态机对齐;predicate 保留英文谓词词汇。
_LANG_DIRECTIVE = {
    "zh": (
        " 重要:所有自然语言字段(scene_summary、每个 object 的 appearance/location/state、每条 "
        "relation 的 description)一律用简体中文填写。但 object 的 name 字段保留仿真原名"
        "(如 cube_1、knife、middle_drawer、fridge),relation 的 predicate 保留英文谓词"
        "(如 on/in/left_of)。"
    ),
    "en": "",
}


def build_input_text(state: dict, lang: str = "zh") -> str:
    """拼用户侧文本提示:说明两图含义 + 注入数值状态(物体/把手/关节/夹爪)做 grounding。"""
    state_json = json.dumps(state, ensure_ascii=False, indent=2, default=_jsonable)
    tail = "\n用中文描述每个物体及其相互关系。" if lang == "zh" else ""
    return (
        "Two images follow:\n"
        "  1) FRONT view  -- third-person static camera looking at the table.\n"
        "  2) WRIST view  -- eye-in-hand camera on the robot's gripper.\n\n"
        "Ground-truth simulator state (use it to resolve names, occlusions, and open/closed amounts; "
        "positions are metres, drawer joints are prismatic metres, door joints are radians):\n"
        f"{state_json}\n\n"
        "Task: describe each object and the relations between them. Map image objects to the sim "
        "names above wherever possible. For appliances/drawers/doors, state open vs closed using the "
        "joint values (a drawer joint > ~0.05 m is open; a door joint > ~0.1 rad is open)."
        f"{tail}"
    )


def system_prompt(lang: str = "zh") -> str:
    return _SYSTEM + _LANG_DIRECTIVE.get(lang, "")


def _jsonable(o):
    try:
        import numpy as np  # noqa: PLC0415 - optional, only if numpy types leak in

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
