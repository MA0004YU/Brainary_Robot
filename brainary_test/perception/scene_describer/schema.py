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
            "description": "Every distinct object you can identify across the provided views.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Use the sim object name (e.g. cube_1, knife, fridge, "
                                       "middle_drawer) when the object maps to one in the provided "
                                       "state; otherwise a short descriptive name. "
                                       "NOTE: when the caller supplies a candidate id list, this "
                                       "field is replaced by a strict enum (see build_scene_schema).",
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

# 调用方没给候选清单时,GPT 只能自由起名(white_cup / yellow_curved_object …),下游 _resolve 靠
# 别名表硬猜、措辞一变就 unresolved。给了清单就把 name 收成【严格枚举】:模型只能从仿真真实存在的
# id 里选,名字对齐问题从源头消失。枚举里额外留一个 OTHER_ID 给"确实不在清单里的东西"(桌子/背景/
# 幻觉),这类对象下游会被当不可抓目标跳过 —— 这正是我们要的行为。
OTHER_ID = "other"


def build_scene_schema(candidates=None) -> dict:
    """返回 SCENE_SCHEMA;给了 candidates 就把 objects[].name 换成 enum=candidates+[OTHER_ID]。

    strict json_schema 下 enum 是硬约束,模型无法输出清单外的名字。
    """
    import copy

    schema = copy.deepcopy(SCENE_SCHEMA)
    ids = [str(c) for c in (candidates or []) if str(c).strip()]
    if not ids:
        return schema
    if OTHER_ID not in ids:
        ids = ids + [OTHER_ID]
    name_prop = schema["properties"]["objects"]["items"]["properties"]["name"]
    name_prop["enum"] = ids
    name_prop["description"] = (
        "The simulator id of this object. MUST be exactly one of the enum values. "
        f"Use '{OTHER_ID}' ONLY for something that is genuinely none of the listed ids "
        "(background, table surface, an object not in the list). Never invent a new name -- "
        "put your own wording in 'appearance' instead."
    )
    return schema


_SYSTEM = (
    "You are a robotics scene-understanding model for a Franka Panda tabletop manipulation setup. "
    "You receive several camera views (each labelled in the user message) and ground-truth "
    "numeric state from the simulator. "
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


def build_input_text(state: dict, lang: str = "zh", candidates=None, view_names=None) -> str:
    """拼用户侧文本提示:说明两图含义 + 注入数值状态(物体/把手/关节/夹爪)做 grounding。

    candidates:仿真里真实存在的物体 id 清单。给了就在提示里显式列出并要求 name 只能从中选
    (schema 那边同时上 enum 硬约束,双保险)。
    """
    state_json = json.dumps(state, ensure_ascii=False, indent=2, default=_jsonable)
    tail = "\n用中文描述每个物体及其相互关系。" if lang == "zh" else ""
    cand_block = ""
    ids = [str(c) for c in (candidates or []) if str(c).strip()]
    if ids:
        cand_block = (
            "\nALLOWED OBJECT IDS (the 'name' field MUST be exactly one of these, or "
            f"'{OTHER_ID}'):\n"
            + "\n".join(f"  - {i}" for i in ids)
            + f"\n  - {OTHER_ID}   (only if it is none of the above)\n"
            "Do NOT invent descriptive names like 'white_cup' or 'yellow_curved_object' -- "
            "pick the matching id above and put your wording in 'appearance'.\n"
        )
    # 视角说明按【实际发过来的图】动态生成 —— 以前写死"两张图 front+wrist",而 front 离桌面 ~2.4m
    # (每个物体才二三十像素、还被机械臂挡)、wrist 只是个特写,导致香蕉/橘子漏检、红盒子被糊成
    # "一组小物体"。top(俯视)才是这个分拣任务信息量最大的一路,必须一起发。
    _VIEW_DESC = {
        "top": "TOP view    -- overhead camera straight down at the table. BEST view for "
               "enumerating every object and its layout; rely on it most.",
        "front": "FRONT view  -- third-person static camera ~2.4 m away; objects are small and the "
                 "robot arm occludes part of the table. Use only as a cross-check.",
        "wrist": "WRIST view  -- eye-in-hand close-up on the gripper; shows only whatever is right "
                 "in front of the hand, at high magnification.",
        "left": "LEFT view   -- static camera from the robot's left side.",
        "right": "RIGHT view  -- static camera from the robot's right side.",
    }
    names = list(view_names or ["front", "wrist"])
    head = f"{len(names)} images follow (in this order):\n" + "".join(
        f"  {i + 1}) {_VIEW_DESC.get(n, n.upper() + ' view')}\n" for i, n in enumerate(names)) + "\n"
    return (
        head +
        "Ground-truth simulator state (use it to resolve names, occlusions, and open/closed amounts; "
        "positions are metres, drawer joints are prismatic metres, door joints are radians):\n"
        f"{state_json}\n"
        f"{cand_block}\n"
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
