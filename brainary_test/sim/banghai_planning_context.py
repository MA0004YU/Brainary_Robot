#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
#  PlanningContext —— 规划模块接收的"记忆上下文"数据结构(参考副本 / schema)
# ======================================================================================
#  给规划(planning)同学的【一个文件看全】参考:PlanningContext 的全部字段 + 说明 + 序列化方法。
#
#  ★ 这是从记忆模块同步过来的【参考副本】,真身(运行时用的那个)在:
#       projects/xiaoyu/brainary_memory_pkg (7.2)/memory_module/planning_interface.py
#     记忆同学出新版时,以真身为准;本文件仅供查阅/独立 import schema。
#
#  你【不用自己造】PlanningContext,是从记忆拿一个填好的实例:
#       from memory_module import PerceptionMemoryPipeline
#       ctx = pipe.get_planning_context()      # -> 返回一个 PlanningContext 对象
#       ctx.visible_objects / ctx.available_skills / ctx.constraints ...
#       ctx.to_prompt_text()                   # 人类可读文本(可塞进 LLM prompt)
#       ctx.to_dict() / ctx.to_json() / ctx.save("ctx.json")   # 转 dict / JSON / 落盘
#
#  规划需要的【三样】对应字段:
#     ① 物体      -> visible_objects / manipulable_objects / object_attributes / scene_description
#     ② 可执行动作 -> available_skills / manipulable_objects(每物体的 affordances) / recommended_skills
#     ③ 物理约束   -> constraints{category_rules, no_category_mixing, collision_avoidance, occlusions}
# ======================================================================================

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Spine brain 定义的完整 skillset
DEFAULT_AVAILABLE_SKILLS: List[str] = [
    "move_above", "descend", "grasp", "lift", "place", "retreat", "wait", "align_orientation", "reach",
]
# 物体没有历史 affordance 记录时的默认值
DEFAULT_AFFORDANCES: List[str] = ["grasp", "place"]


@dataclass
class PlanningContext:
    """规划模块从记忆系统收到的唯一结构化产物。字段说明见下。

    ── 感知(当前帧)──
    task_instruction     : 当前 episode 的自然语言任务
    task_type            : 推断的任务类别(如 "pick_and_place")
    visible_objects      : 本帧感知到的物体名                        ← ①物体
    scene_description    : VLM 感知的场景文本摘要
    object_attributes    : {物体: {color, shape, texture, confidence, ...}}

    ── 工作记忆 ──
    memory_objects       : 已知但当前不一定可见的物体
    current_location     : 机器人语义位置(如 "table")
    visited_locations    : {位置: 状态}
    held_object          : 当前夹着的物体(无则 None)
    recent_actions       : 本 episode 最近 ≤10 条动作记录

    ── 语义记忆(历史)──
    task_success_rate    : 该任务类型历史成功率 (0.0–1.0)
    common_steps         : 最常见动作序列
    recommended_skills   : 历史最优 Blueprint 的技能序列               ← ②动作(推荐)
    object_location_priors : {物体: [位置, ...]} 按频率排序
    user_preferences     : {功能需求: {preferred:[...], avoided:[...]}}

    ── 情景记忆 ──
    similar_episodes     : 最多 K 条相似历史 episode 摘要

    ── 三个新增(7.2)──
    manipulable_objects  : {物体: [affordance, ...]} 可操作物体+能力    ← ①物体 / ②动作
                           来源:语义记忆 ObjectKB.affordances,无历史默认 ["grasp","place"]
    available_skills     : 当前可调用的完整 skill 列表                 ← ②可执行动作
                           来源:脊脑接口静态定义 + TaskSchema 动态补充
    constraints          : 物理规则与环境约束                          ← ③物理约束
                           - category_rules   : {类别: 目标篮子}  (Planner 注入)
                           - no_category_mixing : bool  是否禁止跨类别混放
                           - collision_avoidance : bool  是否启用碰撞检测
                           - occlusions       : [{blocked, blocker, blocker_on_list, target}]
                                                (Simulation 经 pipeline.record_occlusion() 写入)
                           来源:working memory intent["constraints"](Planner 注入)
                                 + clock milestones 里的 occlusion 记录
    """

    task_instruction: str = ""
    task_type: str = ""

    # ① 感知(当前帧)
    visible_objects: List[str] = field(default_factory=list)
    scene_description: str = ""
    object_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 工作记忆:先前步骤已知物体
    memory_objects: Dict[str, Any] = field(default_factory=dict)

    # 导航上下文
    current_location: str = "unknown"
    visited_locations: Dict[str, str] = field(default_factory=dict)
    held_object: Optional[str] = None

    # 语义记忆:任务历史
    task_success_rate: Optional[float] = None
    common_steps: List[str] = field(default_factory=list)
    recommended_skills: List[str] = field(default_factory=list)

    # 语义记忆:空间先验
    object_location_priors: Dict[str, List[str]] = field(default_factory=dict)

    # 语义记忆:用户偏好
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    # 情景记忆:相似历史
    similar_episodes: List[Dict[str, Any]] = field(default_factory=list)

    # 工作记忆:最近动作
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)

    # ── 7.2 三个新增字段 ──
    manipulable_objects: Dict[str, List[str]] = field(default_factory=dict)   # ①物体+②可操作能力
    available_skills: List[str] = field(default_factory=list)                 # ②可执行动作(完整 skillset)
    constraints: Dict[str, Any] = field(default_factory=dict)                 # ③物理规则与环境约束

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str) -> None:
        """原子落盘到 JSON 文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(p)

    def to_prompt_text(self) -> str:
        """转人类可读文本(可直接注入 LLM prompt)。与记忆真身实现保持一致。"""
        lines = [
            f"Task: {self.task_instruction}",
            f"Task Type: {self.task_type or 'unknown'}",
            "",
            f"Visible Objects: {', '.join(self.visible_objects) or 'none'}",
            f"Current Location: {self.current_location}",
            f"Held Object: {self.held_object or 'none'}",
        ]
        if self.manipulable_objects:
            lines.append("Manipulable Objects (with affordances):")
            for obj, affs in list(self.manipulable_objects.items())[:8]:
                lines.append(f"  {obj}: [{', '.join(affs)}]")
        if self.available_skills:
            lines.append(f"Available Skills: {', '.join(self.available_skills)}")
        if self.constraints:
            lines.append("Constraints:")
            cat_rules = self.constraints.get("category_rules", {})
            if cat_rules:
                lines.append("  Category Rules:")
                for cat, basket in cat_rules.items():
                    lines.append(f"    {cat} → {basket}")
            if self.constraints.get("no_category_mixing"):
                lines.append("  No category mixing allowed across baskets.")
            if self.constraints.get("collision_avoidance"):
                lines.append("  Collision avoidance is active.")
            for oc in self.constraints.get("occlusions", []):
                on_list = oc.get("blocker_on_list", False)
                action = f"→ move directly to {oc.get('target')}" if on_list else "→ temporarily move aside"
                lines.append(f"  {oc.get('blocker')} blocks {oc.get('blocked')} {action}")
        if self.recommended_skills:
            lines.append(f"Recommended Skill Sequence: {' → '.join(self.recommended_skills)}")
        if self.task_success_rate is not None:
            lines.append(f"Historical Success Rate: {self.task_success_rate:.1%}")
        if self.object_location_priors:
            lines.append("Known Object Locations:")
            for obj, locs in list(self.object_location_priors.items())[:5]:
                lines.append(f"  {obj}: likely at {', '.join(str(l) for l in locs[:2])}")
        if self.similar_episodes:
            lines.append(f"Similar Past Episodes ({len(self.similar_episodes)} found):")
            for ep in self.similar_episodes[:2]:
                status = "SUCCESS" if ep.get("success") else "FAILED"
                task_str = ep.get("task_instruction", ep.get("task", ""))[:60]
                skills = ep.get("blueprint_skills") or ep.get("objects_encountered", [])
                lines.append(f"  [{status}] {task_str}")
                if isinstance(skills, list) and skills:
                    lines.append(f"    skills: {' → '.join(str(s) for s in skills[:6])}")
        if self.recent_actions:
            n = min(3, len(self.recent_actions))
            lines.append(f"Recent Actions (last {n}):")
            for act in self.recent_actions[-n:]:
                status = "ok" if act.get("success") else "fail"
                lines.append(f"  step {act.get('step')}: {act.get('action')} [{status}]"
                             + (f" — {act.get('feedback', '')[:40]}" if act.get("feedback") else ""))
        return "\n".join(lines)


if __name__ == "__main__":
    # 打印一个空 PlanningContext 的 schema(全部字段 + 默认值),方便查看结构。
    import dataclasses
    print("PlanningContext 字段:")
    for f in dataclasses.fields(PlanningContext):
        print(f"  {f.name}: {f.type}")
    print("\n空实例 JSON:")
    print(PlanningContext().to_json())
