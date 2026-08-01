"""Safety Critic 运行编排（适配层）。

把「流水线的 plan + 记忆快照」喂给 vendored SafetyCritic，逐动作评价安全性，写诊断文件。
被两处复用：
  - Monitor/safety_critic/__main__.py   （独立运行，吃现成 output/*.json）
  - main.py 第 5 阶段 run_safety_critic  （端到端流水线里，规划/SSP 之后）

数据来源：
  - planned_actions.json：规划出的动作序列 [{id, action, target, depends_on}]
  - memory_snapshot.json：working.observation（场景对象/位置/手持物）→ 建 scene_graph_dict

评价语义（已与用户确认）：**评完整条 plan**。逐步跑 SafetyCritic：
  - 每步得 malicious / not malicious 裁决；
  - 遇到第一个 malicious 记录"按原语义会在此中断（halt_here=True）"，但**继续评完后续步**给全貌；
  - past_actions 只累加通过（not malicious）的动作，保持 critic 原始语义。

动作字符串：planned_actions 的 target 是中文名（无 PROTEA 的 "<obj> (id)" 形态），
  故 critic 走自然语言裁判路径，input_mode="natural_language"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Monitor.safety_critic.core.context import SafetyContext
from Monitor.safety_critic.core.enums import RiskLevel
from Monitor.safety_critic.modules.m2_scene_parser import SceneSafetyParser
from Monitor.safety_critic.modules.m5_safety_critic import SafetyCritic


@dataclass
class CriticRunResult:
    reviews: list[dict] = field(default_factory=list)   # 逐步裁决
    num_steps: int = 0
    num_malicious: int = 0
    first_halt_index: int | None = None                 # 按原语义会中断的第一步
    total_tokens: int = 0

    def summary(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "num_malicious": self.num_malicious,
            "first_halt_index": self.first_halt_index,
            "overall": "unsafe" if self.num_malicious else "safe",
            "total_tokens": self.total_tokens,
        }


def _action_to_text(step: dict) -> str:
    """把一个 planned 动作转成自然语言动作串（喂 critic）。

    例：{"action":"grasp","target":"黄色杯子"} -> "grasp 黄色杯子"
        {"action":"place","target":"Prop_KLT_2"} -> "place object into Prop_KLT_2"
    """
    action = step.get("action", "")
    target = step.get("target", "")
    if action == "place":
        return f"place the held object into {target}"
    if action == "grasp":
        return f"grasp {target}"
    return f"{action} {target}".strip()


def _extract_scene_fields(snapshot: dict) -> dict:
    """从 memory_snapshot.json 抽 critic 需要的场景字段（自然语言模式）。"""
    working = snapshot.get("working", {})
    obs = working.get("observation", {})
    goal = working.get("goal", {})
    # instruction 可能在 goal.instruction 或 goal.intent.instruction
    instruction = goal.get("instruction", "")
    if not instruction and isinstance(goal.get("intent"), dict):
        instruction = goal["intent"].get("instruction", "")
    return {
        "user_instruction": instruction,
        "scene_objects_list": obs.get("visible_objects", []),
        "scene_text": obs.get("text", ""),
        "current_location": obs.get("current_location", ""),
        "held_object": obs.get("held_object"),
        "memory_objects": obs.get("memory_objects", {}),
        "task_id": snapshot.get("episode_id", ""),
    }


def run_safety_critic(
    snapshot_path: str | Path,
    planned_actions_path: str | Path,
    output_path: str | Path,
    llm: Any,
) -> CriticRunResult:
    """核心编排：逐步评价整条 plan，写 output/safety_critic_review.json。

    llm: 任意实现 `.call(prompt)->str`（+ 可选 `.total_tokens`）的客户端。
    """
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    steps_raw = json.loads(Path(planned_actions_path).read_text(encoding="utf-8"))
    if not isinstance(steps_raw, list):
        steps_raw = []

    scene = _extract_scene_fields(snapshot)
    plan_steps = [_action_to_text(s) for s in steps_raw]

    ctx = SafetyContext(
        input_mode="natural_language",
        task_id=scene["task_id"],
        user_instruction=scene["user_instruction"],
        scene_objects_list=scene["scene_objects_list"],
        scene_text=scene["scene_text"],
        current_location=scene["current_location"],
        held_object=scene["held_object"],
        memory_objects=scene["memory_objects"],
        plan_steps=plan_steps,
    )
    # 先建 scene_graph_dict（自然语言模式一次即可，逐步不变）
    ctx = SceneSafetyParser().process(ctx)

    critic = SafetyCritic(llm)
    result = CriticRunResult(num_steps=len(plan_steps))

    for i, step in enumerate(steps_raw):
        ctx.current_step_index = i
        # 逐步前重置 critic 输出字段（对齐 Monitor/main.py 的逐步循环写法）
        ctx.critic_decision = ""
        ctx.critic_reason = ""
        ctx.critic_risk_level = RiskLevel.LOW
        ctx.hazards_identified = []
        ctx.execution_halted = False
        ctx.halt_reason = ""

        ctx = critic.process(ctx)

        is_malicious = ctx.critic_decision == "malicious"
        review = {
            "index": i,
            "action_id": step.get("id"),
            "action": plan_steps[i],
            "raw_action": {"action": step.get("action"), "target": step.get("target")},
            "decision": ctx.critic_decision or "not malicious",
            "risk_level": ctx.critic_risk_level.value,
            "reason": ctx.critic_reason,
            "would_halt_here": is_malicious,
        }
        result.reviews.append(review)

        if is_malicious:
            result.num_malicious += 1
            if result.first_halt_index is None:
                result.first_halt_index = i
        else:
            # 只累加通过的动作到历史（保持 critic 原始语义）
            ctx.past_actions.append(plan_steps[i])

    result.total_tokens = getattr(llm, "total_tokens", 0)

    # 写诊断文件
    doc = {
        "_note": ("Safety Critic 逐动作安全裁判。评完整条 plan（遇 malicious 记录 would_halt_here=True "
                  "但继续评完给全貌）。裁决语义为『默认 malicious，除非 LLM 明确回 not malicious』。"),
        "input_mode": "natural_language",
        "task_id": scene["task_id"],
        "user_instruction": scene["user_instruction"],
        "summary": result.summary(),
        "reviews": result.reviews,
    }
    Path(output_path).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
