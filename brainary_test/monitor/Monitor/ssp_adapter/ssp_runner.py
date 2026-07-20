"""SSP 运行编排（适配层）。

把整条 记忆 -> G_P -> SSP -> 回写 串起来，被两处复用：
  - adapters/__main__.py   （独立运行，吃现成 output/*.json，不碰 torch/API）
  - main.py 第 4 阶段 run_ssp（端到端流水线里，记忆之后）

driving 模式：action_conditioned（已与用户确认）。
  逐个读 planned_actions.json 的动作 -> 映射成 SSP CandidateAction -> query_risk -> 聚合。
  grasp -> pick（SSP 无 "grasp" 类型，用 "pick"）；target 为图中物体节点。
  place -> place；但 planned place 的 target 是篮子 id（Prop_KLT_*），不在场景图里，
         无法绑定节点 -> 记为 skipped（如实标注，不伪造）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ssp.ontology.schema import ActionParticipant, CandidateAction
from ssp.parser import SceneSafetyParser

from Monitor.ssp_adapter import constraint_writer as cw
from Monitor.ssp_adapter.memory_to_gp import GPBuildResult, build_graph_from_memory

# grasp/place -> SSP ACTION_TYPES 映射
_ACTION_TYPE_MAP = {
    "grasp": "pick",
    "pick": "pick",
    "place": "place",
}


@dataclass
class SSPRunResult:
    gp_build: GPBuildResult
    diagnostics: dict            # 顶层诊断（写 ssp_safety_constraints.json）
    candidate_constraints: list  # 聚合后的候选约束（合并进 planning_input）
    num_activated: int


def _load_actions(planned_actions_path: str | Path) -> list[dict]:
    p = Path(planned_actions_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _map_to_candidate_action(
    step: dict, name_to_id: dict[str, str]
) -> tuple[CandidateAction | None, str | None]:
    """把一个 planned 动作映射成 SSP CandidateAction。

    返回 (action, skipped_reason)；无法映射时 action=None + 原因。
    """
    raw_action = step.get("action", "")
    target_name = step.get("target", "")
    ssp_type = _ACTION_TYPE_MAP.get(raw_action)
    if ssp_type is None:
        return None, f"未知动作类型 '{raw_action}'"

    target_id = name_to_id.get(target_name)
    if target_id is None:
        # place 的 target 是篮子 Prop_KLT_*，不在场景图 -> 如实跳过
        return None, f"目标 '{target_name}' 不在场景图中（篮子/容器 id 无对应节点）"

    action = CandidateAction(
        id=step.get("id", "unknown"),
        type=ssp_type,
        participants=[ActionParticipant(role="target", entity_id=target_id)],
    )
    return action, None


def _run_scene_intrinsic(
    gp,
    parser,
    output_dir: Path,
    planning_input_path: str | Path,
) -> SSPRunResult:
    """scene_intrinsic 模式：不依赖 plan，直接对场景做全景风险解析。

    用途：SSP 在 planner **之前**跑 —— 读记忆场景 -> 出候选约束 -> 回流给 planner。
    注意：scene_intrinsic 不评估动作级 activation_conditions，会把所有实例化 factor 都算激活，
          风险条数通常比 action 模式多（见 docs）。这是"场景固有风险"视角。
    """
    result = parser.query_risk(gp.graph)  # 无 action = scene_intrinsic
    diag = cw.build_diagnostics(result, gp.id_to_name)
    candidates = cw.collect_candidate_constraints(diag)  # 无 from_action
    summary = diag["summary"]

    top = {
        "_boundary": diag["_boundary"],
        "scope": "scene_intrinsic",
        "summary": summary,
        "num_actions": 0,
        "scene_diagnostics": diag,
        "demo_notes": gp.notes,
    }
    cw.write_diagnostics_file(top, output_dir / "ssp_safety_constraints.json")
    cw.merge_into_planning_input(candidates, summary, planning_input_path)

    return SSPRunResult(
        gp_build=gp,
        diagnostics=top,
        candidate_constraints=candidates,
        num_activated=summary["num_activated"],
    )


def run_ssp_pipeline(
    snapshot_path: str | Path,
    planning_input_path: str | Path,
    output_dir: str | Path,
    templates_dir: str | Path,
    planned_actions_path: str | Path | None = None,
    mode: str = "scene_intrinsic",
) -> SSPRunResult:
    """核心编排。返回 SSPRunResult，副作用是写 output 文件（含更新 planning_input）。

    mode:
      - "scene_intrinsic"（默认）：不依赖 plan，读场景出候选约束。用于 SSP 在 planner **之前**跑，
        约束回流给 planner。
      - "action_conditioned"：逐个读 planned_actions.json 的动作精细校验。需要 planned_actions_path。
        用于 planner **之后**的逐动作诊断。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 记忆 -> G_P
    gp = build_graph_from_memory(snapshot_path)

    # 写调试用 G_P
    (output_dir / "ssp_perceptual_graph.json").write_text(
        json.dumps(gp.to_debug_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) 加载 SSP
    parser = SceneSafetyParser(templates_dir=Path(templates_dir))

    # scene_intrinsic：planner 之前，不需要 plan
    if mode == "scene_intrinsic":
        return _run_scene_intrinsic(gp, parser, output_dir, planning_input_path)

    # 3) action_conditioned：逐个 planned 动作跑
    if planned_actions_path is None:
        raise ValueError("action_conditioned 模式需要 planned_actions_path")
    steps = _load_actions(planned_actions_path)
    per_action: list[dict] = []
    for step in steps:
        action, skip = _map_to_candidate_action(step, gp.name_to_id)
        target_name = step.get("target")
        if action is None:
            per_action.append(
                {
                    "action_id": step.get("id", "?"),
                    "action_type": step.get("action"),
                    "target_name": target_name,
                    "target_id": gp.name_to_id.get(target_name or ""),
                    "skipped_reason": skip,
                }
            )
            continue
        result = parser.query_risk(gp.graph, action)
        diag = cw.build_diagnostics(result, gp.id_to_name)
        per_action.append(
            {
                "action_id": action.id,
                "action_type": action.type,
                "target_name": target_name,
                "target_id": gp.name_to_id.get(target_name or ""),
                "diagnostics": diag,
            }
        )

    # 4) 组装 + 双写
    top, candidates = cw.assemble_action_diagnostics(
        scope="action_conditioned",
        per_action=per_action,
        id_to_name=gp.id_to_name,
        demo_notes=gp.notes,
    )
    cw.write_diagnostics_file(top, output_dir / "ssp_safety_constraints.json")
    cw.merge_into_planning_input(candidates, top["summary"], planning_input_path)

    return SSPRunResult(
        gp_build=gp,
        diagnostics=top,
        candidate_constraints=candidates,
        num_activated=top["summary"]["num_activated"],
    )
