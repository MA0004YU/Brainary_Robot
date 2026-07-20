"""约束回写器（适配层）。

把 SSP 的 QueryResult 写成两处（双写，对齐用户要求）：
  1. 独立文件 output/ssp_safety_constraints.json —— 完整诊断（三分类风险 + CT-id + 证据）
  2. 合并进 output/memory_planning_input.json 的 constraints.safety_constraints 子键（供规划消费）

⚠️⚠️ SSP 边界红线（务必如实标注，回写内容里也会写进去）：
  SSP 只输出**候选约束模板 id（CT-*）** + 实体绑定 + 证据；
  它**绝不生成最终 LTL 约束**，也**不做 accept/reject**（那是下游 L3 的职责）。
  因此回写的 safety_constraints 全部是 "candidate"，规划层不应把它当已裁决的硬约束。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssp.query.result import QueryResult

# 回写里统一声明的边界，防止下游误用
_BOUNDARY_NOTE = (
    "SSP 仅输出候选约束模板 id (CT-*) + 实体绑定 + 证据；不生成最终 LTL、不做 accept/reject。"
    "accept/reject 与 LTL 实例化是下游 L3 约束层的职责。"
)


def _risk_vector_to_dict(risk: Any) -> dict:
    """RiskVector -> 精简 dict（severity/likelihood 是 dict[RiskEventType,float]）。

    只保留非空维度，避免写出 14 维全 0 的噪声。
    """
    if risk is None:
        return {}
    sev = {k.value if hasattr(k, "value") else str(k): v for k, v in (risk.severity or {}).items()}
    lik = {k.value if hasattr(k, "value") else str(k): v for k, v in (risk.likelihood or {}).items()}
    out: dict[str, Any] = {}
    if sev:
        out["severity"] = sev
    if lik:
        out["likelihood"] = lik
    out["confidence"] = getattr(risk, "confidence", 1.0)
    return out


def build_diagnostics(
    result: QueryResult,
    id_to_name: dict[str, str],
    demo_notes: list[str] | None = None,
) -> dict:
    """把 QueryResult 转成完整诊断结构（写独立文件用）。

    id_to_name: G_P 的 ASCII id -> 中文名，回写里同时给出，便于人读。
    """

    def _named(entity_id: str) -> str:
        return id_to_name.get(entity_id, entity_id)

    activated = []
    for ev in result.activated_risk_events:
        activated.append(
            {
                "factor_id": ev.factor_id,
                "re_type": ev.re_type.value,
                "hazard_id": ev.hazard_id,
                "hazard_name": _named(ev.hazard_id),
                "target_id": ev.target_id,
                "target_name": _named(ev.target_id),
                "constraint_template_ids": list(ev.constraint_template_ids),  # CT-*
                "activation_strength": ev.activation_strength,
                "activation_evidence": list(ev.activation_evidence),
                "confidence": ev.confidence,
                "risk": _risk_vector_to_dict(ev.risk),
                "residual_risk": _risk_vector_to_dict(ev.residual_risk),
                "status": "candidate",  # 候选，非裁决
            }
        )

    suppressed = []
    for ev in result.suppressed_events:
        suppressed.append(
            {
                "factor_id": ev.factor_id,
                "re_type": ev.re_type.value,
                "hazard_id": ev.hazard_id,
                "hazard_name": _named(ev.hazard_id),
                "target_id": ev.target_id,
                "target_name": _named(ev.target_id),
                "suppressor_ids": list(ev.suppressor_ids),
                "hard_mitigation_evidence": list(ev.hard_mitigation_evidence),
            }
        )

    inactive = []
    for ev in result.inactive_risk_events:
        inactive.append(
            {
                "factor_id": ev.factor_id,
                "re_type": ev.re_type.value,
                "hazard_id": ev.hazard_id,
                "target_id": ev.target_id,
                "reason": ev.reason,
            }
        )

    meta = result.parser_meta
    return {
        "_boundary": _BOUNDARY_NOTE,
        "scope": result.scope,
        "summary": {
            "num_activated": meta.num_activated,
            "num_suppressed": meta.num_suppressed,
            "num_inactive": meta.num_inactive,
            "num_entities": meta.num_entities,
            "num_factor_nodes": meta.num_factor_nodes,
            "converged": meta.converged,
            "warnings": list(meta.warnings),
        },
        "activated_risk_events": activated,
        "suppressed_events": suppressed,
        "inactive_risk_events": inactive,
        "demo_notes": list(demo_notes or []),
    }


def collect_candidate_constraints(diagnostics: dict, action_id: str | None = None) -> list[dict]:
    """从单个诊断里抽出规划层要用的候选约束（CT-id + 绑定 + 证据）。

    action_id: 若来自某个具体动作，带上以便回写溯源。
    """
    constraints = []
    for ev in diagnostics.get("activated_risk_events", []):
        for ct_id in ev["constraint_template_ids"]:
            item = {
                "constraint_template_id": ct_id,   # CT-*
                "re_type": ev["re_type"],
                "hazard": ev["hazard_name"],
                "target": ev["target_name"],
                "activation_evidence": ev["activation_evidence"],
                "status": "candidate",             # 未裁决
            }
            if action_id is not None:
                item["from_action"] = action_id
            constraints.append(item)
    return constraints


def write_diagnostics_file(diagnostics: dict, out_path: str | Path) -> Path:
    """写独立诊断文件 output/ssp_safety_constraints.json。"""
    out_path = Path(out_path)
    out_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def _dedup_constraints(constraints: list[dict]) -> list[dict]:
    """按 (CT-id, re_type, hazard, target) 去重，from_action 合并成列表。"""
    merged: dict[tuple, dict] = {}
    for c in constraints:
        key = (c["constraint_template_id"], c["re_type"], c["hazard"], c["target"])
        if key in merged:
            fa = c.get("from_action")
            if fa and fa not in merged[key]["from_actions"]:
                merged[key]["from_actions"].append(fa)
        else:
            item = {
                "constraint_template_id": c["constraint_template_id"],
                "re_type": c["re_type"],
                "hazard": c["hazard"],
                "target": c["target"],
                "activation_evidence": c["activation_evidence"],
                "status": "candidate",
                "from_actions": [c["from_action"]] if c.get("from_action") else [],
            }
            merged[key] = item
    return list(merged.values())


def merge_into_planning_input(
    candidate_constraints: list[dict],
    summary: dict,
    planning_input_path: str | Path,
) -> Path:
    """把（已聚合去重的）候选约束合并进 memory_planning_input.json。

    只做加法：读现有文件 -> 在 constraints 下加 safety_constraints 子键 -> 写回。
    不动 category_rules / no_category_mixing / collision_avoidance。
    """
    planning_input_path = Path(planning_input_path)
    if not planning_input_path.exists():
        raise FileNotFoundError(f"planning_input 不存在: {planning_input_path}")

    data = json.loads(planning_input_path.read_text(encoding="utf-8"))
    constraints = data.setdefault("constraints", {})
    constraints["safety_constraints"] = {
        "_boundary": _BOUNDARY_NOTE,
        "source": "SSP (Scene Safety Parser)",
        "candidate_constraints": _dedup_constraints(candidate_constraints),
        "summary": summary,
    }
    planning_input_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return planning_input_path


def assemble_action_diagnostics(
    scope: str,
    per_action: list[dict],
    id_to_name: dict[str, str],
    demo_notes: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    """把多个动作的诊断组装成顶层结构，并返回聚合后的候选约束列表。

    per_action: [{"action_id","action_type","target_name","target_id","diagnostics"|"skipped_reason"}, ...]
    返回 (顶层诊断 dict, 聚合候选约束 list)。
    """
    all_candidates: list[dict] = []
    agg = {"num_activated": 0, "num_suppressed": 0, "num_inactive": 0}
    action_entries = []
    for pa in per_action:
        entry = {
            "action_id": pa["action_id"],
            "action_type": pa["action_type"],
            "target_name": pa.get("target_name"),
            "target_id": pa.get("target_id"),
        }
        if pa.get("skipped_reason"):
            entry["skipped_reason"] = pa["skipped_reason"]
        else:
            diag = pa["diagnostics"]
            entry["diagnostics"] = diag
            all_candidates.extend(collect_candidate_constraints(diag, pa["action_id"]))
            agg["num_activated"] += diag["summary"]["num_activated"]
            agg["num_suppressed"] += diag["summary"]["num_suppressed"]
            agg["num_inactive"] += diag["summary"]["num_inactive"]
        action_entries.append(entry)

    top = {
        "_boundary": _BOUNDARY_NOTE,
        "scope": scope,
        "summary": agg,
        "num_actions": len(per_action),
        "per_action": action_entries,
        "demo_notes": list(demo_notes or []),
    }
    return top, all_candidates
