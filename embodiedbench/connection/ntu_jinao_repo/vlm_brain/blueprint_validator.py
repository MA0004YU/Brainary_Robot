"""Simulator-independent validation for VLM-generated skill blueprints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_DIR = REPO_ROOT / "source" / "standalone" / "franka_state_machine_cerebellum"
if STAGE2_DIR.as_posix() not in sys.path:
    sys.path.insert(0, STAGE2_DIR.as_posix())

from skill_blueprint_schema import (  # noqa: E402
    CONDITION_TYPES,
    LOGIC_TYPES,
    NODE_TYPES,
    PARALLEL_MODES,
    PERFORMANCE_METRICS,
    POSITION_GOAL_SKILLS,
    SKILL_TYPES,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


def validate_blueprint(blueprint: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(blueprint, dict):
        return ValidationResult(valid=False, errors=["blueprint_root_must_be_object"])

    _require(blueprint, "blueprint_id", errors)
    _require(blueprint, "task", errors)
    graph = blueprint.get("execution_graph")
    if not isinstance(graph, dict):
        return ValidationResult(valid=False, errors=errors + ["execution_graph_must_be_object"], warnings=warnings)

    logic = graph.get("logic")
    if logic is not None and logic not in LOGIC_TYPES:
        errors.append(f"unsupported_logic:{logic}")
    start = graph.get("start")
    nodes = graph.get("nodes")
    if not isinstance(start, str) or not start:
        errors.append("execution_graph.start_missing_or_invalid")
    if not isinstance(nodes, dict) or not nodes:
        errors.append("execution_graph.nodes_missing_or_invalid")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)
    if isinstance(start, str) and start and start not in nodes:
        errors.append(f"start_node_not_found:{start}")

    for node_id, node in nodes.items():
        _validate_node(str(node_id), node, nodes, errors, warnings)

    _validate_edges(nodes, errors)
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _validate_node(node_id: str, node: Any, nodes: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    if not isinstance(node, dict):
        errors.append(f"{node_id}:node_must_be_object")
        return
    node_type = node.get("type")
    if node_type not in NODE_TYPES:
        errors.append(f"{node_id}:unsupported_node_type:{node_type}")
        return

    if "execution_policy" in node:
        warnings.append(f"{node_id}:execution_policy_present_validator_allows_but_stage2_loader_may_reject")
    if "brain_hook" in node:
        warnings.append(f"{node_id}:brain_hook_present_validator_allows_but_stage2_loader_may_reject")

    if node_type == "skill":
        _validate_skill_node(node_id, node, errors)
    elif node_type == "parallel":
        _validate_parallel_node(node_id, node, errors)
    elif node_type == "condition":
        _validate_condition_node(node_id, node, errors)
    elif node_type == "terminal":
        if node.get("result") not in {"success", "failure"}:
            errors.append(f"{node_id}:terminal_result_must_be_success_or_failure")

    if node_type in {"skill", "parallel"} and not node.get("performance_query"):
        errors.append(f"{node_id}:performance_query_required")
    if node_type in {"skill", "parallel"}:
        unknown_metrics = sorted(set(node.get("performance_query") or []) - PERFORMANCE_METRICS)
        if unknown_metrics:
            errors.append(f"{node_id}:unsupported_performance_metrics:{unknown_metrics}")


def _validate_skill_node(node_id: str, node: dict[str, Any], errors: list[str]) -> None:
    skill = node.get("skill")
    if skill not in SKILL_TYPES:
        errors.append(f"{node_id}:unsupported_skill:{skill}")
    if skill != "wait" and not node.get("target"):
        errors.append(f"{node_id}:target_required")
    if not isinstance(node.get("params", {}), dict):
        errors.append(f"{node_id}:params_must_be_object")
    if not node.get("next"):
        errors.append(f"{node_id}:next_required")


def _validate_parallel_node(node_id: str, node: dict[str, Any], errors: list[str]) -> None:
    if node.get("parallel_mode", "all_success") not in PARALLEL_MODES:
        errors.append(f"{node_id}:unsupported_parallel_mode:{node.get('parallel_mode')}")
    goals = node.get("goals")
    if not isinstance(goals, dict) or not goals:
        errors.append(f"{node_id}:goals_required")
        return
    unknown_goals = sorted(set(goals) - {"position_goal", "orientation_goal"})
    if unknown_goals:
        errors.append(f"{node_id}:unsupported_parallel_goals:{unknown_goals}")
    if "position_goal" not in goals and "orientation_goal" not in goals:
        errors.append(f"{node_id}:parallel_requires_position_or_orientation_goal")
    if "position_goal" in goals:
        _validate_parallel_goal(node_id, "position_goal", goals["position_goal"], POSITION_GOAL_SKILLS, errors)
    if "orientation_goal" in goals:
        _validate_parallel_goal(node_id, "orientation_goal", goals["orientation_goal"], {"align_orientation"}, errors)
    if not node.get("next"):
        errors.append(f"{node_id}:next_required")


def _validate_parallel_goal(
    node_id: str,
    goal_name: str,
    goal: Any,
    allowed_skills: set[str],
    errors: list[str],
) -> None:
    if not isinstance(goal, dict):
        errors.append(f"{node_id}:{goal_name}_must_be_object")
        return
    skill = goal.get("skill")
    if skill not in allowed_skills:
        errors.append(f"{node_id}:{goal_name}_unsupported_skill:{skill}")
    if not goal.get("target"):
        errors.append(f"{node_id}:{goal_name}_target_required")
    if not isinstance(goal.get("params", {}), dict):
        errors.append(f"{node_id}:{goal_name}_params_must_be_object")


def _validate_condition_node(node_id: str, node: dict[str, Any], errors: list[str]) -> None:
    condition = node.get("condition")
    if not isinstance(condition, dict):
        errors.append(f"{node_id}:condition_must_be_object")
        return
    name = condition.get("name")
    if name not in CONDITION_TYPES:
        errors.append(f"{node_id}:unsupported_condition:{name}")
    if not node.get("if_true") or not node.get("if_false"):
        errors.append(f"{node_id}:if_true_and_if_false_required")


def _validate_edges(nodes: dict[str, Any], errors: list[str]) -> None:
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        for field_name in ("next", "on_failure", "if_true", "if_false"):
            target = node.get(field_name)
            if target and target not in nodes:
                errors.append(f"{node_id}:{field_name}_points_to_missing_node:{target}")


def _require(mapping: dict[str, Any], key: str, errors: list[str]) -> None:
    if not mapping.get(key):
        errors.append(f"{key}_required")
