"""Small deterministic repairs for VLM-generated skill_blueprint JSON."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from blueprint_parser import ParseResult, extract_json_from_text
from blueprint_validator import ValidationResult, validate_blueprint


ALLOWED_TOP_FIELDS = {"blueprint_id", "task", "execution_graph"}
ALLOWED_GRAPH_FIELDS = {"start", "logic", "nodes"}
ALLOWED_NODE_FIELDS = {
    "id",
    "type",
    "skill",
    "target",
    "params",
    "performance_query",
    "parallel_mode",
    "goals",
    "condition",
    "next",
    "on_failure",
    "if_true",
    "if_false",
    "result",
    "failure_reason",
    "execution_policy",
    "brain_hook",
}


def repair_blueprint_from_text(raw_text: str) -> tuple[dict[str, Any] | None, ValidationResult, ParseResult]:
    parse_result = extract_json_from_text(raw_text)
    if not parse_result.ok or parse_result.blueprint is None:
        return None, ValidationResult(valid=False, errors=[parse_result.error or "parse_failed"]), parse_result
    repaired = repair_blueprint(parse_result.blueprint)
    validation = validate_blueprint(repaired)
    return repaired if validation.valid else None, validation, parse_result


def repair_blueprint(blueprint: dict[str, Any]) -> dict[str, Any]:
    repaired = deepcopy(blueprint)
    for key in list(repaired):
        if key not in ALLOWED_TOP_FIELDS:
            repaired.pop(key, None)
    repaired.setdefault("blueprint_id", "vlm_generated_blueprint")
    repaired.setdefault("task", "generated skill blueprint")
    graph = repaired.setdefault("execution_graph", {})
    for key in list(graph):
        if key not in ALLOWED_GRAPH_FIELDS:
            graph.pop(key, None)
    graph.setdefault("logic", "sequence")
    nodes = graph.setdefault("nodes", {})
    if isinstance(nodes, dict):
        for node_id, node in list(nodes.items()):
            if not isinstance(node, dict):
                nodes.pop(node_id, None)
                continue
            for key in list(node):
                if key not in ALLOWED_NODE_FIELDS:
                    node.pop(key, None)
        _ensure_terminal(nodes, "t_success", "success")
        _ensure_terminal(nodes, "t_failure", "failure")
        if not graph.get("start") and nodes:
            graph["start"] = next(iter(nodes))
    return repaired


def _ensure_terminal(nodes: dict[str, Any], node_id: str, result: str) -> None:
    if node_id not in nodes:
        node = {"type": "terminal", "result": result}
        if result == "failure":
            node["failure_reason"] = "skill_failed"
        nodes[node_id] = node
