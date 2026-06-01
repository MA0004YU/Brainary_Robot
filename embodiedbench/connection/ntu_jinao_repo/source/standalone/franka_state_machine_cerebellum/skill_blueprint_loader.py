"""Load and validate stage-2 skill blueprint JSON files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from skill_blueprint_schema import (
    CONDITION_TYPES,
    LOGIC_TYPES,
    NODE_TYPES,
    PARALLEL_GOAL_PARAM_FIELDS,
    PARALLEL_MODES,
    PARALLEL_NODE_PARAM_FIELDS,
    PERFORMANCE_METRICS,
    POSITION_GOAL_SKILLS,
    SKILL_PARAM_FIELDS,
    SKILL_TYPES,
)


TOP_LEVEL_FIELDS = {"blueprint_id", "task", "execution_graph"}
GRAPH_FIELDS = {"start", "nodes", "logic"}
COMMON_NODE_FIELDS = {"id", "type"}
SKILL_NODE_FIELDS = COMMON_NODE_FIELDS | {"skill", "target", "params", "performance_query", "next", "on_failure"}
CONDITION_NODE_FIELDS = COMMON_NODE_FIELDS | {"condition", "if_true", "if_false"}
TERMINAL_NODE_FIELDS = COMMON_NODE_FIELDS | {"result", "failure_reason"}
PARALLEL_NODE_FIELDS = COMMON_NODE_FIELDS | {
    "parallel_mode",
    "goals",
    "params",
    "performance_query",
    "next",
    "on_failure",
}
PARALLEL_GOAL_FIELDS = {"skill", "target", "params"}

REQUIRED_SKILL_PARAMS = {
    "move_above": {"height_offset", "xy_offset", "speed", "position_tolerance", "timeout_steps"},
    "reach": {"speed", "position_tolerance", "timeout_steps"},
    "descend": {"speed", "position_tolerance", "timeout_steps"},
    "grasp": {"close_wait_steps", "check", "timeout_steps"},
    "lift": {"lift_height", "speed", "position_tolerance", "timeout_steps"},
    "place": {"place_height", "release_height", "open_wait_steps", "position_tolerance", "target_tolerance", "timeout_steps"},
    "retreat": {"retreat_height", "speed", "timeout_steps"},
    "wait": {"wait_steps", "gripper"},
    "align_orientation": {"orientation_mode", "orientation_tolerance", "angular_speed", "timeout_steps"},
}

REQUIRED_PARALLEL_NODE_PARAMS = {"timeout_steps", "gripper"}

REQUIRED_POSITION_GOAL_PARAMS = {
    "move_above": {"height_offset", "xy_offset", "speed", "position_tolerance"},
    "reach": {"speed", "position_tolerance"},
}

REQUIRED_ORIENTATION_GOAL_PARAMS = {"orientation_mode", "orientation_tolerance", "angular_speed"}


@dataclass
class BlueprintNode:
    """One node in the stage-2 execution graph."""

    node_id: str
    type: str
    skill: str | None = None
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    performance_query: list[str] = field(default_factory=list)
    next: str | None = None
    on_failure: str | None = None
    condition: dict[str, Any] | None = None
    if_true: str | None = None
    if_false: str | None = None
    result: str | None = None
    failure_reason: str | None = None
    parallel_mode: str | None = None
    goals: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, node_id: str, data: dict[str, Any]) -> "BlueprintNode":
        if not isinstance(data, dict):
            raise ValueError(f"Node '{node_id}' must be an object.")
        node_type = str(data.get("type", ""))
        if node_type not in NODE_TYPES:
            raise ValueError(f"Node '{node_id}' has unsupported type '{node_type}'.")
        _validate_unknown_fields(f"Node '{node_id}'", data, _allowed_node_fields(node_type))
        declared_id = data.get("id")
        if declared_id is not None and str(declared_id) != node_id:
            raise ValueError(f"Node '{node_id}' has mismatched id field '{declared_id}'.")
        params = data.get("params", {})
        if "params" in data and not isinstance(params, dict):
            raise ValueError(f"Node '{node_id}' field 'params' must be an object.")
        performance_query = data.get("performance_query", [])
        if "performance_query" in data and not isinstance(performance_query, list):
            raise ValueError(f"Node '{node_id}' field 'performance_query' must be a list.")
        condition = data.get("condition")
        if node_type == "condition" and not isinstance(condition, dict):
            raise ValueError(f"Condition node '{node_id}' field 'condition' must be an object.")
        goals = data.get("goals")
        if node_type == "parallel" and goals is not None and not isinstance(goals, dict):
            raise ValueError(f"Parallel node '{node_id}' field 'goals' must be an object.")
        node = cls(
            node_id=node_id,
            type=node_type,
            skill=data.get("skill"),
            target=data.get("target"),
            params=dict(params),
            performance_query=[str(item) for item in performance_query],
            next=data.get("next"),
            on_failure=data.get("on_failure"),
            condition=condition,
            if_true=data.get("if_true"),
            if_false=data.get("if_false"),
            result=data.get("result"),
            failure_reason=data.get("failure_reason"),
            parallel_mode=data.get("parallel_mode"),
            goals=dict(goals) if isinstance(goals, dict) else goals,
        )
        node.validate()
        return node

    def validate(self) -> None:
        if self.type == "skill":
            self._validate_skill_node()
        elif self.type == "condition":
            self._validate_condition_node()
        elif self.type == "terminal":
            self._validate_terminal_node()
        elif self.type == "parallel":
            self._validate_parallel_node()

    def _validate_skill_node(self) -> None:
        if self.skill not in SKILL_TYPES:
            raise ValueError(f"Skill node '{self.node_id}' has unsupported skill '{self.skill}'.")
        if not self.target and self.skill != "wait":
            raise ValueError(f"Skill node '{self.node_id}' is missing required field 'target'.")
        if not self.next:
            raise ValueError(f"Skill node '{self.node_id}' is missing required field 'next'.")
        if not isinstance(self.params, dict):
            raise ValueError(f"Skill node '{self.node_id}' field 'params' must be an object.")
        if not isinstance(self.performance_query, list):
            raise ValueError(f"Skill node '{self.node_id}' field 'performance_query' must be a list.")
        allowed_params = SKILL_PARAM_FIELDS[str(self.skill)]
        unknown_params = sorted(set(self.params) - allowed_params)
        if unknown_params:
            raise ValueError(f"Skill node '{self.node_id}' has unsupported params for '{self.skill}': {unknown_params}")
        missing_params = sorted(REQUIRED_SKILL_PARAMS[str(self.skill)] - set(self.params))
        if missing_params:
            raise ValueError(f"Skill node '{self.node_id}' is missing required params for '{self.skill}': {missing_params}")
        if self.skill == "reach" and "target_pose" not in self.params and "target_ref" not in self.params:
            raise ValueError(f"Skill node '{self.node_id}' reach params require 'target_pose' or 'target_ref'.")
        if self.skill == "descend" and self.params.get("target_height") is None and self.params.get("relative_z") is None:
            raise ValueError(f"Skill node '{self.node_id}' descend params require 'target_height' or 'relative_z'.")
        unknown_metrics = sorted(set(self.performance_query) - PERFORMANCE_METRICS)
        if unknown_metrics:
            raise ValueError(f"Skill node '{self.node_id}' has unsupported performance metrics: {unknown_metrics}")

    def _validate_condition_node(self) -> None:
        if not self.condition:
            raise ValueError(f"Condition node '{self.node_id}' is missing 'condition'.")
        condition_name = str(self.condition.get("name", ""))
        if condition_name not in CONDITION_TYPES:
            raise ValueError(f"Condition node '{self.node_id}' has unsupported condition '{condition_name}'.")
        if not self.if_true or not self.if_false:
            raise ValueError(f"Condition node '{self.node_id}' requires 'if_true' and 'if_false'.")

    def _validate_terminal_node(self) -> None:
        if self.result not in {"success", "failure"}:
            raise ValueError(f"Terminal node '{self.node_id}' requires result 'success' or 'failure'.")

    def _validate_parallel_node(self) -> None:
        # TODO: nested parallel nodes inside goals are not supported in v1.
        if not self.goals or not isinstance(self.goals, dict):
            raise ValueError(f"Parallel node '{self.node_id}' requires non-empty 'goals'.")
        position_goal = self.goals.get("position_goal")
        orientation_goal = self.goals.get("orientation_goal")
        if position_goal is None and orientation_goal is None:
            raise ValueError(
                f"Parallel node '{self.node_id}' goals must contain at least one of "
                "'position_goal' or 'orientation_goal'."
            )
        unknown_goals = sorted(set(self.goals) - {"position_goal", "orientation_goal"})
        if unknown_goals:
            raise ValueError(f"Parallel node '{self.node_id}' has unsupported goals: {unknown_goals}")
        parallel_mode = str(self.parallel_mode or "all_success")
        if parallel_mode not in PARALLEL_MODES:
            raise ValueError(
                f"Parallel node '{self.node_id}' has unsupported parallel_mode '{parallel_mode}'. "
                f"Supported: {sorted(PARALLEL_MODES)}"
            )
        if not self.next:
            raise ValueError(f"Parallel node '{self.node_id}' is missing required field 'next'.")
        if not isinstance(self.params, dict):
            raise ValueError(f"Parallel node '{self.node_id}' field 'params' must be an object.")
        unknown_params = sorted(set(self.params) - PARALLEL_NODE_PARAM_FIELDS)
        if unknown_params:
            raise ValueError(f"Parallel node '{self.node_id}' has unsupported params: {unknown_params}")
        missing_params = sorted(REQUIRED_PARALLEL_NODE_PARAMS - set(self.params))
        if missing_params:
            raise ValueError(f"Parallel node '{self.node_id}' is missing required params: {missing_params}")
        if not isinstance(self.performance_query, list):
            raise ValueError(f"Parallel node '{self.node_id}' field 'performance_query' must be a list.")
        unknown_metrics = sorted(set(self.performance_query) - PERFORMANCE_METRICS)
        if unknown_metrics:
            raise ValueError(f"Parallel node '{self.node_id}' has unsupported performance metrics: {unknown_metrics}")
        if position_goal is not None:
            _validate_parallel_goal(self.node_id, "position_goal", position_goal, POSITION_GOAL_SKILLS)
        if orientation_goal is not None:
            _validate_parallel_goal(self.node_id, "orientation_goal", orientation_goal, {"align_orientation"})

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}


@dataclass
class SkillBlueprint:
    """Validated blueprint containing a directed execution graph."""

    blueprint_id: str
    task: str
    start: str
    nodes: dict[str, BlueprintNode]
    source_path: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "SkillBlueprint":
        path = Path(path).expanduser()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        blueprint = cls.from_dict(data)
        blueprint.source_path = path.as_posix()
        return blueprint

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillBlueprint":
        if not isinstance(data, dict):
            raise ValueError("Blueprint JSON root must be an object.")
        _validate_unknown_fields("Blueprint", data, TOP_LEVEL_FIELDS)
        if not data.get("blueprint_id"):
            raise ValueError("Blueprint is missing required field 'blueprint_id'.")
        if not data.get("task"):
            raise ValueError("Blueprint is missing required field 'task'.")
        if not isinstance(data.get("execution_graph"), dict):
            raise ValueError("Blueprint execution_graph must be an object.")
        graph = dict(data.get("execution_graph", {}))
        _validate_unknown_fields("Blueprint execution_graph", graph, GRAPH_FIELDS)
        graph_logic = graph.get("logic")
        if graph_logic is not None and graph_logic not in LOGIC_TYPES:
            raise ValueError(f"Blueprint execution_graph has unsupported logic '{graph_logic}'.")
        start = str(graph.get("start", ""))
        raw_nodes = graph.get("nodes", {})
        if not start:
            raise ValueError("Blueprint execution_graph is missing required field 'start'.")
        if not isinstance(raw_nodes, dict) or not raw_nodes:
            raise ValueError("Blueprint execution_graph.nodes must be a non-empty object.")
        nodes = {node_id: BlueprintNode.from_dict(node_id, node_data) for node_id, node_data in raw_nodes.items()}
        if start not in nodes:
            raise ValueError(f"Blueprint start node '{start}' does not exist.")
        _validate_edges(nodes)
        _validate_acyclic_graph(start, nodes)
        return cls(
            blueprint_id=str(data.get("blueprint_id", "blueprint_unnamed")),
            task=str(data.get("task", "")),
            start=start,
            nodes=nodes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "task": self.task,
            "execution_graph": {
                "start": self.start,
                "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            },
            "source_path": self.source_path,
        }


def _validate_parallel_goal(node_id: str, goal_name: str, goal: Any, allowed_skills: set[str]) -> None:
    if not isinstance(goal, dict):
        raise ValueError(f"Parallel node '{node_id}' goal '{goal_name}' must be an object.")
    _validate_unknown_fields(f"Parallel node '{node_id}' goal '{goal_name}'", goal, PARALLEL_GOAL_FIELDS)
    skill = str(goal.get("skill", ""))
    if skill not in allowed_skills:
        raise ValueError(
            f"Parallel node '{node_id}' goal '{goal_name}' has unsupported skill '{skill}'. "
            f"Allowed: {sorted(allowed_skills)}"
        )
    if not goal.get("target"):
        raise ValueError(f"Parallel node '{node_id}' goal '{goal_name}' is missing required field 'target'.")
    params = goal.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"Parallel node '{node_id}' goal '{goal_name}' field 'params' must be an object.")
    allowed_params = PARALLEL_GOAL_PARAM_FIELDS[skill]
    unknown_params = sorted(set(params) - allowed_params)
    if unknown_params:
        raise ValueError(f"Parallel node '{node_id}' goal '{goal_name}' has unsupported params: {unknown_params}")
    if skill in REQUIRED_POSITION_GOAL_PARAMS:
        missing = sorted(REQUIRED_POSITION_GOAL_PARAMS[skill] - set(params))
        if missing:
            raise ValueError(f"Parallel node '{node_id}' goal '{goal_name}' is missing required params: {missing}")
        if skill == "reach" and "target_pose" not in params and "target_ref" not in params:
            raise ValueError(f"Parallel node '{node_id}' goal '{goal_name}' reach params require 'target_pose' or 'target_ref'.")
    if skill == "align_orientation":
        missing = sorted(REQUIRED_ORIENTATION_GOAL_PARAMS - set(params))
        if missing:
            raise ValueError(f"Parallel node '{node_id}' goal '{goal_name}' is missing required params: {missing}")


def _validate_edges(nodes: dict[str, BlueprintNode]) -> None:
    for node in nodes.values():
        destinations = []
        if node.next:
            destinations.append(node.next)
        if node.on_failure:
            destinations.append(node.on_failure)
        if node.if_true:
            destinations.append(node.if_true)
        if node.if_false:
            destinations.append(node.if_false)
        for dest in destinations:
            if dest not in nodes:
                raise ValueError(f"Node '{node.node_id}' references missing node '{dest}'.")
    # TODO: add optional static cycle checks when loop support is intentionally introduced.


def _allowed_node_fields(node_type: str) -> set[str]:
    if node_type == "skill":
        return SKILL_NODE_FIELDS
    if node_type == "condition":
        return CONDITION_NODE_FIELDS
    if node_type == "terminal":
        return TERMINAL_NODE_FIELDS
    if node_type == "parallel":
        return PARALLEL_NODE_FIELDS
    return COMMON_NODE_FIELDS


def _validate_unknown_fields(context: str, data: dict[str, Any], allowed_fields: set[str]) -> None:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValueError(f"{context} has unsupported fields: {unknown}")


def _validate_acyclic_graph(start: str, nodes: dict[str, BlueprintNode]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, path: list[str]) -> None:
        if node_id in visiting:
            cycle = " -> ".join(path + [node_id])
            raise ValueError(f"Execution graph contains a cycle/loop, which is unsupported in this version: {cycle}")
        if node_id in visited:
            return
        visiting.add(node_id)
        node = nodes[node_id]
        for next_id in _node_destinations(node):
            visit(next_id, path + [node_id])
        visiting.remove(node_id)
        visited.add(node_id)

    visit(start, [])


def _node_destinations(node: BlueprintNode) -> list[str]:
    destinations = []
    if node.next:
        destinations.append(node.next)
    if node.on_failure:
        destinations.append(node.on_failure)
    if node.if_true:
        destinations.append(node.if_true)
    if node.if_false:
        destinations.append(node.if_false)
    return destinations
