"""Execution graph interpreter for stage-2 skill blueprints."""

from __future__ import annotations

from typing import Any

from condition_evaluator import ConditionEvaluator
from performance_collector import PerformanceCollector
from primitive_skills import PrimitiveSkillExecutor


class SkillGraphExecutor:
    """Runs skill, condition, and terminal nodes from a validated blueprint."""

    def __init__(
        self,
        blueprint,
        cerebellum,
        logger,
        primitive_executor: PrimitiveSkillExecutor,
        condition_evaluator: ConditionEvaluator,
        performance_collector: PerformanceCollector,
        episode_id: str,
        max_episode_steps: int,
    ):
        self.blueprint = blueprint
        self.cerebellum = cerebellum
        self.logger = logger
        self.primitive_executor = primitive_executor
        self.condition_evaluator = condition_evaluator
        self.performance_collector = performance_collector
        self.episode_id = episode_id
        self.max_episode_steps = max_episode_steps

    def execute(self, initial_scene: dict[str, Any], env_id: str, target_mode: str, seed: int, trajectory_file: str | None) -> dict[str, Any]:
        current_node_id = self.blueprint.start
        visited_nodes: list[str] = []
        skill_trace: list[dict[str, Any]] = []
        condition_trace: list[dict[str, Any]] = []
        terminal_node: str | None = None
        terminal_result: str | None = None
        failure_reason: str | None = None

        while current_node_id:
            if self.cerebellum.global_step >= self.max_episode_steps:
                failure_reason = "episode_timeout"
                break
            node = self.blueprint.nodes[current_node_id]
            visited_nodes.append(current_node_id)

            if node.type == "skill":
                result = self.primitive_executor.execute(node)
                self.condition_evaluator.set_last_skill_result(result)
                skill_trace.append(result.to_trace())
                sample = self.performance_collector.collect(self.episode_id, self.blueprint.blueprint_id, node, result)
                self.logger.log_predictor_sample(sample)
                if result.success:
                    current_node_id = node.next
                elif node.on_failure:
                    current_node_id = node.on_failure
                else:
                    failure_reason = result.failure_reason or f"{node.skill}_failed"
                    break
                # TODO: support safe skill-boundary blueprint reload behind --reload_at skill.
                continue

            if node.type == "parallel":
                result = self.primitive_executor.execute_parallel(node)
                self.condition_evaluator.set_last_skill_result(result)
                skill_trace.append(result.to_trace())
                sample = self.performance_collector.collect(self.episode_id, self.blueprint.blueprint_id, node, result)
                self.logger.log_predictor_sample(sample)
                if result.success:
                    current_node_id = node.next
                elif node.on_failure:
                    current_node_id = node.on_failure
                else:
                    failure_reason = result.failure_reason or "parallel_failed"
                    break
                continue

            if node.type == "condition":
                result, details = self.condition_evaluator.evaluate(node.condition or {})
                next_node = node.if_true if result else node.if_false
                condition_trace.append(
                    {
                        "node_id": node.node_id,
                        "condition": (node.condition or {}).get("name"),
                        "condition_spec": node.condition,
                        "result": result,
                        "details": details,
                        "next_node": next_node,
                    }
                )
                current_node_id = next_node
                continue

            if node.type == "terminal":
                terminal_node = node.node_id
                terminal_result = node.result
                failure_reason = node.failure_reason if node.result == "failure" else None
                break

            raise RuntimeError(f"Unsupported node type: {node.type}")

        if terminal_node is None and current_node_id in self.blueprint.nodes:
            node = self.blueprint.nodes[current_node_id]
            if node.type == "terminal":
                terminal_node = node.node_id
                terminal_result = node.result
        if terminal_result is None:
            terminal_result = "failure" if failure_reason else "success"

        final_scene = self._scene_state()
        final_position_error = self._final_position_error(final_scene)
        final_success = terminal_result == "success" and failure_reason is None
        episode = {
            "episode_id": self.episode_id,
            "blueprint_id": self.blueprint.blueprint_id,
            "task": self.blueprint.task,
            "env_id": env_id,
            "target_mode": target_mode,
            "seed": seed,
            "initial_scene": initial_scene,
            "execution_graph_summary": {
                "num_nodes": len(self.blueprint.nodes),
                "visited_nodes": visited_nodes,
                "terminal_node": terminal_node,
                "terminal_result": "success" if final_success else "failure",
                "failure_reason": None if final_success else (failure_reason or "terminal_failure"),
            },
            "skill_execution_trace": skill_trace,
            "condition_trace": condition_trace,
            "final_result": {
                "success": final_success,
                "failure_reason": None if final_success else (failure_reason or "terminal_failure"),
                "final_cube_pose": final_scene["cube_pose"],
                "final_ee_pose": final_scene["ee_pose"],
                "final_position_error": final_position_error,
            },
            "trajectory_file": trajectory_file,
        }
        self.logger.log_episode(episode)
        return episode

    def _scene_state(self) -> dict[str, Any]:
        state = self.cerebellum.get_scene_state()
        return {
            "ee_pose": state["ee_pose_w"],
            "cube_pose": state["cube_pose_w"],
            "target_pose": state["target_pose_w"],
            "gripper_width": state["gripper_width"],
            "robot_joint_pos": state["robot_joint_pos"],
            "step_index": state["step_index"],
        }

    def _final_position_error(self, scene: dict[str, Any]) -> float:
        cube = scene["cube_pose"]
        target = scene["target_pose"]
        return ((float(cube[0]) - float(target[0])) ** 2 + (float(cube[1]) - float(target[1])) ** 2) ** 0.5
