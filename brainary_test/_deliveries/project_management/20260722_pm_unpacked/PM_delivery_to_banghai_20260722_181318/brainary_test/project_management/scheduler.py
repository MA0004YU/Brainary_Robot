from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, Iterable, List, Sequence, Set

from .types import ActionNode, PMConfig, ScheduledAction


MANIPULATION_ACTIONS = {
    "grasp", "place", "move_above", "descend", "lift", "retreat",
    "reach", "align_orientation", "open_gripper", "close_gripper",
    "open_drawer", "close_drawer", "open_door", "close_door", "operate_coffee",
}


class ParallelScheduler:
    def __init__(self, config: PMConfig | None = None):
        self.config = config or PMConfig()

    def parse_plan(self, plan: Sequence[dict | ActionNode]) -> List[ActionNode]:
        nodes: List[ActionNode] = []
        for item in plan:
            if isinstance(item, ActionNode):
                node = item
            else:
                node = ActionNode.from_dict(item, default_duration=self.config.default_duration)
            if not node.id:
                raise ValueError(f"plan 中存在缺少 id 的动作: {item}")
            if not node.action:
                raise ValueError(f"动作 {node.id} 缺少 action 字段")
            if not node.resources:
                node.resources = self.infer_resources(node)
            nodes.append(node)
        self.validate(nodes)
        return nodes

    def validate(self, nodes: Sequence[ActionNode]) -> None:
        ids = [node.id for node in nodes]
        duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
        if duplicates:
            raise ValueError(f"plan 中存在重复 id: {duplicates}")
        id_set = set(ids)
        for node in nodes:
            missing = [dep for dep in node.depends_on if dep not in id_set]
            if missing:
                raise ValueError(f"动作 {node.id} 依赖不存在的节点: {missing}")
        self._topological_order(nodes)

    def schedule(self, plan: Sequence[dict | ActionNode]) -> List[ScheduledAction]:
        nodes = self.parse_plan(plan)
        by_id = {node.id: node for node in nodes}
        completed: Set[str] = set()
        scheduled: List[ScheduledAction] = []
        wave = 0
        current_time = 0.0

        while len(completed) < len(nodes):
            ready = [
                node for node in nodes
                if node.id not in completed and all(dep in completed for dep in node.depends_on)
            ]
            if not ready:
                remaining = sorted(set(by_id) - completed)
                raise ValueError(f"无法继续调度，可能存在环或不可满足依赖: {remaining}")

            ready.sort(key=self._ready_sort_key)
            selected: List[ActionNode] = []
            occupied: Set[str] = set()
            for node in ready:
                resources = set(node.resources)
                if occupied.isdisjoint(resources) and len(selected) < self.config.max_concurrent_workers:
                    selected.append(node)
                    occupied.update(resources)

            if not selected:
                selected = [ready[0]]

            group_ids = [node.id for node in selected]
            wave_duration = max(max(node.duration, 0.0) for node in selected)
            for node in selected:
                scheduled.append(ScheduledAction(
                    node=node,
                    wave=wave,
                    start_time=current_time,
                    end_time=current_time + max(node.duration, 0.0),
                    parallel_group=group_ids,
                ))
                completed.add(node.id)
            current_time += wave_duration
            wave += 1
        return scheduled

    def _ready_sort_key(self, node: ActionNode):
        action_order = 0 if node.action == "place" else 1 if node.action == "grasp" else 2
        return (-node.priority, action_order, self._natural_id_key(node.id))

    @staticmethod
    def _natural_id_key(value: str):
        parts = re.split(r"(\d+)", value)
        return tuple(int(part) if part.isdigit() else part for part in parts)

    def infer_resources(self, node: ActionNode) -> List[str]:
        action = node.action.lower()
        target = str(node.target or "none")
        if action == "wait":
            return []
        if action in MANIPULATION_ACTIONS:
            resources = ["robot_arm"]
            if action in {"grasp", "place", "open_gripper", "close_gripper"}:
                resources.append("gripper")
            if node.target is not None:
                resources.append(f"target:{target}")
            return resources
        if node.target is not None:
            return [f"target:{target}"]
        return [f"action:{action}"]

    def _topological_order(self, nodes: Sequence[ActionNode]) -> List[str]:
        outgoing: Dict[str, List[str]] = defaultdict(list)
        indegree: Dict[str, int] = {node.id: 0 for node in nodes}
        for node in nodes:
            for dep in node.depends_on:
                outgoing[dep].append(node.id)
                indegree[node.id] += 1

        queue = sorted([node_id for node_id, degree in indegree.items() if degree == 0], key=self._natural_id_key)
        order: List[str] = []
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for nxt in sorted(outgoing[node_id], key=self._natural_id_key):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
                    queue.sort(key=self._natural_id_key)
        if len(order) != len(nodes):
            cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
            raise ValueError(f"plan 不是 DAG，存在环依赖: {cyclic}")
        return order


def group_by_wave(schedule: Iterable[ScheduledAction]) -> List[List[ScheduledAction]]:
    buckets: Dict[int, List[ScheduledAction]] = defaultdict(list)
    for item in schedule:
        buckets[item.wave].append(item)
    return [buckets[idx] for idx in sorted(buckets)]
