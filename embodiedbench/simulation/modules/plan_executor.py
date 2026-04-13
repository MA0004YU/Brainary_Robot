"""
将 TaskPlan 转为 NavigationAction 序列并在 Sapien 引擎中逐步执行（仿真优先）。
"""
from __future__ import annotations

import json
import time
from typing import List, Tuple

from schemas.data_types import NavigationAction, Pose, RobotInfo, SceneObject, SimulationResult
from schemas.task_plan import ScheduledSubtask, TaskPlan, topological_order
from modules.sapien_engine import SapienNavigationEngine


def subtask_to_navigation_action(st: ScheduledSubtask) -> NavigationAction:
    dir_x, dir_y = st.direction
    norm = (dir_x * dir_x + dir_y * dir_y) ** 0.5
    if norm > 0:
        dir_x, dir_y = dir_x / norm, dir_y / norm
    else:
        dir_x, dir_y = 1.0, 0.0
    return NavigationAction(
        action_type=st.action_type,
        direction=(dir_x, dir_y),
        duration=float(st.duration),
        apply_force=float(st.apply_force),
    )


def execute_plan_in_simulation(
    plan: TaskPlan,
    scene_objects: List[SceneObject],
    *,
    robot_start: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    robot_info: RobotInfo | None = None,
    use_gui: bool = False,
) -> List[SimulationResult]:
    """
    按拓扑序执行子任务；返回每步 SimulationResult。
    """
    robot_info = robot_info or RobotInfo()
    ordered = topological_order(plan.subtasks)
    engine = SapienNavigationEngine(robot_info, use_gui=use_gui)
    results: List[SimulationResult] = []

    try:
        engine.build_scene_objects(scene_objects)
        engine.spawn_robot(initial_pose=Pose(position=robot_start))

        for st in ordered:
            if st.type == "wait":
                if st.wait_seconds > 0:
                    time.sleep(min(st.wait_seconds, 5.0))
                p = engine.robot_actor.get_pose().p
                results.append(
                    SimulationResult(
                        action_success=True,
                        final_robot_pose=Pose(
                            position=(float(p[0]), float(p[1]), float(p[2]))
                        ),
                        distance_moved=0.0,
                        error_reason="",
                    )
                )
                print(f"[Plan→Sim] step {st.id} (wait) {st.wait_seconds:.2f}s")
                continue

            action = subtask_to_navigation_action(st)
            print(
                f"[Plan→Sim] step {st.id} ({st.type}) -> "
                f"{action.action_type} dir={action.direction} F={action.apply_force:.1f}N"
            )
            res = engine.execute_action(action)
            results.append(res)
            if not res.action_success:
                print(f"[Plan→Sim] 步骤 {st.id} 未达预期位移，原因: {res.error_reason}")
    finally:
        engine.destroy()

    return results


def load_task_plan_json(path: str) -> TaskPlan:
    with open(path, "r", encoding="utf-8") as f:
        return TaskPlan.from_dict(json.load(f))
