"""
Simulation body: same get_camera_images / set_velocity_command call shape as hardware.

Each set_velocity_command maps (vx, vy, yaw_rate, period) to one NavigationAction step in Sapien.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_SIM_PATH = Path(__file__).resolve().parent.parent / "simulation"
if str(_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_SIM_PATH))

from modules.sapien_engine import SapienNavigationEngine  # noqa: E402
from modules.perception import NavigationPerception  # noqa: E402
from schemas.data_types import NavigationAction, Pose, RobotInfo, SceneObject  # noqa: E402


def _blank_cameras_bgr(mean_value: int = 200) -> Dict[str, np.ndarray]:
    """Same keys and HxWx3 uint8 BGR as hardware; default bright to trigger straight-ahead policy."""
    img = np.full((480, 640, 3), mean_value, dtype=np.uint8)
    return {"head": img, "left": img.copy(), "right": img.copy()}


def _build_demo_scene_objects() -> list[SceneObject]:
    fake_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_depth = np.full((480, 640), 2.0, dtype=np.float32)
    intrinsic = np.array(
        [[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]]
    )
    extrinsic = np.eye(4)
    extrinsic[2, 3] = 1.0
    perception = NavigationPerception()
    return perception.process_image_to_objects(fake_rgb, fake_depth, intrinsic, extrinsic)


def _velocity_to_action(
    vx: float, vy: float, yaw_rate: float, period: float, max_force: float
) -> NavigationAction:
    # Couple yaw into heading: bias velocity direction in the plane
    dx = float(vx) - 0.2 * float(yaw_rate)
    dy = float(vy)
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dir_x, dir_y = 1.0, 0.0
    else:
        dir_x, dir_y = dx / norm, dy / norm
    speed = min(0.8, norm)
    apply_force = min(max_force, 25.0 + 75.0 * (speed / 0.8))
    duration = float(max(0.12, min(1.2, 0.35 * max(period, 0.5))))
    return NavigationAction(
        action_type="move_forward",
        direction=(dir_x, dir_y),
        duration=duration,
        apply_force=apply_force,
    )


class SapienVelocityBody:
    def __init__(self, use_gui: bool = False, camera_mean: int = 200):
        self.use_gui = use_gui
        self.camera_mean = camera_mean
        self._robot_info = RobotInfo(radius=0.25, height=0.5, max_push_force=100.0)
        self._engine: Optional[SapienNavigationEngine] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._engine = SapienNavigationEngine(self._robot_info, use_gui=self.use_gui)
        objs = _build_demo_scene_objects()
        self._engine.build_scene_objects(objs)
        self._engine.spawn_robot(initial_pose=Pose(position=(0.0, 0.0, 0.0)))
        self._started = True
        print("[task_planning] SapienVelocityBody: scene and robot ready.")

    def get_camera_images(self) -> Optional[Dict[str, Any]]:
        return _blank_cameras_bgr(self.camera_mean)

    def set_velocity_command(
        self, vx: float, vy: float, yaw_rate: float, period: float
    ) -> None:
        if self._engine is None or self._engine.robot_actor is None:
            return
        action = _velocity_to_action(
            vx, vy, yaw_rate, period, self._robot_info.max_push_force
        )
        self._engine.execute_action(action)

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.destroy()
            self._engine = None
        self._started = False
