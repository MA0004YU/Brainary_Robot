"""Protocol for the robot 'body': same surface API as repair WalkController for the planner loop."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class RobotWalkBody(Protocol):
    """Matches the brain-facing API of repair/g1_walk_controller.WalkController."""

    def start(self) -> None:
        """Connect DDS / build simulation scene, etc."""

    def get_camera_images(self) -> Optional[Dict[str, Any]]:
        """Return dict with keys head, left, right -> (480,640,3) uint8 BGR, or None."""

    def set_velocity_command(
        self, vx: float, vy: float, yaw_rate: float, period: float
    ) -> None:
        """Same semantics as WalkController.set_velocity_command."""

    def shutdown(self) -> None:
        """Stop motion, release DDS / tear down simulation."""
