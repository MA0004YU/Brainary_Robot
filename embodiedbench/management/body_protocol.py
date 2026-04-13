from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class RobotWalkBody(Protocol):
    def start(self) -> None: ...

    def get_camera_images(self) -> Optional[Dict[str, Any]]: ...

    def set_velocity_command(
        self, vx: float, vy: float, yaw_rate: float, period: float
    ) -> None: ...

    def shutdown(self) -> None: ...
