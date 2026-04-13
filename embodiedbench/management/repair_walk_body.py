from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RepairWalkBody:
    def __init__(self, init_sleep_s: float = 3.0):
        self._init_sleep_s = init_sleep_s
        self._controller = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        root = Path(__file__).resolve().parents[2]
        repair = root / "repair"
        r = str(repair)
        if r not in sys.path:
            sys.path.insert(0, r)
        from g1_walk_controller import WalkController

        self._controller = WalkController()
        self._thread = threading.Thread(target=self._controller.run, daemon=True)
        self._thread.start()
        print("[management] RepairWalkBody: waiting for DDS / controller...")
        time.sleep(self._init_sleep_s)

    def get_camera_images(self) -> Optional[Dict[str, Any]]:
        if self._controller is None:
            return None
        return self._controller.get_camera_images()

    def set_velocity_command(
        self, vx: float, vy: float, yaw_rate: float, period: float
    ) -> None:
        if self._controller is not None:
            self._controller.set_velocity_command(vx, vy, yaw_rate, period)

    def shutdown(self) -> None:
        if self._controller is not None:
            self._controller.set_velocity_command(0.0, 0.0, 0.0, 0.8)
