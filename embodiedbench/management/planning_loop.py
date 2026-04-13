from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .body_protocol import RobotWalkBody
    from .heuristic_policy import BrightnessAvoidancePolicy


class TaskPlanningLoop:
    def __init__(self, body: "RobotWalkBody", policy: Optional["BrightnessAvoidancePolicy"] = None):
        from .heuristic_policy import BrightnessAvoidancePolicy as _P

        self.body = body
        self.policy = policy or _P()
        self.running = True

    def run_closed_loop(
        self, loop_hz: float = 10.0, max_ticks: Optional[int] = None
    ) -> None:
        print(
            "[management] Closed loop started "
            "(vision -> analyze -> decide -> velocity, aligned with brain_api_example)..."
        )
        dt = 1.0 / loop_hz
        tick = 0
        self.body.start()
        try:
            while self.running:
                if max_ticks is not None and tick >= max_ticks:
                    break
                loop_start = time.time()

                images = self.body.get_camera_images()
                if images:
                    vision_info = self.policy.analyze_vision(images)
                    vx, vy, yaw_rate, period = self.policy.decide_action(vision_info)
                    self.body.set_velocity_command(vx, vy, yaw_rate, period)
                    print(
                        f"[management] tick={tick} brightness={vision_info.get('brightness', 0):.1f} "
                        f"obstacle={vision_info.get('obstacle_detected')} "
                        f"-> vx={vx:.2f} vy={vy:.2f} yaw={yaw_rate:.2f}"
                    )
                else:
                    self.body.set_velocity_command(0.0, 0.0, 0.0, 0.8)
                    print("[management] No vision; zero velocity.")

                tick += 1
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except KeyboardInterrupt:
            print("\n[management] KeyboardInterrupt")
        finally:
            self.body.set_velocity_command(0.0, 0.0, 0.0, 0.8)
            self.running = False
            self.body.shutdown()
            print("[management] Loop stopped.")

    def stop(self) -> None:
        self.running = False
