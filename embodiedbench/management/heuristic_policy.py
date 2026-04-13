from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class BrightnessAvoidancePolicy:
    def analyze_vision(self, images: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not images or "head" not in images or images["head"] is None:
            return {"obstacle_detected": False, "obstacle_direction": "none", "brightness": 0.0}

        front_img = images["head"]
        h, w = front_img.shape[:2]
        center_region = front_img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        avg_brightness = float(np.mean(center_region))
        obstacle_detected = avg_brightness < 80

        left_brightness = float(np.mean(front_img[:, : w // 2]))
        right_brightness = float(np.mean(front_img[:, w // 2 :]))

        if obstacle_detected:
            direction = "left_clear" if left_brightness > right_brightness else "right_clear"
        else:
            direction = "front_clear"

        return {
            "obstacle_detected": obstacle_detected,
            "obstacle_direction": direction,
            "brightness": avg_brightness,
        }

    def decide_action(self, vision_info: Dict[str, Any]) -> Tuple[float, float, float, float]:
        base_speed = 0.3
        turn_rate = 0.5
        period = 0.8

        if not vision_info["obstacle_detected"]:
            return (base_speed, 0.0, 0.0, period)
        if vision_info["obstacle_direction"] == "left_clear":
            return (base_speed * 0.5, 0.0, turn_rate, period)
        if vision_info["obstacle_direction"] == "right_clear":
            return (base_speed * 0.5, 0.0, -turn_rate, period)
        return (0.0, 0.0, 0.0, period)
