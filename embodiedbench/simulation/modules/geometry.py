import math
from typing import Tuple

class NavigationGeometryChecker:
    def __init__(self, robot_radius: float):
        self.robot_radius = robot_radius

    def check_passable_width(self, gap_width: float) -> bool:
        """
        静态物理常识校验：路口宽度必须大于机器人直径，且留有安全余量
        """
        required_width = (self.robot_radius * 2) * 1.1 # 10% 容错余量
        return gap_width >= required_width

    def calculate_distance(self, pos1: Tuple[float, ...], pos2: Tuple[float, ...]) -> float:
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
