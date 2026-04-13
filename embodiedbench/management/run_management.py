#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .heuristic_policy import BrightnessAvoidancePolicy
from .planning_loop import TaskPlanningLoop
from .repair_walk_body import RepairWalkBody
from .sapien_velocity_body import SapienVelocityBody


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", choices=("repair", "sapien"), default="sapien")
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    if args.body == "repair":
        body = RepairWalkBody()
    else:
        body = SapienVelocityBody(use_gui=not args.no_gui)

    loop = TaskPlanningLoop(body, BrightnessAvoidancePolicy())
    loop.run_closed_loop(loop_hz=args.loop_hz, max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
