#!/usr/bin/env python3
"""
CLI: same closed loop as repair/brain_api_example, with body = real WalkController or Sapien.

From repository root:
  python -m embodiedbench.management.run_management --body sapien --max-ticks 30
  python -m embodiedbench.management.run_management --body repair
"""
from __future__ import annotations

import argparse

from .heuristic_policy import BrightnessAvoidancePolicy
from .planning_loop import TaskPlanningLoop
from .repair_walk_body import RepairWalkBody
from .sapien_velocity_body import SapienVelocityBody


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Management closed loop (aligned with brain_api_example)"
    )
    parser.add_argument(
        "--body",
        choices=("repair", "sapien"),
        default="sapien",
        help="repair: WalkController+DDS; sapien: same loop, commands mapped to Sapien",
    )
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Exit after N ticks (sim demos); omit to run until Ctrl+C",
    )
    parser.add_argument("--no-gui", action="store_true", help="Sapien without GUI window")
    args = parser.parse_args()

    if args.body == "repair":
        body = RepairWalkBody()
    else:
        body = SapienVelocityBody(use_gui=not args.no_gui)

    loop = TaskPlanningLoop(body, BrightnessAvoidancePolicy())
    loop.run_closed_loop(loop_hz=args.loop_hz, max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
