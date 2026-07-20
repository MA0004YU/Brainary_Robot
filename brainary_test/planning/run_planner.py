#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planning.llm_client import LLMClient
from planning.task_planner import TaskPlanner

def main():
    parser = argparse.ArgumentParser(description="Task Planner")
    parser.add_argument("--use-ltm", action="store_true", help="Enable Long Term Memory via EmbodiedLTM")
    args = parser.parse_args()

    input_file = ROOT / "output" / "memory_planning_input.json"
    output_file = ROOT / "output" / "planned_actions.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Please run main.py first to generate memory output.")
        return 1

    print(f"Reading input from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        planning_input = json.load(f)

    # Initialize components
    try:
        llm = LLMClient()
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return 1

    planner = TaskPlanner(llm, use_ltm=args.use_ltm)

    # Generate plan
    print("Generating task dependency graph...")
    plan_graph = planner.generate_plan(planning_input)

    if not plan_graph:
        print("Error: Planner failed to generate a valid plan.")
        return 1

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing planned actions to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(plan_graph, f, ensure_ascii=False, indent=2)

    print("\n=== Planner Finished ===")
    print(f"Generated {len(plan_graph)} steps.")
    for step in plan_graph:
        print(f"  {step.get('id')}: {step.get('action')} {step.get('target')} (Depends on: {step.get('depends_on', [])})")

    return 0

if __name__ == "__main__":
    sys.exit(main())
