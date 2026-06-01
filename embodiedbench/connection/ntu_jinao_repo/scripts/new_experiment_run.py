#!/usr/bin/env python3
"""Create a new minute-stamped experiment run under data/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.run_layout import ExperimentRun  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unified experiment run directories.")
    parser.add_argument("--note", default="", help="Optional note in manifest.")
    parser.add_argument("--run_id", default=None, help="Override run id (YYYY-MM-DD_HHMM).")
    args = parser.parse_args()

    run = ExperimentRun.create(run_id=args.run_id, note=args.note)
    print(f"run_id={run.run_id}")
    print(f"manifest={run.manifest_path.resolve()}")
    print(f"vlm_inputs={run.vlm_inputs.resolve()}")
    print(f"vlm_outputs={run.vlm_outputs.resolve()}")
    print(f"predictor_outputs={run.predictor_outputs.resolve()}")
    print(f"execution_data={run.execution_data.resolve()}")


if __name__ == "__main__":
    main()
