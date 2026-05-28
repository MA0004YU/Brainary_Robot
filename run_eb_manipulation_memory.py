#!/usr/bin/env python3
"""
Launcher for EB-Manipulation with the architectural memory stack.

Prepends this repo and the simulator subtree to sys.path so `embodiedbench` and `amsolver`
resolve the same way as in the upstream EmbodiedBench project.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EB_MAN = os.path.join(ROOT, "embodiedbench", "envs", "eb_manipulation")
sys.path.insert(0, ROOT)
sys.path.insert(0, EB_MAN)


def main() -> None:
    parser = argparse.ArgumentParser(description="EB-Manipulation + working/long-term memory")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--model_type", type=str, default="remote")
    parser.add_argument("--eval_sets", type=lambda s: s.split(","), default=["base"])
    parser.add_argument("--n_shots", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--down_sample_ratio", type=float, default=1.0)
    parser.add_argument("--language_only", type=int, default=0)
    parser.add_argument("--chat_history", type=int, default=0)
    parser.add_argument("--detection_box", type=int, default=1)
    parser.add_argument("--multiview", type=int, default=0)
    parser.add_argument("--multistep", type=int, default=0)
    parser.add_argument("--visual_icl", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="memory_arch")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--use_architectural_memory",
        type=int,
        default=1,
        help="1 enables WM/LTM hooks; perception defaults to NullPerception until a custom PerceptionInterface is wired.",
    )
    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "eval_sets": args.eval_sets,
        "n_shots": args.n_shots,
        "resolution": args.resolution,
        "language_only": args.language_only,
        "down_sample_ratio": args.down_sample_ratio,
        "chat_history": args.chat_history,
        "detection_box": args.detection_box,
        "multiview": args.multiview,
        "multistep": args.multistep,
        "visual_icl": args.visual_icl,
        "exp_name": args.exp_name,
        "tp": args.tp,
        "use_architectural_memory": args.use_architectural_memory,
        "selected_indexes": [],
    }

    os.chdir(ROOT)
    from embodiedbench.evaluator.eb_manipulation_evaluator import EB_ManipulationEvaluator

    ev = EB_ManipulationEvaluator(config)
    ev.check_config_valid()
    ev.evaluate_main()


if __name__ == "__main__":
    main()
