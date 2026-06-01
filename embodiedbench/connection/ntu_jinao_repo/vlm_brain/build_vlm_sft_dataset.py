"""Build Qwen3-VL SFT JSONL skeletons from collected VLM/episode artifacts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VLM SFT dataset JSONL.")
    parser.add_argument("--episodes_jsonl", default=None)
    parser.add_argument("--vlm_inputs_jsonl", default=None)
    parser.add_argument("--predictor_feedback_dir", default=None)
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_val", required=True)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    samples: list[dict[str, Any]] = []
    if args.vlm_inputs_jsonl and Path(args.vlm_inputs_jsonl).exists():
        samples.extend(_read_vlm_inputs(Path(args.vlm_inputs_jsonl)))
    if args.episodes_jsonl and Path(args.episodes_jsonl).exists():
        samples.extend(_read_episode_generation_samples(Path(args.episodes_jsonl)))

    if not samples:
        print("[WARN] No SFT samples found. Writing empty train/val JSONL files.")

    random.Random(args.seed).shuffle(samples)
    n_val = int(round(len(samples) * args.val_ratio))
    val = samples[:n_val]
    train = samples[n_val:]
    _write_jsonl(Path(args.output_train), train)
    _write_jsonl(Path(args.output_val), val)
    print(f"[INFO] Wrote SFT train={len(train)} val={len(val)}")


def _read_vlm_inputs(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    samples = []
    for row in rows:
        sample_type = row.get("sample_type", "blueprint_generation")
        image_path = row.get("image_path")
        samples.append(
            {
                "sample_type": sample_type,
                "text_only": not bool(image_path and Path(image_path).exists()),
                "image_path": image_path,
                "input": {
                    "task": row.get("task"),
                    "scene_state": row.get("scene_state"),
                    "original_blueprint": row.get("original_blueprint"),
                    "predictor_feedback": row.get("predictor_feedback"),
                },
                "output": row.get("skill_blueprint") or row.get("revised_blueprint"),
            }
        )
    return samples


def _read_episode_generation_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    for row in _read_jsonl(path):
        blueprint_id = row.get("blueprint_id")
        if not blueprint_id:
            continue
        samples.append(
            {
                "sample_type": "blueprint_generation",
                "text_only": True,
                "image_path": row.get("image_path"),
                "input": {
                    "task": row.get("task", "pick the cube and place it on the target"),
                    "scene_state": row.get("initial_scene") or row.get("scene_state"),
                },
                "output": row.get("skill_blueprint") or row.get("blueprint"),
            }
        )
    return samples


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with path.expanduser().open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
