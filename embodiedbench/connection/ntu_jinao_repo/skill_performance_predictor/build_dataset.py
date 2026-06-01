"""Build processed train/val/test tensors from predictor_dataset.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from dataset_schema import NUMERIC_FEATURE_NAMES, PREDICTOR_SCHEMA_VERSION, REGRESSION_TARGET_NAMES, default_vocab
from feature_extractor import extract_sample

SMALL_DATASET_WARNING = "Dataset is too small for reliable training; this run is for pipeline validation only."


def main() -> None:
    parser = argparse.ArgumentParser(description="Build skill performance predictor dataset.")
    parser.add_argument("--input_jsonl", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--val_ratio", default=0.15, type=float)
    parser.add_argument("--test_ratio", default=0.15, type=float)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_jsonl(input_path)
    if len(samples) < 50:
        print(f"[WARN] {SMALL_DATASET_WARNING}")
    extracted = [extract_sample(sample) for sample in samples]
    splits = _split_by_episode(extracted, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)

    vocab = default_vocab()
    train_stats = _feature_stats(splits["train"])
    label_stats = _label_stats(splits["train"])

    for split_name, split_samples in splits.items():
        tensor_data = _tensorize(split_samples, vocab, train_stats)
        torch.save(tensor_data, output_dir / f"{split_name}.pt")

    _write_json(output_dir / "feature_stats.json", train_stats)
    _write_json(output_dir / "label_stats.json", label_stats)
    _write_json(output_dir / "vocab.json", vocab)
    _write_json(
        output_dir / "dataset_info.json",
        {
            "input_jsonl": input_path.as_posix(),
            "predictor_schema_version": PREDICTOR_SCHEMA_VERSION,
            "num_samples": len(extracted),
            "num_episodes": len({sample["episode_id"] for sample in extracted}),
            "split_counts": {name: len(value) for name, value in splits.items()},
            "split_episode_counts": {
                name: len({sample["episode_id"] for sample in value}) for name, value in splits.items()
            },
            "numeric_feature_names": NUMERIC_FEATURE_NAMES,
            "regression_target_names": REGRESSION_TARGET_NAMES,
            "small_dataset_warning": SMALL_DATASET_WARNING if len(samples) < 50 else None,
        },
    )
    print(f"[INFO] Wrote processed dataset to: {output_dir.resolve()}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_idx}: {exc}") from exc
    return rows


def _split_by_episode(
    samples: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    train_ratio /= total_ratio
    val_ratio /= total_ratio

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_id"]].append(sample)
    episodes = sorted(by_episode)
    random.Random(seed).shuffle(episodes)

    n_episode = len(episodes)
    n_train = int(round(n_episode * train_ratio))
    n_val = int(round(n_episode * val_ratio))
    if n_episode > 0 and n_train == 0:
        n_train = 1
    if n_train + n_val > n_episode:
        n_val = max(0, n_episode - n_train)

    split_episodes = {
        "train": episodes[:n_train],
        "val": episodes[n_train : n_train + n_val],
        "test": episodes[n_train + n_val :],
    }
    return {
        split: [sample for episode_id in episode_ids for sample in by_episode[episode_id]]
        for split, episode_ids in split_episodes.items()
    }


def _feature_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        mean = [0.0 for _ in NUMERIC_FEATURE_NAMES]
        std = [1.0 for _ in NUMERIC_FEATURE_NAMES]
        return {"mean": mean, "std": std, "feature_names": NUMERIC_FEATURE_NAMES}

    features = torch.tensor([sample["numeric_features"] for sample in samples], dtype=torch.float32)
    mean_tensor = features.mean(dim=0)
    std_tensor = features.std(dim=0, unbiased=False)
    std_tensor = torch.where(std_tensor < 1.0e-6, torch.ones_like(std_tensor), std_tensor)
    return {
        "mean": mean_tensor.tolist(),
        "std": std_tensor.tolist(),
        "feature_names": NUMERIC_FEATURE_NAMES,
    }


def _label_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "regression_target_names": REGRESSION_TARGET_NAMES,
            "regression_missing_fraction": [1.0 for _ in REGRESSION_TARGET_NAMES],
        }
    mask = torch.tensor([sample["regression_mask"] for sample in samples], dtype=torch.float32)
    missing_fraction = (1.0 - mask).mean(dim=0).tolist()
    return {
        "regression_target_names": REGRESSION_TARGET_NAMES,
        "regression_missing_fraction": missing_fraction,
    }


def _tensorize(samples: list[dict[str, Any]], vocab: dict[str, dict[str, int]], stats: dict[str, Any]) -> dict[str, Any]:
    feature_count = len(NUMERIC_FEATURE_NAMES)
    regression_count = len(REGRESSION_TARGET_NAMES)
    if not samples:
        return {
            "numeric_features": torch.empty((0, feature_count), dtype=torch.float32),
            "skill_ids": torch.empty((0,), dtype=torch.long),
            "target_ids": torch.empty((0,), dtype=torch.long),
            "success": torch.empty((0,), dtype=torch.float32),
            "timeout": torch.empty((0,), dtype=torch.float32),
            "failure_reason": torch.empty((0,), dtype=torch.long),
            "regression_targets": torch.empty((0, regression_count), dtype=torch.float32),
            "regression_mask": torch.empty((0, regression_count), dtype=torch.float32),
            "metadata": [],
        }

    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    numeric = torch.tensor([sample["numeric_features"] for sample in samples], dtype=torch.float32)
    numeric = (numeric - mean) / std

    unknown_skill = vocab["skill"]["unknown"]
    unknown_target = vocab["target"]["unknown"]
    unknown_failure = vocab["failure_reason"]["unknown"]
    return {
        "numeric_features": numeric,
        "skill_ids": torch.tensor([vocab["skill"].get(sample["skill"], unknown_skill) for sample in samples]),
        "target_ids": torch.tensor([vocab["target"].get(sample["target"], unknown_target) for sample in samples]),
        "success": torch.tensor([sample["success"] for sample in samples], dtype=torch.float32),
        "timeout": torch.tensor([sample["timeout"] for sample in samples], dtype=torch.float32),
        "failure_reason": torch.tensor(
            [vocab["failure_reason"].get(sample["failure_reason"], unknown_failure) for sample in samples],
            dtype=torch.long,
        ),
        "regression_targets": torch.tensor([sample["regression_targets"] for sample in samples], dtype=torch.float32),
        "regression_mask": torch.tensor([sample["regression_mask"] for sample in samples], dtype=torch.float32),
        "metadata": [
            {
                "sample_id": sample["sample_id"],
                "episode_id": sample["episode_id"],
                "node_id": sample["node_id"],
                "skill": sample["skill"],
                "target": sample["target"],
                "performance_query": sample["performance_query"],
            }
            for sample in samples
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
