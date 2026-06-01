"""Qwen3-VL LoRA training skeleton with dry-run validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or start Qwen3-VL LoRA training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--dry_run", default=None)
    args = parser.parse_args()

    config = _read_json(Path(args.config))
    dry_run = _str_to_bool(args.dry_run) if args.dry_run is not None else bool(config.get("dry_run", True))
    train_rows = _read_jsonl_if_exists(Path(args.train_jsonl))
    val_rows = _read_jsonl_if_exists(Path(args.val_jsonl))
    if not train_rows:
        print(f"[WARN] No training samples found at {args.train_jsonl}. This is not a failure for dry-run.")
    _validate_rows(train_rows, "train")
    _validate_rows(val_rows, "val")
    print(f"[INFO] LoRA config ok. train_samples={len(train_rows)} val_samples={len(val_rows)} dry_run={dry_run}")
    print(f"[INFO] Base model: {config.get('base_model')}")
    print(f"[INFO] Output dir: {config.get('output_dir')}")
    if dry_run:
        return

    try:
        import peft  # noqa: F401
        from qwen3vl_loader import load_qwen3vl_model
    except Exception as exc:
        raise RuntimeError("Missing LoRA dependencies. Install peft and vlm_brain/requirements_vlm.txt first.") from exc
    try:
        load_qwen3vl_model({"model_name_or_path": config["base_model"], "torch_dtype": "auto", "device": "auto"})
    except RuntimeError as exc:
        raise RuntimeError(f"Could not start non-dry-run LoRA training: {exc}") from exc
    print("[WARN] Full LoRA optimizer/trainer loop is intentionally left as a Stage IV+ implementation step.")


def _validate_rows(rows: list[dict[str, Any]], split: str) -> None:
    for idx, row in enumerate(rows):
        if "input" not in row or "output" not in row:
            raise ValueError(f"{split}:{idx}: expected keys 'input' and 'output'")
        if row.get("image_path") and not Path(str(row["image_path"])).exists():
            row["text_only"] = True


def _read_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.expanduser().exists():
        print(f"[WARN] JSONL not found: {path}")
        return []
    rows = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    main()
