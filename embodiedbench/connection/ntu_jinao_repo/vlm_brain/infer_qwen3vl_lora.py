"""Qwen3-VL LoRA inference entrypoint placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL LoRA inference.")
    parser.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora_checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.lora_checkpoint).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {checkpoint}")
    raise NotImplementedError(
        "LoRA inference wiring is reserved for the next stage. "
        "The checkpoint path exists; load PEFT adapter here when training artifacts are available."
    )


if __name__ == "__main__":
    main()
