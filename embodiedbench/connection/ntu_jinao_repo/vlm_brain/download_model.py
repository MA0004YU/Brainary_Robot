"""Download/cache Qwen3-VL weights outside the repository."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-VL to Hugging Face cache.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Missing huggingface_hub. Install vlm_brain/requirements_vlm.txt first.") from exc

    repo_root = Path(__file__).resolve().parents[1]
    cache_path = Path(snapshot_download(repo_id=args.model_name))
    if repo_root in cache_path.parents:
        raise RuntimeError(f"Refusing to cache model inside repository: {cache_path}")
    print(f"[INFO] Model cached at: {cache_path}")


if __name__ == "__main__":
    main()
