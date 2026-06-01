"""PyTorch dataset wrapper for processed predictor splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class SkillPerformanceDataset(Dataset):
    """Dataset backed by a processed ``*.pt`` split file."""

    def __init__(self, split_path: str | Path):
        self.split_path = Path(split_path)
        try:
            self.data: dict[str, Any] = torch.load(self.split_path, map_location="cpu", weights_only=False)
        except TypeError:
            self.data = torch.load(self.split_path, map_location="cpu")

    def __len__(self) -> int:
        return int(self.data["numeric_features"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "numeric_features": self.data["numeric_features"][index].float(),
            "skill_id": self.data["skill_ids"][index].long(),
            "target_id": self.data["target_ids"][index].long(),
            "success": self.data["success"][index].float(),
            "timeout": self.data["timeout"][index].float(),
            "failure_reason": self.data["failure_reason"][index].long(),
            "regression_targets": self.data["regression_targets"][index].float(),
            "regression_mask": self.data["regression_mask"][index].float(),
        }


def load_split(data_dir: str | Path, split: str) -> SkillPerformanceDataset:
    return SkillPerformanceDataset(Path(data_dir) / f"{split}.pt")
