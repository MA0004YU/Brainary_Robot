"""Evaluate a trained skill performance predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset import SkillPerformanceDataset
from model import MultiTaskSkillPerformancePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate skill performance predictor.")
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--device", default="auto", type=str)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = _build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SkillPerformanceDataset(Path(args.data_dir).expanduser() / f"{args.split}.pt")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    report = evaluate(model, loader, checkpoint, device)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def evaluate(model: MultiTaskSkillPerformancePredictor, loader: DataLoader, checkpoint: dict[str, Any], device: torch.device) -> dict[str, Any]:
    regression_names = checkpoint["regression_target_names"]
    totals = {
        "sample_count": 0,
        "success_correct": 0,
        "timeout_correct": 0,
        "failure_correct": 0,
        "reg_abs_error": torch.zeros(len(regression_names), dtype=torch.float64),
        "reg_mask": torch.zeros(len(regression_names), dtype=torch.float64),
    }
    success_scores: list[float] = []
    success_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["numeric_features"], batch["skill_id"], batch["target_id"])
            batch_size = int(batch["success"].shape[0])
            totals["sample_count"] += batch_size
            success_prob = torch.sigmoid(outputs["success_logits"])
            timeout_prob = torch.sigmoid(outputs["timeout_logits"])
            totals["success_correct"] += ((success_prob >= 0.5) == (batch["success"] >= 0.5)).sum().item()
            totals["timeout_correct"] += ((timeout_prob >= 0.5) == (batch["timeout"] >= 0.5)).sum().item()
            failure_pred = outputs["failure_reason_logits"].argmax(dim=-1)
            totals["failure_correct"] += (failure_pred == batch["failure_reason"]).sum().item()

            reg_abs = (outputs["regression"] - batch["regression_targets"]).abs() * batch["regression_mask"]
            totals["reg_abs_error"] += reg_abs.detach().cpu().double().sum(dim=0)
            totals["reg_mask"] += batch["regression_mask"].detach().cpu().double().sum(dim=0)
            success_scores.extend(success_prob.detach().cpu().tolist())
            success_labels.extend(batch["success"].detach().cpu().long().tolist())

    n = max(1, totals["sample_count"])
    per_target_mae = {}
    missing_fraction = {}
    for idx, name in enumerate(regression_names):
        mask_count = float(totals["reg_mask"][idx].item())
        per_target_mae[name] = None if mask_count == 0 else float(totals["reg_abs_error"][idx].item() / mask_count)
        missing_fraction[name] = 1.0 if n == 0 else float(1.0 - mask_count / n)

    overall_mask = float(totals["reg_mask"].sum().item())
    report: dict[str, Any] = {
        "predictor_schema_version": checkpoint.get("predictor_schema_version", "predictor_v0"),
        "sample_count": totals["sample_count"],
        "regression_target_count": len(regression_names),
        "success_accuracy": totals["success_correct"] / n,
        "timeout_accuracy": totals["timeout_correct"] / n,
        "failure_reason_accuracy": totals["failure_correct"] / n,
        "regression_mae": per_target_mae,
        "overall_regression_mae": None
        if overall_mask == 0
        else float(totals["reg_abs_error"].sum().item() / overall_mask),
        "regression_missing_fraction": missing_fraction,
    }
    auc = _try_success_auc(success_labels, success_scores)
    if auc is not None:
        report["success_auc"] = auc
    return report


def _build_model_from_checkpoint(checkpoint: dict[str, Any]) -> MultiTaskSkillPerformancePredictor:
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    return MultiTaskSkillPerformancePredictor(
        numeric_dim=len(checkpoint["numeric_feature_names"]),
        num_skills=len(vocab["skill"]),
        num_targets=len(vocab["target"]),
        num_failure_reasons=len(vocab["failure_reason"]),
        num_regression_targets=len(checkpoint["regression_target_names"]),
        **config.get("model", {}),
    )


def _try_success_auc(labels: list[int], scores: list[float]) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


if __name__ == "__main__":
    main()
