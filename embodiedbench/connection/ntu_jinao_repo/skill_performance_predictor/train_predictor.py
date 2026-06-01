"""Train the multi-task skill performance predictor."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SkillPerformanceDataset
from dataset_schema import NUMERIC_FEATURE_NAMES, PREDICTOR_SCHEMA_VERSION, REGRESSION_TARGET_NAMES
from model import MultiTaskSkillPerformancePredictor

SMALL_DATASET_WARNING = "Dataset is too small for reliable training; this run is for pipeline validation only."


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skill performance predictor.")
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--device", default="auto", type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _read_json(Path(args.config).expanduser())
    train_cfg = config.setdefault("training", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr

    seed = int(train_cfg.get("seed", 0))
    _set_seed(seed)
    device = _resolve_device(args.device)

    train_dataset = SkillPerformanceDataset(data_dir / "train.pt")
    val_dataset = SkillPerformanceDataset(data_dir / "val.pt")
    if len(train_dataset) < 50:
        print(f"[WARN] {SMALL_DATASET_WARNING}")

    with (data_dir / "vocab.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    with (data_dir / "feature_stats.json").open("r", encoding="utf-8") as f:
        feature_stats = json.load(f)
    with (data_dir / "label_stats.json").open("r", encoding="utf-8") as f:
        label_stats = json.load(f)

    model_cfg = config.get("model", {})
    model = MultiTaskSkillPerformancePredictor(
        numeric_dim=len(NUMERIC_FEATURE_NAMES),
        num_skills=len(vocab["skill"]),
        num_targets=len(vocab["target"]),
        num_failure_reasons=len(vocab["failure_reason"]),
        num_regression_targets=len(REGRESSION_TARGET_NAMES),
        **model_cfg,
    ).to(device)

    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )

    loss_weights = config.get("loss_weights", {})
    unknown_failure_idx = int(vocab["failure_reason"].get("unknown", -100))
    loss_fns = {
        "bce": nn.BCEWithLogitsLoss(),
        "ce": nn.CrossEntropyLoss(ignore_index=unknown_failure_idx),
        "smooth_l1": nn.SmoothL1Loss(reduction="none"),
    }

    best_val_loss = math.inf
    train_log_path = output_dir / "train_log.jsonl"
    with train_log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, int(train_cfg.get("epochs", 100)) + 1):
            train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                loss_fns,
                loss_weights,
                optimizer=optimizer,
                grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)),
            )
            val_metrics = _run_epoch(model, val_loader, device, loss_fns, loss_weights, optimizer=None)
            if len(val_dataset) == 0:
                val_metrics = train_metrics

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_file.flush()

            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"success_accuracy={val_metrics['success_accuracy']:.4f} "
                f"timeout_accuracy={val_metrics['timeout_accuracy']:.4f} "
                f"failure_reason_accuracy={val_metrics['failure_reason_accuracy']:.4f} "
                f"regression_mae={val_metrics['regression_mae']:.4f}"
            )

            checkpoint = _checkpoint(model, config, vocab, feature_stats, label_stats)
            torch.save(checkpoint, output_dir / "last_model.pt")
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(checkpoint, output_dir / "best_model.pt")

    _write_json(output_dir / "config_used.json", config)
    _write_json(output_dir / "eval_report.json", val_metrics)
    shutil.copyfile(data_dir / "feature_stats.json", output_dir / "feature_stats.json")
    shutil.copyfile(data_dir / "vocab.json", output_dir / "vocab.json")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fns: dict[str, nn.Module],
    loss_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = _metric_totals()

    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(batch["numeric_features"], batch["skill_id"], batch["target_id"])
        losses = _compute_losses(outputs, batch, loss_fns, loss_weights)
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        _accumulate_metrics(totals, outputs, batch, float(losses["total"].detach().cpu().item()))

    return _finalize_metrics(totals)


def _compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_fns: dict[str, nn.Module],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    success_loss = loss_fns["bce"](outputs["success_logits"], batch["success"])
    timeout_loss = loss_fns["bce"](outputs["timeout_logits"], batch["timeout"])
    failure_reason_loss = loss_fns["ce"](outputs["failure_reason_logits"], batch["failure_reason"])
    per_target_reg = loss_fns["smooth_l1"](outputs["regression"], batch["regression_targets"])
    mask = batch["regression_mask"]
    regression_loss = (per_target_reg * mask).sum() / mask.sum().clamp_min(1.0)
    total = (
        float(weights.get("success", 1.0)) * success_loss
        + float(weights.get("timeout", 1.0)) * timeout_loss
        + float(weights.get("failure_reason", 1.0)) * failure_reason_loss
        + float(weights.get("regression", 1.0)) * regression_loss
    )
    return {"total": total}


def _metric_totals() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "success_correct": 0.0,
        "timeout_correct": 0.0,
        "failure_correct": 0.0,
        "failure_count": 0.0,
        "reg_abs_error_sum": 0.0,
        "reg_mask_sum": 0.0,
    }


def _accumulate_metrics(totals: dict[str, float], outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss: float) -> None:
    batch_size = int(batch["success"].shape[0])
    totals["loss_sum"] += loss * batch_size
    totals["sample_count"] += batch_size
    totals["success_correct"] += ((torch.sigmoid(outputs["success_logits"]) >= 0.5) == (batch["success"] >= 0.5)).sum().item()
    totals["timeout_correct"] += ((torch.sigmoid(outputs["timeout_logits"]) >= 0.5) == (batch["timeout"] >= 0.5)).sum().item()
    failure_pred = outputs["failure_reason_logits"].argmax(dim=-1)
    totals["failure_correct"] += (failure_pred == batch["failure_reason"]).sum().item()
    totals["failure_count"] += batch_size
    reg_abs = (outputs["regression"] - batch["regression_targets"]).abs() * batch["regression_mask"]
    totals["reg_abs_error_sum"] += reg_abs.sum().item()
    totals["reg_mask_sum"] += batch["regression_mask"].sum().item()


def _finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    n = max(1.0, totals["sample_count"])
    return {
        "loss": totals["loss_sum"] / n,
        "success_accuracy": totals["success_correct"] / n,
        "timeout_accuracy": totals["timeout_correct"] / n,
        "failure_reason_accuracy": totals["failure_correct"] / max(1.0, totals["failure_count"]),
        "regression_mae": totals["reg_abs_error_sum"] / max(1.0, totals["reg_mask_sum"]),
    }


def _checkpoint(model: nn.Module, config: dict[str, Any], vocab: dict[str, Any], feature_stats: dict[str, Any], label_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "predictor_schema_version": PREDICTOR_SCHEMA_VERSION,
        "vocab": vocab,
        "feature_stats": feature_stats,
        "label_stats": label_stats,
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "regression_target_names": REGRESSION_TARGET_NAMES,
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
