"""Multi-task MLP model for skill performance prediction.

The regression head size is schema-driven so predictor_v0/v1 datasets can use
different continuous target sets without changing the model class.
"""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskSkillPerformancePredictor(nn.Module):
    """Predict classification and regression performance labels for one skill sample."""

    def __init__(
        self,
        numeric_dim: int,
        num_skills: int,
        num_targets: int,
        num_failure_reasons: int,
        num_regression_targets: int,
        skill_embedding_dim: int = 8,
        target_embedding_dim: int = 6,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128, 64]
        self.skill_embedding = nn.Embedding(num_skills, skill_embedding_dim)
        self.target_embedding = nn.Embedding(num_targets, target_embedding_dim)

        input_dim = numeric_dim + skill_embedding_dim + target_embedding_dim
        self.backbone = _build_backbone(input_dim, hidden_dims, dropout, activation)
        backbone_dim = hidden_dims[-1]

        self.success_head = nn.Linear(backbone_dim, 1)
        self.timeout_head = nn.Linear(backbone_dim, 1)
        self.failure_reason_head = nn.Linear(backbone_dim, num_failure_reasons)
        self.regression_head = nn.Linear(backbone_dim, num_regression_targets)

    def forward(self, numeric_features: torch.Tensor, skill_id: torch.Tensor, target_id: torch.Tensor) -> dict[str, torch.Tensor]:
        skill_emb = self.skill_embedding(skill_id)
        target_emb = self.target_embedding(target_id)
        features = torch.cat([numeric_features, skill_emb, target_emb], dim=-1)
        shared = self.backbone(features)
        return {
            "success_logits": self.success_head(shared).squeeze(-1),
            "timeout_logits": self.timeout_head(shared).squeeze(-1),
            "failure_reason_logits": self.failure_reason_head(shared),
            "regression": self.regression_head(shared),
        }


def _build_backbone(input_dim: int, hidden_dims: list[int], dropout: float, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    activation_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU
    for idx, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if idx < len(hidden_dims) - 1:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_layer())
        if idx < len(hidden_dims) - 1 and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)
