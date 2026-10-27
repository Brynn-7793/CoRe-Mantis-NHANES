"""Concept-aligned residual classifier for frozen foundation embeddings."""

from __future__ import annotations

import torch
from torch import nn


class ConceptResidualNet(nn.Module):
    """Add reliability-gated residual evidence to auditable concepts."""

    def __init__(
        self,
        embedding_dim: int,
        num_concepts: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.concept_decoder = nn.Linear(hidden_dim, num_concepts)
        self.concept_weights = nn.Parameter(torch.zeros(num_concepts))
        self.intercept = nn.Parameter(torch.zeros(()))
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )
        self.completeness_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self.residual_scale_logit = nn.Parameter(torch.tensor(-1.5))

    def forward(
        self,
        embeddings: torch.Tensor,
        concepts: torch.Tensor,
        reliabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if reliabilities is None:
            reliabilities = torch.ones_like(concepts)
        if reliabilities.shape != concepts.shape:
            raise ValueError("reliabilities and concepts must have identical shapes")

        adapted = self.adapter(embeddings)
        reconstructed_concepts = self.concept_decoder(adapted)
        reliable_concepts = concepts * reliabilities
        concept_contributions = reliable_concepts * self.concept_weights
        concept_logit = concept_contributions.sum(dim=-1)

        reliability_score = reliabilities.mean(dim=-1)
        reconstruction_disagreement = (
            (reconstructed_concepts.detach() - concepts).abs() * reliabilities
        ).mean(dim=-1)
        gate_features = torch.stack(
            (reliability_score, reconstruction_disagreement),
            dim=-1,
        )
        completeness_gate = torch.sigmoid(self.completeness_gate(gate_features)).squeeze(-1)
        residual_gate = reliability_score * completeness_gate
        residual_scale = torch.sigmoid(self.residual_scale_logit)
        raw_residual_logit = self.residual_head(adapted).squeeze(-1)
        residual_logit = raw_residual_logit * residual_scale * residual_gate
        logit = self.intercept + concept_logit + residual_logit
        return {
            "logit": logit,
            "concept_logit": concept_logit,
            "concept_contributions": concept_contributions,
            "reconstructed_concepts": reconstructed_concepts,
            "raw_residual_logit": raw_residual_logit,
            "residual_logit": residual_logit,
            "residual_gate": residual_gate,
            "residual_scale": residual_scale,
            "reliability_score": reliability_score,
        }


def concept_residual_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    concepts: torch.Tensor,
    reliabilities: torch.Tensor,
    positive_weight: torch.Tensor,
    concept_weight: float = 0.20,
    residual_weight: float = 0.02,
    gate_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint discrimination, concept alignment, and residual-capacity loss."""
    binary_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["logit"],
        labels,
        pos_weight=positive_weight,
    )
    reconstruction_error = nn.functional.smooth_l1_loss(
        outputs["reconstructed_concepts"], concepts, reduction="none"
    )
    reconstruction_loss = (reconstruction_error * reliabilities).sum() / reliabilities.sum().clamp_min(1.0)
    residual_penalty = outputs["residual_logit"].abs().mean()
    gate_penalty = outputs["residual_gate"].mean()
    loss = (
        binary_loss
        + concept_weight * reconstruction_loss
        + residual_weight * residual_penalty
        + gate_weight * gate_penalty
    )
    return loss, {
        "binary": binary_loss.detach(),
        "concept": reconstruction_loss.detach(),
        "residual": residual_penalty.detach(),
        "gate": gate_penalty.detach(),
    }
