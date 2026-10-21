"""Notebook-scale NeuroVFM stack.

The released model (Kondepudi et al., Nature Medicine 2026) has three stages:

1. Vol-JEPA pretraining of a 3D ViT on unlabeled CT/MRI volumes
2. a frozen encoder plus a Classify-Then-Aggregate MIL head
   (74 MRI labels or 82 CT labels in the release)
3. a findings LLM (Qwen3-8B) on top of the frozen encoder

This module implements stages 1 and 2. Stage 3 is a language-model
fine-tune and is not reproduced here. Nothing in this file is a
medical device; weights are randomly initialized.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import init_weights
from vol_jepa import VolumeViT, VolJEPA


class ClassifyThenAggregate(nn.Module):
    """Study-level head used by NeuroVFM diagnostic training.

    Each token gets a class logit. Attention is a per-class softmax over
    tokens in the study. The study logit is the attention-weighted sum of
    token logits, then a learned scale and bias.
    """

    def __init__(self, dim: int, num_labels: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.attn_v = nn.Linear(dim, hidden_dim)
        self.attn_gate = nn.Linear(dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, num_labels)
        self.token_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )
        self.output_scale = nn.Parameter(torch.ones(num_labels))
        self.output_bias = nn.Parameter(torch.zeros(num_labels))
        self.apply(init_weights)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, N, D] -> study logits [B, num_labels]."""
        scores = self.attn_score(torch.tanh(self.attn_v(tokens)) * torch.sigmoid(self.attn_gate(tokens)))
        attention = torch.softmax(scores, dim=1)
        token_logits = self.token_mlp(tokens)
        study_logits = (attention * token_logits).sum(dim=1)
        return study_logits * self.output_scale + self.output_bias

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(tokens))


class NeuroVFM(nn.Module):
    """Vol-JEPA encoder plus a multi-label diagnostic head.

    `pretrain_step` runs one Vol-JEPA update.
    `finetune_step` freezes that encoder and trains only the MIL head.
    """

    def __init__(
        self,
        num_labels: int = 4,
        patch_size: tuple[int, int, int] = (2, 4, 4),
        content_dim: int = 72,
        pe_dim: int = 24,
        depth: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.pretraining = VolJEPA(
            patch_size=patch_size,
            content_dim=content_dim,
            pe_dim=pe_dim,
            depth=depth,
            num_heads=num_heads,
            predictor_depth=depth,
        )
        embed_dim = self.pretraining.embed_dim
        self.diagnostic_head = ClassifyThenAggregate(embed_dim, num_labels, hidden_dim=embed_dim)
        self._encoder_frozen = False

    @property
    def encoder(self) -> VolumeViT:
        return self.pretraining.student

    def embed(self, volume: torch.Tensor) -> torch.Tensor:
        """volume [B, 1, D, H, W] -> token embeddings [B, N, D]."""
        tokens, coords = self.pretraining.tokenize(volume)
        return self.encoder(tokens, coords)

    def predict(self, volume: torch.Tensor) -> torch.Tensor:
        """volume -> multi-label probabilities [B, num_labels]."""
        return self.diagnostic_head.predict(self.embed(volume))

    def pretrain_loss(self, volume: torch.Tensor) -> torch.Tensor:
        if self._encoder_frozen:
            raise RuntimeError("encoder is frozen; call pretrain_loss before freeze_encoder")
        return self.pretraining(volume)

    def classification_loss(self, volume: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """labels: multi-hot float tensor [B, num_labels]."""
        embeddings = self.embed(volume).detach()
        logits = self.diagnostic_head(embeddings)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True

    def update_target(self, momentum: float = 0.994) -> None:
        self.pretraining.update_target(momentum)
