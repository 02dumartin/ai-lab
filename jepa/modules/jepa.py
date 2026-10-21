"""Minimal Joint-Embedding Predictive Architecture.

LeCun's JEPA predicts the embedding of an unobserved part of the input.
The loss lives in representation space. The predictor is conditioned on
where the target is, not on the target values themselves.

I-JEPA and Vol-JEPA are this same diagram with different tokenizers
and mask samplers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    TransformerStack,
    clone_as_teacher,
    ema_update,
    init_weights,
    sinusoidal_positions,
)


class JEPA(nn.Module):
    """Predict target-token embeddings from context-token embeddings.

    Args:
        token_dim: dimension of each input token.
        embed_dim: encoder width.
        depth: encoder depth. The predictor is shallower on purpose.
        num_heads: attention heads.
        max_len: positions are encoded up to this length.
    """

    def __init__(
        self,
        token_dim: int = 32,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        predictor_depth: int = 1,
        max_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.input_proj = nn.Linear(token_dim, embed_dim)
        self.context_encoder = TransformerStack(embed_dim, depth, num_heads)
        self.predictor = nn.ModuleDict(
            {
                "in_proj": nn.Linear(embed_dim, embed_dim),
                "stack": TransformerStack(embed_dim, predictor_depth, num_heads),
                "out_proj": nn.Linear(embed_dim, embed_dim),
            }
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.apply(init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.target_encoder = clone_as_teacher(
            nn.Sequential(self.input_proj, self.context_encoder)
        )

    def positions(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [...] long -> [..., embed_dim]."""
        coords = idx.unsqueeze(-1)
        return sinusoidal_positions(coords, self.embed_dim)

    def sample_mask(self, length: int, target_len: int, device: torch.device):
        """One contiguous target span. Context is everything else."""
        if target_len >= length:
            raise ValueError("target_len must be shorter than the sequence")
        start = torch.randint(0, length - target_len + 1, (), device=device).item()
        target_idx = torch.arange(start, start + target_len, device=device)
        keep = torch.ones(length, dtype=torch.bool, device=device)
        keep[target_idx] = False
        context_idx = keep.nonzero().flatten()
        return context_idx, target_idx

    def predict(self, context: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """context: [B, Nc, D]. Returns [B, Nt, D]."""
        target_pos = self.positions(target_idx).to(dtype=context.dtype)
        mask_tokens = self.mask_token + target_pos.unsqueeze(0)
        tokens = torch.cat(
            [self.predictor["in_proj"](context), self.predictor["in_proj"](mask_tokens.expand(context.size(0), -1, -1))],
            dim=1,
        )
        tokens = self.predictor["stack"](tokens)
        return self.predictor["out_proj"](tokens[:, -target_idx.numel() :])

    def forward(self, tokens: torch.Tensor, target_len: int = 4) -> torch.Tensor:
        """tokens: [B, N, C]. Returns a scalar Smooth-L1 loss in embedding space."""
        length = tokens.size(1)
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        context_idx, target_idx = self.sample_mask(length, target_len, tokens.device)
        pos = self.positions(torch.arange(length, device=tokens.device)).to(dtype=tokens.dtype)

        context = self.context_encoder(self.input_proj(tokens[:, context_idx]) + pos[context_idx])

        with torch.no_grad():
            target = self.target_encoder[1](self.target_encoder[0](tokens) + pos)
            target = F.layer_norm(target, (target.size(-1),))
            target = target[:, target_idx]

        prediction = self.predict(context, target_idx)
        return F.smooth_l1_loss(prediction, target)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        student = nn.Sequential(self.input_proj, self.context_encoder)
        ema_update(self.target_encoder, student, momentum)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
