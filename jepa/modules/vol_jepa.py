"""Vol-JEPA, the encoder pretraining used by NeuroVFM.

This follows the public NeuroVFM training loop, scaled to a notebook:

- a volume is cut into anisotropic patches (official CT/MRI tokens are 4x16x16)
- positional encoding is concatenated, not added (738 content + 30 PE = 768)
- the student encodes a small context (~20-25% for CT/MRI)
- the teacher encodes the whole volume; its outputs are layer-normalized
- a predictor fills learnable mask tokens at the target coordinates
- the loss is Smooth L1 between predictor outputs and teacher targets
- the teacher is an EMA of the student encoder, not of the predictor

Packed sequences and FlashAttention from the official repo are replaced
by a dense [B, N, D] batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import TransformerStack, clone_as_teacher, ema_update, init_weights, sinusoidal_positions


def sample_vol_masks(
    grid: tuple[int, int, int],
    enc_scale: float = 0.25,
    hw_scale: tuple[float, float] = (0.6, 0.8),
    aspect: tuple[float, float] = (0.3, 3.0),
    npred: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3D multi-block masks.

    A target cuboid covers the full depth and a large in-plane block, matching
    d_pred_mask_scale=(1, 1) in the NeuroVFM collator. If the remaining context
    is still larger than enc_scale, a contiguous depth slab is dropped.

    Returns context indices, target indices, and a [D, H, W] long view
    (1 = context, 2 = target).
    """
    depth, height, width = grid
    context = torch.ones(depth, height, width, dtype=torch.bool)
    for _ in range(npred):
        scale = torch.empty(1).uniform_(*hw_scale).item()
        keep = max(1, int(round(height * width * scale)))
        ratio = torch.empty(1).uniform_(*aspect).item()
        block_h = int(round((keep * ratio) ** 0.5))
        block_w = int(round((keep / ratio) ** 0.5))
        block_h = max(1, min(height, block_h))
        block_w = max(1, min(width, block_w))
        top = torch.randint(0, height - block_h + 1, ()).item()
        left = torch.randint(0, width - block_w + 1, ()).item()
        context[:, top : top + block_h, left : left + block_w] = False

    total = depth * height * width
    if float(context.sum()) / total > enc_scale and depth > 1:
        drop_frac = 1.0 - (enc_scale * total) / max(float(context.sum()), 1.0)
        n_drop = min(depth - 1, max(1, int(round(drop_frac * depth))))
        start = torch.randint(0, depth - n_drop + 1, ()).item()
        context[start : start + n_drop] = False

    flat = context.view(-1)
    if int(flat.sum()) == 0:
        flat[0] = True
    if int((~flat).sum()) == 0:
        flat[-1] = False

    view = torch.ones(depth, height, width, dtype=torch.long)
    view[~context] = 2
    return flat.nonzero().flatten(), (~flat).nonzero().flatten(), view


class VolumeViT(nn.Module):
    """3D ViT over pre-tokenized voxels.

    tokens: [B, N, token_dim]
    coords: [N, 3] integer patch coordinates (d, h, w), shared by the batch
    """

    def __init__(
        self,
        token_dim: int,
        content_dim: int,
        pe_dim: int,
        depth: int,
        num_heads: int,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.pe_dim = pe_dim
        self.embed_dim = content_dim + pe_dim
        self.token_proj = nn.Linear(token_dim, content_dim)
        self.stack = TransformerStack(self.embed_dim, depth, num_heads)

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        content = self.token_proj(tokens)
        pe = sinusoidal_positions(coords, self.pe_dim).to(dtype=content.dtype)
        hidden = torch.cat([content, pe.unsqueeze(0).expand(tokens.size(0), -1, -1)], dim=-1)
        return self.stack(hidden)


class VolJEPA(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (2, 4, 4),
        content_dim: int = 72,
        pe_dim: int = 24,
        depth: int = 2,
        num_heads: int = 4,
        predictor_depth: int = 2,
        enc_scale: float = 0.25,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pe_dim = pe_dim
        self.enc_scale = enc_scale
        token_dim = patch_size[0] * patch_size[1] * patch_size[2]
        embed_dim = content_dim + pe_dim
        self.embed_dim = embed_dim

        self.student = VolumeViT(token_dim, content_dim, pe_dim, depth, num_heads)
        # Official predictor concatenates PE onto the encoder output, then
        # projects back to the encoder width. Mask tokens live in that width.
        self.predictor_in = nn.Linear(embed_dim + pe_dim, embed_dim)
        self.predictor = TransformerStack(embed_dim, predictor_depth, num_heads)
        self.predictor_out = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.apply(init_weights)
        nn.init.zeros_(self.mask_token)
        self.teacher = clone_as_teacher(self.student)

    def tokenize(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """volume [B, 1, D, H, W] -> tokens [B, N, P], coords [N, 3]."""
        if volume.ndim != 5 or volume.size(1) != 1:
            raise ValueError("expected volume [B, 1, D, H, W]")
        pd, ph, pw = self.patch_size
        batch, _, depth, height, width = volume.shape
        if depth % pd or height % ph or width % pw:
            raise ValueError(f"volume {(depth, height, width)} is not divisible by patch {self.patch_size}")
        gd, gh, gw = depth // pd, height // ph, width // pw
        tokens = volume.view(batch, 1, gd, pd, gh, ph, gw, pw)
        tokens = tokens.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        tokens = tokens.view(batch, gd * gh * gw, pd * ph * pw)
        zs, ys, xs = torch.meshgrid(
            torch.arange(gd, device=volume.device),
            torch.arange(gh, device=volume.device),
            torch.arange(gw, device=volume.device),
            indexing="ij",
        )
        coords = torch.stack([zs, ys, xs], dim=-1).view(-1, 3)
        return tokens, coords

    def _with_predictor_pe(self, hidden: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        pe = sinusoidal_positions(coords, self.pe_dim).to(dtype=hidden.dtype)
        if hidden.ndim == 2:
            return torch.cat([hidden, pe], dim=-1)
        return torch.cat([hidden, pe.unsqueeze(0).expand(hidden.size(0), -1, -1)], dim=-1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        tokens, coords = self.tokenize(volume)
        gd = int(coords[:, 0].max().item()) + 1
        gh = int(coords[:, 1].max().item()) + 1
        gw = int(coords[:, 2].max().item()) + 1
        context_idx, target_idx, _ = sample_vol_masks((gd, gh, gw), enc_scale=self.enc_scale)
        context_idx = context_idx.to(volume.device)
        target_idx = target_idx.to(volume.device)

        context = self.student(tokens[:, context_idx], coords[context_idx])
        with torch.no_grad():
            target = self.teacher(tokens, coords)
            target = F.layer_norm(target, (target.size(-1),))
            target = target[:, target_idx]

        context_pe = self._with_predictor_pe(context, coords[context_idx])
        mask = self.mask_token.expand(target_idx.numel(), -1)
        mask = mask.unsqueeze(0).expand(volume.size(0), -1, -1)
        mask_pe = self._with_predictor_pe(mask, coords[target_idx])
        pred_tokens = torch.cat([self.predictor_in(context_pe), self.predictor_in(mask_pe)], dim=1)
        pred_tokens = self.predictor(pred_tokens)
        prediction = self.predictor_out(pred_tokens[:, -target_idx.numel() :])
        return F.smooth_l1_loss(prediction, target)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float = 0.994) -> None:
        ema_update(self.teacher, self.student, momentum)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
