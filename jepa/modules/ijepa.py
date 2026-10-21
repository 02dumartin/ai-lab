"""Image JEPA (Assran et al., 2023), scaled down for a notebook.

Differences from the abstract JEPA in jepa.py:
- tokens are image patches
- targets are several spatial blocks, not one span
- the context block is large, then target patches are removed from it
- the target encoder runs on the whole image
- the loss is mean squared error, averaged over target blocks
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import TransformerStack, clone_as_teacher, ema_update, init_weights, sinusoidal_positions


def sample_block(grid: int, scale: tuple[float, float], aspect: tuple[float, float]) -> torch.Tensor:
    """Boolean [grid, grid] mask of one rectangle. True means inside the block."""
    area = grid * grid
    for _ in range(30):
        target_area = area * torch.empty(1).uniform_(*scale).item()
        ratio = torch.empty(1).uniform_(*aspect).item()
        height = int(round((target_area * ratio) ** 0.5))
        width = int(round((target_area / ratio) ** 0.5))
        if height < 1 or width < 1 or height > grid or width > grid:
            continue
        top = torch.randint(0, grid - height + 1, ()).item()
        left = torch.randint(0, grid - width + 1, ()).item()
        mask = torch.zeros(grid, grid, dtype=torch.bool)
        mask[top : top + height, left : left + width] = True
        return mask
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    mask[: max(1, grid // 4), : max(1, grid // 4)] = True
    return mask


def sample_ijepa_masks(
    grid: int = 8,
    num_targets: int = 4,
    pred_scale: tuple[float, float] = (0.15, 0.2),
    pred_aspect: tuple[float, float] = (0.75, 1.5),
    ctx_scale: tuple[float, float] = (0.85, 1.0),
    ctx_aspect: tuple[float, float] = (0.75, 1.5),
    min_context_frac: float = 0.15,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Sample I-JEPA multi-block masks.

    Returns context indices [Nc], a list of target-index tensors, and a
    [grid, grid] long map with 0 = dropped, 1 = context, 2+ = target block id.
    """
    targets = []
    union = torch.zeros(grid, grid, dtype=torch.bool)
    for _ in range(num_targets):
        block = sample_block(grid, pred_scale, pred_aspect)
        targets.append(block.flatten().nonzero().flatten())
        union |= block

    context = torch.zeros(grid, grid, dtype=torch.bool)
    min_keep = max(1, int(min_context_frac * grid * grid))
    for _ in range(30):
        block = sample_block(grid, ctx_scale, ctx_aspect)
        candidate = block & ~union
        if int(candidate.sum()) >= min_keep:
            context = candidate
            break
    else:
        context = ~union
        if int(context.sum()) < min_keep:
            context = torch.ones(grid, grid, dtype=torch.bool)
            context[union] = False

    view = torch.zeros(grid, grid, dtype=torch.long)
    view[context] = 1
    for block_id, index in enumerate(targets, start=2):
        view.view(-1)[index] = block_id
    return context.flatten().nonzero().flatten(), targets, view


class IJEPA(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 96,
        depth: int = 2,
        num_heads: int = 4,
        predictor_depth: int = 2,
        num_targets: int = 4,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        patch_dim = in_chans * patch_size * patch_size

        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.context_encoder = TransformerStack(embed_dim, depth, num_heads)
        self.predictor_in = nn.Linear(embed_dim, embed_dim)
        self.predictor = TransformerStack(embed_dim, predictor_depth, num_heads)
        self.predictor_out = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.apply(init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.target_encoder = clone_as_teacher(
            nn.Sequential(self.patch_proj, self.context_encoder)
        )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """images [B, C, H, W] -> patches [B, N, C*P*P] in row-major order."""
        batch, channels, _, _ = images.shape
        grid, patch = self.grid, self.patch_size
        patches = images.view(batch, channels, grid, patch, grid, patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        return patches.view(batch, grid * grid, channels * patch * patch)

    def positions(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(self.grid, device=device),
            torch.arange(self.grid, device=device),
            indexing="ij",
        )
        coords = torch.stack([ys, xs], dim=-1).view(-1, 2)
        return sinusoidal_positions(coords, self.embed_dim).to(dtype=dtype)

    def encode(self, patches: torch.Tensor, pos: torch.Tensor, index: torch.Tensor | None = None) -> torch.Tensor:
        if index is not None:
            patches = patches[:, index]
            pos = pos[index]
        hidden = self.patch_proj(patches) + pos
        return self.context_encoder(hidden)

    def encode_targets(self, patches: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        hidden = self.target_encoder[0](patches) + pos
        hidden = self.target_encoder[1](hidden)
        return F.layer_norm(hidden, (hidden.size(-1),))

    def predict_block(self, context: torch.Tensor, target_idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        mask_tokens = self.mask_token + pos[target_idx].unsqueeze(0)
        mask_tokens = mask_tokens.expand(context.size(0), -1, -1)
        tokens = torch.cat([self.predictor_in(context), self.predictor_in(mask_tokens)], dim=1)
        tokens = self.predictor(tokens)
        return self.predictor_out(tokens[:, -target_idx.numel() :])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(images)
        pos = self.positions(images.device, patches.dtype)
        context_idx, targets, _ = sample_ijepa_masks(self.grid, num_targets=self.num_targets)
        context_idx = context_idx.to(images.device)
        targets = [idx.to(images.device) for idx in targets]

        context = self.encode(patches, pos, context_idx)
        with torch.no_grad():
            target_tokens = self.encode_targets(patches, pos)

        losses = []
        for target_idx in targets:
            prediction = self.predict_block(context, target_idx, pos)
            losses.append(F.mse_loss(prediction, target_tokens[:, target_idx]))
        return torch.stack(losses).mean()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        student = nn.Sequential(self.patch_proj, self.context_encoder)
        ema_update(self.target_encoder, student, momentum)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
