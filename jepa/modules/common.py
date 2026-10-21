"""Shared transformer pieces for the JEPA tutorials.

Official I-JEPA / Vol-JEPA stacks use a ViT. These blocks keep that
structure and drop FlashAttention, packed sequences, and stochastic depth
so the forward pass can be read next to the papers.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention block. Input and output are [B, N, D]."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def sinusoidal_positions(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of integer coordinates.

    coords: [..., axes]
    returns: [..., dim] with dim divisible by 2 * axes.
    """
    if coords.ndim < 2:
        raise ValueError("coords must end with an axis dimension")
    n_axes = coords.shape[-1]
    if dim % (2 * n_axes) != 0:
        raise ValueError(f"dim {dim} must be divisible by {2 * n_axes}")
    half = dim // (2 * n_axes)
    freq = torch.arange(half, device=coords.device, dtype=torch.float32)
    freq = 1.0 / (10000 ** (freq / max(half, 1)))
    parts = []
    for axis in range(n_axes):
        angles = coords[..., axis].float().unsqueeze(-1) * freq
        parts.append(torch.sin(angles))
        parts.append(torch.cos(angles))
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """teacher <- m * teacher + (1 - m) * student."""
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def clone_as_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    return freeze_module(teacher)
