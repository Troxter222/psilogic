"""Tiny reference architectures used by preset / auto-config tests."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyViT(nn.Module):
    """Minimal ViT with the canonical parameter naming conventions.

    8x8 input, 4x4 patches -> 4 patch tokens + 1 cls token.
    """

    def __init__(self, dim: int = 16, num_classes: int = 10) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, dim))
        self.attn_qkv = nn.Linear(dim, dim * 3)
        self.attn_out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp_fc1 = nn.Linear(dim, dim * 2)
        self.mlp_fc2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1)
        tokens = tokens + self.pos_embed

        q, k, v = self.attn_qkv(tokens).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / (self._dim**0.5)
        attended = torch.softmax(scores, dim=-1) @ v
        tokens = self.norm1(tokens + self.attn_out_proj(attended))
        tokens = self.norm2(tokens + self.mlp_fc2(F.relu(self.mlp_fc1(tokens))))
        return self.head(tokens[:, 0])


class _ToyGPTBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.c_attn = nn.Linear(dim, dim * 3)
        self.c_proj = nn.Linear(dim, dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp_c_fc = nn.Linear(dim, dim * 4)
        self.mlp_c_proj = nn.Linear(dim * 4, dim)
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        q, k, v = self.c_attn(h).chunk(3, dim=-1)
        t = x.shape[1]
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = q @ k.transpose(-2, -1) / (self._dim**0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        x = x + self.c_proj(torch.softmax(scores, dim=-1) @ v)
        h = self.ln_2(x)
        return x + self.mlp_c_proj(F.gelu(self.mlp_c_fc(h)))


class ToyGPT(nn.Module):
    """Minimal GPT with wte/wpe/lm_head naming; optional weight tying."""

    def __init__(
        self,
        vocab_size: int = 64,
        block_size: int = 16,
        dim: int = 16,
        tied: bool = True,
    ) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(block_size, dim)
        self.h = nn.ModuleList([_ToyGPTBlock(dim)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tied:
            self.lm_head.weight = self.wte.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))


class ToyCNN(nn.Module):
    """Minimal CNN classifier (8x8 RGB input)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(16 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))
