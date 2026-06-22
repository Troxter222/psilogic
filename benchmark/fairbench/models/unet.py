"""A compact UNet and DDPM wrapper for unconditional 64x64 image generation.

The UNet follows the standard DDPM design (Ho et al., 2020): sinusoidal
timestep embeddings, residual blocks with GroupNorm + SiLU, a self-attention
block at the bottleneck, and skip connections between the down/up paths. The
:class:`DDPM` wrapper holds the linear noise schedule and exposes a single
``loss`` method (epsilon-prediction MSE) plus an ancestral ``sample`` loop.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps (Transformer-style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention over a feature map."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        attn = torch.softmax((q @ k) / math.sqrt(C), dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):
    """Compact UNet predicting noise epsilon for DDPM.

    The down and up paths are built as symmetric *stages* with an explicit
    skip-channel ledger, so every skip pushed on the way down is consumed
    exactly once on the way up at the matching spatial resolution. This
    structural symmetry removes the classic UNet shape-mismatch bug.

    Args:
        in_ch: Image channels (3 for RGB).
        base_ch: Base channel width.
        ch_mults: Channel multipliers per resolution level.
        attn_at: Resolution-level indices that get a self-attention block.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 64,
        ch_mults: list[int] = (1, 2, 2),
        attn_at: list[int] = (1,),
    ):
        super().__init__()
        t_dim = base_ch * 4
        self.t_mlp = nn.Sequential(nn.Linear(base_ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.base_ch = base_ch
        n_levels = len(ch_mults)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # --- Down path: one stage per level (ResBlock [+ Attn] [+ Downsample]).
        # ``skip_chs`` tracks the channel count of every tensor we stash for the
        # up path: the in_conv output plus one per level (the post-attn output).
        self.down_stages = nn.ModuleList()
        skip_chs: list[int] = [base_ch]
        cur = base_ch
        for level, mult in enumerate(ch_mults):
            out = base_ch * mult
            is_last = level == n_levels - 1
            stage = nn.ModuleDict(
                {
                    "res": ResBlock(cur, out, t_dim),
                    "attn": AttnBlock(out) if level in attn_at else nn.Identity(),
                    "down": nn.Identity() if is_last else Down(out),
                }
            )
            self.down_stages.append(stage)
            cur = out
            skip_chs.append(cur)

        # --- Bottleneck.
        self.mid1 = ResBlock(cur, cur, t_dim)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, t_dim)

        # --- Up path: one stage per stashed skip (LIFO). Each stage concatenates
        # the popped skip, applies a ResBlock [+ Attn], then upsamples unless it
        # is already at full resolution.
        self.up_stages = nn.ModuleList()
        # Output channels per up-stage: levels high->low, then a final base stage.
        up_outs = [base_ch * m for m in reversed(ch_mults)] + [base_ch]
        # Which up-stages upsample: the first (n_levels - 1) move to a higher res.
        for j, out in enumerate(up_outs):
            skip_ch = skip_chs.pop()
            level_for_attn = (n_levels - 1 - j) if j < n_levels else -1
            stage = nn.ModuleDict(
                {
                    "res": ResBlock(cur + skip_ch, out, t_dim),
                    "attn": AttnBlock(out) if level_for_attn in attn_at else nn.Identity(),
                    "up": Up(out) if j < n_levels - 1 else nn.Identity(),
                }
            )
            self.up_stages.append(stage)
            cur = out

        self.out_norm = nn.GroupNorm(min(8, cur), cur)
        self.out_conv = nn.Conv2d(cur, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.t_mlp(timestep_embedding(t, self.base_ch))
        h = self.in_conv(x)
        skips = [h]
        for stage in self.down_stages:
            h = stage["res"](h, temb)
            h = stage["attn"](h)
            skips.append(h)
            h = stage["down"](h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for stage in self.up_stages:
            h = torch.cat([h, skips.pop()], dim=1)
            h = stage["res"](h, temb)
            h = stage["attn"](h)
            h = stage["up"](h)

        return self.out_conv(F.silu(self.out_norm(h)))


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model wrapper around a UNet.

    Holds a linear beta schedule and the derived alpha-bar coefficients, and
    exposes the standard epsilon-prediction training objective.
    """

    def __init__(
        self, model: UNet, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: sample x_t ~ q(x_t | x_0)."""
        sab = self.sqrt_alpha_bars[t][:, None, None, None]
        somab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sab * x0 + somab * noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Epsilon-prediction MSE objective over uniformly sampled timesteps."""
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n: int, img_ch: int, img_size: int, device: torch.device) -> torch.Tensor:
        """Ancestral sampling from pure noise to an image batch in [-1, 1]."""
        x = torch.randn(n, img_ch, img_size, img_size, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            eps = self.model(x, t)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
            mean = (1.0 / torch.sqrt(alpha)) * (x - coef * eps)
            if step > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1.0, 1.0)
