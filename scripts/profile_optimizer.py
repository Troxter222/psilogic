#!/usr/bin/env python3
"""Profile PsiLogic step time vs AdamW on a ViT-like multi-parameter model."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic._cuda import is_fused_available


class TinyViTLike(nn.Module):
    def __init__(self, depth: int = 12, dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU()) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Deep-copy parameters so later optimizer steps do not poison comparisons."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _restore_state(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(snapshot)


def _median_step_ms(
    factory: Callable[[], torch.optim.Optimizer],
    model: nn.Module,
    device: str,
    *,
    init_state: dict[str, torch.Tensor],
    n_warmup: int = 10,
    n_timed: int = 50,
) -> float:
    _restore_state(model, init_state)
    opt = factory()
    crit = nn.CrossEntropyLoss()
    x = torch.randn(32, 128, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    times: list[float] = []
    for i in range(n_warmup + n_timed):
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            times.append(elapsed_ms)
    return statistics.median(times)


def _profile_psilogic(
    model: nn.Module,
    device: str,
    fused: bool,
    *,
    init_state: dict[str, torch.Tensor],
) -> None:
    _restore_state(model, init_state)
    opt = PsiLogic(
        model.parameters(),
        lr=3e-4,
        gamma=0.04,
        chaos_warmup=0,
        use_fused_cuda=fused,
        use_foreach=not fused,
    )
    crit = nn.CrossEntropyLoss()
    x = torch.randn(32, 128, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            *([torch.profiler.ProfilerActivity.CUDA] if device == "cuda" else []),
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()

    label = "fused" if fused else "foreach"
    print(f"\n=== PsiLogic ({label}) kernel breakdown ===")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total", row_limit=15
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PsiLogic optimizer step time")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--profile", action="store_true", help="Run torch.profiler breakdown")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = TinyViTLike(depth=args.depth).to(args.device)
    init_state = _snapshot_state(model)

    adamw_ms = _median_step_ms(
        lambda: torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=args.device == "cuda"),
        model,
        args.device,
        init_state=init_state,
    )
    psi_foreach_ms = _median_step_ms(
        lambda: PsiLogic(
            model.parameters(),
            lr=3e-4,
            gamma=0.04,
            chaos_warmup=0,
            use_fused_cuda=False,
            use_foreach=args.device == "cuda",
        ),
        model,
        args.device,
        init_state=init_state,
    )

    print(f"Device: {args.device}")
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,} ({len(list(model.parameters()))} tensors)"
    )
    print(f"AdamW median step: {adamw_ms:.3f} ms")
    print(
        f"PsiLogic (foreach) median step: {psi_foreach_ms:.3f} ms ({psi_foreach_ms / adamw_ms:.2f}x)"
    )

    if args.device == "cuda" and is_fused_available():
        psi_fused_ms = _median_step_ms(
            lambda: PsiLogic(
                model.parameters(),
                lr=3e-4,
                gamma=0.04,
                chaos_warmup=0,
                use_fused_cuda=True,
            ),
            model,
            args.device,
            init_state=init_state,
        )
        print(
            f"PsiLogic (fused) median step: {psi_fused_ms:.3f} ms ({psi_fused_ms / adamw_ms:.2f}x)"
        )
        if args.profile:
            _profile_psilogic(model, args.device, fused=True, init_state=init_state)
    elif args.profile:
        _profile_psilogic(model, args.device, fused=False, init_state=init_state)


if __name__ == "__main__":
    main()
