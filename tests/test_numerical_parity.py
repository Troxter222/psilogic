"""Numerical parity: scalar reference vs foreach / fused CUDA backends."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic._cuda import is_fused_available


def _clone_model(model: nn.Module) -> nn.Module:
    clone = copy.deepcopy(model)
    for p in clone.parameters():
        if p.grad is not None:
            p.grad = None
    return clone


def _clone_optimizer(opt: PsiLogic) -> PsiLogic:
    state = copy.deepcopy(opt.state_dict())
    clone = PsiLogic(opt.param_groups[0]["params"], lr=opt.param_groups[0]["lr"])
    clone.load_state_dict(state)
    return clone


def _assert_state_close(
    ref: PsiLogic,
    other: PsiLogic,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> None:
    for group_ref, group_other in zip(ref.param_groups, other.param_groups):
        for p_ref, p_other in zip(group_ref["params"], group_other["params"]):
            assert torch.allclose(p_ref, p_other, rtol=rtol, atol=atol), (
                f"param mismatch max={ (p_ref - p_other).abs().max().item() }"
            )
            s_ref = ref.state[p_ref]
            s_other = other.state[p_other]
            for key in ("m", "v", "fast", "slow", "gn_avg"):
                if key not in s_ref:
                    continue
                assert torch.allclose(s_ref[key], s_other[key], rtol=rtol, atol=atol), (
                    f"{key} mismatch max={ (s_ref[key] - s_other[key]).abs().max().item() }"
                )
            assert s_ref["t"] == s_other["t"]


def _infer_input(
    model: nn.Module, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a forward-compatible random batch for simple test models."""
    if isinstance(model, nn.Sequential):
        if isinstance(model[0], nn.Conv2d):
            conv = model[0]
            h = w = 8
            x = torch.randn(8, conv.in_channels, h, w, device=device, dtype=dtype)
        elif isinstance(model[0], nn.Linear):
            x = torch.randn(8, model[0].in_features, device=device, dtype=dtype)
        else:
            raise ValueError("Cannot infer input shape for sequential model")
    elif hasattr(model, "layers"):
        in_features = model.layers[0][0].in_features  # type: ignore[index]
        x = torch.randn(8, in_features, device=device, dtype=dtype)
    else:
        raise ValueError("Cannot infer input shape for model")
    with torch.no_grad():
        out = model(x)
    y = torch.randn_like(out)
    return x, y


def _run_parity(
    *,
    model_factory: Callable[[], nn.Module],
    kwargs: dict[str, Any],
    n_steps: int,
    backend: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> None:
    torch.manual_seed(42)
    model_ref = model_factory().to(device=device, dtype=dtype)
    model_other = _clone_model(model_ref)

    opt_ref = PsiLogic(model_ref.parameters(), use_foreach=False, **kwargs)
    if backend == "foreach":
        opt_other = PsiLogic(model_other.parameters(), use_foreach=True, use_fused_cuda=False, **kwargs)
    elif backend == "fused":
        opt_other = PsiLogic(
            model_other.parameters(),
            use_foreach=False,
            use_fused_cuda=True,
            **kwargs,
        )
    else:
        raise ValueError(backend)

    crit = nn.MSELoss()
    x, y = _infer_input(model_ref, device, dtype)

    for step in range(n_steps):
        opt_ref.zero_grad()
        opt_other.zero_grad()
        loss_ref = crit(model_ref(x), y)
        loss_other = crit(model_other(x), y)
        loss_ref.backward()
        loss_other.backward()
        opt_ref.step()
        opt_other.step()

    rtol = 1e-5 if dtype == torch.bfloat16 else 1e-6
    atol = 1e-6 if dtype == torch.bfloat16 else 1e-7
    _assert_state_close(opt_ref, opt_other, rtol=rtol, atol=atol)


def _mixed_shape_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 1e-3, "gamma": 0.05, "chaos_warmup": 0},
        {"lr": 1e-3, "gamma": 0.0, "chaos_warmup": 0},
        {"lr": 1e-3, "gamma": 0.03, "quantum_decay": 2e-4, "chaos_warmup": 0},
        {"lr": 1e-3, "gamma": 0.04, "agc_clip": 0.02, "chaos_warmup": 0},
        {"lr": 1e-3, "gamma": 0.04, "grad_centralize": False, "chaos_warmup": 0},
        {"lr": 1e-3, "gamma": 0.03, "lion_mode": True, "chaos_warmup": 0},
        {"lr": 1e-3, "gamma": 0.02, "chaos_warmup": 50, "gamma_T_max": 200},
        {"lr": 1e-3, "gamma": 0.03, "adaptive_tau": False, "chaos_tau": 0.4, "chaos_warmup": 0},
    ],
    ids=[
        "default",
        "gamma_zero",
        "quantum_decay",
        "agc",
        "no_gc",
        "lion",
        "warmup",
        "absolute_tau",
    ],
)
def test_foreach_matches_scalar_cpu(kwargs: dict[str, Any]) -> None:
    def factory() -> nn.Sequential:
        torch.manual_seed(7)
        return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))

    _run_parity(model_factory=factory, kwargs=kwargs, n_steps=60, backend="foreach")


def test_foreach_matches_scalar_mixed_shapes_cpu() -> None:
    _run_parity(
        model_factory=_mixed_shape_model,
        kwargs={"lr": 1e-3, "gamma": 0.04, "quantum_decay": 1e-4, "chaos_warmup": 0},
        n_steps=40,
        backend="foreach",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_foreach_matches_scalar_cuda(dtype: torch.dtype) -> None:
    def factory() -> nn.Sequential:
        torch.manual_seed(11)
        return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16))

    _run_parity(
        model_factory=factory,
        kwargs={"lr": 1e-3, "gamma": 0.05, "quantum_decay": 2e-4, "chaos_warmup": 0},
        n_steps=80,
        backend="foreach",
        device="cuda",
        dtype=dtype,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not is_fused_available(), reason="Triton fused CUDA path unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_matches_scalar_cuda(dtype: torch.dtype) -> None:
    def factory() -> nn.Sequential:
        torch.manual_seed(13)
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    _run_parity(
        model_factory=factory,
        kwargs={
            "lr": 1e-3,
            "gamma": 0.04,
            "quantum_decay": 2e-4,
            "agc_clip": 0.02,
            "chaos_warmup": 0,
        },
        n_steps=100,
        backend="fused",
        device="cuda",
        dtype=dtype,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not is_fused_available(), reason="Triton fused CUDA path unavailable")
def test_fused_matches_scalar_vit_like_cuda() -> None:
    """Many-parameter model similar to ViT overhead profile."""

    class TinyViTLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU()) for _ in range(12)]
            )
            self.head = nn.Linear(64, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = x + layer(x)
            return self.head(x)

    _run_parity(
        model_factory=TinyViTLike,
        kwargs={"lr": 3e-4, "gamma": 0.04, "chaos_warmup": 0, "agc_clip": 0.02},
        n_steps=50,
        backend="fused",
        device="cuda",
    )
