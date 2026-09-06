"""Bit-exact equivalence: PsiLogic(gamma=0) vs torch.optim.AdamW on scalar CPU path."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic

_ADAMW_BASE_KWARGS = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
}

_PSI_ADAMW_KWARGS = {
    **_ADAMW_BASE_KWARGS,
    "gamma": 0.0,
    "quantum_decay": 0.0,
    "grad_centralize": False,
    "agc_clip": 0.0,
    "chaos_warmup": 0,
    "use_foreach": False,
    "use_fused_cuda": False,
}


def _make_model() -> nn.Sequential:
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))


def _fixed_batch(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    x = torch.randn(8, 16)
    with torch.no_grad():
        y = torch.randn_like(model(x))
    return x, y


def _run_mirror_steps(
    *,
    weight_decay: float,
    n_steps: int = 40,
) -> tuple[nn.Sequential, nn.Sequential, PsiLogic, torch.optim.AdamW]:
    model_psi = _make_model()
    model_adam = copy.deepcopy(model_psi)
    x, y = _fixed_batch(model_psi)
    crit = nn.MSELoss()

    opt_psi = PsiLogic(model_psi.parameters(), weight_decay=weight_decay, **_PSI_ADAMW_KWARGS)
    opt_adam = torch.optim.AdamW(
        model_adam.parameters(),
        weight_decay=weight_decay,
        foreach=False,
        **_ADAMW_BASE_KWARGS,
    )

    for _ in range(n_steps):
        opt_psi.zero_grad()
        opt_adam.zero_grad()
        crit(model_psi(x), y).backward()
        for p_psi, p_adam in zip(model_psi.parameters(), model_adam.parameters(), strict=True):
            p_adam.grad = p_psi.grad.detach().clone()
        opt_psi.step()
        opt_adam.step()

    return model_psi, model_adam, opt_psi, opt_adam


@pytest.mark.parametrize("weight_decay", [0.0, 1e-4])
def test_gamma_zero_matches_adamw(weight_decay: float) -> None:
    """With all PsiLogic extras disabled, scalar CPU path must match AdamW."""
    model_psi, model_adam, opt_psi, opt_adam = _run_mirror_steps(weight_decay=weight_decay)

    for p_psi, p_adam in zip(model_psi.parameters(), model_adam.parameters(), strict=True):
        assert torch.allclose(p_psi, p_adam, rtol=0.0, atol=5e-5), (
            f"param mismatch max={(p_psi - p_adam).abs().max().item()}"
        )

    for group_psi, group_adam in zip(opt_psi.param_groups, opt_adam.param_groups, strict=True):
        for p_psi, p_adam in zip(group_psi["params"], group_adam["params"], strict=True):
            s_psi = opt_psi.state[p_psi]
            s_adam = opt_adam.state[p_adam]
            assert torch.allclose(s_psi["m"], s_adam["exp_avg"], rtol=1e-6, atol=1e-7), (
                f"m mismatch max={(s_psi['m'] - s_adam['exp_avg']).abs().max().item()}"
            )
            assert torch.allclose(s_psi["v"], s_adam["exp_avg_sq"], rtol=1e-6, atol=1e-7), (
                f"v mismatch max={(s_psi['v'] - s_adam['exp_avg_sq']).abs().max().item()}"
            )
