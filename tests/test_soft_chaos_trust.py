"""Soft chaos gate + trust-damped Adam update behaviour."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic._chaos import soft_chaos_signal, trust_from_soft_chaos
from psilogic.debug import chaos_stats


def test_soft_chaos_rises_with_fast_slow_excess() -> None:
    slow = torch.tensor([1.0])
    fast_calm = torch.tensor([1.0])
    fast_hot = torch.tensor([3.0])
    disagree = torch.tensor([0.0])
    calm = soft_chaos_signal(
        slow,
        fast_calm,
        disagree,
        adaptive_tau=True,
        chaos_tau=0.4,
        tau_scale=2.0,
        eps=1e-8,
    )
    hot = soft_chaos_signal(
        slow,
        fast_hot,
        disagree,
        adaptive_tau=True,
        chaos_tau=0.4,
        tau_scale=2.0,
        eps=1e-8,
    )
    assert float(hot) > float(calm)
    assert float(hot) > 0.2


def test_trust_less_than_one_when_chaos_high() -> None:
    soft = torch.tensor([0.8])
    trust = trust_from_soft_chaos(
        soft, gamma_eff=0.05, p_ext=1.0, chaos_gain=1.0, max_cancel=0.05
    )
    assert float(trust) < 1.0
    assert float(trust) == pytest.approx(0.96, abs=1e-5)


def test_trust_damping_shrinks_update_vs_gamma_zero() -> None:
    """With forced high soft_chaos, gamma>0 must move less than gamma=0 twin."""
    torch.manual_seed(0)
    model_a = nn.Linear(8, 4, bias=False)
    model_b = copy.deepcopy(model_a)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    crit = nn.MSELoss()

    # Short warmup so chaos is active immediately; high gamma / max_cancel.
    opt_damped = PsiLogic(
        model_a.parameters(),
        lr=1e-2,
        weight_decay=0.0,
        gamma=0.5,
        max_cancel=0.5,
        chaos_warmup=0,
        agc_clip=0.0,
        grad_centralize=False,
        use_foreach=False,
        use_fused_cuda=False,
    )
    opt_plain = PsiLogic(
        model_b.parameters(),
        lr=1e-2,
        weight_decay=0.0,
        gamma=0.0,
        chaos_warmup=0,
        agc_clip=0.0,
        grad_centralize=False,
        use_foreach=False,
        use_fused_cuda=False,
    )

    # Warm up momentums with identical grads so disagree can be nonzero.
    for _ in range(3):
        for opt, model in ((opt_damped, model_a), (opt_plain, model_b)):
            opt.zero_grad()
            crit(model(x), y).backward()
        # Share grads so paths differ only by trust.
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
            p_b.grad = p_a.grad.detach().clone()
        opt_damped.step()
        opt_plain.step()

    # Flip the target so grads fight existing momentum → high disagree.
    y_flip = -y
    before_a = model_a.weight.detach().clone()
    before_b = model_b.weight.detach().clone()
    opt_damped.zero_grad()
    opt_plain.zero_grad()
    crit(model_a(x), y_flip).backward()
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        p_b.grad = p_a.grad.detach().clone()
    opt_damped.step()
    opt_plain.step()

    delta_damped = (model_a.weight.detach() - before_a).norm().item()
    delta_plain = (model_b.weight.detach() - before_b).norm().item()
    assert delta_damped < delta_plain

    stats = chaos_stats(opt_damped)
    assert stats[0]["soft_chaos_mean"] > 0.0


def test_gamma_zero_still_matches_adamw_path() -> None:
    """Smoke: gamma=0 leaves soft_chaos recorded as zero after a step."""
    model = nn.Linear(4, 2)
    opt = PsiLogic(
        model.parameters(),
        lr=1e-3,
        gamma=0.0,
        chaos_warmup=0,
        use_foreach=False,
        use_fused_cuda=False,
    )
    x = torch.randn(3, 4)
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()
    assert chaos_stats(opt)[0]["soft_chaos_mean"] == 0.0
