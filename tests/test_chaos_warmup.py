"""Auto-scaling chaos warmup and warm-in ramp tests."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic._chaos import effective_warmup, resolve_warmup


class TestResolveWarmup:
    def test_explicit_warmup_honoured(self):
        assert resolve_warmup(7, 100_000) == 7
        assert resolve_warmup(0, 100_000) == 0

    def test_auto_scales_with_total_steps(self):
        assert resolve_warmup(-1, 100_000) == 5_000
        assert resolve_warmup(-1, 40_000) == 2_000

    def test_auto_floor_at_500(self):
        assert resolve_warmup(-1, 1_000) == 500
        assert resolve_warmup(-1, 50) == 500

    def test_auto_fallback_when_horizon_unknown(self):
        assert resolve_warmup(-1, 0) == 200


class TestEffectiveWarmup:
    def test_zero_chaos_at_step_zero(self):
        assert effective_warmup(0, 0, 0) == 0.0
        assert effective_warmup(0, 10_000, -1) == 0.0

    def test_zero_chaos_during_warmup(self):
        # auto warmup for 10k steps is max(500, 500) = 500
        for step in (1, 100, 499, 500):
            assert effective_warmup(step, 10_000, -1) == 0.0

    def test_full_chaos_after_ramp(self):
        # warmup 500, ramp 125 -> full chaos at step 625
        assert effective_warmup(625, 10_000, -1) == 1.0
        assert effective_warmup(10_000, 10_000, -1) == 1.0

    def test_ramp_is_monotonic_and_bounded(self):
        gains = [effective_warmup(step, 10_000, -1) for step in range(490, 700)]
        assert all(0.0 <= g <= 1.0 for g in gains)
        assert all(b >= a for a, b in zip(gains, gains[1:]))
        assert 0.0 < effective_warmup(560, 10_000, -1) < 1.0

    def test_warmup_zero_gives_full_chaos_immediately(self):
        assert effective_warmup(1, 0, 0) == 1.0


class TestWarmupIntegration:
    def test_matches_chaos_free_optimizer_during_warmup(self):
        """During auto warmup PsiLogic must be bit-identical to its gamma=0 self."""
        torch.manual_seed(42)
        model_chaos = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 2))
        model_plain = copy.deepcopy(model_chaos)
        x = torch.randn(4, 8)
        y = torch.randn(4, 2)
        crit = nn.MSELoss()

        # auto warmup = max(500, 10000 // 20) = 500 >> 20 steps below
        opt_chaos = PsiLogic(
            model_chaos.parameters(), lr=1e-2, gamma=0.05, chaos_warmup=-1, gamma_T_max=10_000
        )
        opt_plain = PsiLogic(
            model_plain.parameters(), lr=1e-2, gamma=0.0, chaos_warmup=-1, gamma_T_max=10_000
        )

        for _ in range(20):
            for model, opt in ((model_chaos, opt_chaos), (model_plain, opt_plain)):
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()

        for p_chaos, p_plain in zip(model_chaos.parameters(), model_plain.parameters()):
            assert torch.allclose(p_chaos, p_plain, atol=1e-7), (
                "Chaos leaked into updates during the warmup window"
            )

    def test_converges_with_auto_warmup(self):
        torch.manual_seed(0)
        model = nn.Linear(16, 4)
        opt = PsiLogic(model.parameters(), lr=1e-2, chaos_warmup=-1, gamma_T_max=1_000)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))

        with torch.no_grad():
            initial = crit(model(x), y).item()
        for _ in range(10):
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        assert crit(model(x), y).item() < initial
