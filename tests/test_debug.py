"""psilogic.debug diagnostics tests."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic, get_chaos_metrics
from psilogic.debug import chaos_stats, layer_norms, norm_history


def _train(model: nn.Module, opt: PsiLogic, n: int = 5) -> None:
    crit = nn.CrossEntropyLoss()
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    for _ in range(n):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()


class TestChaosStats:
    def test_structure_and_ranges(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        opt = PsiLogic(model.parameters(), lr=1e-2, chaos_warmup=0)
        _train(model, opt)

        stats = chaos_stats(opt)
        assert len(stats) == len(opt.param_groups)
        entry = stats[0]
        for key in (
            "group",
            "n_params",
            "step",
            "fast_mean",
            "slow_mean",
            "ratio_mean",
            "soft_chaos_mean",
            "spike_rate",
        ):
            assert key in entry
        assert entry["step"] == 5
        assert entry["n_params"] == sum(1 for _ in model.parameters())
        assert 0.0 <= entry["spike_rate"] <= 1.0
        assert entry["soft_chaos_mean"] >= 0.0
        assert math.isfinite(entry["fast_mean"]) and entry["fast_mean"] > 0

    def test_empty_before_first_step(self):
        model = nn.Linear(4, 2)
        opt = PsiLogic(model.parameters(), lr=1e-3)
        stats = chaos_stats(opt)
        assert stats[0]["n_params"] == 0
        assert stats[0]["step"] == 0

    def test_rejects_foreign_optimizer(self):
        opt = torch.optim.AdamW(nn.Linear(4, 2).parameters())
        with pytest.raises(TypeError):
            chaos_stats(opt)


class TestGetChaosMetrics:
    def test_per_param_metrics(self):
        torch.manual_seed(0)
        model = nn.Linear(16, 4)
        opt = PsiLogic(model.parameters(), lr=1e-2)
        _train(model, opt, n=3)

        metrics = get_chaos_metrics(opt.state[model.weight])
        assert metrics["step"] == 3
        assert metrics["fast"] > 0 and metrics["slow"] > 0
        assert metrics["ratio"] == pytest.approx(metrics["fast"] / metrics["slow"], rel=1e-6)

    def test_empty_state_is_safe(self):
        assert get_chaos_metrics({}) == {
            "step": 0.0,
            "fast": 0.0,
            "slow": 0.0,
            "ratio": 0.0,
            "gn_avg": 0.0,
            "soft_chaos": 0.0,
        }


class TestNormHistory:
    def test_records_every_step(self):
        torch.manual_seed(0)
        model = nn.Linear(16, 4)
        opt = PsiLogic(model.parameters(), lr=1e-2)
        tracker = norm_history(opt, model)
        _train(model, opt, n=4)
        tracker.close()

        assert tracker.steps == 4
        assert set(tracker.history) == {name for name, _ in model.named_parameters()}
        for series in tracker.history.values():
            assert len(series) == 4
            assert all(math.isfinite(v) for v in series)

    def test_close_stops_recording(self):
        torch.manual_seed(0)
        model = nn.Linear(16, 4)
        opt = PsiLogic(model.parameters(), lr=1e-2)
        tracker = norm_history(opt, model)
        _train(model, opt, n=2)
        tracker.close()
        _train(model, opt, n=2)
        assert tracker.steps == 2

    def test_context_manager(self):
        torch.manual_seed(0)
        model = nn.Linear(16, 4)
        opt = PsiLogic(model.parameters(), lr=1e-2)
        with norm_history(opt, model) as tracker:
            _train(model, opt, n=3)
        assert tracker.steps == 3
        assert opt.step.__func__ is type(opt).step  # original restored


def test_layer_norms_matches_manual():
    model = nn.Linear(4, 2)
    norms = layer_norms(model)
    assert norms["weight"] == pytest.approx(float(model.weight.norm()))
    assert norms["bias"] == pytest.approx(float(model.bias.norm()))
