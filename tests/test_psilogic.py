"""
PsiLogic v6 — Comprehensive Test Suite
=======================================

Tests cover:
  - Basic optimizer correctness (loss decreases)
  - All four classes: PsiLogic, PsiLogicNLP, PsiLogicGPT, PsiLogicViT
  - All preset helper functions
  - Edge cases: zero grad, single param, large batch
  - State serialization: save/load checkpoint
  - Hyperparameter validation (invalid inputs must raise)
  - Lion mode
  - nlp_param_groups helper
  - Chaos gating modes: adaptive_tau and absolute
  - gamma_T_max cosine decay
  - chaos_warmup behaviour
"""

import copy
import io

import pytest
import torch
import torch.nn as nn

from psilogic import (
    PsiLogic,
    PsiLogicGPT,
    PsiLogicNLP,
    PsiLogicViT,
    gpt_scratch_defaults,
    nlp_defaults,
    nlp_param_groups,
    vision_defaults,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_model():
    torch.manual_seed(0)
    return nn.Linear(16, 4)


@pytest.fixture
def simple_data():
    torch.manual_seed(1)
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    return x, y


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


def _run_steps(model, optimizer, x, y, criterion, n=10):
    """Helper: run N gradient steps and return (initial_loss, final_loss)."""
    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()
    for _ in range(n):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final_loss = criterion(model(x), y).item()
    return initial_loss, final_loss


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergence:
    def test_loss_decreases(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init, f"Loss did not decrease: {init:.4f} → {final:.4f}"

    def test_loss_decreases_nlp(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogicNLP(simple_model.parameters(), lr=1e-2)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init

    def test_loss_decreases_gpt(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogicGPT(simple_model.parameters(), lr=1e-2)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init

    def test_loss_decreases_vit(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogicViT(simple_model.parameters(), lr=1e-2)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init

    def test_loss_decreases_no_weight_decay(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2, weight_decay=0.0)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init

    def test_loss_decreases_gamma_zero(self, simple_model, simple_data, criterion):
        """With gamma=0 PsiLogic degrades to AdamW — must still converge."""
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2, gamma=0.0)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parameter groups / preset helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPresets:
    def test_nlp_defaults_keys(self):
        d = nlp_defaults(total_steps=1000)
        required = {"betas", "weight_decay", "gamma", "adaptive_tau",
                    "tau_scale", "max_cancel", "gamma_T_max", "use_foreach"}
        assert required.issubset(d.keys())
        assert d["gamma_T_max"] == 1000

    def test_vision_defaults_quantum_decay_disabled(self):
        d = vision_defaults()
        assert d["quantum_decay"] == 0.0, "BUG-A fix: QD must be disabled for vision"

    def test_gpt_defaults_quantum_decay_disabled(self):
        d = gpt_scratch_defaults()
        assert d["quantum_decay"] == 0.0

    def test_gpt_defaults_weight_decay(self):
        d = gpt_scratch_defaults()
        assert d["weight_decay"] == 0.1, "GPT scratch should use 0.1 weight decay"

    def test_nlp_param_groups_structure(self):
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
        )
        groups = nlp_param_groups(model, lr=3e-4)
        assert isinstance(groups, list)
        assert all("params" in g for g in groups)
        assert all(len(g["params"]) > 0 for g in groups)

    def test_nlp_param_groups_optimizer_accepts(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        groups = nlp_param_groups(model, lr=3e-4)
        # Should not raise
        opt = PsiLogic(groups, **nlp_defaults(100))
        assert opt is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:
    @pytest.mark.parametrize("bad_kwargs,match", [
        ({"lr": -1e-3},          "lr"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"gamma": -0.1},        "gamma"),
        ({"quantum_decay": -1},  "quantum_decay"),
        ({"agc_clip": -0.1},     "agc_clip"),
        ({"max_cancel": 0.0},    "max_cancel"),
        ({"max_cancel": 1.1},    "max_cancel"),
        ({"betas": (1.0, 0.99)}, "beta1"),
        ({"betas": (0.9, -0.1)}, "beta2"),
    ])
    def test_invalid_hparam_raises(self, bad_kwargs, match):
        model = nn.Linear(4, 2)
        with pytest.raises((AssertionError, ValueError)):
            PsiLogic(model.parameters(), **bad_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chaos gating modes
# ─────────────────────────────────────────────────────────────────────────────

class TestChaosGating:
    def test_adaptive_tau_mode(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(
            simple_model.parameters(),
            lr=1e-2,
            adaptive_tau=True,
            tau_scale=2.0,
            chaos_warmup=0,
        )
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init

    def test_absolute_tau_mode(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(
            simple_model.parameters(),
            lr=1e-2,
            adaptive_tau=False,
            chaos_tau=0.01,   # very low — will trigger often
            chaos_warmup=0,
        )
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init

    def test_chaos_warmup_delays_activation(self):
        """Optimizer with a very long warmup should behave like plain AdamW early on."""
        torch.manual_seed(42)
        model_psi  = nn.Linear(8, 2)
        model_adam = copy.deepcopy(model_psi)
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        crit = nn.CrossEntropyLoss()

        opt_psi  = PsiLogic(model_psi.parameters(),  lr=1e-2, chaos_warmup=10_000)
        opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=1e-2)

        for _ in range(5):
            for m, opt in [(model_psi, opt_psi), (model_adam, opt_adam)]:
                opt.zero_grad()
                crit(m(x), y).backward()
                opt.step()

        # With warmup=10000 and only 5 steps, params should be very close
        for p_psi, p_adam in zip(model_psi.parameters(), model_adam.parameters()):
            assert torch.allclose(p_psi, p_adam, atol=1e-4), \
                "During warmup, PsiLogic should match AdamW closely"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cosine gamma decay
# ─────────────────────────────────────────────────────────────────────────────

class TestGammaDecay:
    def test_gamma_t_max_converges(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(
            simple_model.parameters(),
            lr=1e-2,
            gamma_T_max=50,
        )
        init, final = _run_steps(simple_model, opt, x, y, criterion, n=20)
        assert final < init


# ─────────────────────────────────────────────────────────────────────────────
# 6. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_grad_no_crash(self):
        model = nn.Linear(4, 2)
        opt = PsiLogic(model.parameters(), lr=1e-3)
        # step without any backward — should not raise
        opt.zero_grad()
        opt.step()

    def test_closure_interface(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2)

        def closure():
            opt.zero_grad()
            loss = criterion(simple_model(x), y)
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None
        assert loss.item() > 0

    def test_1d_param_no_grad_centralization_crash(self):
        """Bias terms are 1-D; grad centralization must skip them silently."""
        model = nn.Linear(4, 2)  # has bias (1-D param)
        opt = PsiLogic(model.parameters(), lr=1e-2, grad_centralize=True)
        x = torch.randn(2, 4)
        model(x).sum().backward()
        opt.step()  # must not raise

    def test_single_parameter(self):
        p = nn.Parameter(torch.randn(4))
        opt = PsiLogic([p], lr=1e-2)
        loss = p.pow(2).sum()
        loss.backward()
        opt.step()
        assert p.grad is not None

    def test_multiple_param_groups(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        groups = [
            {"params": list(model[0].parameters()), "lr": 1e-2, "gamma": 0.05},
            {"params": list(model[1].parameters()), "lr": 1e-3, "gamma": 0.01},
        ]
        opt = PsiLogic(groups)
        x = torch.randn(2, 4)
        model(x).sum().backward()
        opt.step()

    def test_no_grad_centralize(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2, grad_centralize=False)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init

    def test_agc_disabled(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2, agc_clip=0.0)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lion mode
# ─────────────────────────────────────────────────────────────────────────────

class TestLionMode:
    def test_lion_mode_converges(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-3, lion_mode=True)
        init, final = _run_steps(simple_model, opt, x, y, criterion)
        assert final < init


# ─────────────────────────────────────────────────────────────────────────────
# 8. State serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_state_dict_round_trip(self, simple_model, simple_data, criterion):
        """Saving and loading state_dict must produce identical results."""
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2)

        # Warm up optimizer state
        for _ in range(3):
            opt.zero_grad()
            criterion(simple_model(x), y).backward()
            opt.step()

        # Save
        buf = io.BytesIO()
        torch.save({
            "model": simple_model.state_dict(),
            "optimizer": opt.state_dict(),
        }, buf)

        # Restore into fresh copies
        model2 = nn.Linear(16, 4)
        opt2   = PsiLogic(model2.parameters(), lr=1e-2)
        buf.seek(0)
        ckpt = torch.load(buf, weights_only=False)
        model2.load_state_dict(ckpt["model"])
        opt2.load_state_dict(ckpt["optimizer"])

        # One more step — outputs must be identical
        opt.zero_grad()
        criterion(simple_model(x), y).backward()
        opt.step()

        opt2.zero_grad()
        criterion(model2(x), y).backward()
        opt2.step()

        for p1, p2 in zip(simple_model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), \
                "Params diverged after checkpoint round-trip"

    def test_state_dict_has_expected_keys(self, simple_model, simple_data, criterion):
        x, y = simple_data
        opt = PsiLogic(simple_model.parameters(), lr=1e-2)
        opt.zero_grad()
        criterion(simple_model(x), y).backward()
        opt.step()

        sd = opt.state_dict()
        assert "state" in sd
        assert "param_groups" in sd
        # Each tracked parameter should have m, v, fast, slow, t
        for param_state in sd["state"].values():
            for key in ("m", "v", "fast", "slow", "t"):
                assert key in param_state, f"Missing key '{key}' in optimizer state"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_identical_seeds_identical_loss(self, simple_data, criterion):
        x, y = simple_data

        def run():
            torch.manual_seed(7)
            m   = nn.Linear(16, 4)
            opt = PsiLogic(m.parameters(), lr=1e-2)
            losses = []
            for _ in range(5):
                opt.zero_grad()
                loss = criterion(m(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            return losses

        assert run() == run(), "Same seed must produce identical loss curves"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Import and version
# ─────────────────────────────────────────────────────────────────────────────

class TestPackage:
    def test_version_string(self):
        import psilogic
        assert hasattr(psilogic, "__version__")
        parts = psilogic.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_all_exports_importable(self):
        from psilogic import (  # noqa: F401
            PsiLogic,
            PsiLogicGPT,
            PsiLogicNLP,
            PsiLogicViT,
            gpt_scratch_defaults,
            nlp_defaults,
            nlp_param_groups,
            vision_defaults,
        )