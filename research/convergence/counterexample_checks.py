"""Finite supporting checks for counterexamples.md; these are not proofs.

Run from the repository with ``python research/convergence/counterexample_checks.py``.
Only the research output JSON is written. Canonical optimizer source is imported
without changes. CUDA execution is not required or claimed.
"""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

from psilogic import PsiLogic
from psilogic._chaos import trust_from_soft_chaos
from psilogic.optimizer import _apply_agc


def scalar_optimizer(x: float, **overrides):
    p = torch.nn.Parameter(torch.tensor([x], dtype=torch.float64))
    kwargs = dict(
        lr=1.0,
        betas=(0.0, 0.0),
        eps=0.01,
        gamma=0.0,
        weight_decay=0.0,
        quantum_decay=0.0,
        grad_centralize=False,
        agc_clip=0.0,
        use_foreach=False,
        use_fused_cuda=False,
    )
    kwargs.update(overrides)
    return p, PsiLogic([p], **kwargs)


def set_gradient_and_step(p, opt, gradient):
    p.grad = torch.full_like(p, gradient)
    opt.step()


def main():
    results = {
        "scope": "Finite exact-algebra and CPU implementation checks; not convergence proofs or GPU validation",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "source": str(REPO / "psilogic" / "optimizer.py"),
    }

    # Exact rational two-cycle, followed by the actual scalar path with dyadic
    # values that remain exactly representable in its float32 moment buffers.
    e, eta = F(1, 8), F(1)
    a = eta / 2 - e
    x = a
    orbit = []
    for t in range(20):
        assert x == (-1) ** t * a
        orbit.append(str(x))
        x = x - eta * x / (abs(x) + e)
    p, opt = scalar_optimizer(float(a), eps=float(e), lr=float(eta))
    for t in range(20):
        assert p.item() == float((-1) ** t * a)
        p.grad = p.detach().clone()
        opt.step()
    results["C1"] = {"passed": True, "exact_orbit": orbit[:4], "mean_gradient_squared": str(a * a)}

    exact_direction = F(1, 10) * F(10) / (F(10) + F(1, 100)) - F(9, 10) / (1 + F(1, 100))
    assert exact_direction == -F(79990, 101101)
    first_branches = []
    for g, prob in [(10.0, 0.1), (-1.0, 0.9)]:
        p, opt = scalar_optimizer(0.0, lr=0.1, gamma=1.0, p_ext=1.0, chaos_warmup=0, tau_scale=2.0)
        set_gradient_and_step(p, opt, g)
        first_branches.append((prob, p.item(), opt.state[p]["soft_chaos"].item()))
    assert first_branches[0][2] == first_branches[1][2]
    first_obj_change = sum(prob * (0.5 * (x + 0.1) ** 2 - 0.005) for prob, x, _ in first_branches)
    assert first_obj_change > 0
    results["C2"] = {
        "passed": True,
        "exact_mean_normalized_gradient": str(exact_direction),
        "expected_objective_increase": first_obj_change,
        "first_step_soft_chaos": first_branches[0][2],
    }

    conditional_branches = []
    for g, prob in [(10.0, 0.1), (-1.0, 0.9)]:
        p, opt = scalar_optimizer(
            0.0,
            lr=0.1,
            gamma=1.0,
            p_ext=1.0,
            chaos_warmup=0,
            gamma_T_max=0,
            gamma_auto=False,
            adaptive_tau=True,
            tau_scale=2.0,
            max_cancel=1.0,
        )
        set_gradient_and_step(p, opt, -1.0)
        delta = p.item()
        state = opt.state[p]
        assert state["m"].item() == -1
        assert all(state[key].item() == 1 for key in ("fast", "slow", "gn_avg"))
        set_gradient_and_step(p, opt, g)
        soft = opt.state[p]["soft_chaos"].item()
        trust = 1.0 - min(soft, 0.99)
        conditional_branches.append((prob, g, trust, p.item(), delta))
    trust_pos, trust_neg = conditional_branches[0][2], conditional_branches[1][2]
    assert trust_pos < float(F(353, 800))
    assert trust_neg > float(F(299, 400))
    weighted = sum(prob * g * trust for prob, g, trust, _, _ in conditional_branches)
    assert weighted < -float(F(463, 2000))
    delta = conditional_branches[0][4]
    assert delta == conditional_branches[1][4]
    lipschitz = 1.1 / delta
    objective = lambda val: 0.5 * lipschitz * (val - 1.0 / lipschitz) ** 2
    conditional_increase = sum(
        prob * (objective(x) - objective(delta)) for prob, _, _, x, _ in conditional_branches
    )
    assert conditional_increase > 0
    results["C3"] = {
        "passed": True,
        "positive_branch_trust": trust_pos,
        "negative_branch_trust": trust_neg,
        "conditional_mean_trust_times_gradient": weighted,
        "rigorous_upper_bound": str(-F(463, 2000)),
        "conditional_expected_objective_increase": conditional_increase,
    }

    for lion in (False, True):
        p = torch.nn.Parameter(torch.zeros((1, 2), dtype=torch.float64))
        opt = PsiLogic(
            [p],
            lr=0.1,
            weight_decay=0.2,
            quantum_decay=0.3,
            gamma=1.0,
            p_ext=1.0,
            grad_centralize=True,
            agc_clip=1.0,
            chaos_warmup=0,
            lion_mode=lion,
            use_foreach=False,
            use_fused_cuda=False,
        )
        for t in range(20):
            opt.param_groups[0]["lr"] = 0.1 / (t + 1)
            p.grad = p.detach() - torch.ones_like(p)
            opt.step()
            assert torch.count_nonzero(p).item() == 0
            assert torch.count_nonzero(opt.state[p]["m"]).item() == 0
            assert torch.count_nonzero(opt.state[p]["v"]).item() == 0
    results["C4"] = {
        "passed": True,
        "steps_per_mode": 20,
        "modes": ["Adam", "Lion"],
        "gradient_norm_squared": 2,
    }

    p = torch.tensor([1.0], dtype=torch.float64)
    clipped = [
        _apply_agc(torch.tensor([g], dtype=torch.float64), p, 1.0).item() for g in (10.0, -1.0)
    ]
    assert clipped == [1.0, -1.0]
    assert F(1, 10) * int(clipped[0]) + F(9, 10) * int(clipped[1]) == -F(4, 5)
    results["C5"] = {"passed": True, "clipped_gradients": clipped, "exact_mean_clipped": "-4/5"}

    p, opt = scalar_optimizer(2.0, lr=0.1, weight_decay=0.5)
    set_gradient_and_step(p, opt, 0.0)
    assert abs(p.item() - 1.9) < 1e-14
    results["C6"] = {
        "passed": True,
        "initial_minimizer": 2.0,
        "next_parameter": p.item(),
        "objective_increase": 0.5 * (p.item() - 2.0) ** 2,
    }

    p, opt = scalar_optimizer(0.0, eps=0.0)
    set_gradient_and_step(p, opt, 0.0)
    assert torch.isnan(p).all()
    amplified = trust_from_soft_chaos(
        torch.tensor([0.5]), gamma_eff=1.0, p_ext=-10.0, chaos_gain=1.0, max_cancel=1.0
    ).item()
    assert amplified == 6.0
    results["C7"] = {
        "passed": True,
        "zero_epsilon_produces_nan": True,
        "negative_p_ext_trust": amplified,
    }

    norm = torch.tensor(1.0, dtype=torch.float16)
    scalar_rms = norm / math.sqrt(3)
    batched_rms = norm.reshape(1) / torch.tensor([math.sqrt(3)], dtype=torch.float32)
    assert scalar_rms.item() != batched_rms.item()
    results["backend_rms_rounding"] = {
        "passed": True,
        "scalar_dtype": str(scalar_rms.dtype),
        "batched_dtype": str(batched_rms.dtype),
        "scalar_value": scalar_rms.item(),
        "batched_value": batched_rms.item(),
    }

    output = Path(__file__).with_name("counterexample_results.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
