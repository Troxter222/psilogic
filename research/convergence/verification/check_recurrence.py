#!/usr/bin/env python3
"""Independent real-equation oracle against CPU scalar/foreach execution.

No psilogic helpers are used in the oracle. Arithmetic is evaluated in float64;
comparisons allow fp32 moment/EMA rounding. This is correspondence evidence,
not a convergence proof or a GPU/DDP/finite-precision equivalence theorem.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from psilogic import PsiLogic


def reference(x, supplied_gradient, state, c):
    if supplied_gradient is None:
        return x, state, {"active": False, "missing": True}
    g = supplied_gradient.double().clone()
    if c["agc_clip"] > 0:
        g = g * min(
            1.0, c["agc_clip"] * max(float(x.norm()), 0.001) / max(float(g.norm()), 0.000001)
        )
    decay_gradient = g.clone()  # after AGC, before centralization
    if c["grad_centralize"] and g.ndim > 1:
        g = g - g.mean(dim=tuple(range(1, g.ndim)), keepdim=True)
    if state is None:
        state = {
            "t": 0,
            "m": torch.zeros_like(x),
            "v": torch.zeros_like(x),
            "fast": 0.0,
            "slow": 0.0,
            "gn_avg": 0.0,
            "soft_chaos": 0.0,
        }
    old_m = state["m"]
    t = state["t"] + 1
    eps = c["eps"]
    b1, b2 = c["betas"]
    nrm = float(g.norm())
    horizon = c["gamma_T_max"]
    cosine = 0.5 * (1 + math.cos(math.pi * min(t / horizon, 1))) if horizon > 0 else 1.0
    gamma = c["gamma"] * cosine
    qd = c["quantum_decay"] * cosine
    w = c["chaos_warmup"]
    if w < 0:
        w = max(500, horizon // 20) if horizon > 0 else 200
    gain = 0.0 if t <= w else min(1.0, (t - w) / max(1, w // 4))
    active = gamma > 0 and gain > 0
    disagreement = (
        0.0
        if t == 1
        else 0.5
        * (1 - max(-1.0, min(1.0, float((g * old_m).sum()) / (nrm * float(old_m.norm()) + eps))))
    )
    s = nrm / math.sqrt(g.numel())
    if t == 1:
        average, fast, slow = s, 1.0, 1.0
    else:
        average = 0.99 * state["gn_avg"] + 0.01 * s
        z = s / (average + eps)
        fast = 0.9 * state["fast"] + 0.1 * z
        slow = 0.99 * state["slow"] + 0.01 * z
    auto_factor = 1.0
    if c["gamma_auto"] and t > 1 and gamma > 0 and slow < 0.1:
        auto_factor = max(slow / 0.1, 0.1)
        gamma *= auto_factor
    soft, trust = 0.0, 1.0
    if active:
        if c["adaptive_tau"]:
            logit = (fast / (slow + eps) - 1) * 4 / max(c["tau_scale"] - 1, 0.25)
        else:
            logit = (slow / max(c["chaos_tau"], eps) - 1) * 4
        excess = 1 / (1 + math.exp(-logit))
        soft = math.tanh(slow) * (excess + disagreement) / 2
        trust = 1 - min(soft * gamma * c["p_ext"] * gain, c["max_cancel"], 0.99)
    y = x * (1 - c["lr"] * c["weight_decay"])
    if active and qd > 0:
        y = y * (1 - c["lr"] * qd * gain * (1 - soft) * decay_gradient.abs().tanh())
    if c["lion_mode"]:
        direction = (b1 * old_m + (1 - b1) * g).sign()
        m, v = b2 * old_m + (1 - b2) * g, state["v"]
    else:
        m = b1 * old_m + (1 - b1) * g
        v = b2 * state["v"] + (1 - b2) * g.square()
        direction = math.sqrt(1 - b2**t) / (1 - b1**t) * m / (v.sqrt() + eps)
    y = y - c["lr"] * trust * direction
    return (
        y,
        {"t": t, "m": m, "v": v, "fast": fast, "slow": slow, "gn_avg": average, "soft_chaos": soft},
        {
            "active": active,
            "gain": gain,
            "trust": trust,
            "auto_factor": auto_factor,
            "cosine": cosine,
            "missing": False,
        },
    )


def run_case(name, overrides, steps=24, mode="alternating", foreach=False):
    c = dict(
        lr=0.03,
        betas=(0.8, 0.93),
        weight_decay=0.07,
        gamma=0.7,
        p_ext=1.8,
        quantum_decay=0.13,
        eps=0.07,
        grad_centralize=False,
        chaos_tau=0.4,
        chaos_warmup=0,
        adaptive_tau=True,
        tau_scale=2.0,
        max_cancel=0.55,
        agc_clip=0.0,
        gamma_T_max=0,
        lion_mode=False,
        gamma_auto=False,
        use_foreach=False,
        use_fused_cuda=False,
    )
    c.update(overrides)
    initial = [
        torch.tensor([[0.4, -0.7, 0.2], [1.1, 0.1, -0.3]], dtype=torch.float32),
        torch.tensor([-0.2, 0.5, 0.8], dtype=torch.float32),
    ]
    params = [torch.nn.Parameter(x.clone()) for x in initial]
    xs, states = [x.double() for x in initial], [None, None]
    opt = PsiLogic(params, **c)
    errors = {k: 0.0 for k in ["parameter", "m", "v", "fast", "slow", "gn_avg", "soft_chaos"]}
    saw = {
        "inactive": False,
        "active": False,
        "ramp": False,
        "cap": False,
        "auto_reduce": False,
        "auto_floor": False,
        "missing": False,
        "zero_first_norm": False,
        "unequal_steps": False,
        "trust_above_one": False,
    }
    comparisons = 0
    for outer in range(steps):
        grads = []
        for j, x in enumerate(initial):
            if mode == "missing" and (outer + j) % (j + 3) == 0:
                g = None
            elif mode == "quiet" and outer > 0 or mode == "zero_first" and outer == 0:
                g = torch.zeros_like(x)
            else:
                idx = torch.arange(x.numel(), dtype=torch.float32).reshape(x.shape)
                g = (idx + 1) * (0.11 + 0.03 * (outer % 4)) * (
                    -1 if (outer + j) % 3 == 1 else 1
                ) + 0.07 * j
                if outer % 7 == 6:
                    g *= 9
            grads.append(g)
            params[j].grad = g
            xs[j], states[j], diagnostic = reference(xs[j], g, states[j], c)
            saw["missing"] |= diagnostic["missing"]
            saw["active"] |= diagnostic["active"]
            saw["inactive"] |= not diagnostic["active"]
            if not diagnostic["missing"]:
                saw["ramp"] |= 0 < diagnostic["gain"] < 1
                saw["cap"] |= abs(diagnostic["trust"] - (1 - min(c["max_cancel"], 0.99))) < 1e-14
                saw["auto_reduce"] |= diagnostic["auto_factor"] < 1
                saw["auto_floor"] |= diagnostic["auto_factor"] == 0.1
                saw["trust_above_one"] |= diagnostic["trust"] > 1
                saw["zero_first_norm"] |= states[j]["t"] == 1 and float(g.norm()) == 0
        if foreach:
            with torch.no_grad():
                opt._step_foreach(opt.param_groups[0])
        else:
            opt.step()
        if all(s is not None for s in states):
            saw["unequal_steps"] |= states[0]["t"] != states[1]["t"]
        for j, p in enumerate(params):
            errors["parameter"] = max(
                errors["parameter"], float((p.detach().double() - xs[j]).abs().max())
            )
            torch.testing.assert_close(p.detach().double(), xs[j], atol=2e-5, rtol=2e-5)
            comparisons += 1
            if states[j] is None:
                assert p not in opt.state
                continue
            actual = opt.state[p]
            assert actual["t"] == states[j]["t"]
            for key in errors:
                if key == "parameter":
                    continue
                target = torch.as_tensor(states[j][key], dtype=torch.float64).reshape(
                    actual[key].shape
                )
                errors[key] = max(errors[key], float((actual[key].double() - target).abs().max()))
                torch.testing.assert_close(actual[key].double(), target, atol=2e-5, rtol=2e-5)
                comparisons += 1
    return {
        "case": name,
        "backend": "forced_foreach_cpu" if foreach else "scalar_cpu",
        "steps": steps,
        "comparisons": comparisons,
        "max_absolute_errors": errors,
        "observed_branches": saw,
        "status": "pass",
    }


def main():
    cases = [
        ("adam_all_active", {}, 24, "alternating"),
        ("lion_all_active", {"lion_mode": True}, 24, "alternating"),
        ("agc_then_centralize_qd", {"agc_clip": 0.17, "grad_centralize": True}, 24, "alternating"),
        ("zero_norm_initialization", {}, 24, "zero_first"),
        ("explicit_warmup_ramp_cosine", {"chaos_warmup": 8, "gamma_T_max": 20}, 24, "alternating"),
        ("auto_warmup_unknown_horizon", {"chaos_warmup": -1}, 253, "alternating"),
        ("auto_warmup_known_horizon", {"chaos_warmup": -1, "gamma_T_max": 800}, 629, "alternating"),
        ("gamma_auto_reduction_and_floor", {"gamma_auto": True}, 475, "quiet"),
        ("cap_at_099", {"gamma": 100.0, "max_cancel": 1.0}, 24, "alternating"),
        ("absolute_gate_tau_guard", {"adaptive_tau": False, "chaos_tau": 0.0}, 24, "alternating"),
        ("adaptive_tau_margin_guard", {"tau_scale": 0.5}, 24, "alternating"),
        ("gamma_zero_disables_qd", {"gamma": 0.0, "quantum_decay": 9.0}, 24, "alternating"),
        ("zero_learning_rate", {"lr": 0.0}, 24, "alternating"),
        ("negative_p_ext_amplifies", {"p_ext": -2.0}, 24, "alternating"),
        ("missing_and_unequal_counters", {}, 24, "missing"),
        ("lion_missing_counters", {"lion_mode": True}, 24, "missing"),
        ("no_momentum_bias_epsilon", {"betas": (0.0, 0.93), "gamma": 0.0}, 24, "alternating"),
    ]
    records = [run_case(*case, foreach=backend) for backend in [False, True] for case in cases]
    by_name = {r["case"]: r for r in records if r["backend"] == "scalar_cpu"}
    for case, flag in [
        ("explicit_warmup_ramp_cosine", "ramp"),
        ("gamma_auto_reduction_and_floor", "auto_floor"),
        ("cap_at_099", "cap"),
        ("zero_norm_initialization", "zero_first_norm"),
        ("missing_and_unequal_counters", "unequal_steps"),
        ("negative_p_ext_amplifies", "trust_above_one"),
    ]:
        assert by_name[case]["observed_branches"][flag], (case, flag)
    for case in ["gamma_zero_disables_qd"]:
        assert not by_name[case]["observed_branches"]["active"]
    result = {
        "status": "pass",
        "torch_version": torch.__version__,
        "cases": len(records),
        "comparisons": sum(r["comparisons"] for r in records),
        "tolerance": {"atol": 2e-5, "rtol": 2e-5},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "psilogic/optimizer.py", ROOT / "psilogic/_chaos.py"]
        },
        "scope": "Independent float64 equations; CPU float32 scalar and forced CPU foreach. No GPU/DDP claim.",
        "records": records,
    }
    destination = Path(__file__).with_name("recurrence_results.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
