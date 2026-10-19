#!/usr/bin/env python3
"""Exact algebra plus finite numerical stress tests; not formal proof.

Run from any directory with Python, NumPy and SymPy. Writes a sibling JSON.
No package source is changed or optimizer helper used.
"""

from fractions import Fraction as F
from pathlib import Path
import json
import math
import numpy as np
import sympy as s


def check():
    exact = {}
    a, M, al, L, C, X, delta = s.symbols("a M alpha L C X Delta", positive=True)
    exact["absorption_margin"] = str(F(1, 2) - F(1, 8) - F(1, 16))
    assert F(1, 2) - F(1, 8) - F(1, 16) >= F(1, 4)
    assert 2 + 4 * F(1, 16) == F(9, 4)
    expr = s.Rational(9, 4) * (4 * delta / (al * a) + 4 * M**2 * C * X / a**2) + 4 * C * X
    target = 9 * delta / (al * a) + C * (9 * M**2 / a**2 + 4) * X
    assert s.simplify(expr - target) == 0
    exact["final_constants_identity"] = str(s.simplify(expr))
    m, q, b, r, eps = s.symbols("m q b r eps", positive=True)
    assert s.simplify((q / b) * m / (r + eps) - (m / b) / (r / q + eps / q)) == 0
    exact["epsilon_identity"] = "zero symbolic residual"
    exact["young_identity"] = str(s.factor(a * m**2 / 2 + M**2 * r**2 / (2 * a) - M * r * m))
    assert (
        s.simplify(
            a * m**2 / 2 + M**2 * r**2 / (2 * a) - M * r * m - (a * m - M * r) ** 2 / (2 * a)
        )
        == 0
    )

    rng = np.random.default_rng(20261016)
    max_noise_ratio = max_drift_ratio = 0.0
    tracking_cases = 0
    for beta in [0.0, 0.05, 0.5, 0.9, 0.99, 0.999]:
        for _ in range(50):
            T = int(rng.integers(1, 65))
            d = 5
            h = rng.normal(size=(T, d))
            noise = rng.normal(size=(T, d))
            n = []
            drift = []
            bh = []
            for t in range(1, T + 1):
                w = (1 - beta) * beta ** np.arange(t - 1, -1, -1) / (1 - beta**t)
                assert abs(w.sum() - 1) < 1e-12
                bh.append(w @ h[:t])
                n.append(w @ noise[:t])
                drift.append(bh[-1] - h[t - 1])
                if t > 1:
                    rho = beta * (1 - beta ** (t - 1)) / (1 - beta**t)
                    np.testing.assert_allclose(
                        drift[-1], rho * (drift[-2] + h[t - 2] - h[t - 1]), rtol=1e-11, atol=1e-11
                    )
            noise_energy = float(np.square(n).sum())
            drift_energy = float(np.square(drift).sum())
            noise_upper = float(np.square(noise).sum()) / (1 - beta)
            drift_upper = (beta / (1 - beta)) ** 2 * float(np.square(np.diff(h, axis=0)).sum())
            assert noise_energy <= noise_upper + 1e-9
            assert drift_energy <= drift_upper + 1e-9
            max_noise_ratio = max(max_noise_ratio, noise_energy / noise_upper if noise_upper else 0)
            max_drift_ratio = max(max_drift_ratio, drift_energy / drift_upper if drift_upper else 0)
            tracking_cases += 1

    dynamic_records = []
    for beta in [0.0, 0.3, 0.9, 0.99]:
        for with_noise in [False, True]:
            lo = 0.25
            hi = 1.4
            Lval = 1.0
            K = beta / (1 - beta)
            alpha = lo / (4 * Lval * hi * hi * max(1, K))
            x = rng.normal(size=4)
            gap = float((1 - np.cos(x)).sum())
            moment = np.zeros(4)
            ge = ne = me = ee = 0.0
            for t in range(1, 201):
                h = np.sin(x)
                noise = 0.1 * rng.normal(size=4) if with_noise else np.zeros(4)
                g = h + noise
                # Deliberately current-gradient-dependent scaling within its bounds.
                H = lo + (hi - lo) / (1 + np.exp(-g))
                moment = beta * moment + (1 - beta) * g
                mh = moment / (1 - beta**t)
                err = mh - h
                fx = float((1 - np.cos(x)).sum())
                xn = x - alpha * H * mh
                fn = float((1 - np.cos(xn)).sum())
                descent = (
                    fx
                    - alpha * (lo / 2 - Lval * alpha * hi * hi / 2) * float(mh @ mh)
                    + alpha * hi * hi / (2 * lo) * float(err @ err)
                )
                assert fn <= descent + 1e-12
                ge += float(h @ h)
                ne += float(noise @ noise)
                me += float(mh @ mh)
                ee += float(err @ err)
                Q = K * K * Lval * Lval * alpha * alpha * hi * hi
                assert ee <= 2 / (1 - beta) * ne + 2 * Q * me + 1e-10
                bound = 9 * gap / (alpha * lo) + 13 / (1 - beta) * (hi / lo) ** 2 * ne
                assert ge <= bound + 1e-8
                x = xn
            dynamic_records.append(
                {
                    "beta": beta,
                    "noise": with_noise,
                    "steps": 200,
                    "gradient_energy": ge,
                    "theorem_bound": bound,
                }
            )

    # Example only: exposes conservatism at common defaults, not a recommended LR.
    beta = 0.9
    beta2 = 0.999
    ep = 1e-8
    c = 0.95
    G = 1.0
    Lv = 1.0
    lower = c / (G + ep / math.sqrt(1 - beta2))
    upper = 1 / ep
    step_limit = lower / (4 * Lv * upper**2 * max(1, beta / (1 - beta)))
    return {
        "status": "pass",
        "scope": "Symbolic algebra and finite floating-point stress checks; not theorem formalization",
        "seed": 20261016,
        "exact_checks": exact,
        "tracking_cases": tracking_cases,
        "max_noise_energy_ratio": max_noise_ratio,
        "max_drift_energy_ratio": max_drift_ratio,
        "dynamic_records": dynamic_records,
        "conservative_default_example": {
            "L": Lv,
            "G": G,
            "a": lower,
            "M": upper,
            "alpha_upper": step_limit,
        },
    }


if __name__ == "__main__":
    result = check()
    Path(__file__).with_name("proof_algebra_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("dynamic_records", "exact_checks")},
            indent=2,
        )
    )
