#!/usr/bin/env python3
"""Follow-up NLP suite: bare extras + soft tau (does cancel actually help?).

Hypothesis from the big 39-cell sweep
-------------------------------------
* Default GPT-scratch PsiLogic loses to AdamW mostly because of AGC +
  ``grad_centralize``, not because of chaos cancel.
* With ``tau_scale=3`` the adaptive gate almost never opens (fast/slow ≈ 1),
  so gamma / max_cancel / p_ext ablations were bit-identical.
* ``combo_bare`` (agc=0, centralize=False) was the only significant win on the
  paper-like setup; ``tiny_model`` was a huge win.

This script tests the natural next cells **without changing library defaults**:

* paper setup × bare × tau_scale ∈ {1.2, 1.5, 2.0, 3.0}
* bare + mild gamma variants
* long_ctx × default vs bare vs bare+soft-tau  (was catastrophic)
* tiny_model × default vs bare vs bare+soft-tau
* controls: paper_default, bare_only, soft_tau_only (isolate factors)

Also reports mean ``psi/spike_rate`` / ``fast−slow`` so you can see whether
the cancel gate actually opened.

Examples
--------
List::

    .venv/bin/python scripts/nlp_followup_bare_tau.py --list

Smoke::

    .venv/bin/python scripts/nlp_followup_bare_tau.py --smoke --device cpu --no-amp

Full (TinyStories already cached)::

    .venv/bin/python scripts/nlp_followup_bare_tau.py \\
        --data-root ./data --offline --output-dir ./results/nlp_followup_bare_tau

Core-only (paper × bare × tau grid, ~6 cells)::

    .venv/bin/python scripts/nlp_followup_bare_tau.py --suite core --offline --data-root ./data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[1]
_BENCH = _REPO / "benchmark"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from fairbench.config import (  # noqa: E402
    ArenaConfig,
    BenchmarkConfig,
    HardwareConfig,
    LoggingConfig,
    SweepConfig,
    TrainConfig,
)
from fairbench.metrics import paired_ttest  # noqa: E402
from fairbench.runner import BenchmarkRunner  # noqa: E402


# --------------------------------------------------------------------------- #
# Presets (overrides on top of NLPArena GPT-scratch base)
# --------------------------------------------------------------------------- #

# Proven win from the big sweep.
_BARE: dict[str, Any] = {"agc_clip": 0.0, "grad_centralize": False}


def _bare_tau(tau: float, **extra: Any) -> dict[str, Any]:
    return {**_BARE, "tau_scale": tau, **extra}


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str  # control | core | long_ctx | tiny | smoke
    why: str
    max_steps: int = 2000
    batch_size: int = 32
    sweep_steps: int = 500
    num_lrs: int = 7
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 128
    train_chars: int = 2_000_000
    val_chars: int = 200_000
    steps_per_epoch: int = 1000
    warmup_steps: int = 100
    eval_every: int = 200
    use_scheduler: bool = True
    fixed_lr: Optional[float] = None
    psi_overrides: dict[str, Any] = field(default_factory=dict)
    toggles: str = ""


@dataclass
class ExpResult:
    name: str
    family: str
    why: str
    toggles: str
    winner: str
    delta_ppl: float
    adamw_ppl: float
    adamw_ppl_std: float
    psilogic_ppl: float
    psilogic_ppl_std: float
    adamw_loss: float
    psilogic_loss: float
    p_value: float
    n_seeds: int
    # Cancel-gate diagnostics from steps.csv (PsiLogic only).
    spike_rate_mean: float
    fast_minus_slow_mean: float
    chaos_t_mean: float
    max_steps: int
    n_layer: int
    n_embd: int
    block_size: int
    psi_overrides_json: str
    output_dir: str


def _paper(**kwargs: Any) -> dict[str, Any]:
    base = dict(
        max_steps=2000,
        batch_size=32,
        sweep_steps=500,
        num_lrs=7,
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=128,
        eval_every=200,
        warmup_steps=100,
    )
    base.update(kwargs)
    return base


def _toggles(ov: dict[str, Any]) -> str:
    if not ov:
        return "psi=default"
    return " ".join(f"{k}={v}" for k, v in sorted(ov.items()))


def build_control_experiments() -> list[Experiment]:
    """Isolate factors that the big sweep implicated."""
    cells: list[tuple[str, str, dict[str, Any]]] = [
        ("paper_default", "Library GPT-scratch defaults (known loser vs AdamW)", {}),
        ("bare_only", "AGC off + no centralize (combo_bare replicate)", dict(_BARE)),
        (
            "soft_tau_only",
            "tau_scale=1.5 only (keep AGC+centralize) — does soft tau alone help?",
            {"tau_scale": 1.5},
        ),
    ]
    return [
        Experiment(
            name=n,
            family="control",
            why=w,
            toggles=_toggles(ov),
            psi_overrides=ov,
            **_paper(),
        )
        for n, w, ov in cells
    ]


def build_core_experiments() -> list[Experiment]:
    """Paper setup × bare × tau grid (+ a couple gamma checks)."""
    cells: list[tuple[str, str, dict[str, Any]]] = [
        ("bare_tau12", "bare + tau=1.2 (aggressive gate)", _bare_tau(1.2)),
        ("bare_tau15", "bare + tau=1.5", _bare_tau(1.5)),
        ("bare_tau20", "bare + tau=2.0", _bare_tau(2.0)),
        ("bare_tau30", "bare + tau=3.0 (default tau, bare extras)", _bare_tau(3.0)),
        (
            "bare_tau15_gamma01",
            "bare + tau=1.5 + gentle gamma",
            _bare_tau(1.5, gamma=0.01),
        ),
        (
            "bare_tau15_gamma05",
            "bare + tau=1.5 + strong gamma",
            _bare_tau(1.5, gamma=0.05),
        ),
        (
            "bare_tau15_warmup0",
            "bare + tau=1.5 + chaos from step 0",
            _bare_tau(1.5, chaos_warmup=0),
        ),
    ]
    # Deduplicate bare_tau15 if also in controls when suite=all — fine to
    # keep one definition; select_experiments merges by name.
    return [
        Experiment(
            name=n,
            family="core",
            why=w,
            toggles=_toggles(ov),
            psi_overrides=ov,
            **_paper(),
        )
        for n, w, ov in cells
    ]


def build_long_ctx_experiments() -> list[Experiment]:
    """long_ctx was Δ=-14.8 — retry with bare / soft tau."""
    kw = _paper(block_size=256, batch_size=16)
    cells: list[tuple[str, str, dict[str, Any]]] = [
        ("long_default", "ctx=256, default psi (known catastrophe)", {}),
        ("long_bare", "ctx=256, bare extras", dict(_BARE)),
        ("long_bare_tau15", "ctx=256, bare + tau=1.5", _bare_tau(1.5)),
    ]
    return [
        Experiment(
            name=n,
            family="long_ctx",
            why=w,
            toggles=_toggles(ov) + " ctx=256",
            psi_overrides=ov,
            **kw,
        )
        for n, w, ov in cells
    ]


def build_tiny_experiments() -> list[Experiment]:
    """tiny_model was Δ=+9.6 — see if bare/soft-tau amplifies it."""
    kw = _paper(n_layer=2, n_head=2, n_embd=128, batch_size=32)
    cells: list[tuple[str, str, dict[str, Any]]] = [
        ("tiny_default", "2L/128d default psi (known big win)", {}),
        ("tiny_bare", "2L/128d bare", dict(_BARE)),
        ("tiny_bare_tau15", "2L/128d bare + tau=1.5", _bare_tau(1.5)),
    ]
    return [
        Experiment(
            name=n,
            family="tiny",
            why=w,
            toggles=_toggles(ov) + " model=2L/128",
            psi_overrides=ov,
            **kw,
        )
        for n, w, ov in cells
    ]


def build_smoke_experiments() -> list[Experiment]:
    tiny = dict(
        max_steps=10,
        sweep_steps=5,
        num_lrs=2,
        batch_size=4,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        train_chars=50_000,
        val_chars=10_000,
        steps_per_epoch=10,
        warmup_steps=2,
        eval_every=5,
        fixed_lr=1e-3,
    )
    return [
        Experiment(
            name="smoke_default",
            family="smoke",
            why="Smoke default",
            toggles="psi=default",
            **tiny,
        ),
        Experiment(
            name="smoke_bare_tau15",
            family="smoke",
            why="Smoke bare+tau1.5",
            toggles=_toggles(_bare_tau(1.5)),
            psi_overrides=_bare_tau(1.5),
            **tiny,
        ),
    ]


def all_experiments() -> dict[str, Experiment]:
    items = (
        build_control_experiments()
        + build_core_experiments()
        + build_long_ctx_experiments()
        + build_tiny_experiments()
    )
    # Later families win on name collision (bare_tau15 in control+core → keep core).
    out: dict[str, Experiment] = {}
    for e in items:
        out[e.name] = e
    return out


# --------------------------------------------------------------------------- #
# FairBench glue
# --------------------------------------------------------------------------- #


def build_config(
    exp: Experiment,
    *,
    seeds: list[int],
    data_root: str,
    output_dir: str,
    device: str,
    offline: bool,
    synthetic: bool,
    amp: bool,
    num_workers: int,
) -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        arenas=["nlp"],
        optimizers=["adamw", "psilogic"],
        sweep=SweepConfig(
            lr_min=1e-5,
            lr_max=1e-2,
            num_lrs=exp.num_lrs,
            max_steps=exp.sweep_steps,
            max_epochs=max(1, exp.sweep_steps // 50),
        ),
        train=TrainConfig(
            seeds=list(seeds),
            max_steps=exp.max_steps,
            max_epochs=max(1, exp.max_steps // 50),
            eval_every=exp.eval_every,
            grad_clip=1.0,
            warmup_steps=exp.warmup_steps,
            use_scheduler=exp.use_scheduler,
        ),
        hardware=HardwareConfig(
            device=device,
            amp=amp,
            amp_dtype="bfloat16" if device.startswith("cuda") else "float16",
            use_foreach=True,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        ),
        logging=LoggingConfig(
            output_dir=output_dir,
            tensorboard=False,
            wandb=False,
            plots=False,
            log_every=max(1, exp.eval_every // 10),
        ),
        arena_configs={
            "nlp": ArenaConfig(
                name="nlp",
                batch_size=exp.batch_size,
                data_root=data_root,
                extra={
                    "n_layer": exp.n_layer,
                    "n_head": exp.n_head,
                    "n_embd": exp.n_embd,
                    "block_size": exp.block_size,
                    "train_chars": exp.train_chars,
                    "val_chars": exp.val_chars,
                    "steps_per_epoch": exp.steps_per_epoch,
                    "psilogic_overrides": dict(exp.psi_overrides),
                },
            )
        },
        synthetic=synthetic,
        offline=offline,
    )
    cfg.seeds = list(seeds)  # type: ignore[attr-defined]
    if exp.fixed_lr is not None:
        cfg.fixed_lrs = {"nlp": {"adamw": exp.fixed_lr, "psilogic": exp.fixed_lr}}
    return cfg


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _chaos_diagnostics(output_dir: Path) -> tuple[float, float, float]:
    """Mean spike_rate / fast_minus_slow / chaos_t over PsiLogic train steps."""
    rows = _read_csv(output_dir / "steps.csv")
    spikes, gaps, chaos = [], [], []
    for r in rows:
        if r.get("optimizer") != "psilogic":
            continue
        if r.get("split") not in (None, "", "train"):
            # keep train rows; empty split also appears on some logs
            if r.get("split") == "val":
                continue
        s = _f(r, "psi/spike_rate")
        g = _f(r, "psi/fast_minus_slow")
        c = _f(r, "psi/chaos_t")
        if math.isfinite(s):
            spikes.append(s)
        if math.isfinite(g):
            gaps.append(g)
        if math.isfinite(c):
            chaos.append(c)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return mean(spikes), mean(gaps), mean(chaos)


def summarize_run(exp: Experiment, output_dir: Path) -> ExpResult:
    agg_rows = _read_csv(output_dir / "aggregate.csv")
    by_opt = {r["optimizer"]: r for r in agg_rows if r.get("arena") == "nlp"}
    adamw = by_opt.get("adamw", {})
    psi = by_opt.get("psilogic", {})

    aw_ppl = _f(adamw, "perplexity_mean")
    aw_std = _f(adamw, "perplexity_std", 0.0)
    psi_ppl = _f(psi, "perplexity_mean")
    psi_std = _f(psi, "perplexity_std", 0.0)
    aw_loss = _f(adamw, "val_loss_mean")
    psi_loss = _f(psi, "val_loss_mean")
    n = int(_f(psi, "perplexity_n", _f(adamw, "perplexity_n", 0)))

    p_value = float("nan")
    for row in _read_csv(output_dir / "significance.csv"):
        if (
            row.get("arena") == "nlp"
            and row.get("metric") == "perplexity"
            and row.get("baseline") == "adamw"
            and row.get("reference") == "psilogic"
        ):
            p_value = _f(row, "p_value")
            break

    if math.isnan(p_value):
        summary = _read_csv(output_dir / "summary.csv")
        a = [
            _f(r, "final_perplexity")
            for r in summary
            if r.get("optimizer") == "psilogic" and r.get("failed", "False") in ("False", "0", "")
        ]
        b = [
            _f(r, "final_perplexity")
            for r in summary
            if r.get("optimizer") == "adamw" and r.get("failed", "False") in ("False", "0", "")
        ]
        if a and b and len(a) == len(b):
            tt = paired_ttest(a, b)
            p_value = tt.p_value
            n = tt.n_pairs

    delta = aw_ppl - psi_ppl
    if math.isnan(delta):
        winner = "unknown"
    elif abs(delta) < 1e-6:
        winner = "tie"
    elif delta > 0:
        winner = "psilogic"
    else:
        winner = "adamw"

    spike, gap, chaos = _chaos_diagnostics(output_dir)

    return ExpResult(
        name=exp.name,
        family=exp.family,
        why=exp.why,
        toggles=exp.toggles,
        winner=winner,
        delta_ppl=delta,
        adamw_ppl=aw_ppl,
        adamw_ppl_std=aw_std,
        psilogic_ppl=psi_ppl,
        psilogic_ppl_std=psi_std,
        adamw_loss=aw_loss,
        psilogic_loss=psi_loss,
        p_value=p_value,
        n_seeds=n,
        spike_rate_mean=spike,
        fast_minus_slow_mean=gap,
        chaos_t_mean=chaos,
        max_steps=exp.max_steps,
        n_layer=exp.n_layer,
        n_embd=exp.n_embd,
        block_size=exp.block_size,
        psi_overrides_json=json.dumps(exp.psi_overrides, sort_keys=True),
        output_dir=str(output_dir),
    )


def run_experiment(exp: Experiment, args: argparse.Namespace) -> ExpResult:
    out = Path(args.output_dir) / exp.family / exp.name
    out.mkdir(parents=True, exist_ok=True)

    agg = out / "aggregate.csv"
    if agg.is_file() and not args.force:
        print(f"[skip] {exp.family}/{exp.name} (use --force to rerun)")
        return summarize_run(exp, out)

    cfg = build_config(
        exp,
        seeds=list(args.seeds),
        data_root=args.data_root,
        output_dir=str(out),
        device=args.device,
        offline=args.offline,
        synthetic=args.synthetic or args.smoke,
        amp=not args.no_amp,
        num_workers=args.num_workers,
    )
    meta = {
        "name": exp.name,
        "family": exp.family,
        "why": exp.why,
        "toggles": exp.toggles,
        "psi_overrides": exp.psi_overrides,
        "hypothesis": (
            "bare extras (no AGC/centralize) + softer tau_scale so cancel gate opens"
        ),
        "experiment": asdict(exp),
    }
    with (out / "experiment.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\n=== [{exp.family}] {exp.name} ===")
    print(f"  {exp.why}")
    print(f"  toggles: {exp.toggles}")
    print(
        f"  steps={exp.max_steps}  GPT={exp.n_layer}L/{exp.n_head}H/{exp.n_embd}d  "
        f"ctx={exp.block_size}  bs={exp.batch_size}"
    )
    BenchmarkRunner(cfg).run()
    if not agg.is_file():
        raise RuntimeError(
            f"Experiment {exp.name} produced no aggregate.csv under {out}. "
            "Arena likely failed (missing data / OOM)."
        )
    return summarize_run(exp, out)


def ensure_tinystories(data_root: str, *, offline: bool, synthetic: bool) -> None:
    if synthetic:
        return
    from fairbench.datasets import download_tinystories, tinystories_ready

    root = os.path.abspath(data_root)
    if tinystories_ready(root):
        print(f"TinyStories cache OK: {root}/tinystories/")
        return
    if offline:
        raise SystemExit(
            f"TinyStories not found under {root}/tinystories/.\n"
            "Download once:\n"
            f"  .venv/bin/python scripts/nlp_followup_bare_tau.py --download-only "
            f"--data-root {data_root}\n"
            "Then rerun with --offline."
        )
    print(f"TinyStories missing under {root}; downloading...")
    download_tinystories(root)
    if not tinystories_ready(root):
        raise SystemExit(f"Download finished but cache still missing under {root}/tinystories/")
    print(f"TinyStories ready: {root}/tinystories/")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_table(results: list[ExpResult]) -> None:
    ranked = sorted(
        results,
        key=lambda r: (-r.delta_ppl if math.isfinite(r.delta_ppl) else -1e9, r.name),
    )
    print("\n" + "=" * 130)
    print("Follow-up: bare + soft tau  (Δppl = AdamW − Ψ; >0 ⇒ Ψ wins)")
    print("=" * 130)
    print(
        f"{'#':>2} {'name':<20} {'fam':<8} {'win':<9} {'Δppl':>7} "
        f"{'AdamW':>11} {'Ψ':>11} {'p':>6} {'spike':>6} {'Δf-s':>7}  toggles"
    )
    print("-" * 130)
    for i, r in enumerate(ranked, 1):
        aw = f"{r.adamw_ppl:.2f}±{r.adamw_ppl_std:.2f}" if math.isfinite(r.adamw_ppl) else "n/a"
        psi = (
            f"{r.psilogic_ppl:.2f}±{r.psilogic_ppl_std:.2f}"
            if math.isfinite(r.psilogic_ppl)
            else "n/a"
        )
        p = f"{r.p_value:.3f}" if math.isfinite(r.p_value) else "n/a"
        delta = f"{r.delta_ppl:+.3f}" if math.isfinite(r.delta_ppl) else "n/a"
        spike = f"{r.spike_rate_mean:.3f}" if math.isfinite(r.spike_rate_mean) else "n/a"
        gap = (
            f"{r.fast_minus_slow_mean:+.4f}"
            if math.isfinite(r.fast_minus_slow_mean)
            else "n/a"
        )
        mark = "✓" if r.winner == "psilogic" else ("·" if r.winner == "tie" else "✗")
        note = r.toggles
        if len(note) > 36:
            note = note[:33] + "..."
        print(
            f"{i:>2} {r.name:<20} {r.family:<8} {mark}{r.winner:<8} {delta:>7} "
            f"{aw:>11} {psi:>11} {p:>6} {spike:>6} {gap:>7}  {note}"
        )
    print("=" * 130)

    wins = sum(1 for r in results if r.winner == "psilogic")
    sig = [
        r
        for r in results
        if r.winner == "psilogic" and math.isfinite(r.p_value) and r.p_value < 0.05
    ]
    print(f"PsiLogic wins {wins}/{len(results)}; significant (p<0.05): {len(sig)}/{len(results)}")

    # Decision helper
    core = [r for r in results if r.family in ("control", "core")]
    if core:
        best = max(core, key=lambda r: r.delta_ppl if math.isfinite(r.delta_ppl) else -1e9)
        default = next((r for r in core if r.name == "paper_default"), None)
        print("\nDecision hint (paper-like cells):")
        if default and math.isfinite(default.delta_ppl):
            print(f"  paper_default Δ={default.delta_ppl:+.3f}  spike={default.spike_rate_mean:.4f}")
        print(
            f"  best among control/core: {best.name}  Δ={best.delta_ppl:+.3f}  "
            f"spike={best.spike_rate_mean:.4f}  ({best.toggles})"
        )
        if best.winner == "psilogic" and math.isfinite(best.p_value) and best.p_value < 0.05:
            print(
                "  → Candidate for changing NLPArena/gpt_scratch defaults "
                "(agc=0, centralize=False, and/or lower tau_scale)."
            )
        else:
            print("  → Not yet strong enough to change library defaults; inspect spike column.")


def write_csv(results: list[ExpResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ExpResult.__dataclass_fields__)
    ranked = sorted(
        results,
        key=lambda r: (-r.delta_ppl if math.isfinite(r.delta_ppl) else -1e9, r.name),
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in ranked:
            w.writerow(asdict(r))
    print(f"Wrote CSV: {path}")


def write_markdown(results: list[ExpResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        results,
        key=lambda r: (-r.delta_ppl if math.isfinite(r.delta_ppl) else -1e9, r.name),
    )
    lines = [
        "# NLP follow-up: bare extras + soft tau",
        "",
        "Hypothesis: turn off AGC/centralize and lower `tau_scale` so cancel fires.",
        "",
        "| # | name | family | winner | Δppl | AdamW | Ψ | p | spike_rate | fast−slow | toggles |",
        "|--:|------|--------|--------|-----:|------:|--:|--:|-----------:|----------:|---------|",
    ]
    for i, r in enumerate(ranked, 1):
        aw = f"{r.adamw_ppl:.2f}±{r.adamw_ppl_std:.2f}" if math.isfinite(r.adamw_ppl) else "n/a"
        psi = (
            f"{r.psilogic_ppl:.2f}±{r.psilogic_ppl_std:.2f}"
            if math.isfinite(r.psilogic_ppl)
            else "n/a"
        )
        p = f"{r.p_value:.3f}" if math.isfinite(r.p_value) else "n/a"
        delta = f"{r.delta_ppl:+.3f}" if math.isfinite(r.delta_ppl) else "n/a"
        spike = f"{r.spike_rate_mean:.4f}" if math.isfinite(r.spike_rate_mean) else "n/a"
        gap = (
            f"{r.fast_minus_slow_mean:+.5f}"
            if math.isfinite(r.fast_minus_slow_mean)
            else "n/a"
        )
        lines.append(
            f"| {i} | `{r.name}` | {r.family} | **{r.winner}** | {delta} | {aw} | {psi} | "
            f"{p} | {spike} | {gap} | {r.toggles} |"
        )
    wins = sum(1 for r in results if r.winner == "psilogic")
    lines += ["", f"**PsiLogic wins:** {wins}/{len(results)}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Markdown: {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Follow-up NLP: bare extras + soft tau_scale vs AdamW.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--suite",
        choices=["all", "control", "core", "long_ctx", "tiny"],
        default="all",
        help="Subset of experiment families.",
    )
    p.add_argument("--only", default=None, help="Comma-separated experiment names.")
    p.add_argument("--list", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data-root", default=str(_REPO / "data"))
    p.add_argument("--output-dir", default=str(_REPO / "results" / "nlp_followup_bare_tau"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-experiments", type=int, default=0)
    p.add_argument("--catalog-size", action="store_true")
    return p.parse_args(argv)


def select_experiments(args: argparse.Namespace) -> dict[str, Experiment]:
    if args.smoke:
        return {e.name: e for e in build_smoke_experiments()}

    catalog = all_experiments()
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        missing = [n for n in wanted if n not in catalog]
        if missing:
            raise SystemExit(f"Unknown: {missing}\nAvailable: {', '.join(catalog)}")
        return {n: catalog[n] for n in wanted}

    if args.suite == "control":
        return {e.name: e for e in build_control_experiments()}
    if args.suite == "core":
        # Include paper_default + bare_only as anchors next to the tau grid.
        anchors = {e.name: e for e in build_control_experiments() if e.name in ("paper_default", "bare_only")}
        core = {e.name: e for e in build_core_experiments()}
        return {**anchors, **core}
    if args.suite == "long_ctx":
        return {e.name: e for e in build_long_ctx_experiments()}
    if args.suite == "tiny":
        return {e.name: e for e in build_tiny_experiments()}
    return catalog


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    catalog = all_experiments()

    if args.catalog_size:
        print(
            f"control={len(build_control_experiments())}  "
            f"core={len(build_core_experiments())}  "
            f"long_ctx={len(build_long_ctx_experiments())}  "
            f"tiny={len(build_tiny_experiments())}  "
            f"all_unique={len(catalog)}"
        )
        return 0

    if args.list:
        print(f"{'name':<22} {'family':<9} {'steps':>5}  toggles")
        print("-" * 90)
        for e in catalog.values():
            print(f"{e.name:<22} {e.family:<9} {e.max_steps:>5}  {e.toggles}")
        print("-" * 90)
        print(f"Total unique: {len(catalog)}")
        return 0

    if args.download_only:
        ensure_tinystories(args.data_root, offline=False, synthetic=False)
        return 0

    try:
        experiments = select_experiments(args)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.smoke:
        args.synthetic = True

    try:
        ensure_tinystories(
            args.data_root,
            offline=args.offline,
            synthetic=args.synthetic or args.smoke,
        )
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    names = list(experiments.keys())
    if args.max_experiments and args.max_experiments > 0:
        names = names[: args.max_experiments]
        experiments = {n: experiments[n] for n in names}

    print(
        f"Planning {len(experiments)} follow-up experiments | seeds={args.seeds} | "
        f"out={args.output_dir}"
    )

    results: list[ExpResult] = []
    failures = 0
    for exp in experiments.values():
        try:
            results.append(run_experiment(exp, args))
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {exp.family}/{exp.name}: {exc}", file=sys.stderr)
            results.append(
                ExpResult(
                    name=exp.name,
                    family=exp.family,
                    why=exp.why,
                    toggles=exp.toggles,
                    winner="failed",
                    delta_ppl=float("nan"),
                    adamw_ppl=float("nan"),
                    adamw_ppl_std=float("nan"),
                    psilogic_ppl=float("nan"),
                    psilogic_ppl_std=float("nan"),
                    adamw_loss=float("nan"),
                    psilogic_loss=float("nan"),
                    p_value=float("nan"),
                    n_seeds=0,
                    spike_rate_mean=float("nan"),
                    fast_minus_slow_mean=float("nan"),
                    chaos_t_mean=float("nan"),
                    max_steps=exp.max_steps,
                    n_layer=exp.n_layer,
                    n_embd=exp.n_embd,
                    block_size=exp.block_size,
                    psi_overrides_json=json.dumps(exp.psi_overrides, sort_keys=True),
                    output_dir=str(Path(args.output_dir) / exp.family / exp.name),
                )
            )

    print_table(results)
    out = Path(args.output_dir)
    write_csv(results, out / "leaderboard.csv")
    write_markdown(results, out / "leaderboard.md")

    ok = sum(1 for r in results if r.winner not in ("failed", "unknown"))
    if ok == 0:
        print("No successful experiments.", file=sys.stderr)
        return 1
    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
