#!/usr/bin/env python3
"""Big NLP sweep: when / how PsiLogic beats AdamW.

Runs FairBench TinyStories + small-GPT with AdamW vs many PsiLogic configs:

* **regimes** — change train setup (steps, model size, context, LR protocol)
* **ablations** — on a fixed paper-like setup, toggle PsiLogic knobs on/off
  and try value variants (gamma, AGC, centralize, chaos, lion, …)
* **combos** — a few multi-knob presets that often matter together

Outputs a large comparable table (CSV + Markdown + console), sorted by
Δperplexity (AdamW − PsiLogic; positive ⇒ PsiLogic wins).

Examples
--------
List every planned experiment::

    python scripts/nlp_when_psilogic_beats_adamw.py --list

Quick dry-run (synthetic, tiny)::

    python scripts/nlp_when_psilogic_beats_adamw.py --smoke

Full suite (needs TinyStories under --data-root)::

    python scripts/nlp_when_psilogic_beats_adamw.py \\
        --data-root ./data --offline --output-dir ./results/nlp_vs_adamw

Only ablations / only regimes / subset::

    python scripts/nlp_when_psilogic_beats_adamw.py --suite ablations
    python scripts/nlp_when_psilogic_beats_adamw.py --suite regimes
    python scripts/nlp_when_psilogic_beats_adamw.py --only abl_no_agc,paper_2k
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

# FairBench lives under benchmark/, not as an installed package.
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
# Experiment definition
# --------------------------------------------------------------------------- #

# GPT-from-scratch defaults (same as NLPArena.psilogic_kwargs base).
_PSI_BASE: dict[str, Any] = {
    "gamma": 0.02,
    "chaos_tau": 0.40,
    "adaptive_tau": True,
    "tau_scale": 2.0,
    "max_cancel": 0.03,
    "agc_clip": 0.0,
    "grad_centralize": False,
    "quantum_decay": 0.0,
    "p_ext": 1.0,
    "chaos_warmup": -1,
    "gamma_auto": False,
    "lion_mode": False,
}


@dataclass(frozen=True)
class Experiment:
    """One AdamW-vs-PsiLogic comparison cell."""

    name: str
    family: str  # regime | ablation | combo | smoke
    why: str
    max_steps: int = 2000
    batch_size: int = 64
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
    # Merged on top of NLPArena GPT-scratch defaults.
    psi_overrides: dict[str, Any] = field(default_factory=dict)
    # Human-readable toggle summary for the big table.
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
    max_steps: int
    n_layer: int
    n_embd: int
    block_size: int
    fixed_lr: Optional[float]
    psi_overrides_json: str
    output_dir: str


# --------------------------------------------------------------------------- #
# Grid builders
# --------------------------------------------------------------------------- #


def _paper_like(**kwargs: Any) -> dict[str, Any]:
    """Shared kwargs for ablation/combo cells on the paper NLP setup."""
    base = dict(
        max_steps=2000,
        batch_size=32,  # safer on 4–8 GB GPUs (GTX 1650 etc.)
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


def build_regime_experiments() -> list[Experiment]:
    return [
        Experiment(
            name="short_500",
            family="regime",
            why="Very short budget — chaos barely warms up",
            max_steps=500,
            batch_size=32,
            sweep_steps=150,
            num_lrs=5,
            eval_every=100,
            warmup_steps=50,
            toggles="steps=500",
        ),
        Experiment(
            name="paper_2k",
            family="regime",
            why="FairBench / PAPER reference (2k steps, 4L/256d, ctx=128)",
            max_steps=2000,
            batch_size=32,
            toggles="steps=2000 default-psi",
        ),
        Experiment(
            name="long_5k",
            family="regime",
            why="Longer train — late cancel / γ decay matter more",
            max_steps=5000,
            batch_size=32,
            eval_every=250,
            toggles="steps=5000",
        ),
        Experiment(
            name="wider_gpt",
            family="regime",
            why="Larger GPT (6L/6H/384d)",
            max_steps=2000,
            n_layer=6,
            n_head=6,
            n_embd=384,
            batch_size=16,
            toggles="model=6L/384",
        ),
        Experiment(
            name="long_ctx",
            family="regime",
            why="Longer context (block=256)",
            max_steps=2000,
            block_size=256,
            batch_size=16,
            toggles="ctx=256",
        ),
        Experiment(
            name="tiny_model",
            family="regime",
            why="Under-capacity GPT (2L/128d)",
            max_steps=2000,
            n_layer=2,
            n_head=2,
            n_embd=128,
            batch_size=32,
            toggles="model=2L/128",
        ),
        Experiment(
            name="fixed_lr_3e4",
            family="regime",
            why="No LR sweep; shared LR≈3e-4 isolates optimizer",
            max_steps=2000,
            batch_size=32,
            fixed_lr=3.1622776601683794e-4,
            num_lrs=1,
            sweep_steps=1,
            toggles="fixed_lr=3.16e-4",
        ),
        Experiment(
            name="no_scheduler",
            family="regime",
            why="Constant LR after warmup disabled entirely",
            max_steps=2000,
            batch_size=32,
            use_scheduler=False,
            warmup_steps=0,
            toggles="scheduler=off",
        ),
        Experiment(
            name="small_batch",
            family="regime",
            why="Noisier grads (bs=16)",
            max_steps=2000,
            batch_size=16,
            toggles="bs=16",
        ),
    ]


def build_ablation_experiments() -> list[Experiment]:
    """One-factor toggles / value sweeps on the paper_2k train setup."""
    cells: list[tuple[str, str, dict[str, Any]]] = [
        # --- feature on/off ---
        ("abl_baseline", "Default GPT-scratch PsiLogic", {}),
        ("abl_no_centralize", "OFF grad_centralize", {"grad_centralize": False}),
        ("abl_no_agc", "OFF AGC (agc_clip=0)", {"agc_clip": 0.0}),
        ("abl_no_adaptive_tau", "OFF adaptive_tau (absolute chaos_tau)", {"adaptive_tau": False}),
        ("abl_gamma0", "OFF cancel (gamma=0 ≈ AdamW+extras)", {"gamma": 0.0}),
        ("abl_gamma_auto", "ON gamma_auto", {"gamma_auto": True}),
        ("abl_lion", "ON lion_mode (sign-momentum + cancel)", {"lion_mode": True}),
        ("abl_quantum", "ON quantum_decay=2e-4", {"quantum_decay": 2e-4}),
        ("abl_warmup0", "chaos_warmup=0 (chaos from step 0)", {"chaos_warmup": 0}),
        ("abl_warmup_fixed", "chaos_warmup=1000 (late chaos)", {"chaos_warmup": 1000}),
        ("abl_gamma_cosine", "ON cosine γ decay (gamma_T_max=2000)", {"gamma_T_max": 2000}),
        # --- value sweeps ---
        ("abl_gamma_low", "gamma=0.01 (gentle)", {"gamma": 0.01}),
        ("abl_gamma_high", "gamma=0.05 (strong)", {"gamma": 0.05}),
        ("abl_gamma_hot", "gamma=0.08 (aggressive)", {"gamma": 0.08}),
        ("abl_tau_strict", "tau_scale=4.0 (fewer spikes)", {"tau_scale": 4.0}),
        ("abl_tau_loose", "tau_scale=2.0 (more spikes)", {"tau_scale": 2.0}),
        ("abl_cancel_tight", "max_cancel=0.01", {"max_cancel": 0.01}),
        ("abl_cancel_loose", "max_cancel=0.05", {"max_cancel": 0.05}),
        ("abl_pext_half", "p_ext=0.5", {"p_ext": 0.5}),
        ("abl_pext_2", "p_ext=2.0", {"p_ext": 2.0}),
        ("abl_agc_strong", "agc_clip=0.02", {"agc_clip": 0.02}),
        ("abl_chaos_tau_low", "chaos_tau=0.2 (abs mode friend)", {"chaos_tau": 0.2}),
    ]
    out: list[Experiment] = []
    for name, why, ov in cells:
        # Human toggle string from overrides (or "default").
        if not ov:
            toggles = "psi=default"
        else:
            toggles = " ".join(f"{k}={v}" for k, v in sorted(ov.items()))
        out.append(
            Experiment(
                name=name,
                family="ablation",
                why=why,
                toggles=toggles,
                psi_overrides=ov,
                **_paper_like(),
            )
        )
    return out


def build_combo_experiments() -> list[Experiment]:
    """Multi-knob presets worth comparing as a package."""
    combos: list[tuple[str, str, dict[str, Any]]] = [
        (
            "combo_bare",
            "Cancel only: no AGC, no centralize",
            {"agc_clip": 0.0, "grad_centralize": False},
        ),
        (
            "combo_adamish",
            "Extras without cancel (gamma=0, AGC+centralize)",
            {"gamma": 0.0},
        ),
        (
            "combo_stable",
            "Strict tau + low gamma + auto γ",
            {"tau_scale": 4.0, "gamma": 0.01, "gamma_auto": True},
        ),
        (
            "combo_aggressive",
            "High γ + quantum + loose cancel",
            {"gamma": 0.05, "quantum_decay": 2e-4, "max_cancel": 0.05, "p_ext": 1.5},
        ),
        (
            "combo_early_chaos",
            "No warmup + loose tau (chaos early)",
            {"chaos_warmup": 0, "tau_scale": 2.0},
        ),
        (
            "combo_late_chaos",
            "Long warmup + strict tau",
            {"chaos_warmup": 1000, "tau_scale": 4.0},
        ),
        (
            "combo_lion_cancel",
            "Lion update + default cancel",
            {"lion_mode": True},
        ),
        (
            "combo_nlp_preset",
            "Library nlp_defaults-ish on from-scratch GPT",
            {
                "gamma": 0.03,
                "quantum_decay": 2e-4,
                "tau_scale": 2.0,
                "max_cancel": 0.05,
                "agc_clip": 0.01,
            },
        ),
    ]
    out: list[Experiment] = []
    for name, why, ov in combos:
        toggles = " ".join(f"{k}={v}" for k, v in sorted(ov.items()))
        out.append(
            Experiment(
                name=name,
                family="combo",
                why=why,
                toggles=toggles,
                psi_overrides=ov,
                **_paper_like(),
            )
        )
    return out


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
            why="Smoke default psi",
            toggles="psi=default",
            **tiny,
        ),
        Experiment(
            name="smoke_no_agc",
            family="smoke",
            why="Smoke AGC off",
            toggles="agc_clip=0",
            psi_overrides={"agc_clip": 0.0},
            **tiny,
        ),
        Experiment(
            name="smoke_gamma_high",
            family="smoke",
            why="Smoke gamma=0.05",
            toggles="gamma=0.05",
            psi_overrides={"gamma": 0.05},
            **tiny,
        ),
    ]


def all_experiments() -> dict[str, Experiment]:
    items = build_regime_experiments() + build_ablation_experiments() + build_combo_experiments()
    return {e.name: e for e in items}


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
    # Ensure gamma_T_max tracks budget unless the ablation sets it explicitly.
    psi = dict(exp.psi_overrides)
    if "gamma_T_max" not in psi and exp.max_steps > 0:
        # NLPArena does not set gamma_T_max by default; ablations that care
        # about cosine-γ can set it. Leave unset for default arena behavior
        # unless the experiment asked for a specific value.
        pass

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
                    "psilogic_overrides": psi,
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
        max_steps=exp.max_steps,
        n_layer=exp.n_layer,
        n_embd=exp.n_embd,
        block_size=exp.block_size,
        fixed_lr=exp.fixed_lr,
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
        "psi_base": _PSI_BASE,
        "psi_overrides": exp.psi_overrides,
        "experiment": {k: v for k, v in asdict(exp).items()},
    }
    with (out / "experiment.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\n=== [{exp.family}] {exp.name} ===")
    print(f"  {exp.why}")
    print(f"  toggles: {exp.toggles or '(none)'}")
    print(
        f"  steps={exp.max_steps}  GPT={exp.n_layer}L/{exp.n_head}H/{exp.n_embd}d  "
        f"ctx={exp.block_size}  bs={exp.batch_size}"
    )
    BenchmarkRunner(cfg).run()
    if not agg.is_file():
        raise RuntimeError(
            f"Experiment {exp.name} produced no aggregate.csv under {out}. "
            "Usually the arena failed before training (missing dataset / OOM). "
            "Fix the cause and rerun; do not trust leaderboard rows with winner=unknown."
        )
    return summarize_run(exp, out)


def ensure_tinystories(data_root: str, *, offline: bool, synthetic: bool) -> None:
    """Fail fast (or download) so we don't burn a whole suite on missing data."""
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
            "You passed --offline, so the suite cannot download it.\n\n"
            "Either download once (no --offline):\n"
            f"  .venv/bin/python scripts/nlp_when_psilogic_beats_adamw.py --download-only "
            f"--data-root {data_root}\n"
            "or from the benchmark package:\n"
            f"  cd benchmark && python -m fairbench.download --data-root {root}\n"
            "Then rerun your suite with --offline."
        )
    print(f"TinyStories missing under {root}; downloading subset...")
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
    print("\n" + "=" * 120)
    print("PsiLogic vs AdamW on NLP — full comparison  (Δppl = AdamW − Ψ; >0 ⇒ Ψ wins)")
    print("=" * 120)
    print(
        f"{'#':>3} {'name':<22} {'fam':<9} {'win':<9} {'Δppl':>8} "
        f"{'AdamW':>12} {'ΨLogic':>12} {'p':>7}  toggles / why"
    )
    print("-" * 120)
    for i, r in enumerate(ranked, 1):
        aw = f"{r.adamw_ppl:.2f}±{r.adamw_ppl_std:.2f}" if math.isfinite(r.adamw_ppl) else "n/a"
        psi = (
            f"{r.psilogic_ppl:.2f}±{r.psilogic_ppl_std:.2f}"
            if math.isfinite(r.psilogic_ppl)
            else "n/a"
        )
        p = f"{r.p_value:.3f}" if math.isfinite(r.p_value) else "n/a"
        delta = f"{r.delta_ppl:+.3f}" if math.isfinite(r.delta_ppl) else "n/a"
        mark = "✓" if r.winner == "psilogic" else ("·" if r.winner == "tie" else "✗")
        note = r.toggles or r.why
        if len(note) > 42:
            note = note[:39] + "..."
        print(
            f"{i:>3} {r.name:<22} {r.family:<9} {mark}{r.winner:<8} {delta:>8} "
            f"{aw:>12} {psi:>12} {p:>7}  {note}"
        )
    print("=" * 120)

    wins = sum(1 for r in results if r.winner == "psilogic")
    sig = [
        r
        for r in results
        if r.winner == "psilogic" and math.isfinite(r.p_value) and r.p_value < 0.05
    ]
    print(f"PsiLogic wins {wins}/{len(results)} cells (mean perplexity).")
    print(f"Significant wins (paired p<0.05): {len(sig)}/{len(results)}")

    # Per-family breakdown.
    families = sorted({r.family for r in results})
    for fam in families:
        sub = [r for r in results if r.family == fam]
        w = sum(1 for r in sub if r.winner == "psilogic")
        best = max(sub, key=lambda r: r.delta_ppl if math.isfinite(r.delta_ppl) else -1e9)
        print(
            f"  [{fam}] Ψ wins {w}/{len(sub)}; "
            f"best Δ={best.delta_ppl:+.3f} @ {best.name} ({best.toggles or best.why})"
        )


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
        "# PsiLogic vs AdamW — NLP experiment matrix",
        "",
        "Sorted by **Δperplexity = AdamW − PsiLogic** (positive ⇒ PsiLogic better).",
        "",
        "| # | name | family | winner | Δppl | AdamW PPL | ΨLogic PPL | p | toggles | why |",
        "|--:|------|--------|--------|-----:|----------:|-----------:|--:|---------|-----|",
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
        lines.append(
            f"| {i} | `{r.name}` | {r.family} | **{r.winner}** | {delta} | {aw} | {psi} | {p} | "
            f"{r.toggles} | {r.why} |"
        )
    wins = sum(1 for r in results if r.winner == "psilogic")
    lines += [
        "",
        f"**PsiLogic wins:** {wins}/{len(results)}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Markdown: {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    catalog = all_experiments()
    p = argparse.ArgumentParser(
        description="Large NLP AdamW vs PsiLogic matrix (regimes + ablations + combos).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--suite",
        choices=["all", "regimes", "ablations", "combos"],
        default="all",
        help="Which experiment families to run.",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment names (overrides --suite).",
    )
    p.add_argument("--list", action="store_true", help="List experiments and exit.")
    p.add_argument("--smoke", action="store_true", help="Tiny synthetic 3-cell dry run.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data-root", default=str(_REPO / "data"))
    p.add_argument("--output-dir", default=str(_REPO / "results" / "nlp_vs_adamw"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Download TinyStories into --data-root and exit (needed before --offline).",
    )
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        help="Cap how many experiments to run (0 = no cap). Useful for overnight slices.",
    )
    p.add_argument(
        "--catalog-size",
        action="store_true",
        help="Print how many experiments --suite all would run and exit.",
    )
    # stash for --list help text
    p.set_defaults(_catalog=catalog)
    return p.parse_args(argv)


def select_experiments(args: argparse.Namespace) -> dict[str, Experiment]:
    if args.smoke:
        return {e.name: e for e in build_smoke_experiments()}

    catalog = all_experiments()
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        missing = [n for n in wanted if n not in catalog]
        if missing:
            raise SystemExit(f"Unknown experiments: {missing}\nAvailable: {', '.join(catalog)}")
        return {n: catalog[n] for n in wanted}

    if args.suite == "regimes":
        return {e.name: e for e in build_regime_experiments()}
    if args.suite == "ablations":
        return {e.name: e for e in build_ablation_experiments()}
    if args.suite == "combos":
        return {e.name: e for e in build_combo_experiments()}
    return catalog


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    catalog = all_experiments()

    if args.catalog_size:
        print(
            f"regimes={len(build_regime_experiments())}  "
            f"ablations={len(build_ablation_experiments())}  "
            f"combos={len(build_combo_experiments())}  "
            f"all={len(catalog)}"
        )
        return 0

    if args.list:
        print(f"{'name':<22} {'family':<9} {'steps':>5}  toggles / why")
        print("-" * 100)
        for e in catalog.values():
            note = e.toggles or e.why
            print(f"{e.name:<22} {e.family:<9} {e.max_steps:>5}  {note}")
        print("-" * 100)
        print(
            f"Total: {len(catalog)} "
            f"(regimes={len(build_regime_experiments())}, "
            f"ablations={len(build_ablation_experiments())}, "
            f"combos={len(build_combo_experiments())})"
        )
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

    # Real NLP runs need TinyStories (or --synthetic). Fail before the suite.
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

    print(f"Planning {len(experiments)} experiments | seeds={args.seeds} | out={args.output_dir}")

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
                    max_steps=exp.max_steps,
                    n_layer=exp.n_layer,
                    n_embd=exp.n_embd,
                    block_size=exp.block_size,
                    fixed_lr=exp.fixed_lr,
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
        print(
            "\nNo successful experiments. Likely missing TinyStories or every run crashed.\n"
            f"  .venv/bin/python scripts/nlp_when_psilogic_beats_adamw.py --download-only "
            f"--data-root {args.data_root}",
            file=sys.stderr,
        )
        return 1
    if failures:
        print(f"\nCompleted with {failures} failed experiment(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
