#!/usr/bin/env python3
"""Audit the archived measurements and regenerate the paper's tables and figures.

This is post-processing only: it never runs training or changes source CSVs.
Requires numpy, scipy and matplotlib. Run from any working directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "arxiv/generated"
FIG = ROOT / "arxiv/figures"
SNAPSHOTS = {
    "reference": ROOT / "benchmark/results/full",
    "later": ROOT / "benchmark/results/fused_h100",
}
OPTS = ("adam", "adamw", "lion", "psilogic")
NAMES = ("Adam", "AdamW", "Lion", "PsiLogic")
ARENAS = ("nlp", "vit", "resnet", "diffusion")
METRICS = {"nlp": "perplexity", "vit": "val_acc", "resnet": "val_acc", "diffusion": "val_loss"}
LABELS = {
    "nlp": "NLP perplexity (lower)",
    "vit": "ViT accuracy (higher)",
    "resnet": "ResNet accuracy (higher)",
    "diffusion": "Diffusion MSE (lower)",
}
COLORS = ("#68788a", "#2678b2", "#c58a24", "#b74853")


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def export_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values, digits):
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def holm(rows):
    """One family: all available arena x baseline primary-quality tests."""
    previous = 0.0
    for rank, row in enumerate(sorted(rows, key=lambda r: r["p_raw"])):
        previous = max(previous, min(1.0, (len(rows) - rank) * row["p_raw"]))
        row["p_holm"] = previous


def pformat(p):
    return f"{p:.3g}"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tables, data, all_stats, inputs = {}, {}, [], []
    for label, directory in SNAPSHOTS.items():
        summary = read(directory / "summary.csv")
        if any(r.get("failed", "").lower() != "false" for r in summary):
            raise ValueError(f"Failed or ambiguous runs in {directory}")
        keyed = {}
        for r in summary:
            key = (r["arena"], r["optimizer"], int(r["seed"]))
            if key in keyed:
                raise ValueError(f"Duplicate measurement: {label} {key}")
            keyed[key] = r
        grouped = defaultdict(list)
        for (arena, opt, _seed), r in sorted(keyed.items()):
            grouped[arena, opt].append(r)
        for key, rows in grouped.items():
            if [int(r["seed"]) for r in rows] != [0, 1, 2]:
                raise ValueError(f"Expected seeds 0,1,2: {label} {key}")
        arenas = [a for a in ARENAS if (a, "psilogic") in grouped]
        for arena in arenas:
            for opt in OPTS:
                if (arena, opt) not in grouped:
                    raise ValueError(f"Missing optimizer {label} {arena} {opt}")
        # Verify the hand-authored paper cannot inherit stale aggregate values.
        for r in read(directory / "aggregate.csv"):
            for field, value in r.items():
                if not value or not field.endswith(("_mean", "_std", "_n")):
                    continue
                metric, kind = field.rsplit("_", 1)
                values = [float(x["final_" + metric]) for x in grouped[r["arena"], r["optimizer"]]]
                expected = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values),
                    "n": len(values),
                }[kind]
                if not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-10):
                    raise ValueError(
                        f"Aggregate mismatch {label}: {r['arena']} {r['optimizer']} {field}"
                    )
        data[label] = grouped
        primary_rows, loss_rows, time_rows, memory_rows, lr_rows = [], [], [], [], []
        tests = []
        for arena in arenas:
            metric = METRICS[arena]
            digits = 2 if arena == "nlp" else 5 if arena == "diffusion" else 4
            vals = {o: [float(r["final_" + metric]) for r in grouped[arena, o]] for o in OPTS}
            best = (min if arena in ("nlp", "diffusion") else max)(
                OPTS, key=lambda o: statistics.mean(vals[o])
            )
            cells = [mean_sd(vals[o], digits) for o in OPTS]
            cells[OPTS.index(best)] = "**" + cells[OPTS.index(best)] + "**"
            primary_rows.append([LABELS[arena], *cells])
            loss_rows.append(
                [
                    arena.upper(),
                    *[
                        mean_sd(
                            [float(r["final_val_loss"]) for r in grouped[arena, o]],
                            5 if arena == "diffusion" else 3,
                        )
                        for o in OPTS
                    ],
                ]
            )
            times = {o: [float(r["final_wall_time_s"]) for r in grouped[arena, o]] for o in OPTS}
            ratio = statistics.mean(times["psilogic"]) / statistics.mean(times["adamw"])
            time_rows.append(
                [arena.upper(), *[mean_sd(times[o], 2) for o in OPTS], f"{ratio:.3f}×"]
            )
            memory_rows.append(
                [
                    arena.upper(),
                    *[
                        mean_sd([float(r["final_peak_vram_mb"]) for r in grouped[arena, o]], 1)
                        for o in OPTS
                    ],
                ]
            )
            lr_rows.append(
                [arena.upper(), *[f"{float(grouped[arena, o][0]['lr']):.3g}" for o in OPTS]]
            )
            for baseline in OPTS[:-1]:
                test = ttest_rel(vals["psilogic"], vals[baseline])
                tests.append(
                    {
                        "snapshot": label,
                        "arena": arena,
                        "metric": metric,
                        "baseline": baseline,
                        "n_pairs": 3,
                        "mean_difference": statistics.mean(vals["psilogic"])
                        - statistics.mean(vals[baseline]),
                        "t_stat": float(test.statistic),
                        "p_raw": float(test.pvalue),
                    }
                )
        holm(tests)
        all_stats.extend(tests)
        tables[label + "_quality"] = markdown(["Primary metric", *NAMES], primary_rows)
        tables[label + "_loss"] = markdown(["Validation loss", *NAMES], loss_rows)
        tables[label + "_time"] = markdown(["Arena", *NAMES, "Psi/AdamW"], time_rows)
        tables[label + "_memory"] = markdown(["Peak allocated MiB", *NAMES], memory_rows)
        tables[label + "_lr"] = markdown(["Arena", *NAMES], lr_rows)
        tables[label + "_stats"] = markdown(
            ["Arena", "vs Adam raw / Holm", "vs AdamW raw / Holm", "vs Lion raw / Holm"],
            [
                [
                    a.upper(),
                    *[
                        f"{pformat(r['p_raw'])} / {pformat(r['p_holm'])}"
                        for r in tests
                        if r["arena"] == a
                    ],
                ]
                for a in arenas
            ],
        )
        tables[label + "_seeds"] = markdown(
            ["Arena / seed", *NAMES],
            [
                [
                    f"{a.upper()} / {seed}",
                    *[f"{float(keyed[a, o, seed]['final_' + METRICS[a]]):.5f}" for o in OPTS],
                ]
                for a in arenas
                for seed in (0, 1, 2)
            ],
        )
        inputs.extend(
            directory / f
            for f in (
                "summary.csv",
                "aggregate.csv",
                "config.json",
                "lr_sweep.csv",
                "significance.csv",
            )
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    for ax, arena in zip(axes, ARENAS[:3]):
        for j, opt in enumerate(OPTS):
            v = [float(r["final_" + METRICS[arena]]) for r in data["later"][arena, opt]]
            ax.errorbar(
                j, np.mean(v), yerr=np.std(v, ddof=1), fmt="o", color=COLORS[j], capsize=4, ms=7
            )
            ax.scatter(np.array([j - 0.12, j, j + 0.12]), v, color=COLORS[j], alpha=0.65, s=18)
        ax.set(title=LABELS[arena], xticks=range(4), xticklabels=NAMES)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.15)
    save(fig, "later_quality")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout="constrained", sharey=True)
    for ax, label in zip(axes, SNAPSHOTS):
        arenas = [a for a in ARENAS if (a, "psilogic") in data[label]]
        for j, opt in enumerate(("adamw", "psilogic")):
            values = [[float(r["final_wall_time_s"]) for r in data[label][a, opt]] for a in arenas]
            ax.bar(
                np.arange(len(arenas)) + (j - 0.5) * 0.34,
                [np.mean(v) for v in values],
                0.34,
                yerr=[np.std(v, ddof=1) for v in values],
                capsize=3,
                label=("AdamW", "PsiLogic")[j],
                color=(COLORS[1], COLORS[3])[j],
            )
        for j, a in enumerate(arenas):
            p = np.mean([float(r["final_wall_time_s"]) for r in data[label][a, "psilogic"]])
            w = np.mean([float(r["final_wall_time_s"]) for r in data[label][a, "adamw"]])
            ax.text(j, max(p, w) + 10, f"{p / w:.3f}x", ha="center", fontsize=9)
        ax.set(
            xticks=range(len(arenas)),
            xticklabels=[a.upper() for a in arenas],
            title="Reference snapshot"
            if label == "reference"
            else "Later partial snapshot (no diffusion)",
            ylim=(0, 225),
        )
        ax.grid(axis="y", alpha=0.15)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Recorded training-run wall time (s)")
    save(fig, "wall_time_snapshots")

    # Only the reference has step histories. Do not invent later learning curves.
    hist = read(SNAPSHOTS["reference"] / "steps.csv")
    inputs.append(SNAPSHOTS["reference"] / "steps.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    for ax, arena in zip(axes.flat, ARENAS):
        metric = METRICS[arena]
        for opt, name, color in zip(OPTS, NAMES, COLORS):
            by_step = defaultdict(dict)
            for r in hist:
                if (
                    r["arena"] == arena
                    and r["optimizer"] == opt
                    and r["split"] == "val"
                    and r.get(metric)
                ):
                    step, seed = int(r["step"]), int(r["seed"])
                    if seed in by_step[step]:
                        raise ValueError(f"Duplicate curve point: {arena} {opt} {step} {seed}")
                    by_step[step][seed] = float(r[metric])
            if not by_step or any(set(v) != {0, 1, 2} for v in by_step.values()):
                raise ValueError(f"Incomplete reference curve: {arena} {opt}")
            x = sorted(by_step)
            y = np.array([list(by_step[s].values()) for s in x])
            avg, sd = y.mean(axis=1), y.std(axis=1, ddof=1)
            ax.plot(x, avg, color=color, label=name, lw=1.8)
            ax.fill_between(x, avg - sd, avg + sd, color=color, alpha=0.12)
        ax.set(title=LABELS[arena], xlabel="Optimizer step (reference snapshot)")
        if arena in ("nlp", "diffusion"):
            ax.set_yscale("log")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False)
    save(fig, "reference_learning_curves")

    profile_path = SNAPSHOTS["later"] / "profile_20260924T174634Z.txt"
    profile_text = profile_path.read_text(encoding="utf-8")
    entries = re.findall(
        r"^(AdamW|PsiLogic \((?:foreach|fused)\)) median step: ([0-9.]+) ms", profile_text, re.M
    )
    if len(entries) != 3 or entries[0][0] != "AdamW":
        raise ValueError("Incomplete saved optimizer-only profile")
    baseline_ms = float(entries[0][1])
    tables["later_profile"] = markdown(
        ["Method", "Saved median step (ms)", "Ratio to AdamW"],
        [[name, f"{float(ms):.3f}", f"{float(ms) / baseline_ms:.2f}×"] for name, ms in entries],
    )
    inputs.extend([profile_path, SNAPSHOTS["later"] / "PROOF.md"])
    export_csv("paired_primary_tests.csv", all_stats)
    manuscript = ROOT / "PAPER.md"
    text = manuscript.read_text(encoding="utf-8")
    for key, table in tables.items():
        (OUT / f"{key}.md").write_text(table + "\n", encoding="utf-8")
        pattern = rf"(<!-- BEGIN GENERATED: {key} -->\n).*?(<!-- END GENERATED: {key} -->)"
        text = re.sub(pattern, lambda m, table=table: m[1] + table + "\n" + m[2], text, flags=re.S)
    manuscript.write_text(text.rstrip() + "\n", encoding="utf-8")
    code = [
        ROOT / p
        for p in (
            "psilogic/optimizer.py",
            "psilogic/_chaos.py",
            "psilogic/_cuda/step.py",
            "psilogic/_cuda/kernels.py",
            "benchmark/fairbench/runner.py",
            "scripts/paper_results.py",
            "scripts/profile_optimizer.py",
            "scripts/run_h100_gate1c.sh",
            "scripts/render_paper.py",
            "scripts/build_arxiv_pdf.py",
        )
    ]
    manifest = {
        "operation": "post-processing of existing measurements; no new training",
        "statistics": "two-sided seed-paired t-tests; Holm across primary-quality tests within each snapshot (reference:12, later:9)",
        "source_revision_warning": "Hashes identify files used for this audit, not the unknown exact revisions that produced the archived runs.",
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs + code
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Audited {sum(len(v) for v in data.values())} arena/optimizer groups; wrote 21 paired comparisons and 3 figures."
    )


if __name__ == "__main__":
    main()
