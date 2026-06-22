"""Learning-curve plotting with mean +/- std confidence bands.

Consumes the long-format step CSV produced by the benchmark and renders one
figure per (arena, metric): each optimizer is a line (mean over seeds) with a
shaded +/-1 std band. Matplotlib is optional -- if it is not installed the
functions log a warning and return without error.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .logging_utils import LOGGER


def _load_rows(csv_path: str) -> list[dict]:
    import csv

    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(x: Optional[str]) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v
    except (TypeError, ValueError):
        return None


def plot_learning_curves(
    step_csv: str,
    output_dir: str,
    split: str = "val",
    metrics: Optional[list[str]] = None,
) -> list[str]:
    """Render learning curves with shaded std bands from a step CSV.

    Args:
        step_csv: Long-format CSV with columns
            ``arena, optimizer, seed, step, split, <metrics...>``.
        output_dir: Directory to write PNGs into.
        split: Which split to plot (``"val"`` or ``"train"``).
        metrics: Restrict to these metric columns; ``None`` plots all numeric
            metrics found.

    Returns:
        List of written PNG file paths.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend for servers
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dep
        LOGGER.warning("Plotting disabled (matplotlib/numpy missing: %s).", exc)
        return []

    if not os.path.exists(step_csv):
        LOGGER.warning("Step CSV not found for plotting: %s", step_csv)
        return []

    rows = [r for r in _load_rows(step_csv) if r.get("split") == split]
    if not rows:
        return []

    reserved = {"arena", "optimizer", "seed", "lr", "step", "epoch", "split", "gpu_name", "device"}
    gpu_label = next((r.get("gpu_name") for r in rows if r.get("gpu_name")), None)
    if metrics is None:
        metrics = sorted(
            {
                k
                for r in rows
                for k in r
                if k not in reserved and not k.startswith("psi/") and _to_float(r[k]) is not None
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    arenas = sorted({r["arena"] for r in rows})

    for arena in arenas:
        for metric in metrics:
            # series[optimizer][step] -> list of values across seeds
            series: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
            for r in rows:
                if r["arena"] != arena:
                    continue
                val = _to_float(r.get(metric))
                step = _to_float(r.get("step"))
                if val is None or step is None:
                    continue
                series[r["optimizer"]][int(step)].append(val)

            if not series:
                continue

            fig, ax = plt.subplots(figsize=(7, 4.5))
            plotted = False
            for optimizer in sorted(series):
                steps = sorted(series[optimizer])
                means = np.array([np.mean(series[optimizer][s]) for s in steps])
                stds = np.array([np.std(series[optimizer][s], ddof=0) for s in steps])
                steps_arr = np.array(steps)
                ax.plot(steps_arr, means, label=optimizer, linewidth=1.8)
                ax.fill_between(steps_arr, means - stds, means + stds, alpha=0.18)
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            title = f"{arena} — {split} {metric} (mean ± std)"
            if gpu_label:
                title += f"\nGPU: {gpu_label}"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            safe_metric = metric.replace("/", "_")
            out_path = os.path.join(output_dir, f"{arena}_{split}_{safe_metric}.png")
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            written.append(out_path)

    return written
