"""Metric collection, hardware profiling, statistics and CSV export.

This module provides three concerns kept deliberately separate:

* :class:`StepTimer` / :func:`peak_vram_mb` -- wall-clock and memory profiling.
* :class:`RunRecord` / :class:`ResultStore` -- structured, CSV-backed results.
* :func:`aggregate` / :func:`welch_ttest` -- Mean+/-Std and significance tests
  that turn raw per-seed numbers into paper-ready statistics.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------- #
# Hardware profiling
# --------------------------------------------------------------------------- #


def cuda_sync(device: Optional[torch.device] = None) -> None:
    """Synchronize CUDA so host-side timers measure real GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


class StepTimer:
    """Accurate wall-clock timing for training steps/epochs under CUDA async.

    Use as a context manager around the region to time::

        with StepTimer(device) as t:
            ... train step ...
        seconds = t.elapsed
    """

    def __init__(self, device: Optional[torch.device] = None, sync: bool = True):
        self.device = device
        self.sync = sync
        self.elapsed: float = 0.0

    def __enter__(self) -> StepTimer:
        if self.sync:
            cuda_sync(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.sync:
            cuda_sync(self.device)
        self.elapsed = time.perf_counter() - self._start


def reset_peak_vram(device: Optional[torch.device] = None) -> None:
    """Reset the CUDA peak-memory counter before a measured region."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device: Optional[torch.device] = None) -> float:
    """Peak allocated CUDA memory in MiB since the last reset (0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return 0.0


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #


@dataclass
class StepRecord:
    """A single logged training/eval step (the long-format CSV row)."""

    arena: str
    optimizer: str
    seed: int
    lr: float
    step: int
    epoch: int
    split: str  # "train" | "val"
    metrics: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        row = {
            "arena": self.arena,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "lr": self.lr,
            "step": self.step,
            "epoch": self.epoch,
            "split": self.split,
        }
        row.update(self.metrics)
        return row


@dataclass
class RunRecord:
    """Final summary of one (arena, optimizer, seed) training run."""

    arena: str
    optimizer: str
    seed: int
    lr: float
    final_metrics: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        row = {
            "arena": self.arena,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "lr": self.lr,
        }
        row.update({f"final_{k}": v for k, v in self.final_metrics.items()})
        return row


class CSVLogger:
    """Append-only CSV writer that tolerates a growing set of metric columns.

    The header is rewritten if a new metric key appears, so heterogeneous
    arenas (each with their own metrics) can share one file safely.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fieldnames: list[str] = []
        self._rows: list[dict[str, Any]] = []
        if os.path.exists(path):
            os.remove(path)

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        for k in row:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        self._flush()

    def _flush(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@dataclass
class Aggregate:
    """Mean +/- standard deviation over seeds for a single metric."""

    mean: float
    std: float
    n: int
    values: list[float] = field(default_factory=list)

    def as_str(self, fmt: str = ".4f") -> str:
        return f"{self.mean:{fmt}} +/- {self.std:{fmt}}"


def aggregate(values: Sequence[float]) -> Aggregate:
    """Compute sample mean and (unbiased, ddof=1) std over seeds."""
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(vals)
    if n == 0:
        return Aggregate(float("nan"), float("nan"), 0, [])
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return Aggregate(mean, std, n, vals)


@dataclass
class TTestResult:
    """Outcome of a two-sample test comparing two optimizers on one metric."""

    t_stat: float
    p_value: float
    df: float
    cohens_d: float
    significant: bool


def welch_ttest(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> TTestResult:
    """Welch's two-sample t-test (unequal variances) with Cohen's d effect size.

    Welch's variant is preferred over Student's pooled test because optimizer
    runs routinely have unequal variance across seeds. Uses ``scipy`` when
    available for an exact p-value, otherwise falls back to a normal
    approximation of the survival function (adequate for reporting trends).
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return TTestResult(float("nan"), float("nan"), float("nan"), float("nan"), False)

    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0.0:
        # Identical-variance, different-mean edge case.
        t = float("inf") if ma != mb else 0.0
        return TTestResult(t, 0.0 if ma != mb else 1.0, float(na + nb - 2), float("inf"), ma != mb)

    t = (ma - mb) / se
    # Welch-Satterthwaite degrees of freedom.
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))

    try:
        from scipy import stats  # type: ignore

        p = float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        # Normal approximation: p = 2 * (1 - Phi(|t|)).
        p = math.erfc(abs(t) / math.sqrt(2.0))

    # Pooled-SD Cohen's d effect size.
    pooled_sd = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    d = (ma - mb) / pooled_sd if pooled_sd > 0 else float("inf")
    return TTestResult(t, p, df, d, p < alpha)


@dataclass
class PairedTTestResult:
    """Outcome of a paired two-sample test (same seeds/inits on both sides)."""

    t_stat: float
    p_value: float
    df: float
    cohens_dz: float
    n_pairs: int
    significant: bool


def paired_ttest(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> PairedTTestResult:
    """Paired t-test on ``a - b`` for two optimizers sharing seeds/inits.

    The benchmark trains every optimizer from an identical per-seed init on
    an identical seed, so ``a[i]`` and ``b[i]`` are not independent draws --
    they are the same seed's outcome under two optimizers. A paired test
    (on the per-seed differences) is the correct choice here; an unpaired
    test such as Welch's ignores the pairing and understates significance
    when per-seed difficulty varies (which it does: some seeds are easy for
    every optimizer, some are hard for every optimizer).

    ``a`` and ``b`` must already be aligned by seed (``a[i]`` and ``b[i]``
    come from the same seed) and equal length; non-finite entries in either
    position are dropped as a pair before the test runs.
    """
    if len(a) != len(b):
        raise ValueError(
            f"paired_ttest requires equal-length, seed-aligned inputs, got {len(a)} vs {len(b)}"
        )

    pairs = [
        (float(x), float(y))
        for x, y in zip(a, b)
        if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    n = len(pairs)
    if n < 2:
        return PairedTTestResult(float("nan"), float("nan"), float("nan"), float("nan"), n, False)

    diffs = [x - y for x, y in pairs]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var_diff)
    df = float(n - 1)

    if std_diff == 0.0:
        # Every seed moved the same direction by the same amount.
        if mean_diff == 0.0:
            return PairedTTestResult(0.0, 1.0, df, 0.0, n, False)
        return PairedTTestResult(float("inf"), 0.0, df, float("inf"), n, True)

    se = std_diff / math.sqrt(n)
    t = mean_diff / se

    try:
        from scipy import stats  # type: ignore

        p = float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        p = math.erfc(abs(t) / math.sqrt(2.0))

    dz = mean_diff / std_diff
    return PairedTTestResult(t, p, df, dz, n, p < alpha)


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down correction for a family of p-values.

    Returns ``(adjusted_pvalues, reject)`` in the same order as ``pvalues``.
    Holm controls the family-wise error rate (like Bonferroni) but is
    uniformly more powerful, and unlike Benjamini-Hochberg it bounds the
    probability of *any* false positive rather than the expected false
    discovery *proportion* -- the appropriate default when every row in the
    significance table is read as an individual claim ("psilogic beats X on
    metric Y") rather than a screened candidate list where a few false
    positives are tolerable. NaN p-values (e.g. a metric with <2 valid
    pairs) pass through as NaN/not-rejected and are excluded from the
    family size used to scale the correction.
    """
    m_all = len(pvalues)
    valid_idx = [i for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    m = len(valid_idx)

    adjusted: list[float] = [float("nan")] * m_all
    reject: list[bool] = [False] * m_all
    if m == 0:
        return adjusted, reject

    order = sorted(valid_idx, key=lambda i: pvalues[i])
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvalues[idx] * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
        reject[idx] = adjusted[idx] < alpha
    return adjusted, reject
