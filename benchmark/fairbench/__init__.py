"""FairBench: a bias-free cross-domain optimizer benchmark.

A modular, publication-grade framework for comparing optimizers
(Adam, AdamW, Lion, PsiLogic) across four heterogeneous arenas
(NLP language modeling, ViT classification, CNN classification and
unconditional diffusion) under a strict Fair-Play protocol:

    Stage 1 -- per-optimizer learning-rate sweep (removes tuning bias).
    Stage 2 -- multi-seed statistical evaluation with identical model init.

The public entry point is :class:`fairbench.runner.BenchmarkRunner`, driven
by the command-line interface in :mod:`fairbench.cli`.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
