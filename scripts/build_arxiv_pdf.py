#!/usr/bin/env python3
"""Build psilogic-arxiv.pdf from arxiv/paper.tex via Tectonic."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARXIV = ROOT / "arxiv"
TECTONIC = ROOT / "tools" / "tectonic" / "tectonic"
TEX = ARXIV / "paper.tex"
OUT = ROOT / "psilogic-arxiv.pdf"
PLOTS = ROOT / "benchmark" / "results" / "full" / "plots"
FIGURES = ARXIV / "figures"
CACHE = ROOT / ".tectonic-cache"

NEEDED_FIGURES = (
    "vit_val_val_acc.png",
    "nlp_val_perplexity.png",
    "resnet_val_val_acc.png",
    "vit_train_step_time_s.png",
)


def sync_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in NEEDED_FIGURES:
        src = PLOTS / name
        dst = FIGURES / name
        if not src.exists():
            raise FileNotFoundError(f"Missing plot: {src}")
        shutil.copy2(src, dst)


def build_pdf() -> None:
    if not TECTONIC.exists():
        raise FileNotFoundError(
            f"Tectonic not found at {TECTONIC}. "
            "Download from https://github.com/tectonic-typesetting/tectonic/releases"
        )
    sync_figures()
    env = os.environ.copy()
    env["TECTONIC_CACHE_DIR"] = str(CACHE)
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(TECTONIC), str(TEX.name)],
        cwd=ARXIV,
        check=True,
        env=env,
    )
    built = ARXIV / "paper.pdf"
    if not built.exists():
        raise FileNotFoundError(f"Tectonic did not produce {built}")
    shutil.copy2(built, OUT)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_pdf()
    sys.exit(0)
