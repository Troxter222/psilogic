#!/usr/bin/env python3
"""Regenerate analysis, LaTeX and PDF from the single manuscript source PAPER.md."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from render_paper import render

ROOT = Path(__file__).resolve().parents[1]
ARXIV = ROOT / "arxiv"
TECTONIC = ROOT / "tools" / "tectonic" / "tectonic"
TEX = ARXIV / "paper.tex"
OUTDIR = ROOT / "output" / "pdf"
OUT = OUTDIR / "paper.pdf"
CACHE = ROOT / ".tectonic-cache"


def build_pdf(skip_results: bool = False, tex_only: bool = False) -> None:
    if not skip_results:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/paper_results.py")], cwd=ROOT, check=True
        )
    markdown = (ROOT / "PAPER.md").read_text(encoding="utf-8")
    if re.search(r"<!-- BEGIN GENERATED: [^>]+ -->\s*<!-- END GENERATED:", markdown):
        raise ValueError("An empty generated table remains in PAPER.md")
    TEX.write_text(render(markdown, ROOT), encoding="utf-8")
    if tex_only:
        print(f"Wrote {TEX}")
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory=" + str(OUTDIR),
            TEX.name,
        ]
        for _ in range(2):
            result = subprocess.run(command, cwd=ARXIV, capture_output=True, text=True)
            if result.returncode:
                print(result.stdout[-8000:], file=sys.stderr)
                raise RuntimeError(f"LaTeX failed; see {OUTDIR / 'paper.log'}")
    elif TECTONIC.exists():
        env = os.environ.copy()
        env["TECTONIC_CACHE_DIR"] = str(CACHE)
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(TECTONIC), TEX.name, "--outdir", str(OUTDIR)], cwd=ARXIV, check=True, env=env
        )
    else:
        raise RuntimeError("Install pdflatex or provide tools/tectonic/tectonic")
    if not OUT.exists():
        raise FileNotFoundError(f"Compiler did not produce {OUT}")
    for target in (ARXIV / "paper.pdf", ROOT / "psilogic-arxiv.pdf"):
        shutil.copy2(OUT, target)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-results", action="store_true")
    parser.add_argument("--tex-only", action="store_true")
    args = parser.parse_args()
    build_pdf(args.skip_results, args.tex_only)
    sys.exit(0)
