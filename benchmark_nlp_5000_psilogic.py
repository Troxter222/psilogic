#!/usr/bin/env python3
"""Deprecated wrapper — use FairBench NLP arena instead of this script.

The old standalone TinyStories GPT harness drifted from
``benchmark/fairbench/arenas/nlp.py``. Prefer::

    cd benchmark
    python -m fairbench --arenas nlp --data-root ./data \\
        --max-steps 5000 --seeds 0 1 2 --output-dir results/nlp_5000

This file only forwards to that command for compatibility.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deprecated: forwards to python -m fairbench --arenas nlp"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="results/nlp_5000")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args after -- are passed through to fairbench",
    )
    args = parser.parse_args(argv)

    print(
        "WARNING: benchmark_nlp_5000_psilogic.py is deprecated; "
        "forwarding to FairBench NLP arena.",
        file=sys.stderr,
    )

    benchmark_dir = Path(__file__).resolve().parent / "benchmark"
    cmd = [
        sys.executable,
        "-m",
        "fairbench",
        "--arenas",
        "nlp",
        "--data-root",
        args.data_root,
        "--output-dir",
        args.output_dir,
        "--max-steps",
        str(args.max_steps),
        "--seeds",
        *[str(s) for s in args.seeds],
        *args.extra,
    ]
    return subprocess.call(cmd, cwd=benchmark_dir)


if __name__ == "__main__":
    raise SystemExit(main())
