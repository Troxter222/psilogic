"""Pre-download all FairBench datasets for offline / cloud use.

Usage::

    # On your PC (fast home internet):
    python -m fairbench.download --data-root ./data

    # Copy ./data to RunPod, then:
    python -m fairbench --data-root /workspace/data --offline

    # Check what is already cached:
    python -m fairbench.download --data-root ./data --check-only
"""

from __future__ import annotations

import argparse
import sys

from .datasets import (
    DEFAULT_TRAIN_CHARS,
    DEFAULT_VAL_CHARS,
    dataset_status,
    download_all,
    print_upload_instructions,
)
from .logging_utils import LOGGER, setup_console_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fairbench.download",
        description="Pre-download FairBench datasets for offline cloud runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="./data", help="Where to store all datasets.")
    p.add_argument(
        "--train-chars",
        type=int,
        default=DEFAULT_TRAIN_CHARS,
        help="TinyStories train byte budget.",
    )
    p.add_argument(
        "--val-chars", type=int, default=DEFAULT_VAL_CHARS, help="TinyStories val byte budget."
    )
    p.add_argument(
        "--skip-celeba",
        action="store_true",
        help="Skip CelebA (~1.3 GB; diffusion uses synthetic).",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    p.add_argument(
        "--check-only", action="store_true", help="Only print dataset readiness; do not download."
    )
    p.add_argument(
        "--no-instructions", action="store_true", help="Skip upload instructions at the end."
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging()

    if args.check_only:
        status = dataset_status(args.data_root)
        for name, ok in status.items():
            LOGGER.info("%s: %s", name, "OK" if ok else "MISSING")
        return 0 if all(status.values()) else 1

    download_all(
        data_root=args.data_root,
        train_chars=args.train_chars,
        val_chars=args.val_chars,
        skip_celeba=args.skip_celeba,
        force=args.force,
    )
    if not args.no_instructions:
        print_upload_instructions(args.data_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
