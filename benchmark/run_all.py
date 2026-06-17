"""
One-command reproduction of the PsiLogic reference benchmark suite.

    python benchmark/run_all.py --suite v1
    python benchmark/run_all.py --suite v1 --runs 5 --steps 3000
    python benchmark/run_all.py --suite v1 --imagenet-data /datasets/imagenet

Suite "v1" covers the in-harness arenas:

    1. CIFAR-10 / ResNet-18      (cifar10)
    2. CIFAR-100 / ViT-Small     (vit)
    4. BERT-base / SST-2         (bert)
    5. GPT-2 / WikiText-2        (gpt2)
    7. nanoGPT / Tiny Shakespeare (nanogpt)

Arena 3 (ImageNet-1k / ResNet-50) and arena 6 (GPT-2 medium / OpenWebText)
require dedicated multi-GPU runs; arena 3 is launched automatically through
``benchmark/imagenet/train_imagenet.py`` when ``--imagenet-data`` is given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from run_benchmark import format_table, run_benchmark, save_json

SUITES: dict[str, list[str]] = {
    "v1": ["cifar10", "vit", "bert", "gpt2", "nanogpt"],
    "quick": ["cifar10", "nanogpt"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full PsiLogic benchmark suite")
    parser.add_argument("--suite", choices=sorted(SUITES), default="v1")
    parser.add_argument("--optimizers", nargs="+", default=["adam", "adamw", "lion", "psilogic"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-lr-search", action="store_true")
    parser.add_argument("--lr-grid", nargs="+", type=float, default=None)
    parser.add_argument("--lr-tune-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./results/suite"))
    parser.add_argument(
        "--preset", choices=["task", "vit", "auto"], default="task", dest="psilogic_preset"
    )
    parser.add_argument(
        "--imagenet-data",
        type=Path,
        default=None,
        help="Path to ImageNet-1k; enables arena 3 (ResNet-50)",
    )
    parser.add_argument("--imagenet-epochs", type=int, default=90)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_agg, _ = run_benchmark(
        tasks=SUITES[args.suite],
        optimizers=args.optimizers,
        n_runs=args.runs,
        total_steps=args.steps,
        output_dir=args.output_dir,
        psilogic_preset=args.psilogic_preset,
        tune_lr=not args.no_lr_search,
        lr_grid=args.lr_grid,
        lr_tune_steps=args.lr_tune_steps,
    )

    table = format_table(all_agg, title=f"Suite {args.suite} results")
    print(
        "\n"
        + table.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
    )
    save_json([asdict(a) for a in all_agg], args.output_dir / "suite_results.json")

    if args.imagenet_data is not None:
        script = Path(__file__).parent / "imagenet" / "train_imagenet.py"
        for optimizer in args.optimizers:
            if optimizer == "lion":
                continue  # arena 3 reference baselines: adamw vs psilogic
            cmd = [
                sys.executable,
                str(script),
                "--data-dir",
                str(args.imagenet_data),
                "--model",
                "resnet50",
                "--optimizer",
                optimizer,
                "--epochs",
                str(args.imagenet_epochs),
                "--output-dir",
                str(args.output_dir / "imagenet"),
            ]
            print(f"\n[suite] launching arena 3: {' '.join(cmd)}")
            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                print(f"[suite] arena 3 run failed for {optimizer} (exit {completed.returncode})")
                return completed.returncode
    else:
        print(
            "\n[suite] Arena 3 (ImageNet/ResNet-50) skipped — pass --imagenet-data. "
            "Arena 6 (GPT-2 medium/OpenWebText) requires a dedicated multi-GPU run."
        )

    summary_path = args.output_dir / "suite_results.json"
    print(f"\n[suite] Done. Aggregated results: {summary_path}")
    if summary_path.exists():
        with open(summary_path) as f:
            print(json.dumps(json.load(f), indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
