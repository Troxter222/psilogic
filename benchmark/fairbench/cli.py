"""Command-line interface for the FairBench optimizer benchmark.

Examples
--------
Full benchmark (all arenas, all optimizers, 3 seeds)::

    python -m fairbench --output-dir results/full

A single arena with W&B logging::

    python -m fairbench --arenas vit --wandb --wandb-project my-bench

Fast smoke test on synthetic data (no downloads, runs on CPU)::

    python -m fairbench --smoke-test

Resume from a saved JSON config (overrides still apply on top)::

    python -m fairbench --config results/full/config.json
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .config import ARENAS, OPTIMIZERS, ArenaConfig, BenchmarkConfig
from .runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fairbench",
        description="Bias-free cross-domain optimizer benchmark (Adam/AdamW/Lion/PsiLogic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # What to run.
    p.add_argument(
        "--arenas", nargs="+", choices=ARENAS, default=list(ARENAS), help="Arenas to run."
    )
    p.add_argument(
        "--optimizers",
        nargs="+",
        choices=OPTIMIZERS,
        default=list(OPTIMIZERS),
        help="Optimizers to compare.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for Stage-2 multi-seed evaluation.",
    )

    # Stage 1: LR sweep.
    p.add_argument(
        "--lr-min", type=float, default=1e-5, help="Lowest LR in the log-spaced sweep grid."
    )
    p.add_argument(
        "--lr-max", type=float, default=1e-2, help="Highest LR in the log-spaced sweep grid."
    )
    p.add_argument("--num-lrs", type=int, default=7, help="Number of LRs in the sweep grid.")
    p.add_argument("--sweep-steps", type=int, default=500, help="Max steps per LR-sweep trial.")
    p.add_argument("--sweep-epochs", type=int, default=5, help="Max epochs per LR-sweep trial.")
    p.add_argument(
        "--no-sweep", action="store_true", help="Skip Stage 1; use --fixed-lr for all optimizers."
    )
    p.add_argument(
        "--fixed-lr",
        type=float,
        default=None,
        help="Use this LR for every optimizer/arena (implies --no-sweep).",
    )

    # Stage 2: training budget.
    p.add_argument(
        "--max-steps", type=int, default=2000, help="Max optimization steps per full run."
    )
    p.add_argument("--max-epochs", type=int, default=10, help="Max epochs per full run.")
    p.add_argument("--eval-every", type=int, default=200, help="Validation interval in steps.")
    p.add_argument(
        "--grad-clip", type=float, default=1.0, help="Global grad-norm clip (0 disables)."
    )
    p.add_argument("--no-scheduler", action="store_true", help="Disable warmup+cosine LR schedule.")
    p.add_argument(
        "--warmup-steps", type=int, default=100, help="Linear warmup steps before cosine decay."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Per-arena batch size (override per arena via --config).",
    )

    # Hardware / performance.
    p.add_argument("--device", default="cuda", help="Device, e.g. 'cuda', 'cuda:0' or 'cpu'.")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    p.add_argument(
        "--amp-dtype", default="float16", choices=["float16", "bfloat16"], help="AMP compute dtype."
    )
    p.add_argument(
        "--no-foreach", action="store_true", help="Disable foreach/batched optimizer kernels."
    )
    p.add_argument("--compile", action="store_true", help="torch.compile the model (PyTorch >= 2).")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")

    # Data / logging.
    p.add_argument("--data-root", default="./data", help="Dataset download/cache root.")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Do not download datasets; require pre-staged files under --data-root.",
    )
    p.add_argument(
        "--download-datasets",
        action="store_true",
        help="Download all datasets to --data-root and exit (run on PC before cloud upload).",
    )
    p.add_argument("--output-dir", default="./results", help="Results output directory.")
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", default="fairbench-optimizers", help="W&B project name.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (team/user).")
    p.add_argument("--no-plots", action="store_true", help="Disable learning-curve plots.")
    p.add_argument("--log-every", type=int, default=10, help="Scalar logging interval in steps.")

    # Convenience / config.
    p.add_argument(
        "--synthetic", action="store_true", help="Use tiny synthetic datasets (no downloads)."
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Minimal end-to-end run on synthetic data (for CI).",
    )
    p.add_argument(
        "--config", default=None, help="Load a BenchmarkConfig JSON; CLI flags override it."
    )

    return p


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Translate parsed CLI args (and optional JSON) into a BenchmarkConfig."""
    cfg = BenchmarkConfig.from_json(args.config) if args.config else BenchmarkConfig()

    cfg.arenas = list(args.arenas)
    cfg.optimizers = list(args.optimizers)
    cfg.seeds = list(args.seeds)

    cfg.sweep.lr_min = args.lr_min
    cfg.sweep.lr_max = args.lr_max
    cfg.sweep.num_lrs = args.num_lrs
    cfg.sweep.max_steps = args.sweep_steps
    cfg.sweep.max_epochs = args.sweep_epochs

    cfg.train.seeds = list(args.seeds)
    cfg.train.max_steps = args.max_steps
    cfg.train.max_epochs = args.max_epochs
    cfg.train.eval_every = args.eval_every
    cfg.train.grad_clip = args.grad_clip
    cfg.train.use_scheduler = not args.no_scheduler
    cfg.train.warmup_steps = args.warmup_steps

    cfg.hardware.device = args.device
    cfg.hardware.amp = not args.no_amp
    cfg.hardware.amp_dtype = args.amp_dtype
    cfg.hardware.use_foreach = not args.no_foreach
    cfg.hardware.compile_model = args.compile
    cfg.hardware.num_workers = args.num_workers

    cfg.logging.output_dir = args.output_dir
    cfg.logging.tensorboard = not args.no_tensorboard
    cfg.logging.wandb = args.wandb
    cfg.logging.wandb_project = args.wandb_project
    cfg.logging.wandb_entity = args.wandb_entity
    cfg.logging.plots = not args.no_plots
    cfg.logging.log_every = args.log_every

    cfg.synthetic = args.synthetic
    cfg.offline = args.offline

    # Per-arena config: shared batch size + data root.
    for arena in cfg.arenas:
        acfg = cfg.arena_config(arena)
        acfg.batch_size = args.batch_size
        acfg.data_root = args.data_root

    # Fixed LR handling.
    if args.fixed_lr is not None or args.no_sweep:
        lr = args.fixed_lr if args.fixed_lr is not None else 1e-3
        cfg.fixed_lrs = {a: dict.fromkeys(cfg.optimizers, lr) for a in cfg.arenas}

    # Smoke test: shrink everything and force synthetic CPU-friendly settings.
    if args.smoke_test:
        cfg.synthetic = True
        cfg.seeds = [0, 1]
        cfg.train.seeds = [0, 1]
        cfg.sweep.explicit_grid = [1e-4, 1e-3]
        cfg.sweep.max_steps = 5
        cfg.sweep.max_epochs = 1
        cfg.train.max_steps = 10
        cfg.train.max_epochs = 1
        cfg.train.eval_every = 5
        cfg.train.warmup_steps = 2
        cfg.logging.log_every = 1
        for arena in cfg.arenas:
            acfg = cfg.arena_config(arena)
            acfg.batch_size = 4
            acfg.extra.update(
                {
                    "img_size": 32,  # vit/resnet: small images
                    "steps_per_epoch": 10,  # nlp
                    "block_size": 32,  # nlp
                    "train_chars": 50_000,
                    "val_chars": 10_000,  # nlp: tiny corpus
                    "n_layer": 2,
                    "n_head": 2,
                    "n_embd": 64,
                    "base_ch": 16,
                    "timesteps": 50,  # diffusion
                }
            )

    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.download_datasets:
        from .datasets import download_all, print_upload_instructions
        from .logging_utils import setup_console_logging

        setup_console_logging()
        download_all(data_root=args.data_root)
        print_upload_instructions(args.data_root)
        return 0

    cfg = config_from_args(args)
    BenchmarkRunner(cfg).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
