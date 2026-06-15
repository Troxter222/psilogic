# Benchmark Harness

Reproducible multi-arena evaluation of PsiLogic against AdamW, Lion, and other baselines.

## Install

```bash
pip install -e ".[benchmark]"
```

Requires a CUDA GPU for full runs. CPU works for smoke tests with `--steps 10`.

## Quick Start

```bash
# From repository root

# One arena
python benchmark/run_benchmark.py --task cifar10 --runs 3 --optimizers adamw lion psilogic

# Full v1 suite
python benchmark/run_all.py --suite v1

# Quick smoke suite (CIFAR-10 + nanoGPT only)
python benchmark/run_all.py --suite quick --steps 50
```

## Arenas

| Task flag | Model | Dataset | Notes |
|-----------|-------|---------|-------|
| `cifar10` | ResNet-18 | CIFAR-10 | Primary vision benchmark |
| `vit` | ViT-Small | CIFAR-100 | Use `--preset vit` for PsiLogicViT |
| `bert` | BERT-base | SST-2 | Requires `transformers` + `datasets` |
| `gpt2` | GPT-2 | WikiText-2 | From-scratch LM |
| `nanogpt` | char-GPT | Tiny Shakespeare | Low-variance reproducibility check |

## Key Flags

| Flag | Description |
|------|-------------|
| `--task` | Arena name (see table above) |
| `--runs` | Number of independent seeds |
| `--steps` | Training steps (LM tasks) |
| `--epochs` | Training epochs (vision / BERT) |
| `--optimizers` | Space-separated list: `adamw`, `lion`, `psilogic`, `adam`, `sgd` |
| `--preset` | PsiLogic preset: `task`, `vit`, or `auto` |
| `--output-dir` | JSON results directory |

## ImageNet (Arena 3)

Multi-GPU DDP training via `imagenet/train_imagenet.py`:

```bash
torchrun --nproc_per_node=4 benchmark/imagenet/train_imagenet.py \
    --data-dir /path/to/imagenet --model resnet50 --optimizer psilogic --epochs 90
```

Or pass `--imagenet-data` to `run_all.py` to include it in a suite run.

## Ablation Scripts

| Script | Purpose |
|--------|---------|
| `gc_agc_ablation.py` | Gradient centralization / AGC ablation |
| `mirror_ablation.py` | Mirror-term ablation study |

These are standalone experiments — run with `python benchmark/<script>.py --help`.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PSILOGIC_TG_TOKEN` | Optional Telegram bot token for run notifications |
| `PSILOGIC_TG_CHAT` | Telegram chat ID for notifications |

## Output

Results are saved as JSON under `--output-dir` (default `./results/`). Use `run_benchmark.py --list-tasks` to see all available arenas.
