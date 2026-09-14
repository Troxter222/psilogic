# FairBench

PyTorch harness for comparing Adam, AdamW, Lion, and PsiLogic on four tasks.
Each optimizer gets its own learning-rate search. Runs that share a seed start
from the same weights. Results are mean ± std, with a Welch *t*-test (p-value
and Cohen's *d*).

| Optimizer | Source |
|-----------|--------|
| Adam | `torch.optim.Adam` (coupled L2) |
| AdamW | `torch.optim.AdamW` (decoupled decay) |
| Lion | `lion_pytorch` or `pytorch_optimizer` if installed, otherwise the built-in reference |
| PsiLogic | [`psilogic`](https://pypi.org/project/psilogic/) |

## Arenas

| # | Arena | Task | Model | Dataset |
|---|-------|------|-------|---------|
| 1 | `nlp` | language modeling | small GPT (nanoGPT-style, built-in) | TinyStories (HF `datasets`) |
| 2 | `vit` | image classification | `vit_tiny_patch16_224` (`timm`) | CIFAR-100 @ 224×224 |
| 3 | `resnet` | image classification | ResNet-18/34 (`torchvision`) | Tiny ImageNet (auto-download) |
| 4 | `diffusion` | generative modeling | unconditional DDPM + UNet (built-in) | CelebA @ 64×64 |

If `timm`, `datasets`, or `torchmetrics` is missing, or a download fails, that
arena falls back to a built-in model or a synthetic dataset so the run still
finishes.

## Protocol

Fair-Play has two stages.

**Stage 1, LR sweep.** Short budget (`--sweep-steps`) on a log-spaced grid
(`--lr-min` … `--lr-max`, `--num-lrs`). The LR with the best validation metric
is kept. Other hyperparameters stay at each optimizer's defaults.

**Stage 2, seeds.** That LR, `N` seeds (`--seeds`). For a given seed every
optimizer starts from the same snapshot and sees the same `DataLoader` order.

## What gets logged

Quality: train/val loss, val accuracy (ViT, ResNet), perplexity (GPT), MSE and
optional FID (diffusion).

Timing: per-step and per-epoch wall time, throughput, peak VRAM from
`torch.cuda.max_memory_allocated()`. GPU name and VRAM are printed at startup,
stored in `config.json` under `runtime_hardware`, and written into the CSVs
and plots.

PsiLogic extras in `fairbench/probe.py`: `chaos_t`, `fast_t`, `slow_t`,
`fast_t − slow_t`, spike rate.

Statistics: mean ± std, plus Welch *t*-test of PsiLogic against each baseline.

## Hardware

Single GPU. AMP through `torch.amp.autocast` (`GradScaler` on fp16). Batched
kernels where the optimizer supports `foreach=True` / `use_foreach=True`.

On CUDA OOM the run retries with half the batch size, up to twice. If it still
fails, that cell is recorded as a failure and the rest of the benchmark
continues.

## Outputs

Written under `--output-dir`. The H100 reference run is committed at
`results/full/` and is what README, PAPER, and logs.md cite.

```
results/full/
├── config.json          # full run configuration
├── lr_sweep.csv         # stage 1
├── steps.csv            # per-step metrics, long format
├── summary.csv          # one row per (arena, optimizer, seed)
├── aggregate.csv        # mean ± std over seeds
├── significance.csv     # Welch t-test: p-value, Cohen's d
├── tensorboard/
└── plots/               # learning curves, ±std band
```

TensorBoard and Weights & Biases are optional. W&B uses `group` for the
optimizer and `job_type` for the arena.

## Install

```bash
pip install -r requirements.txt
```

Strictly required: `torch`, `torchvision`, `numpy`, `psilogic`. The rest turn
on individual arenas.

## Usage

### Datasets

Toronto and Stanford mirrors can sit at about 20 KB/s on some pods. CIFAR-100
alone can take hours that way. Download once on a normal connection, upload
the folder, run offline.

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

# on your machine, about 2 GB
python -m fairbench.download --data-root ./data

tar -czf fairbench_data.tar.gz -C ./data .

# on the pod
mkdir -p /workspace/data && tar -xzf fairbench_data.tar.gz -C /workspace/data

python -m fairbench --data-root /workspace/data --offline --output-dir results/full
```

Check the cache:

```bash
python -m fairbench.download --data-root ./data --check-only
```

Expected layout:

```
data/
├── tinystories/          # ~2 MB, pre-tokenized TinyStories subset
├── cifar-100-python/     # ~169 MB
├── tiny-imagenet-200/    # ~600 MB extracted
├── celeba/               # ~1.3 GB
└── manifest.json
```

Same download from the main CLI:

```bash
python -m fairbench --download-datasets --data-root ./data
```

### Runs

```bash
python -m fairbench --output-dir results/full

# one arena, W&B on
python -m fairbench --arenas vit --wandb --wandb-project my-bench

# smoke test, synthetic data, no downloads
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0

# skip the sweep
python -m fairbench --arenas resnet --no-sweep --fixed-lr 1e-3

# LaTeX table (booktabs, best in bold, significance stars)
python -m fairbench.analysis --output-dir results/full --metric val_acc --higher-better
```

`python -m fairbench --help` lists the flags.

## Code

```
fairbench/
├── config.py        # dataclass config
├── optimizers.py    # factory + reference Lion
├── probe.py         # PsiLogic chaos diagnostics
├── metrics.py       # timing, VRAM, CSV, mean±std, Welch t-test
├── logging_utils.py # console, TensorBoard, W&B
├── plotting.py      # learning curves
├── utils.py         # seeding, AMP, schedulers, OOM
├── runner.py        # TrainEngine, LRSweeper, BenchmarkRunner
├── analysis.py      # CSV to LaTeX
├── cli.py
├── models/          # GPT, UNet/DDPM
└── arenas/          # nlp, vit, resnet, diffusion
```

## Fairness notes

Adam uses coupled L2. AdamW, Lion, and PsiLogic use decoupled decay. One
optimizer's regularizer is not copied onto another.

Only the learning rate is swept. Everything else stays at that optimizer's
published defaults. PsiLogic's per-arena presets (γ, chaos τ, and the rest)
are the library presets, held fixed for the run.

## License

MIT.
