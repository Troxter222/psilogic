# FairBench — Bias-Free Cross-Domain Optimizer Benchmark

A modular, publication-grade PyTorch framework for a **fair** comparison of four
optimizers across four heterogeneous deep-learning arenas:

| Optimizer | Source |
|-----------|--------|
| **Adam**     | `torch.optim.Adam` (coupled L2) |
| **AdamW**    | `torch.optim.AdamW` (decoupled decay) |
| **Lion**     | `lion_pytorch` / `pytorch_optimizer` if installed, else a clean built-in reference impl |
| **PsiLogic** | [`psilogic`](https://pypi.org/project/psilogic/) — Active-Cancellation optimizer |

The design goal is to **eliminate tuning bias** and meet the methodological bar
of NeurIPS/ICLR submissions: every optimizer gets its own learning-rate search,
all runs start from *identical* weights per seed, results are reported as
**Mean ± Std** over multiple seeds, and differences are checked with a
**Welch t-test** (p-value + Cohen's *d*).

---

## 1. Arenas

| # | Arena | Task | Model | Dataset |
|---|-------|------|-------|---------|
| 1 | `nlp`       | Language modeling   | small GPT (nanoGPT-style, built-in) | TinyStories (HF `datasets`) |
| 2 | `vit`       | Image classification| `vit_tiny_patch16_224` (`timm`)     | CIFAR-100 @ 224×224 |
| 3 | `resnet`    | Image classification| ResNet-18/34 (`torchvision`)        | Tiny ImageNet (auto-download) |
| 4 | `diffusion` | Generative modeling | unconditional DDPM + UNet (built-in)| CelebA @ 64×64 |

Every arena degrades gracefully: if an optional dependency (`timm`, `datasets`,
`torchmetrics`) or a dataset download is unavailable, a built-in model fallback
or a synthetic dataset is used so the benchmark always completes.

## 2. Fair-Play protocol

**Stage 1 — LR sweep (per optimizer).** A short budget (`--sweep-steps`) over a
log-spaced LR grid (`--lr-min … --lr-max`, `--num-lrs`). The LR with the best
validation metric is selected. This removes per-optimizer tuning bias.

**Stage 2 — Multi-seed evaluation.** Using the selected LR, each optimizer is
trained over `N` seeds (`--seeds`). For a given seed, **all optimizers start
from the same initial weights** (snapshotted once and reloaded per optimizer)
and see the **same data order** (seeded `DataLoader`).

## 3. Metrics (paper-ready)

* **Quality:** train/val loss, val accuracy (ViT/ResNet), perplexity (GPT),
  MSE loss & optional **FID** (diffusion).
* **Performance:** per-step / per-epoch wall-clock time, throughput, and peak
  VRAM via `torch.cuda.max_memory_allocated()`.
* **Hardware provenance:** detected GPU name and VRAM are printed at startup,
  logged per run, saved in `config.json` (`runtime_hardware`) and written to
  every CSV/plot for reproducibility.
* **PsiLogic diagnostics:** `chaos_t`, `fast_t`, `slow_t`, `fast_t − slow_t`
  and spike rate over time (see `fairbench/probe.py`).
* **Statistics:** Mean ± Std plus a Welch t-test (PsiLogic vs each baseline).

## 4. Hardware / performance

* Single-GPU oriented; **AMP** via `torch.amp.autocast` (+ `GradScaler` for fp16).
* `foreach=True` / `use_foreach=True` batched optimizer kernels where supported.
* Automatic **CUDA OOM handling**: a run is retried with a halved batch size
  (up to 2×) and, if still failing, recorded as a failure without aborting the
  benchmark.

## 5. Logging & outputs

Written under `--output-dir`. The reference H100 run is committed at
`results/full/` (used by README, PAPER, and logs.md).

```
results/full/
├── config.json          # the full, reproducible run configuration
├── lr_sweep.csv         # Stage-1 trial results
├── steps.csv            # long-format per-step metrics (Pandas-ready)
├── summary.csv          # per (arena, optimizer, seed) final metrics
├── aggregate.csv        # Mean ± Std over seeds
├── significance.csv     # Welch t-test: p-value, Cohen's d
├── tensorboard/         # grouped arena/optimizer/seed
└── plots/               # learning curves with ±std shaded bands
```

TensorBoard and **Weights & Biases** are optional; W&B groups runs by optimizer
(`group`) and arena (`job_type`).

## 6. Installation

```bash
pip install -r requirements.txt
```

Only `torch`, `torchvision`, `numpy` and `psilogic` are strictly required; the
rest unlock individual arenas / features.

## 7. Usage

### Pre-download datasets (recommended for RunPod / slow cloud links)

Toronto / Stanford mirrors can crawl at **~20 KB/s** on some pods (CIFAR-100
would take hours). Download once on your PC, upload the folder, then run offline.

```bash
# Step 0 - set up variable
export HF_HUB_ENABLE_HF_TRANSFER=1

# Step 1 — on your PC (fast home internet), ~2 GB total:
python -m fairbench.download --data-root ./data

# Step 2 — archive and upload to the pod:
tar -czf fairbench_data.tar.gz -C ./data .

# Step 3 — on RunPod / Jupyter:
mkdir -p /workspace/data && tar -xzf fairbench_data.tar.gz -C /workspace/data

# Step 4 — benchmark with zero network downloads:
python -m fairbench --data-root /workspace/data --offline --output-dir results/full
```

Check what is cached:

```bash
python -m fairbench.download --data-root ./data --check-only
```

Expected layout under `data/`:

```
data/
├── tinystories/          # ~2 MB (pre-tokenized TinyStories subset)
├── cifar-100-python/     # ~169 MB
├── tiny-imagenet-200/    # ~600 MB extracted
├── celeba/               # ~1.3 GB
└── manifest.json
```

Alternative one-liner from the main CLI:

```bash
python -m fairbench --download-datasets --data-root ./data
```

### Full benchmark

```bash
python -m fairbench --output-dir results/full

# A single arena, with W&B
python -m fairbench --arenas vit --wandb --wandb-project my-bench

# Fast end-to-end smoke test on synthetic data (no downloads, CPU-friendly)
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0

# Skip the sweep and pin a learning rate
python -m fairbench --arenas resnet --no-sweep --fixed-lr 1e-3

# Generate a LaTeX results table (booktabs, bold-best, significance stars)
python -m fairbench.analysis --output-dir results/full --metric val_acc --higher-better
```

Run `python -m fairbench --help` for the full list of flags.

## 8. Code layout

```
fairbench/
├── config.py        # typed dataclass configuration
├── optimizers.py    # optimizer factory + reference Lion
├── probe.py         # PsiLogic chaos diagnostics (chaos_t, fast/slow)
├── metrics.py       # timing, VRAM, CSV, Mean±Std, Welch t-test
├── logging_utils.py # console + TensorBoard + W&B
├── plotting.py      # learning curves with ±std bands
├── utils.py         # seeding, AMP, schedulers, OOM detection
├── runner.py        # TrainEngine, LRSweeper, BenchmarkRunner
├── analysis.py      # CSV -> LaTeX tables
├── cli.py           # command-line interface
├── models/          # GPT, UNet/DDPM
└── arenas/          # base + nlp / vit / resnet / diffusion adapters
```

## 9. Reproducibility & fairness notes

* Each optimizer runs in its **canonical** form (Adam = coupled L2,
  AdamW/Lion/PsiLogic = decoupled decay); we deliberately do not retrofit one
  algorithm's regularizer onto another.
* Only the **learning rate** is tuned; all other hyperparameters are each
  optimizer's published defaults, held constant across the benchmark.
* PsiLogic per-arena presets (γ, chaos τ, …) mirror the library's published
  architecture presets and are likewise held fixed.

## License

MIT.
