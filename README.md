<div align="center">

# ΨLogic

### Active Cancellation Optimizer for Deep Neural Networks

[![PyPI version](https://img.shields.io/pypi/v/psilogic.svg?cache=1)](https://pypi.org/project/psilogic/)
[![CI](https://github.com/Troxter222/psilogic/actions/workflows/ci.yml/badge.svg)](https://github.com/Troxter222/psilogic/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/psilogic)](https://pypi.org/project/psilogic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.xxxxx-B31B1B.svg)](https://arxiv.org/abs/2607.xxxxx)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18739857.svg)](https://doi.org/10.5281/zenodo.18739857)

```
dΨ/dt = -iĤ·Ψ  −  γ·P·chaos(S_t)·Ψ
         └──────┘   └───────────────┘
          Gradient   Active Cancellation
```

**ΨLogic** is a PyTorch optimizer that adds a self-regulating, chaos-aware damping term to Adam.
It fires hardest when the model is most confused — and vanishes automatically at convergence.
No warmup schedule needed. One-line drop-in for `torch.optim.Adam`.

Tested against **Adam, AdamW, and Lion** on FairBench — four cross-domain arenas on NVIDIA H100.

</div>

---

## Install

```bash
pip install psilogic

# Development, benchmarks, or framework integrations
pip install -e ".[dev]"              # tests + lint
pip install -e ".[integrations]"       # HuggingFace + Lightning
pip install -e ".[benchmark]"          # full benchmark harness
pip install -e ".[cuda]"               # Triton fused GPU step (Linux/Windows + CUDA)
pip install -e ".[all]"                # everything
```

## Project Structure

```
psilogic-pkg/
├── psilogic/                  # installable package
│   ├── optimizer.py           # PsiLogic core optimizer
│   ├── _chaos.py              # chaos detector, warmup, metrics
│   ├── _cuda/                 # optional Triton fused CUDA step backend
│   ├── param_groups.py        # per-architecture param groups (NLP / ViT / GPT)
│   ├── presets.py             # task-specific default hyperparameters
│   ├── convenience.py         # PsiLogicNLP/GPT/ViT/Whisper + auto-config
│   ├── debug.py               # chaos_stats, norm_history diagnostics
│   └── integrations/          # optional HuggingFace & Lightning hooks
├── tests/                     # pytest suite (DDP, AMP, presets, migrations)
├── benchmark/                 # FairBench — bias-free cross-domain harness
│   ├── fairbench/             # 4 arenas × 4 optimizers, LR sweep + Welch t-test
│   ├── results/full/          # aggregate.csv, significance.csv, plots (committed)
│   ├── README.md              # protocol, datasets, usage
│   └── logs.txt               # raw H100 run log (Jun 2026)
├── examples/                  # integration recipes (HF, torchtune)
├── pyproject.toml             # packaging, ruff, pytest, mypy config
├── OLD_RESULTS.md             # archived pre-FairBench benchmark tables
└── CHANGELOG.md
```

## Drop-in Replacement

```python
# Before
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=1e-3)

# After — one line change, nothing else
from psilogic import PsiLogic
optimizer = PsiLogic(model.parameters(), lr=1e-3)
```

---

## Benchmark Results

Source data: [`benchmark/results/full/aggregate.csv`](benchmark/results/full/aggregate.csv) ·
[`significance.csv`](benchmark/results/full/significance.csv) ·
[`logs.md`](logs.md) · [`benchmark/logs.txt`](benchmark/logs.txt)  
Archived pre-FairBench results: [`OLD_RESULTS.md`](OLD_RESULTS.md)

### FairBench · Adam vs AdamW vs Lion vs ΨLogic · **3 seeds** · NVIDIA H100 80GB

> **Primary reference benchmark (Jun 2026).** Stage-1: per-optimizer LR sweep (500 steps, 7 log-spaced LRs).
> Stage-2: 2000 training steps, 3 seeds, identical init per seed, bf16 AMP, `grad_clip=1.0`.
> Welch *t*-test. Full config: [`benchmark/results/full/config.json`](benchmark/results/full/config.json).

#### Quality metrics (mean ± std)

| Arena | Task | Metric | Adam | AdamW | Lion | **ΨLogic** |
|:------|:-----|:-------|:----:|:-----:|:----:|:----------:|
| **NLP** | GPT / TinyStories | Perplexity ↓ | 13.66 ± 0.22 | 8.17 ± 0.08 | 21.04 ± 1.41 | **7.79 ± 0.18\*** |
| **NLP** | GPT / TinyStories | Val loss ↓ | 2.614 ± 0.016 | 2.101 ± 0.010 | 3.045 ± 0.068 | **2.053 ± 0.023** |
| **ViT** | ViT-Tiny / CIFAR-100 | Val acc ↑ | 0.079 ± 0.003 | 0.223 ± 0.002 | 0.213 ± 0.002 | **0.244 ± 0.006\*\*\*** |
| **ResNet** | ResNet-18 / Tiny ImageNet | Val acc ↑ | 0.172 ± 0.004 | 0.219 ± 0.005 | 0.205 ± 0.007 | **0.222 ± 0.001\*\*** |
| **Diffusion** | DDPM / CelebA 64×64 | Val MSE ↓ | **0.01987 ± 0.00006** | **0.01987 ± 0.00006** | 0.02175 ± 0.00025 | 0.02009 ± 0.00045 |

\*ΨLogic vs AdamW on NLP perplexity: *p* = 0.049. Val loss vs AdamW: *p* = 0.054 (not significant).
\*\*ResNet vs Adam: *p* = 0.001; vs AdamW: *p* = 0.44 (numerical tie).
\*\*\*ViT vs all baselines: *p* < 0.02.

#### Performance (mean over 3 seeds)

| Arena | Peak VRAM (MB) | Wall time (s) | ΨLogic / AdamW time |
|:------|:--------------:|:-------------:|:-------------------:|
| NLP | 458 / 445 (Lion) | 55.2 / 45.9 / 38.2 | 1.20× |
| ViT | 1229 / 1208 (Lion) | 176.7 / 98.5 / 98.6 | **1.79×** |
| ResNet | 823 / 777 (Lion) | 67.4 / 47.6 / 46.1 | 1.42× |
| Diffusion | 3781 / 3768 (Lion) | 168.3 / 95.2 / 91.6 | **1.77×** |

**ΨLogic wins 3 of 4 quality arenas** (NLP perplexity, ViT, ResNet vs Adam).
Diffusion ties Adam/AdamW (*p* = 0.49). Main overhead on **Jun 2026 H100 baseline**:
~1.4–1.8× wall time on ViT/diffusion (pre-Triton fusion).

**v0.5+:** install `psilogic[cuda]` for the Triton fused step (`use_fused_cuda=True`,
default when available). Profile locally: `python scripts/profile_optimizer.py`.
Target: **≤1.25× AdamW** step time on ViT-like models.

---

## Discussion

Under FairBench's per-optimizer LR sweep (data in `benchmark/results/full/`), ΨLogic achieves
the best perplexity on NLP (7.79 vs 8.17 AdamW), the best ViT accuracy (0.244 vs 0.223 AdamW,
*p* < 0.02 vs all baselines), and beats Adam on ResNet (0.222 vs 0.172, *p* = 0.001) while
numerically tying AdamW (0.222 vs 0.219, *p* = 0.44). Diffusion MSE ties Adam/AdamW within noise.

Trade-off: ΨLogic uses comparable VRAM (±3%) but 1.2–1.8× more wall time on the
Jun 2026 H100 baseline, peaking at 1.79× on ViT. **v0.5** adds an optional Triton
fused CUDA path (`pip install psilogic[cuda]`) to close that gap without changing
optimizer math. Pre-FairBench experiments are in [`OLD_RESULTS.md`](OLD_RESULTS.md).

---

## The Formula

```
Ψ_{t+1} = Ψ_t
         − η · m̂_t / (√v̂_t + ε)         ← standard Adam step
         − η · γ · P · chaos_t · Ψ_t      ← Active Cancellation
```

The **chaos detector** — dual EMA of normalized gradient norm:

```
gn_t   = ‖∇_t‖₂ / √(numel)

fast_t = 0.90 · fast_{t-1} + 0.10 · gn_t   ← responsive (τ ≈ 10 steps)
slow_t = 0.99 · slow_{t-1} + 0.01 · gn_t   ← stable baseline (τ ≈ 100 steps)

ratio_t = fast_t / (slow_t + ε)
chaos_t = tanh(slow_t) · (1 + 0.5 · tanh(relu(ratio_t − 1)))
```

| Training Phase | `slow_t` | `chaos_t` | Effect |
|:---|:---:|:---:|:---|
| Early — large noisy gradients | high | → 1.0 | Strong damping, prevents overshooting |
| Mid — active descent | medium | 0.4–0.8 | Moderate regularization |
| Late — converging | low | → 0.1 | Minimal interference |
| Converged | ≈ 0 | → 0.0 | **Term vanishes completely** |

---

## API

```python
from psilogic import PsiLogic

optimizer = PsiLogic(
    params,
    lr             = 1e-3,    # learning rate
    betas          = (0.9, 0.999),
    weight_decay   = 1e-4,
    gamma          = 0.05,    # max cancellation strength
    p_ext          = 1.0,     # chaos amplification factor
    quantum_decay  = 0.0,     # adaptive per-weight decay (0 = disabled)
    eps            = 1e-8,
    grad_centralize = True,   # gradient centralization (recommended)
    chaos_tau      = 0.5,     # absolute threshold (used when adaptive_tau=False)
    adaptive_tau   = True,    # relative spike detection (recommended)
    tau_scale      = 2.0,     # fast/slow ratio to trigger chaos
    max_cancel     = 0.05,    # hard clamp on per-step weight shrinkage
    agc_clip       = 0.02,    # adaptive gradient clipping ratio
    gamma_T_max    = 0,       # cosine γ decay over N steps (0 = disabled)
    use_foreach    = True,    # batched CUDA foreach fallback
    use_fused_cuda = True,    # Triton fused step when CUDA+Triton available
)
```

### Task-Specific Presets

```python
from psilogic import PsiLogicNLP, PsiLogicGPT, PsiLogicViT

# BERT / RoBERTa fine-tuning
optimizer = PsiLogicNLP(model.parameters(), lr=3e-4, gamma_T_max=total_steps)

# GPT-2 / nanoGPT from scratch
optimizer = PsiLogicGPT(model.parameters(), lr=3e-4, gamma_T_max=total_steps)

# ViT / CNN vision training
optimizer = PsiLogicViT(model.parameters(), lr=1e-3, gamma_T_max=total_steps)
```

### Auto-Configuration

```python
from psilogic import PsiLogic

# Detects ViT / GPT / NLP / CNN from model architecture and applies matching preset
optimizer = PsiLogic.auto(model, total_steps=len(loader) * epochs)
```

### Framework Integrations

```bash
pip install psilogic[integrations]
```

```python
# HuggingFace Trainer
from psilogic.integrations.hf import psilogic_trainer_class
Trainer = psilogic_trainer_class()
Trainer(model=model, args=training_args, ...)

# PyTorch Lightning
from psilogic.integrations.lightning import configure_psilogic, ChaosMonitorCallback
trainer = configure_psilogic(model, lr=3e-4)
```

See `examples/` for runnable scripts and `examples/torchtune/` for a torchtune YAML.

### Recommended Hyperparameters

| Task | `lr` | `gamma` | `chaos_tau` | `gamma_T_max` |
|:-----|:----:|:-------:|:-----------:|:-------------:|
| Image classification | `1e-3` | `0.05` | `0.3` | `0` |
| NLP / Transformer fine-tuning | `5e-4` | `0.03` | `0.2` | `total_steps` |
| Audio classification | `1e-3` | `0.05` | `0.3` | `0` |
| Language modeling (from scratch) | `3e-4` | `0.02` | `0.4` | `total_steps` |

---

## Reproduce

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic
pip install -e ".[benchmark]"
pip install -r benchmark/requirements.txt

cd benchmark

# 1. Pre-download datasets (~2 GB, recommended for cloud GPUs)
python -m fairbench.download --data-root ./data

# 2. Full FairBench suite (4 arenas × 4 optimizers, LR sweep + 3 seeds)
python -m fairbench --data-root ./data --output-dir results/full

# 3. LaTeX results table
python -m fairbench.analysis --output-dir results/full --metric val_acc --higher-better

# Smoke test (no downloads, CPU-friendly)
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0
```

See [`benchmark/README.md`](benchmark/README.md) for all arenas, flags, and dataset layout.

---

## License

MIT © 2026 Ali (Troxter222)

---

<div align="center">

*"Fire hard when wrong. Disappear when right."*

</div>