<div align="center">

# ΨLogic

### Active Cancellation Optimizer for Deep Neural Networks

[![PyPI version](https://img.shields.io/pypi/v/psilogic.svg?cache=1)](https://pypi.org/project/psilogic/)
[![CI](https://github.com/Troxter222/psilogic/actions/workflows/ci.yml/badge.svg)](https://github.com/Troxter222/psilogic/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/psilogic)](https://pypi.org/project/psilogic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.16268-B31B1B.svg)](https://arxiv.org/abs/2607.16268)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18739857.svg)](https://doi.org/10.5281/zenodo.18739857)

```
dΨ/dt = -iĤ·Ψ  −  γ·P·chaos(S_t)·Ψ
         └──────┘   └───────────────┘
          Gradient   Active Cancellation
```

**ΨLogic** (`psilogic`) is a PyTorch optimizer that adds a self-regulating, chaos-aware
damping term to Adam. It fires hardest when the model is most confused — and vanishes
at convergence. One-line drop-in for `torch.optim.Adam`. Status: **Alpha** (v0.6).

</div>

**TL;DR**

- **Quality (FairBench · H100 · 3 seeds):** best NLP perplexity and ViT accuracy vs Adam/AdamW/Lion; ResNet beats Adam / ties AdamW; diffusion ties Adam/AdamW.
- **Cost:** comparable VRAM (±3%); wall time **1.2–1.8×** AdamW on the Jun 2026 H100 run (pre-Triton fusion). Fused CUDA path ships in v0.5+; H100 re-run still pending.
- **Install:** `pip install psilogic` · optional GPU fusion: `pip install "psilogic[cuda]"`
- **Use:** `from psilogic import PsiLogic` → `PsiLogic(model.parameters(), lr=1e-3)`

---

## Contents

- [Install](#install)
- [30-second start](#30-second-start)
- [When to use](#when-to-use)
- [Drop-in replacement](#drop-in-replacement)
- [How it works](#how-it-works)
- [Benchmark results](#benchmark-results)
- [API](#api)
- [Integrations](#integrations)
- [Reproduce](#reproduce)
- [FAQ](#faq)
- [Citation](#citation)
- [See also](#see-also)

---

## Install

**Requirements:** Python ≥ 3.8 · PyTorch ≥ 1.9 · optional CUDA + [Triton](https://github.com/triton-lang/triton) for the fused step.

```bash
pip install psilogic

# Optional extras
pip install "psilogic[cuda]"           # Triton fused GPU step (Linux/Windows + CUDA; not Darwin)
pip install "psilogic[integrations]"   # HuggingFace Trainer + Lightning
pip install "psilogic[hf]"             # HuggingFace only
pip install "psilogic[benchmark]"      # FairBench harness deps
pip install "psilogic[deepspeed]"      # DeepSpeed (experimental)
pip install "psilogic[all]"            # everything above + dev tools
```

| Extra | What you get |
|:------|:-------------|
| *(none)* | Core optimizer (`torch` only) |
| `cuda` | Triton fused step (`use_fused_cuda=True` when available) |
| `integrations` | HuggingFace + Lightning helpers |
| `hf` | HuggingFace helpers only |
| `benchmark` | FairBench datasets / analysis stack |
| `deepspeed` | DeepSpeed optional path |
| `dev` / `all` | Tests, lint, type-check (see [CONTRIBUTING.md](CONTRIBUTING.md)) |

---

## 30-second start

```python
import torch
import torch.nn as nn
from psilogic import PsiLogic

model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
opt = PsiLogic(model.parameters(), lr=1e-3)
x, y = torch.randn(32, 128), torch.randint(0, 10, (32,))

for _ in range(100):
    opt.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
```

Works with the usual PyTorch stack: LR schedulers, `grad_clip`, AMP, DDP, and
`optimizer.state_dict()` checkpointing — same patterns as Adam.

---

## When to use

| Situation | Guidance |
|:----------|:---------|
| GPT / LM from scratch, ViT / CNN classification | Strong fit — FairBench wins or ties vs AdamW |
| Transformer fine-tuning (BERT-style) | Use `PsiLogicNLP` or `PsiLogic.auto` |
| Diffusion / generative MSE | Expect a **tie** with AdamW, not a free win |
| Strict wall-time budgets on large transformers | Budget **~1.2–1.8×** step time until fused H100 numbers land; try `psilogic[cuda]` |
| Need a plain AdamW twin with identical defaults | Prefer AdamW — ΨLogic adds chaos cancellation (and optional AGC/GC via presets) |

**Compatibility (covered by tests):** AMP / bf16 · DDP · grad accumulation · `torch.compile` · FSDP / DeepSpeed (experimental) · foreach + fused CUDA backends.

---

## Drop-in replacement

```python
# Before
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-3)

# After — one-line change
from psilogic import PsiLogic
optimizer = PsiLogic(model.parameters(), lr=1e-3)
```

**v0.6+ safer defaults:** bare `PsiLogic(...)` leaves `agc_clip=0.0` and
`grad_centralize=False` (plain drop-in). Task helpers still opt in when useful:
`PsiLogicNLP`, `PsiLogicGPT`, `PsiLogicViT`, `PsiLogicWhisper`, or
`PsiLogic.auto(model)`, or kwargs like `agc_clip=0.01, grad_centralize=True`.

---

## How it works

```
Ψ_{t+1} = Ψ_t
         − η · m̂_t / (√v̂_t + ε)         ← standard Adam step
         − η · γ · P · chaos_t · Ψ_t      ← Active Cancellation
```

The **chaos detector** is a dual EMA of scale-normalized gradient norms:

```
gn_t   = ‖∇_t‖₂ / √(numel)
fast_t = 0.90 · fast_{t-1} + 0.10 · gn_t   ← responsive (τ ≈ 10 steps)
slow_t = 0.99 · slow_{t-1} + 0.01 · gn_t   ← baseline (τ ≈ 100 steps)
ratio_t = fast_t / (slow_t + ε)
chaos_t = tanh(slow_t) · (1 + 0.5 · tanh(relu(ratio_t − 1)))
```

| Phase | `chaos_t` | Effect |
|:------|:---------:|:-------|
| Early — large noisy grads | → 1.0 | Strong damping |
| Mid — active descent | 0.4–0.8 | Moderate regularization |
| Late / converged | → 0 | Term vanishes |

Full derivation and related work: [arXiv:2607.16268](https://arxiv.org/abs/2607.16268) · [PAPER.md](PAPER.md).

<details>
<summary><strong>Glossary</strong></summary>

| Symbol / flag | Meaning |
|:--------------|:--------|
| `gamma` (γ) | Max cancellation strength |
| `p_ext` (P) | Chaos amplification |
| `chaos_t` | Soft gate from dual EMA of ‖∇‖ |
| `agc_clip` | Adaptive gradient clipping (0 = off) |
| `grad_centralize` | Subtract spatial mean from grads |
| `use_fused_cuda` | Triton fused step when CUDA+Triton available |

</details>

---

## Benchmark results

Source: [`aggregate.csv`](benchmark/results/full/aggregate.csv) ·
[`significance.csv`](benchmark/results/full/significance.csv) ·
[`config.json`](benchmark/results/full/config.json) ·
[`benchmark/logs.txt`](benchmark/logs.txt)  
Archived pre-FairBench tables: [`OLD_RESULTS.md`](OLD_RESULTS.md)

### FairBench · Adam vs AdamW vs Lion vs ΨLogic · 3 seeds · NVIDIA H100 80GB

> **Primary reference (Jun 2026).** Stage-1: per-optimizer LR sweep (500 steps, 7 log-spaced LRs).
> Stage-2: 2000 steps, 3 seeds, identical init per seed, bf16 AMP, `grad_clip=1.0`.
> Welch *t*-test.

#### Quality (mean ± std)

| Arena | Task | Metric | Adam | AdamW | Lion | **ΨLogic** | vs best baseline |
|:------|:-----|:-------|:----:|:-----:|:----:|:----------:|:----------------:|
| **NLP** | GPT / TinyStories | PPL ↓ | 13.66 ± 0.22 | 8.17 ± 0.08 | 21.04 ± 1.41 | **7.79 ± 0.18\*** | **−4.7%** vs AdamW |
| **NLP** | GPT / TinyStories | Val loss ↓ | 2.614 ± 0.016 | 2.101 ± 0.010 | 3.045 ± 0.068 | **2.053 ± 0.023** | −2.3% vs AdamW (*p*=0.054) |
| **ViT** | ViT-Tiny / CIFAR-100 | Acc ↑ | 0.079 ± 0.003 | 0.223 ± 0.002 | 0.213 ± 0.002 | **0.244 ± 0.006\*\*\*** | **+9.4%** vs AdamW |
| **ResNet** | ResNet-18 / Tiny ImageNet | Acc ↑ | 0.172 ± 0.004 | 0.219 ± 0.005 | 0.205 ± 0.007 | **0.222 ± 0.001\*\*** | +1.4% vs AdamW (*p*=0.44 tie) |
| **Diffusion** | DDPM / CelebA 64×64 | MSE ↓ | **0.01987 ± 0.00006** | **0.01987 ± 0.00006** | 0.02175 ± 0.00025 | 0.02009 ± 0.00045 | +1.1% vs AdamW (*p*=0.49 tie) |

\*NLP PPL vs AdamW: *p* = 0.049. \*\*ResNet vs Adam: *p* = 0.001; vs AdamW: *p* = 0.44. \*\*\*ViT vs all baselines: *p* < 0.02.

**Scorecard:** ΨLogic wins NLP (PPL) and ViT; beats Adam / ties AdamW on ResNet; ties on diffusion.

#### Wall time & VRAM (Jun 2026 H100 · **pre-fusion**)

> These wall times predate the Triton fused path. Install `psilogic[cuda]` for fusion;
> FairBench H100 re-run with fusion is **still pending**. Quality metrics above are unchanged.

| Arena | AdamW peak VRAM | ΨLogic peak VRAM | AdamW time | ΨLogic time | ΨLogic / AdamW |
|:------|:---------------:|:----------------:|:----------:|:-----------:|:--------------:|
| NLP | ~445 MB | ~458 MB | 45.9 s | 55.2 s | 1.20× |
| ViT | ~1208 MB | ~1229 MB | 98.5 s | 176.7 s | **1.79×** |
| ResNet | ~777 MB | ~823 MB | 47.6 s | 67.4 s | 1.42× |
| Diffusion | ~3768 MB | ~3781 MB | 95.2 s | 168.3 s | **1.77×** |

**v0.5+ fused CUDA:** `pip install "psilogic[cuda]"` enables `use_fused_cuda=True` (default when available). Multi-tensor fusion targets **≤1.25× AdamW** step time on Ampere+ (A100/H100). Profile locally:

```bash
python scripts/profile_optimizer.py
```

<details>
<summary><strong>Local microbench (GTX 1650 · Turing · not the H100 target)</strong></summary>

TinyViTLike (50 tensors, ~202k params), Aug 2026:

| Path | Median step | vs AdamW |
|:-----|------------:|:--------:|
| AdamW `foreach=True` | 1.045 ms | 1.00× |
| ΨLogic foreach | 2.124 ms | 2.03× |
| ΨLogic fused (multi-tensor) | 2.047 ms | **1.96×** |

This card is launch-bound. The ≤1.25× Gate 1C target is Ampere+; do not read this table as the FairBench overhead claim.

</details>

---

## API

```python
from psilogic import PsiLogic

optimizer = PsiLogic(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    gamma=0.05,           # max cancellation strength
    p_ext=1.0,            # chaos amplification
    adaptive_tau=True,    # relative spike detection (recommended)
    tau_scale=2.0,
    max_cancel=0.05,      # hard clamp on per-step weight shrinkage
    agc_clip=0.0,         # 0 = off (default); presets may enable
    grad_centralize=False,
    gamma_T_max=0,        # cosine γ decay over N steps (0 = off)
    use_foreach=True,
    use_fused_cuda=True,  # Triton path when available; set False to debug
)
```

### Task helpers (optional)

```python
from psilogic import PsiLogicNLP, PsiLogicGPT, PsiLogicViT, PsiLogicWhisper, PsiLogic

optimizer = PsiLogicNLP(model.parameters(), lr=3e-4, gamma_T_max=total_steps)
optimizer = PsiLogicGPT(model.parameters(), lr=3e-4, gamma_T_max=total_steps)
optimizer = PsiLogicViT(model.parameters(), lr=1e-3, gamma_T_max=total_steps)
optimizer = PsiLogicWhisper(model.parameters(), lr=1e-3, gamma_T_max=total_steps)

# Infer architecture and apply a matching preset
optimizer = PsiLogic.auto(model, total_steps=len(loader) * epochs)
```

### Recommended hyperparameters

| Task | Helper / preset | `lr` | `gamma` | Notes |
|:-----|:----------------|:----:|:-------:|:------|
| Image classification | `PsiLogicViT` / `vision_defaults` | `1e-3` | ~0.04 | Mild AGC + GC on |
| NLP fine-tuning | `PsiLogicNLP` / `nlp_defaults` | `3e-4`–`5e-4` | ~0.03 | Set `gamma_T_max=total_steps` |
| LM from scratch | `PsiLogicGPT` / `gpt_scratch_defaults` | `3e-4` | ~0.02 | Bare defaults (no AGC/GC) |
| Audio / Whisper | `PsiLogicWhisper` | `1e-3` | ~0.05 | See `whisper_defaults` |

### Diagnostics

```python
from psilogic import debug, get_chaos_metrics

print(debug.chaos_stats(optimizer))   # per-group chaos snapshot
# or inspect a single param state:
# get_chaos_metrics(optimizer.state[param])
```

---

## Integrations

```bash
pip install "psilogic[integrations]"
```

```python
# HuggingFace Trainer
from psilogic.integrations.hf import psilogic_trainer_class
Trainer = psilogic_trainer_class()
Trainer(model=model, args=training_args, ...)

# PyTorch Lightning — configure_psilogic returns an optimizer, not a Trainer
import lightning as L
from psilogic.integrations.lightning import configure_psilogic, ChaosMonitorCallback

class LitModel(L.LightningModule):
    def configure_optimizers(self):
        return configure_psilogic(self.model, lr=3e-4, total_steps=10_000)

trainer = L.Trainer(callbacks=[ChaosMonitorCallback(log_every_n_steps=100)])
```

Runnable recipes: [`examples/`](examples/) · [`examples/README.md`](examples/README.md) · torchtune YAML under `examples/torchtune/`.

Canonical FairBench artifacts (citation-grade): [`benchmark/results/full/`](benchmark/results/full/).
Root `results/` and `benchmark/results/local_full/` are local/exploratory.
`./run_fairbench.sh` defaults to a longer local recipe (5 seeds / 5000 steps / fp16), not the paper protocol.

---

## Reproduce

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic
pip install -e ".[benchmark]"
pip install -r benchmark/requirements.txt

# Optional: long H100-style run in a detachable tmux session
./run_fairbench.sh

# Or step-by-step:
cd benchmark
python -m fairbench.download --data-root ./data
python -m fairbench --data-root ./data --output-dir results/full
python -m fairbench.analysis --output-dir results/full --metric val_acc --higher-better

# Smoke test (CPU-friendly, no downloads)
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0
```

Full protocol, arenas, and flags: [`benchmark/README.md`](benchmark/README.md).

---

## FAQ

**Is this a drop-in for Adam or AdamW?**  
API-wise, yes for `torch.optim.Adam` / AdamW-style loops. Math-wise it adds chaos-gated cancellation on top of Adam. From v0.6, bare defaults match a plain Adam-like setup (no AGC / no grad centralization).

**Do I still need LR warmup?**  
Often less critical — chaos damping is strongest early — but schedulers and warmup still work if your recipe uses them.

**Why is it slower than AdamW?**  
Extra chaos state + cancellation work. Jun 2026 FairBench is pre-fusion. Use `psilogic[cuda]`; Ampere+ fusion target is ≤1.25× AdamW step time.

**Can I turn fusion off?**  
Yes: `PsiLogic(..., use_fused_cuda=False)` falls back to foreach, then scalar.

**Breaking change in 0.6?**  
Bare constructor no longer enables AGC / grad centralization. Prefer task helpers or pass kwargs to restore the old behavior. See [CHANGELOG.md](CHANGELOG.md).

**Where do I contribute / report security issues?**  
[CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md).

---

## Citation

If you find the work or code useful, please cite:

```bibtex
@misc{sultonov2026psilogic,
      title={PsiLogic: Chaos-Aware Active Cancellation for Adam with a Fair Cross-Domain Benchmark},
      author={Ali Sultonov},
      year={2026},
      eprint={2607.16268},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.16268},
}
```

> Sultonov, A. (2026). *PsiLogic: Chaos-Aware Active Cancellation for Adam with a Fair Cross-Domain Benchmark*. arXiv preprint arXiv:2607.16268.

---

## License

MIT © 2026 Ali (Troxter222) — see [LICENSE](LICENSE).

---

## See also

| Doc | Purpose |
|:----|:--------|
| [CHANGELOG.md](CHANGELOG.md) | Release notes (incl. v0.6 default change) |
| [ROADMAP.md](ROADMAP.md) | Gates, scorecard, path to v1.0 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, PR workflow, layout |
| [SECURITY.md](SECURITY.md) | Private vulnerability reporting |
| [PAPER.md](PAPER.md) | Paper notes (PDF via `scripts/build_arxiv_pdf.py`) |
| [benchmark/README.md](benchmark/README.md) | FairBench protocol |
| [OLD_RESULTS.md](OLD_RESULTS.md) | Pre-FairBench archive |

---

<div align="center">

*"Fire hard when wrong. Disappear when right."*

</div>
