# ΨLogic

PyTorch optimizer: Adam + chaos-gated damping. Extra term kicks in when gradient
noise spikes and fades when things settle. Drop-in for `torch.optim.Adam`.

Alpha, v0.6. Paper: [arXiv:2607.16268](https://arxiv.org/abs/2607.16268).
PyPI: [`psilogic`](https://pypi.org/project/psilogic/).

```
dΨ/dt = -iĤ·Ψ  −  γ·P·chaos(S_t)·Ψ
         └──────┘   └───────────────┘
          Gradient   Active Cancellation
```

## Install

Python ≥ 3.8, PyTorch ≥ 1.9. Triton optional (fused CUDA step).

```bash
pip install psilogic
pip install "psilogic[cuda]"           # Triton fused step (Linux/Windows + CUDA)
pip install "psilogic[integrations]"   # HuggingFace Trainer + Lightning
pip install "psilogic[hf]"             # HuggingFace only
pip install "psilogic[benchmark]"      # FairBench deps
pip install "psilogic[deepspeed]"      # DeepSpeed (experimental)
pip install "psilogic[all]"            # everything + dev tools
```

Core package depends on `torch` only. `cuda` enables `use_fused_cuda=True` when
available. Dev extras: [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

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

Schedulers, grad clip, AMP, DDP, `state_dict` — same as Adam.

```python
# Before
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-3)

# After
from psilogic import PsiLogic
optimizer = PsiLogic(model.parameters(), lr=1e-3)
```

From v0.6, bare `PsiLogic(...)` has `agc_clip=0.0` and `grad_centralize=False`.
For task defaults use `PsiLogicNLP` / `PsiLogicGPT` / `PsiLogicViT` /
`PsiLogicWhisper`, or `PsiLogic.auto(model)`, or pass the kwargs yourself.

Tested with: AMP / bf16, DDP, grad accumulation, `torch.compile`, foreach +
fused CUDA. FSDP / DeepSpeed are experimental.

**Where it helps:** from-scratch LM and ResNet — ties AdamW, beats Adam.
**Where it doesn't:** the June ViT win did not hold on the Sep 2026 H100 re-run
(Lion ahead). Diffusion was a tie in the paper run; that arena did not finish
in the fused re-run. With `psilogic[cuda]` on H100, FairBench wall time is
1.02–1.10× AdamW. For a plain AdamW twin, just use AdamW.

## How it works

```
Ψ_{t+1} = Ψ_t
         − η · m̂_t / (√v̂_t + ε)         ← Adam
         − η · γ · P · chaos_t · Ψ_t      ← cancellation
```

Chaos is a dual EMA of scale-normalized grad norms:

```
gn_t   = ‖∇_t‖₂ / √(numel)
fast_t = 0.90 · fast_{t-1} + 0.10 · gn_t
slow_t = 0.99 · slow_{t-1} + 0.01 · gn_t
ratio_t = fast_t / (slow_t + ε)
chaos_t = tanh(slow_t) · (1 + 0.5 · tanh(relu(ratio_t − 1)))
```

Early noisy grads → chaos near 1. Converged → near 0. Full writeup:
[arXiv:2607.16268](https://arxiv.org/abs/2607.16268), [PAPER.md](PAPER.md).

| Flag | Meaning |
|:-----|:--------|
| `gamma` | Max cancellation strength |
| `p_ext` | Chaos amplification |
| `agc_clip` | Adaptive grad clip (`0` = off) |
| `grad_centralize` | Subtract spatial mean from grads |
| `use_fused_cuda` | Triton fused step if CUDA+Triton present |

## Benchmarks

Two H100 snapshots. The June table is what the paper cites. The September
table is the fused re-run (`psilogic[cuda]`, Triton on).

Paper CSVs (Jun 2026, pre-fusion):
[`aggregate.csv`](benchmark/results/full/aggregate.csv),
[`significance.csv`](benchmark/results/full/significance.csv),
[`config.json`](benchmark/results/full/config.json).
Fused re-run (24 Sep 2026):
[`aggregate.csv`](benchmark/results/fused_h100/aggregate.csv),
[`significance.csv`](benchmark/results/fused_h100/significance.csv),
[`PROOF.md`](benchmark/results/fused_h100/PROOF.md).
Pre-FairBench archive: [`OLD_RESULTS.md`](OLD_RESULTS.md).

Shared protocol: stage-1 LR sweep (500 steps, 7 LRs), stage-2 2000 steps,
3 seeds, identical init per seed, bf16 AMP, `grad_clip=1.0`, Welch *t*-test.
June stack: PyTorch 2.4.1+cu124. September stack: PyTorch 2.8.0+cu128, CUDA 12.8.

### Quality — paper reference (Jun 2026, pre-fusion)

| Arena | Task | Metric | Adam | AdamW | Lion | ΨLogic | vs best baseline |
|:------|:-----|:-------|:----:|:-----:|:----:|:------:|:----------------:|
| NLP | GPT / TinyStories | PPL ↓ | 13.66 ± 0.22 | 8.17 ± 0.08 | 21.04 ± 1.41 | **7.79 ± 0.18\*** | −4.7% vs AdamW |
| NLP | GPT / TinyStories | Val loss ↓ | 2.614 ± 0.016 | 2.101 ± 0.010 | 3.045 ± 0.068 | **2.053 ± 0.023** | −2.3% vs AdamW (*p*=0.054) |
| ViT | ViT-Tiny / CIFAR-100 | Acc ↑ | 0.079 ± 0.003 | 0.223 ± 0.002 | 0.213 ± 0.002 | **0.244 ± 0.006\*\*\*** | +9.4% vs AdamW |
| ResNet | ResNet-18 / Tiny ImageNet | Acc ↑ | 0.172 ± 0.004 | 0.219 ± 0.005 | 0.205 ± 0.007 | **0.222 ± 0.001\*\*** | +1.4% vs AdamW (*p*=0.44) |
| Diffusion | DDPM / CelebA 64×64 | MSE ↓ | **0.01987 ± 0.00006** | **0.01987 ± 0.00006** | 0.02175 ± 0.00025 | 0.02009 ± 0.00045 | +1.1% vs AdamW (*p*=0.49) |

\*NLP PPL vs AdamW: *p* = 0.049. \*\*ResNet vs Adam: *p* = 0.001; vs AdamW: *p* = 0.44. \*\*\*ViT vs all baselines: *p* < 0.02.

### Quality — fused re-run (24 Sep 2026, H100)

Same FairBench recipe, fusion on. LR sweep winners moved (ViT AdamW/ΨLogic
landed at 3.2e-5, Lion at 1e-5; June had 3.2e-4 / 1e-4). ViT peak VRAM is
~5.5 GB here vs ~1.2 GB in June, so this is not a bit-exact replay of the
paper table.

| Arena | Metric | AdamW | Lion | ΨLogic | vs AdamW |
|:------|:-------|------:|-----:|-------:|:---------|
| NLP | PPL ↓ | 8.20 ± 0.20 | 20.50 ± 1.29 | **8.15 ± 0.16** | −0.6% (*p*=0.44) |
| ViT | Acc ↑ | 0.283 ± 0.005 | **0.305 ± 0.004** | 0.267 ± 0.007 | −5.4% (*p*=0.007) |
| ResNet | Acc ↑ | 0.221 ± 0.005 | 0.208 ± 0.005 | 0.221 ± 0.002 | −0.3% (*p*=0.90) |

NLP still points the right way, but the gap vs AdamW is not significant.
ResNet still ties AdamW and beats Adam (*p*=0.003). ViT does not. Diffusion
did not run: local CelebA was a flat JPEG folder, and `torchvision.datasets.CelebA`
rejected it for missing annotations.

### Wall time (H100, ΨLogic / AdamW)

Target: ≤1.25× AdamW on the FairBench ViT wall clock. September clears it.

| Arena | Jun 2026 pre-fusion | Sep 2026 fused | Sep AdamW | Sep ΨLogic |
|:------|--------------------:|---------------:|----------:|-----------:|
| NLP | 1.20× | **1.10×** | 20.2 s | 22.2 s |
| ViT | 1.79× | **1.02×** | 114.4 s | 117.2 s |
| ResNet | 1.42× | **1.04×** | 35.9 s | 37.2 s |
| Diffusion | 1.77× | — | — | — |

Peak VRAM on the fused run stays within ~1% of AdamW (NLP ~459 MB, ViT ~5.5 GB,
ResNet ~858 MB).

`scripts/profile_optimizer.py` on the same H100, TinyViTLike ~202k params, is
launch-bound and is not the FairBench claim: fused median step 1.37 ms vs
AdamW 0.60 ms (2.27×). Profile locally:

```bash
python scripts/profile_optimizer.py
```

GTX 1650 (Turing, Aug 2026) — TinyViTLike, ~202k params. Launch-bound; not the
Ampere target, don't treat this as the FairBench overhead claim.

| Path | Median step | vs AdamW |
|:-----|------------:|:--------:|
| AdamW `foreach=True` | 1.045 ms | 1.00× |
| ΨLogic foreach | 2.124 ms | 2.03× |
| ΨLogic fused (multi-tensor) | 2.047 ms | 1.96× |

## API

```python
from psilogic import PsiLogic

optimizer = PsiLogic(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    gamma=0.05,
    p_ext=1.0,
    adaptive_tau=True,
    tau_scale=2.0,
    max_cancel=0.05,
    agc_clip=0.0,              # off by default; presets may enable
    grad_centralize=False,
    gamma_T_max=0,             # cosine γ decay over N steps (0 = off)
    use_foreach=True,
    use_fused_cuda=True,       # set False to debug
)
```

```python
from psilogic import PsiLogicNLP, PsiLogicGPT, PsiLogicViT, PsiLogicWhisper, PsiLogic

optimizer = PsiLogicNLP(model.parameters(), lr=3e-4, gamma_T_max=total_steps)
optimizer = PsiLogicGPT(model.parameters(), lr=3e-4, gamma_T_max=total_steps)
optimizer = PsiLogicViT(model.parameters(), lr=1e-3, gamma_T_max=total_steps)
optimizer = PsiLogicWhisper(model.parameters(), lr=1e-3, gamma_T_max=total_steps)
optimizer = PsiLogic.auto(model, total_steps=len(loader) * epochs)
```

| Task | Helper / preset | `lr` | `gamma` | Notes |
|:-----|:----------------|:----:|:-------:|:------|
| Image classification | `PsiLogicViT` / `vision_defaults` | `1e-3` | ~0.04 | Mild AGC + GC on |
| NLP fine-tuning | `PsiLogicNLP` / `nlp_defaults` | `3e-4`–`5e-4` | ~0.03 | Set `gamma_T_max=total_steps` |
| LM from scratch | `PsiLogicGPT` / `gpt_scratch_defaults` | `3e-4` | ~0.02 | No AGC/GC |
| Audio / Whisper | `PsiLogicWhisper` | `1e-3` | ~0.05 | See `whisper_defaults` |

```python
from psilogic import debug, get_chaos_metrics

print(debug.chaos_stats(optimizer))
# get_chaos_metrics(optimizer.state[param])
```

## Integrations

```bash
pip install "psilogic[integrations]"
```

```python
from psilogic.integrations.hf import psilogic_trainer_class
Trainer = psilogic_trainer_class()
Trainer(model=model, args=training_args, ...)

import lightning as L
from psilogic.integrations.lightning import configure_psilogic, ChaosMonitorCallback

class LitModel(L.LightningModule):
    def configure_optimizers(self):
        return configure_psilogic(self.model, lr=3e-4, total_steps=10_000)

trainer = L.Trainer(callbacks=[ChaosMonitorCallback(log_every_n_steps=100)])
```

`configure_psilogic` returns an optimizer, not a Trainer. Examples:
[`examples/`](examples/), [`examples/README.md`](examples/README.md).

Paper FairBench output: [`benchmark/results/full/`](benchmark/results/full/).
Fused H100 re-run: [`benchmark/results/fused_h100/`](benchmark/results/fused_h100/).
Root `results/` and `benchmark/results/local_full/` are local scratch.
`./run_fairbench.sh` is a longer local recipe (5 seeds / 5000 steps / fp16),
not the paper protocol.

## Reproduce

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic
pip install -e ".[benchmark]"
pip install -r benchmark/requirements.txt

./run_fairbench.sh   # optional long run

cd benchmark
python -m fairbench.download --data-root ./data
python -m fairbench --data-root ./data --output-dir results/full
python -m fairbench.analysis --output-dir results/full --metric val_acc --higher-better

# smoke test, no downloads
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0
```

Details: [`benchmark/README.md`](benchmark/README.md).

## Notes

- API matches Adam/AdamW loops; math adds chaos-gated cancellation. v0.6 bare
  defaults = no AGC / no grad centralization.
- Warmup is often less critical (early damping is already strong), but schedulers
  still work.
- FairBench wall time with `psilogic[cuda]` on H100 (24 Sep 2026) is 1.02–1.10×
  AdamW. The June 1.2–1.8× figures are the pre-fusion run. A tiny-model
  microbench can still look like ~2×; that is not the FairBench claim.
- Disable fusion: `PsiLogic(..., use_fused_cuda=False)`.
- v0.6 breaking: bare constructor no longer enables AGC / grad centralization.
  Use helpers or kwargs. See [CHANGELOG.md](CHANGELOG.md).
- Contributing / security: [CONTRIBUTING.md](CONTRIBUTING.md),
  [SECURITY.md](SECURITY.md).

## Citation

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

## License

MIT © 2026 Ali (Troxter222) — [LICENSE](LICENSE).

Also: [CHANGELOG.md](CHANGELOG.md) · [ROADMAP.md](ROADMAP.md) ·
[PAPER.md](PAPER.md) · [OLD_RESULTS.md](OLD_RESULTS.md).
