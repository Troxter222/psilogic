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

**Where it helps:** LM / ViT / CNN from scratch — FairBench wins or ties AdamW.
**Where it doesn't:** diffusion is a tie; if wall-clock is tight, budget
~1.2–1.8× AdamW step time until the fused H100 numbers land (try
`psilogic[cuda]`). For a plain AdamW twin, just use AdamW.

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

FairBench CSVs:
[`aggregate.csv`](benchmark/results/full/aggregate.csv),
[`significance.csv`](benchmark/results/full/significance.csv),
[`config.json`](benchmark/results/full/config.json),
[`benchmark/logs.txt`](benchmark/logs.txt).
Pre-FairBench archive: [`OLD_RESULTS.md`](OLD_RESULTS.md).

Protocol (Jun 2026, H100 80GB, 3 seeds): stage-1 LR sweep (500 steps, 7 LRs),
stage-2 2000 steps, identical init per seed, bf16 AMP, `grad_clip=1.0`, Welch
*t*-test.

| Arena | Task | Metric | Adam | AdamW | Lion | ΨLogic | vs best baseline |
|:------|:-----|:-------|:----:|:-----:|:----:|:------:|:----------------:|
| NLP | GPT / TinyStories | PPL ↓ | 13.66 ± 0.22 | 8.17 ± 0.08 | 21.04 ± 1.41 | **7.79 ± 0.18\*** | −4.7% vs AdamW |
| NLP | GPT / TinyStories | Val loss ↓ | 2.614 ± 0.016 | 2.101 ± 0.010 | 3.045 ± 0.068 | **2.053 ± 0.023** | −2.3% vs AdamW (*p*=0.054) |
| ViT | ViT-Tiny / CIFAR-100 | Acc ↑ | 0.079 ± 0.003 | 0.223 ± 0.002 | 0.213 ± 0.002 | **0.244 ± 0.006\*\*\*** | +9.4% vs AdamW |
| ResNet | ResNet-18 / Tiny ImageNet | Acc ↑ | 0.172 ± 0.004 | 0.219 ± 0.005 | 0.205 ± 0.007 | **0.222 ± 0.001\*\*** | +1.4% vs AdamW (*p*=0.44) |
| Diffusion | DDPM / CelebA 64×64 | MSE ↓ | **0.01987 ± 0.00006** | **0.01987 ± 0.00006** | 0.02175 ± 0.00025 | 0.02009 ± 0.00045 | +1.1% vs AdamW (*p*=0.49) |

\*NLP PPL vs AdamW: *p* = 0.049. \*\*ResNet vs Adam: *p* = 0.001; vs AdamW: *p* = 0.44. \*\*\*ViT vs all baselines: *p* < 0.02.

Wins NLP PPL and ViT. Beats Adam / ties AdamW on ResNet. Tie on diffusion.

### Wall time & VRAM (Jun 2026 H100, pre-fusion)

These times are from before the Triton fused path. `psilogic[cuda]` ships fusion
in v0.5+; FairBench H100 re-run with fusion is still pending. Quality numbers
above are unchanged.

| Arena | AdamW peak VRAM | ΨLogic peak VRAM | AdamW time | ΨLogic time | ΨLogic / AdamW |
|:------|:---------------:|:----------------:|:----------:|:-----------:|:--------------:|
| NLP | ~445 MB | ~458 MB | 45.9 s | 55.2 s | 1.20× |
| ViT | ~1208 MB | ~1229 MB | 98.5 s | 176.7 s | 1.79× |
| ResNet | ~777 MB | ~823 MB | 47.6 s | 67.4 s | 1.42× |
| Diffusion | ~3768 MB | ~3781 MB | 95.2 s | 168.3 s | 1.77× |

Target with multi-tensor fusion: ≤1.25× AdamW step time on Ampere+ (A100/H100).
Profile locally:

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

Citation-grade FairBench output: [`benchmark/results/full/`](benchmark/results/full/).
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
- Slower than AdamW because of chaos state. Jun 2026 FairBench is pre-fusion —
  use `psilogic[cuda]`. Ampere+ target: ≤1.25× AdamW step time.
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
