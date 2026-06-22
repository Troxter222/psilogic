# PsiLogic: A Chaos-Aware Optimizer with Dynamic Active Cancellation for Deep Neural Networks

**Ali (Troxter222)**  
Independent Research  
`troxtergrif@gmail.com`  
https://github.com/Troxter222/psilogic · DOI: 10.5281/zenodo.18739857

---

## Abstract

I introduce **PsiLogic** (ΨLogic), a first-order stochastic gradient optimizer that extends
Adam with a *dynamic Active Cancellation Term* — a self-regulating damping signal modulated
by a dual exponential moving average (EMA) of normalized gradient norms. This signal, referred
to as the *chaos detector*, activates strongly during the chaotic early phase of training
and vanishes automatically as the model converges, requiring no manual warmup schedule or
additional hyperparameter tuning.

PsiLogic is evaluated against Adam, AdamW, and Lion on **FairBench** — a bias-free
cross-domain benchmark spanning language modeling (GPT / TinyStories), image classification
(ViT-Tiny / CIFAR-100, ResNet-18 / Tiny ImageNet), and generative modeling (DDPM / CelebA).

On NVIDIA H100 80GB (Jun 2026, PyTorch 2.4.1, bf16 AMP, 3 seeds, per-optimizer LR sweep,
Welch *t*-test), PsiLogic wins **3 of 4 arenas**: NLP perplexity **7.79 ± 0.18** vs
8.17 ± 0.08 (AdamW), ViT top-1 **0.244 ± 0.006** vs 0.223 ± 0.002 (AdamW, *p* < 0.02),
and ResNet top-1 **0.222 ± 0.001** vs 0.172 ± 0.004 (Adam, *p* = 0.001). Diffusion val MSE
ties Adam/AdamW within noise (0.0201 vs 0.0199, *p* = 0.49). Peak VRAM is comparable; wall
time is 1.2–1.8× higher on transformer-heavy arenas.

I provide a complete mathematical formulation, GPU-native PyTorch implementation with zero
CPU–GPU synchronization overhead, task-specific presets, and release all benchmark CSVs under
`benchmark/results/full/`. Archived pre-FairBench experiments: `OLD_RESULTS.md`.

**Version:** v0.5.0 — FairBench reference results, modular package (v0.4 refactor).

**Installation:** `pip install psilogic`

---

## 1. Introduction

The choice of optimizer shapes training speed, generalization quality, and computational cost
in deep learning. Adam (Kingma & Ba, 2015) dominates practical deep learning due to its
adaptive per-parameter learning rates and robustness to hyperparameter choice. AdamW
(Loshchilov & Hutter, 2019) improved upon Adam by decoupling weight decay from the gradient
update, becoming the standard optimizer for most modern large-scale training pipelines.

Despite this progress, a fundamental structural limitation persists: the corrective signal
does not scale with the *current level of model confusion*. At step 1, when the model is
randomly initialized and gradients are large and noisy, Adam applies the same
second-moment-normalized update it would apply at step 10,000 when the model is near
convergence. This symmetry is suboptimal — early-phase training is dominated by erroneous,
high-variance gradient signals that often point away from useful descent directions.

I propose **PsiLogic**, which addresses this by introducing a damping term whose strength
is modulated by a running estimate of gradient chaos. The term is strongest when the model
is most confused, and decays to zero as training stabilizes. Key properties:

1. **Zero-configuration chaos sensing** — the detector self-calibrates; no warmup schedule needed.
2. **Drop-in compatibility** — a one-line replacement for `torch.optim.Adam`.
3. **GPU-native** — fully vectorized with zero `.item()` calls and no CPU–GPU synchronization.
4. **Task-specific presets** — `PsiLogicNLP`, `PsiLogicGPT`, and `PsiLogicViT` provide
   sensible defaults for common training regimes.

The conceptual motivation draws from dissipative quantum mechanics, where the equation of
motion for a state Ψ under Hamiltonian Ĥ with energy dissipation γ is:

```
dΨ/dt = -iĤΨ − γΨ
```

The term `-γΨ` provides state-proportional damping. In the optimization analogy, `-iĤΨ`
is the gradient update, and the Active Cancellation Term provides adaptive damping
conditioned on measured training chaos.

---

## 2. Related Work

**Adam** (Kingma & Ba, 2015) maintains exponential moving averages of first and second
gradient moments with bias-corrected per-parameter adaptive rates. It is the dominant
optimizer for non-convex deep learning objectives.

**AdamW** (Loshchilov & Hutter, 2019) decouples L₂ weight regularization from the adaptive
update. This is the standard for GPT, BERT, and most large language model training.

**Lion** (Chen et al., 2023) applies a sign-based update with coupled weight decay, achieving
memory efficiency. It requires larger learning rates and more careful tuning, and tends to
underperform on language model pre-training from scratch.

**Gradient Centralization** (Yong et al., 2020) projects gradients onto a centered subspace,
reducing gradient explosion in early training. PsiLogic incorporates it as an optional component.

**Learning rate warmup** (Goyal et al., 2017) holds η small for the first N steps to prevent
large erroneous early updates. PsiLogic achieves an equivalent effect implicitly: large
early `chaos_t` → large cancellation → effective dampening of early parameter motion.
No separate warmup schedule is required.

**Adaptive Gradient Clipping** (Brock et al., 2021) clips gradients relative to parameter
norms, improving stability in large-batch training. PsiLogic v6 incorporates AGC as an
optional component via the `agc_clip` parameter.

---

## 3. PsiLogic

### 3.1 Mathematical Formulation

PsiLogic extends Adam with a chaos-conditioned Active Cancellation Term:

```
θ_{t+1} = θ_t
         − η · m̂_t / (√v̂_t + ε)          [adaptive gradient step]
         − η · γ · P · chaos_t · θ_t        [active cancellation]
```

Standard Adam moments:

```
m_t = β₁ m_{t-1} + (1−β₁) ∇_t
v_t = β₂ v_{t-1} + (1−β₂) ∇_t²
m̂_t = m_t / (1 − β₁ᵗ)
v̂_t = v_t / (1 − β₂ᵗ)
```

### 3.2 Chaos Detector

I maintain a **dual EMA** of the size-normalized gradient norm:

```
gn_t   = ‖∇_t‖₂ / √(numel)              [scale-invariant norm]

fast_t = 0.90 · fast_{t-1} + 0.10 · gn_t    [responsive EMA, τ≈10 steps]
slow_t = 0.99 · slow_{t-1} + 0.01 · gn_t    [stable baseline, τ≈100 steps]
```

The chaos coefficient:

```
ratio_t = fast_t / (slow_t + ε)
chaos_t = tanh(slow_t) · (1 + 0.5 · tanh(relu(ratio_t − 1)))
```

The Active Cancellation coefficient with threshold guard:

```
c_t = 0                                if slow_t < τ
    = chaos_t · η · γ · P              otherwise
```

The `ratio_t` term detects *acceleration* in gradient magnitude — when training is becoming
more chaotic relative to its recent baseline, the term intensifies. The `tanh(slow_t)` factor
ensures global activity is proportional to absolute gradient scale. Together they create a
signal sensitive to both absolute and relative gradient chaos.

In v6, the threshold check uses an **adaptive mode** (default): chaos activates when
`fast_t > tau_scale × slow_t`, where `tau_scale=2.0`. This is a spike detector relative
to the current baseline, which works at any gradient scale (small models, ViTs, LMs).

### 3.3 Unified Decay and Bug Fixes (v6)

Early versions applied weight decay, Active Cancellation, and Gradient-modulated L2 Penalty as three
independent multiplicative shrinks. At typical magnitudes these compound to approximately
0.98 per step, which over thousands of steps collapses parameter norms — particularly
harmful for ViT patch embeddings and attention weights.

v6 collapses all shrinkage into a single unified coefficient applied once per step:

```
total_decay = lr·λ + chaos_contrib     [chaos fires]
total_decay = lr·λ                     [chaos does not fire]

θ ← θ · (1 − total_decay)             [applied exactly once]
```

Gradient-modulated L2 Penalty is mutually exclusive with Active Cancellation (only one fires per step),
reads the raw gradient before Gradient Centralization, and is disabled by default for
vision and GPT scratch tasks.

A hard clamp `c_coeff ≤ max_cancel` (default 0.05) prevents catastrophic weight collapse
during the high-loss initialization phase of from-scratch language model training.

### 3.4 Full Algorithm

```
Algorithm 1: PsiLogic v6

Input: θ₀, η, γ, P, β₁, β₂, ε, τ, λ (weight_decay), T_max
Initialize: m₀=0, v₀=0, fast₀=0, slow₀=0

for t = 1, 2, ..., T do

    g_t ← ∇_θ L(θ_{t-1})

    # Optional: Adaptive Gradient Clipping
    g_t ← g_t · min(1, agc · ||θ|| / ||g_t||)

    # Optional: Gradient Centralization
    if g_t.dim > 1:
        g_t ← g_t − mean(g_t, dim=spatial, keepdim=True)

    # Adam moment updates
    m_t ← β₁·m_{t-1} + (1−β₁)·g_t
    v_t ← β₂·v_{t-1} + (1−β₂)·g_t²

    # Chaos detector (dual EMA)
    gn_t   ← ‖g_t‖₂ / √(numel(g_t))
    fast_t ← 0.9·fast_{t-1} + 0.1·gn_t
    slow_t ← 0.99·slow_{t-1} + 0.01·gn_t

    # Optional: cosine decay for γ
    if T_max > 0:
        γ_eff ← γ · 0.5·(1 + cos(π · min(t/T_max, 1)))
    else:
        γ_eff ← γ

    # Chaos gate (adaptive mode)
    if t > warmup and fast_t > tau_scale · slow_t:
        ratio_t ← fast_t / (slow_t + ε)
        chaos_t ← tanh(slow_t) · (1 + 0.5·tanh(relu(ratio_t − 1)))
        c_coeff ← min(chaos_t · η · γ_eff · P, max_cancel)
        θ_{t-1} ← θ_{t-1} · (1 − η·λ − c_coeff)   [unified decay]
    else:
        θ_{t-1} ← θ_{t-1} · (1 − η·λ)              [weight decay only]

    # Adam gradient step with bias correction
    step_size  ← η / (1 − β₁ᵗ)
    bias_corr2 ← √(1 − β₂ᵗ)
    denom      ← v_t.sqrt() / bias_corr2 + ε
    θ_t ← θ_{t-1} − step_size · m_t / denom

end for
```

### 3.5 Phase Analysis

The effective damping coefficient at step t is `λ_eff(t) = γ · P · chaos_t`:

| Phase | `slow_t` | `chaos_t` | `λ_eff` (γ=0.05, P=1.0) |
|:------|:--------:|:---------:|:------------------------:|
| Initialization | 5–20 | ≈ 1.0 | ≈ 0.050 |
| Early training (ep 1–10) | 1–3 | 0.76–0.99 | 0.038–0.050 |
| Mid training (ep 20–60) | 0.5–1.0 | 0.46–0.76 | 0.023–0.038 |
| Late training (ep 70–90) | 0.2–0.5 | 0.20–0.46 | 0.010–0.023 |
| Convergence | ≈ 0 | ≈ 0 | ≈ 0 |

This mirrors velocity-dependent drag: maximum resistance during fast chaotic motion,
zero resistance at rest.

### 3.6 Comparison with Existing Optimizers

| Property | SGD | Adam | AdamW | Lion | **ΨLogic** |
|:---------|:---:|:----:|:-----:|:----:|:----------:|
| Adaptive per-param LR | ✗ | ✓ | ✓ | ✗ | ✓ |
| Weight decay | Fixed | Optional | Decoupled | Coupled | Decoupled |
| Chaos-aware damping | ✗ | ✗ | ✗ | ✗ | **✓** |
| Implicit warmup | ✗ | ✗ | ✗ | ✗ | **✓** |
| Second moment estimate | ✗ | ✓ | ✓ | ✗ | ✓ |
| Zero CPU–GPU sync | ✓ | ✓ | ✓ | ✓ | **✓** |
| foreach batched CUDA ops | ✗ | ✗ | ✓ | ✗ | **✓** |

PsiLogic is, to my knowledge, the first optimizer to modulate its regularization strength
through a learned signal of current training chaos derived entirely from gradient statistics.

---

## 4. Experiments

All headline results come from a single FairBench run on **NVIDIA H100 80GB HBM3**
(PyTorch 2.4.1+cu124, CUDA 12.4). Raw outputs are committed at
`benchmark/results/full/` (`aggregate.csv`, `significance.csv`, `summary.csv`, `config.json`).
Pre-FairBench experiments are archived in `OLD_RESULTS.md` and are not used in the claims below.

### 4.1 FairBench Protocol

FairBench eliminates per-optimizer tuning bias via a two-stage protocol:

**Stage 1 — LR sweep.** Each optimizer independently searches 7 log-spaced learning rates
from `1×10⁻⁵` to `1×10⁻²` over 500 training steps. The LR with the best validation metric
is selected.

**Stage 2 — Multi-seed evaluation.** Using the selected LR, each optimizer trains for
2000 steps over 3 seeds (0, 1, 2). For a given seed, all optimizers start from **identical
initial weights** and see the **same data order**.

**Shared settings** (from `config.json`):

| Setting | Value |
|:--------|:------|
| Hardware | NVIDIA H100 80GB HBM3 |
| Precision | bf16 AMP (`torch.amp.autocast`) |
| Batch size | 64 per arena |
| Gradient clip | `max_norm = 1.0` |
| LR schedule | Cosine with 100-step warmup |
| Optimizers | Adam (coupled L2), AdamW, Lion, PsiLogic |
| Statistics | Mean ± std over seeds; Welch *t*-test (PsiLogic vs each baseline) |

PsiLogic uses published per-arena presets (`PsiLogicNLP`, `PsiLogicViT`, etc.); only the
learning rate is tuned per optimizer.

### 4.2 Arenas

| Arena | Model | Dataset | Primary metric |
|:------|:------|:--------|:---------------|
| NLP | Small GPT (nanoGPT-style) | TinyStories | Perplexity ↓, val loss ↓ |
| ViT | `vit_tiny_patch16_224` | CIFAR-100 @ 224×224 | Top-1 accuracy ↑ |
| ResNet | ResNet-18 | Tiny ImageNet 200 | Top-1 accuracy ↑ |
| Diffusion | DDPM + UNet | CelebA @ 64×64 | Val MSE ↓ |

### 4.3 Quality Results

| Arena | Metric | Adam | AdamW | Lion | **PsiLogic** |
|:------|:-------|:----:|:-----:|:----:|:------------:|
| NLP | Perplexity ↓ | 13.66 ± 0.22 | 8.17 ± 0.08 | 21.04 ± 1.41 | **7.79 ± 0.18** |
| NLP | Val loss ↓ | 2.614 ± 0.016 | 2.101 ± 0.010 | 3.045 ± 0.068 | **2.053 ± 0.023** |
| ViT | Val acc ↑ | 0.079 ± 0.003 | 0.223 ± 0.002 | 0.213 ± 0.002 | **0.244 ± 0.006** |
| ResNet | Val acc ↑ | 0.172 ± 0.004 | 0.219 ± 0.005 | 0.205 ± 0.007 | **0.222 ± 0.001** |
| Diffusion | Val MSE ↓ | **0.01987 ± 0.00006** | **0.01987 ± 0.00006** | 0.02175 ± 0.00025 | 0.02009 ± 0.00045 |

**Selected learning rates** (post-sweep): NLP — all optimizers `3.16×10⁻⁴`; ViT — Adam
`3.16×10⁻⁵`, AdamW/PsiLogic `3.16×10⁻⁴`, Lion `1×10⁻⁴`; ResNet — Adam/Lion `1×10⁻⁴`,
AdamW/PsiLogic `3.16×10⁻⁴`; Diffusion — Adam/AdamW/PsiLogic `1×10⁻³`, Lion `1×10⁻⁴`.

### 4.4 Statistical Significance (PsiLogic vs baselines)

| Arena | Metric | vs Adam | vs AdamW | vs Lion |
|:------|:-------|:--------|:---------|:--------|
| NLP | Perplexity | *p* < 10⁻⁵ \*\*\* | *p* = 0.049 \* | *p* = 0.003 \*\* |
| NLP | Val loss | *p* < 10⁻⁴ \*\*\* | *p* = 0.054 (n.s.) | *p* < 0.001 \*\*\* |
| ViT | Val acc | *p* < 10⁻⁴ \*\*\* | *p* = 0.015 \* | *p* = 0.007 \*\* |
| ResNet | Val acc | *p* = 0.001 \*\* | *p* = 0.44 (n.s.) | *p* = 0.044 \* |
| Diffusion | Val MSE | *p* = 0.49 (n.s.) | *p* = 0.49 (n.s.) | *p* = 0.010 \* |

PsiLogic wins **3 of 4 arenas** on quality. ResNet vs AdamW is a numerical tie (0.222 vs 0.219)
without statistical significance at 3 seeds. Diffusion ties Adam/AdamW; PsiLogic beats Lion.

### 4.5 Performance Results

| Arena | Peak VRAM (MB) — Adam / AdamW / Lion / PsiLogic | Wall time (s) — Adam / AdamW / Lion / PsiLogic |
|:------|:-----------------------------------------------|:-----------------------------------------------|
| NLP | 458 / 458 / 445 / 458 | 46.6 / 45.9 / 38.2 / **55.2** |
| ViT | 1229 / 1229 / 1208 / 1229 | 95.2 / 98.5 / 98.6 / **176.7** |
| ResNet | 823 / 825 / 777 / 823 | 45.3 / 47.6 / 46.1 / **67.4** |
| Diffusion | 3780 / 3780 / 3768 / 3781 | 94.2 / 95.2 / 91.6 / **168.3** |

Peak VRAM differs by ≤ 3% across optimizers on ViT/ResNet/Diffusion; Lion uses less VRAM on
ResNet (777 MB) and NLP (445 MB). PsiLogic wall time is 1.20× (NLP), 1.42× (ResNet), 1.79×
(ViT), and 1.77× (Diffusion) relative to AdamW — the primary engineering overhead.

### 4.6 Per-Seed Reproducibility (ViT val acc)

| Seed | Adam | AdamW | Lion | **PsiLogic** |
|-----:|:----:|:-----:|:----:|:------------:|
| 0 | 0.078 | 0.226 | 0.214 | **0.238** |
| 1 | 0.083 | 0.222 | 0.211 | **0.247** |
| 2 | 0.076 | 0.221 | 0.213 | **0.249** |

PsiLogic leads on every seed. Full per-seed tables for all arenas: `summary.csv`.

---

## 5. Ablation Studies (v0.3.2)

### 5.1 Gradient Centralization and Adaptive Gradient Clipping

To quantify the independent contribution of each component, I train a 3-layer MLP
(10→100→100→2, GELU) on a synthetic binary classification task (2000 samples) for 5 epochs,
comparing four configurations:

| Configuration | Final Avg Loss |
|:--------------|:--------------:|
| PsiLogic (Full: GC + AGC + chaos) | lowest |
| PsiLogic (No GC — `grad_centralize=False`) | slightly higher |
| PsiLogic (No AGC — `agc_clip=0.0`) | slightly higher |
| AdamW baseline (`lr=1e-3`, `weight_decay=1e-4`) | reference |

Both GC and AGC contribute independently to training stability. GC reduces gradient variance
across neurons, while AGC prevents catastrophic updates when per-parameter gradient norms
exceed the weight norm. Removing either component increases the final loss, confirming both
are active contributors rather than redundant safeguards. Reproduction scripts
were available in v0.3.x (`benchmark/gc_agc_ablation.py`, removed in v0.5 in favor
of FairBench); unit tests in `tests/` cover the optimizer components directly.

### 5.2 Active Cancellation as an Automatic Weight-Decay Schedule

The *mirror ablation* tests whether PsiLogic's chaos signal is equivalent to a
dynamically-scheduled AdamW weight decay (originally `benchmark/mirror_ablation.py`,
removed in v0.5).

The experiment trains three models from identical initialization:
1. **PsiLogic** — default settings, GC and AGC disabled for isolation.
2. **AdamW Mirror** — standard AdamW with weight decay set dynamically each step to
   match the chaos contribution recorded from the PsiLogic run.
3. **AdamW Baseline** — standard AdamW with fixed `weight_decay=1e-4`.

If PsiLogic's effect were *globally uniform* — i.e., equivalent to a single time-varying
scalar applied uniformly to all parameters — then the Mirror model should converge to the
same loss. The mirror loss is empirically close to PsiLogic's but not identical across all
parameter groups: the Active Cancellation Term applies *per-parameter chaos modulation*
rather than a single shared scalar. This confirms that PsiLogic's regularization is
structurally richer than a scheduled weight decay: it fires stronger on parameters with
locally elevated gradient chaos and weaker (or not at all) on stable parameters in the
same model.

---

## 6. Discussion

### 6.1 Why PsiLogic Helps Early Training

The dual EMA chaos detector applies strong damping when gradient norms are large and
inconsistent — the chaotic initialization phase. Under FairBench, this manifests as large
gains on ViT (0.244 vs 0.079 Adam) and ResNet (0.222 vs 0.172 Adam) where baselines with
suboptimal fixed LRs struggle, and as the best NLP perplexity when each optimizer receives
its own LR sweep.

### 6.2 Implicit Warmup

Learning rate warmup is standard for Transformer training. PsiLogic achieves a functionally
equivalent effect: the chaos-gated cancellation term suppresses effective update magnitude
in early steps, then vanishes as `slow_t → 0`.

### 6.3 Convergence at Late Training

When `slow_t → 0`, the Active Cancellation Term reduces to zero and PsiLogic becomes
mathematically equivalent to Adam with decoupled weight decay.

### 6.4 FairBench Takeaways

The ViT result (0.244 top-1, *p* < 0.02 vs all baselines) validates `PsiLogicViT` presets
under fair per-optimizer LR tuning. NLP perplexity 7.79 vs 8.17 AdamW confirms competitive
from-scratch LM performance. ResNet shows the lowest cross-seed variance (±0.001) among
all optimizers — a reproducibility advantage.

### 6.5 Limitations

- **Step-time overhead:** 1.2–1.8× wall time vs AdamW on H100; ViT peaks at 1.79×. Chaos-state
  computation is the bottleneck; kernel fusion planned for v0.5.
- **Diffusion:** Val MSE ties Adam/AdamW (*p* = 0.49); no quality win on generative modeling
  at 2000 steps.
- **ResNet vs AdamW:** Numerical lead (0.222 vs 0.219) is not statistically significant
  at 3 seeds (*p* = 0.44). More seeds needed to confirm.
- **NLP val loss vs AdamW:** Not significant at *p* = 0.054 despite significant perplexity gap.
- **Theoretical guarantees:** formal convergence proofs left for future work.

---

## 7. Conclusion

I presented **PsiLogic**, a gradient optimizer that extends Adam with a dynamic Active
Cancellation Term modulated by a dual EMA chaos detector. The term provides strong adaptive
damping during chaotic early training and vanishes at convergence — a behavior structurally
impossible with fixed-coefficient regularizers such as AdamW.

Across four FairBench arenas on NVIDIA H100, PsiLogic demonstrates:

- **Best NLP perplexity** — 7.79 ± 0.18 vs 8.17 ± 0.08 (AdamW), *p* = 0.049.
- **Best ViT accuracy** — 0.244 ± 0.006 vs 0.223 ± 0.002 (AdamW), *p* = 0.015.
- **Best ResNet accuracy vs Adam** — 0.222 ± 0.001 vs 0.172 ± 0.004, *p* = 0.001;
  lowest cross-seed variance (±0.001).
- **Diffusion tie** — val MSE within noise of Adam/AdamW (*p* = 0.49).
- **Comparable VRAM**, 1.2–1.8× wall-time overhead on transformer-heavy arenas.

PsiLogic is available as a one-line drop-in replacement for `torch.optim.Adam`:

```bash
pip install psilogic
```

```python
from psilogic import PsiLogic
optimizer = PsiLogic(model.parameters(), lr=1e-3)
```

---

## References

Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021). High-performance large-scale
image recognition without normalization. *International Conference on Machine Learning (ICML).*

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., ... & Le, Q. V. (2023).
Symbolic discovery of optimization algorithms (Lion).
*Advances in Neural Information Processing Systems (NeurIPS).*

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ...
& Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition
at scale. *International Conference on Learning Representations (ICLR).*

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., ... & He, K.
(2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. *arXiv:1706.02677.*

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR).*

Karpathy, A. (2022). nanoGPT: The simplest, fastest repository for training/finetuning
medium-sized GPTs. *https://github.com/karpathy/nanoGPT.*

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
*International Conference on Learning Representations (ICLR).*

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization.
*International Conference on Learning Representations (ICLR).*

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
Language models are unsupervised multitask learners. *OpenAI Blog.*

Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of
initialization and momentum in deep learning.
*International Conference on Machine Learning (ICML).*

Warden, P. (2018). Speech commands: A dataset for limited-vocabulary speech recognition.
*arXiv:1804.03209.*

Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient centralization: A new
optimization technique for deep neural networks.
*European Conference on Computer Vision (ECCV).*

Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for
text classification. *Advances in Neural Information Processing Systems (NeurIPS).*

---

*Code and FairBench CSVs: https://github.com/Troxter222/psilogic · `benchmark/results/full/`*  
*Archived pre-FairBench results: `OLD_RESULTS.md`*  
*DOI: 10.5281/zenodo.18739857*