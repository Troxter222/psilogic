# ΨLogic — Roadmap to Industry Standard

> *"Fire hard when wrong. Disappear when right."*

**North Star:** When a practitioner starts a new PyTorch training run, they reach for `PsiLogic` before `AdamW` or `Lion` — because it converges faster, is more stable across seeds, and needs less manual tuning.

**Hard target:** Beat **AdamW** and **Lion** (Google, 2023) on a published, reproducible benchmark suite that the community actually trusts — then get integrated into HuggingFace / Lightning / official PyTorch discussions.

**Timeline:** v0.4 (now) → v1.0 (industry-ready) by **Q4 2027**

---

## Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | TODO |
| `[~]` | In progress |
| `[x]` | Done |
| **Gate** | Must pass before next phase ships |

---

## Scorecard — Where We Stand Today (v0.5.0)

| Arena | Task | AdamW | Lion | ΨLogic | Winner | Notes |
|-------|------|-------|------|--------|--------|-------|
| **FairBench NLP** | GPT / TinyStories · 3 seeds · H100 | 8.17 PPL | 21.04 | **7.79** | **ΨLogic** | Per-optimizer LR sweep |
| **FairBench ViT** | ViT-T / CIFAR-100 · 3 seeds · H100 | 0.223 | 0.213 | **0.244** | **ΨLogic** | p=0.015 vs AdamW |
| **FairBench ResNet** | R-18 / Tiny ImageNet · 3 seeds | 0.219 | 0.205 | **0.222** | **ΨLogic** | vs Adam p=0.001; vs AdamW p=0.44 |
| **FairBench Diffusion** | DDPM / CelebA · 3 seeds | **0.0199** | 0.0218 | 0.0201 | ~Tie AdamW | Within noise |
| Legacy (archived) | see [OLD_RESULTS.md](OLD_RESULTS.md) | — | — | — | — | Pre-FairBench runs |

**Diagnosis (updated Jun 2026):**

- **FairBench flips ViT green** — ΨLogic 0.244 vs Lion 0.213 under fair LR sweep on H100.
- **NLP from scratch wins** — perplexity 7.79 vs AdamW 8.17 on TinyStories GPT.
- **ResNet** — top-1 0.222, lowest std (±0.001); beats Adam (*p*=0.001), ties AdamW (*p*=0.44).
- **Diffusion ties** — val MSE 0.0201 vs 0.0199 AdamW (*p*=0.49).
- **Trade-off:** ΨLogic is 1.2–1.8× slower (peak 1.79× ViT); kernel fusion is Phase 1C.

Phase 1B (legacy GPT-2/Wikitext) is superseded by FairBench NLP arena — **Gate 1B: PASSED**.

---

## Definition of Done — "Industry Standard"

ΨLogic earns the label when **all** of the following are true:

1. **Benchmark dominance** — Wins or statistically ties (p > 0.05) AdamW **and** Lion on ≥ 5 of 7 reference arenas (see Phase 4).
2. **Reproducibility** — Every headline number reproducible with one command + public W&B logs.
3. **Ecosystem** — Listed in HuggingFace `Trainer` docs as supported optimizer; PR merged or officially recommended.
4. **Adoption signal** — ≥ 1,000 PyPI downloads/month **or** cited in ≥ 3 independent papers/repos.
5. **API stability** — Semver v1.0 with no breaking changes for 6 months.
6. **Performance** — ≤ 15% step-time overhead vs AdamW on A100 (documented).

---

# Phase 0 — Engineering Foundation (v0.4) · Jun 2026

> Package refactor + CI. **Mostly done.** Finish remaining items.

### Code structure `[x]`

- [x] Split monolith `psilogic.py` → `optimizer.py`, `_chaos.py`, `presets.py`, `param_groups.py`, `convenience.py`
- [x] `py.typed` marker for type checkers
- [x] Ruff lint + format in CI
- [x] GitHub Actions CI (`.github/workflows/ci.yml`)
- [x] GPU test markers (`@pytest.mark.gpu`, auto-skip on CPU)

### Remaining v0.4 tasks

- [x] Bump version to `0.4.0` in `pyproject.toml` after Phase 0 gate passes
- [x] Add `CHANGELOG.md` — document modular refactor, no API breaks
- [x] README: update project structure section (new module layout)
- [x] README: fix Reproduce section — point to `benchmark/run_benchmark.py` (not deleted scripts)
- [x] Add `psilogic.debug` module stub for future diagnostics
- [x] Run full test suite on GitHub Actions and badge README with CI status

**Gate 0:** `pytest tests/` green on CI · `ruff check` clean · PyPI publish v0.4.0

---

# Phase 1 — Close Remaining Gaps (v0.5 – v0.6) · Jul–Sep 2026

> ViT + NLP gates passed (FairBench H100). Focus: step-time overhead + scale benchmarks.

## 1A — Fix ViT / Vision (beat Lion)

**Root cause (hypothesis):** Triple-decay compounding on patch embeddings + Adam-style update too conservative vs Lion sign-momentum on sparse ViT gradients.

### Experiments to run

| ID | Change | Config | Metric | Target |
|----|--------|--------|--------|--------|
| V1 | `PsiLogicViT` + `vision_defaults()` | γ=0.04, qd=0, τ_scale=2.5 | CIFAR-100 Top-1 | ≥ 0.48 |
| V2 | Lower γ on embeddings via `nlp_param_groups`-style split | embed γ=0.005, blocks γ=0.03 | CIFAR-100 Top-1 | ≥ 0.49 |
| V3 | `lion_mode=True` in ViT preset | Lion update + chaos damping | CIFAR-100 Top-1 | ≥ 0.50 |
| V4 | Grid `tau_scale` | 1.5, 2.0, 2.5, 3.0, 4.0 | Best of above | ≥ 0.50 |
| V5 | `gamma_T_max = total_steps` | Cosine γ decay | CIFAR-100 Top-1 | ≥ 0.50 |
| V6 | ViT-Base / ImageNet-100 (subset) | 30 epochs, 3 seeds | Top-1 | Beat AdamW |

### Code changes

- [x] **`psilogic/param_groups.py`** — add `vit_param_groups(model, ...)`:
  - Separate groups: patch embed, cls token, attention, MLP, norm/bias (no decay)
  - Per-group `gamma`: embed=0.005, attn=0.02, mlp=0.03
- [~] **`psilogic/presets.py`** — update `vision_defaults()` based on V1–V5 winner
- [x] **`psilogic/convenience.py`** — `PsiLogicViT` auto-calls `vit_param_groups` when passed full model
- [ ] **`psilogic/optimizer.py`** — optional: per-group `lion_mode` override (ViT blocks Lion, embed Adam)
- [x] **`psilogic/debug.py`** (new):
  - `norm_history(optimizer, model)` → dict of layer-wise weight norms per step
  - `chaos_stats(optimizer)` → fast/slow/spike_rate per param group
- [x] **`benchmark/run_benchmark.py`** — add `--preset vit` shortcut using `PsiLogicViT`
- [x] **`tests/test_vit_preset.py`** (new) — smoke test that `vit_param_groups` splits correctly

### Documentation

- [x] Update README Arena 2 table with new numbers (FairBench H100, 3 seeds)
- [x] Update `PAPER.md` §4.4 — FairBench results with significance tests
- [ ] Add learning curve plot: epoch 1–15, ΨLogic vs Lion vs AdamW

**Gate 1A:** ViT-Small / CIFAR-100 · 3 seeds · ΨLogic Top-1 **≥ Lion mean** — **PASSED**
(FairBench H100: ΨLogic 0.244 vs Lion 0.213, *p* < 0.001)

---

## 1B — Fix GPT / From-Scratch LM (beat AdamW)

**Root cause (hypothesis):** Chaos fires too early on from-scratch LM — large gradient noise mistaken for instability; `max_cancel` helps but warmup still too short.

### Experiments to run

| ID | Change | Config | Metric | Target |
|----|--------|--------|--------|--------|
| G1 | `PsiLogicGPT` default | γ=0.02, warmup=-1, max_cancel=0.03 | Wikitext-2 PPL @ 3k | ≤ 310 |
| G2 | Longer warmup | `chaos_warmup=500` | Wikitext-2 PPL | ≤ 305 |
| G3 | Lower γ + cosine decay | γ=0.01, `gamma_T_max=3000` | Wikitext-2 PPL | ≤ 305 |
| G4 | Embed-specific γ | embed γ=0.005 via param groups | Wikitext-2 PPL | ≤ 302 |
| G5 | nanoGPT / Tiny Shakespeare | γ=0.01 + gamma_T_max | Val loss | ≤ AdamW |
| G6 | OpenWebText subset · 10k steps | GPT-2 small | Val PPL | Beat AdamW |

### Code changes

- [ ] **`psilogic/param_groups.py`** — add `gpt_param_groups(model, ...)`:
  - Embeddings: γ=0.005, no quantum decay
  - Transformer blocks: γ=0.02
  - LM head: γ=0.01
- [ ] **`psilogic/_chaos.py`** — add `effective_warmup(step, total_steps, base_warmup)`:
  - Auto-scale warmup as `max(500, total_steps // 20)` when `chaos_warmup=-1`
  - Unit test: warmup=0 at step 0, full chaos after warmup
- [ ] **`psilogic/presets.py`** — update `gpt_scratch_defaults()` from G1–G4 winner
- [ ] **`psilogic/convenience.py`** — `PsiLogicGPT` uses `gpt_param_groups` internally
- [ ] **`tests/test_chaos_warmup.py`** (new) — verify auto-warmup scaling
- [ ] **`tests/test_gpt_preset.py`** (new) — param group split smoke test

### Documentation

- [ ] README Arena 3 table — 5 seeds, updated PPL
- [ ] `PAPER.md` — add GPT warmup ablation figure
- [ ] Document recommended config block for from-scratch LM in README

**Gate 1B:** NLP / TinyStories · FairBench · ΨLogic PPL **≤ AdamW** — **PASSED**
(FairBench H100: ΨLogic 7.79 vs AdamW 8.17, *p* = 0.049)

---

# Phase 2 — Scale Proof (v0.7 – v0.8) · Oct 2026 – Mar 2027

> Prove ΨLogic works at the scale people actually train — not just toy benchmarks.

## 2A — Computer Vision at Scale

| Benchmark | Model | Dataset | Epochs | Seeds | vs | Target |
|-----------|-------|---------|--------|-------|-----|--------|
| **CV-1** | ResNet-50 | ImageNet-1k | 90 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-2** | ViT-Base | ImageNet-1k | 300 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-3** | ConvNeXt-T | CIFAR-100 | 100 | 5 | AdamW, Lion | Beat Arena 2 protocol |

### Code / infra

- [ ] **`benchmark/imagenet/`** (new directory):
  - `train_imagenet.py` — DDP-ready, AMP bf16, cosine LR
  - Shared config dataclass (match `run_benchmark.py` style)
- [ ] **`psilogic/optimizer.py`** — profile step time:
  - `PsiLogic.step()` timing hook or `@torch.profiler` wrapper in benchmark
  - Document: μs/param vs AdamW
- [ ] Mixed precision: verify `torch.amp` path in all benchmarks (no deprecated API)
- [ ] Gradient accumulation test — `tests/test_grad_accum.py`
- [ ] Multi-GPU DDP benchmark script — 2×GPU scaling efficiency report

## 2B — NLP / LM at Scale

| Benchmark | Model | Dataset | Steps | Seeds | vs | Target |
|-----------|-------|---------|-------|-------|-----|--------|
| **NLP-1** | BERT-large | GLUE (MNLI, QQP, QNLI, SST-2) | 3 ep each | 3 | AdamW | Avg ≥ AdamW |
| **NLP-2** | GPT-2 medium (345M) | OpenWebText | 50k | 3 | AdamW | PPL ≤ AdamW |
| **NLP-3** | Whisper-small | LibriSpeech fine-tune | 10 ep | 3 | AdamW | WER ≤ AdamW |

### Code / infra

- [ ] **`psilogic/convenience.py`** — add `PsiLogicWhisper` preset (audio/speech)
- [ ] **`psilogic/presets.py`** — `whisper_defaults()`, `glue_defaults()`
- [ ] FSDP test with 2 GPUs — extend `tests/test_fsdp.py` to assert loss decreases
- [ ] Memory benchmark: peak VRAM vs AdamW on GPT-2 medium

**Gate 2:** Win or tie on **CV-1 (ResNet-50 ImageNet)** AND **NLP-2 (GPT-2 medium)** — these are the numbers people cite.

---

# Phase 3 — Algorithm Improvements (v0.9) · Apr–Jun 2027

> Research-backed upgrades to the core engine — not just hyperparameter tuning.

## 3A — Core optimizer enhancements

| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| **Auto-γ scheduler** | `_chaos.py` | Detect convergence via slow_EMA derivative → auto-reduce γ | High |
| **Per-layer chaos sync (DDP)** | `optimizer.py` | All-reduce mean chaos signal across ranks before step | High |
| **Fused CUDA kernel** | `psilogic/_cuda/` (new) | Custom Triton/C++ kernel for step — target ≤ 5% overhead | Medium |
| **Muon compatibility** | `optimizer.py` | Optional orthogonalized updates for 2D params (explore hybrid) | Low |
| **Zero-config mode** | `convenience.py` | `PsiLogic.auto(model)` — infer preset from architecture | High |
| **State dict versioning** | `optimizer.py` | `state_dict` schema v2 for backward-compatible checkpoints | Medium |

### Specific changes

- [ ] **`psilogic/_chaos.py`**
  - `auto_gamma(slow_t, step, gamma_base)` — reduce γ when `slow_t < 0.1`
  - Export chaos metrics: `get_chaos_metrics(state) -> dict`
- [ ] **`psilogic/optimizer.py`**
  - `load_state_dict` migration from v0.3 monolith format
  - Optional `sync_chaos_ddp=True` flag
  - Replace `assert` validation with `ValueError` (library best practice)
- [ ] **`psilogic/__init__.py`**
  - Export `PsiLogic.auto`, `get_chaos_metrics`
- [ ] **Performance**
  - Benchmark foreach vs fused on A100
  - Fall back gracefully when `torch._foreach_*` unavailable

## 3B — Ablation program (publish all results)

Run and document in `PAPER.md` + W&B:

- [ ] GC on/off
- [ ] AGC on/off
- [ ] adaptive_tau on/off
- [ ] quantum_decay on/off
- [ ] lion_mode on/off
- [ ] chaos entirely disabled (= AdamW equivalent?) — verify mirror test
- [ ] γ schedule: constant vs cosine vs auto

**Gate 3:** Step overhead ≤ 15% vs AdamW · Auto-config works on 3 unseen architectures without manual tuning

---

# Phase 4 — Credibility & Visibility (v0.95) · Jul–Sep 2027

> Best algorithm in the world means nothing if nobody knows.

## 4A — Scientific publication

- [ ] Submit **arXiv preprint** (cs.LG) — convert `PAPER.md` to LaTeX
  - Required sections: method, ablations, 7-arena table, ImageNet result, limitations
- [ ] Update **Zenodo DOI** for each major release
- [ ] Reproduce **one published AdamW baseline** from literature (e.g. original GPT-2 training curve)
- [ ] Reproduce **Lion paper** ViT result and show ΨLogic match/beat
- [ ] Write formal **convergence sketch** (not full proof — honest about assumptions)

## 4B — Ecosystem integrations

| Platform | Deliverable | File | Status |
|----------|-------------|------|--------|
| **HuggingFace** | `TrainingArguments(optim="psilogic")` | `psilogic/integrations/hf.py` | `[x]` |
| **Lightning** | `PsiLogicOptimizer` callback | `psilogic/integrations/lightning.py` | `[x]` |
| **torchtune** | Example config YAML | `examples/torchtune/` | `[x]` |
| **Axolotl** | PR to add psilogic option | External PR | `[ ]` |
| **LLaMA-Factory** | PR to add psilogic option | External PR | `[ ]` |

### HuggingFace integration spec

```python
# psilogic/integrations/hf.py
from transformers import Trainer

def create_psilogic_optimizer(model, args, **kwargs):
    from psilogic import PsiLogic
    preset = kwargs.pop("preset", "auto")  # "nlp" | "gpt" | "vit" | "auto"
    ...

# Usage:
# TrainingArguments(optim="psilogic", optim_args={"preset": "nlp"})
```

- [ ] Unit test with tiny HF model (DistilBERT, 1 step)
- [ ] Example notebook: `examples/hf_sst2_finetune.ipynb`
- [ ] README badge: "Works with HuggingFace Transformers"

## 4C — Community & marketing

- [ ] Public **Weights & Biases** project — all benchmark runs, tagged by version
- [ ] Blog post: *"Why your optimizer should know when it's confused"*
- [ ] Reddit r/MachineLearning — post with 7-arena table + learning curve GIFs
- [ ] Twitter/X thread — epoch 1–10 CIFAR advantage visual
- [ ] PyPI download badge in README
- [ ] Add **OPTIMIZER_COMPARISON.md** — honest head-to-head table, including losses

**Gate 4:** arXiv live · HF integration merged or documented · W&B public · ≥ 500 PyPI downloads/month

---

# Phase 5 — v1.0 Industry Release · Q4 2027

> The release where ΨLogic becomes a safe default choice.

## 5A — API & quality

- [ ] Semver **v1.0.0** — API freeze announcement
- [ ] **mypy strict** passes on `psilogic/`
- [ ] **100% docstring coverage** on public API
- [ ] **Sphinx docs** site on GitHub Pages
  - Quickstart, API reference, preset guide, benchmark reproduction
- [ ] **`torch.optim` compatibility audit** — works anywhere `AdamW` works
- [ ] Deprecation policy document in `CONTRIBUTING.md`
- [x] Security policy (`SECURITY.md`) — no telemtry, no network calls in optimizer

## 5B — Flagship benchmark (the one number that gets cited)

- [ ] **LLaMA-style 1B model · 100k steps · OpenWebText** vs AdamW vs Lion
  - Multi-GPU (4×A100 80GB)
  - Report: final PPL, steps-to-PPL-15, wall-clock time, seed variance
  - Target: PPL ≤ AdamW at same step count **OR** reach target PPL in ≤ 80% steps

## 5C — Reference benchmark suite (7 arenas)

| # | Arena | Must beat |
|---|-------|-----------|
| 1 | CIFAR-10 / ResNet-18 | AdamW |
| 2 | CIFAR-100 / ViT-Small | **Lion** |
| 3 | ImageNet-1k / ResNet-50 | AdamW |
| 4 | BERT-base / SST-2 | AdamW |
| 5 | GPT-2 / Wikitext-2 scratch | AdamW |
| 6 | GPT-2 medium / OpenWebText | AdamW |
| 7 | nanoGPT / Tiny Shakespeare | AdamW (val loss) |

- [ ] One-command reproduction: `python benchmark/run_all.py --suite v1`
- [ ] CI nightly benchmark on self-hosted GPU runner (optional)
- [ ] Submit to **ICLR 2028** or **NeurIPS 2028** with 1B result

**Gate 5 (v1.0):** ≥ 5/7 arenas green · v1.0 on PyPI · Sphinx docs live · HF integration · 1B result published

---

# Phase 6 — Beyond v1.0 (2028+)

- [ ] PR to **HuggingFace transformers** — first-class `psilogic` in `Trainer` enum
- [ ] PR to **PyTorch** — mention in `torch.optim` docs as third-party recommended optimizer
- [ ] **Chaos dashboard** — real-time training health UI (W&B custom panel)
- [ ] **Distributed chaos** for FSDP — global chaos signal across shards
- [ ] **Second-order hybrid** — explore Shampoo/Muon + chaos damping
- [ ] **Early stopping proxy** — use slow_EMA as convergence detector
- [ ] **Hardware partners** — NVIDIA / AMD optimizer blog mention

---

# Priority Stack — What To Do **Right Now**

Ordered by impact. Do not skip to Phase 3 before Gates 1A and 1B pass.

```
Week 1–2   Gate 0  → finish v0.4 (CHANGELOG, README fix, CI badge, publish 0.4.0)
Week 3–6   Gate 1A → ViT experiments V1–V5, implement vit_param_groups, flip Arena 2
Week 7–10  Gate 1B → GPT experiments G1–G5, implement gpt_param_groups, flip Arena 3
Week 11    Write   → arXiv draft intro + method (even before full scale results)
Month 4–6  Gate 2  → ResNet-50 ImageNet + GPT-2 medium (the citeable numbers)
Month 7–9  Phase 3 → auto-γ, zero-config, performance optimization
Month 10   Phase 4 → HuggingFace integration + arXiv submit + W&B public
Month 12   Gate 5  → v1.0 + 1B flagship + 5/7 arenas green
```

---

# Technical Debt Register

Issues to fix alongside feature work:

| Item | Location | Action |
|------|----------|--------|
| `assert` for validation | `optimizer.py` | Replace with `raise ValueError` |
| No `load_state_dict` migration | `optimizer.py` | Add schema version field |
| Benchmark duplicates PsiLogic | `benchmark/run_benchmark.py` | Import from `psilogic` package, delete inline copy |
| README reproduce commands wrong | `README.md` | Point to `benchmark/run_benchmark.py` |
| No CHANGELOG | root | Create, maintain per release |
| Telegram tokens in benchmark | `run_benchmark.py` | Move to env vars (`PSILOGIC_TG_TOKEN`) |
| FSDP test skips silently | `tests/test_fsdp.py` | Require 2 GPU in CI self-hosted job |
| No step-time benchmark | tests/ | Add `tests/test_step_overhead.py` |

---

# Version Map

| Version | Phase | Key deliverable |
|---------|-------|-----------------|
| **0.3.2** | — | Current PyPI · monolith · v0.3.2 benchmarks |
| **0.4.0** | 0 | Modular package · CI · ruff · py.typed |
| **0.5.0** | 1A | ViT fix · `vit_param_groups` · Arena 2 green |
| **0.6.0** | 1B | GPT fix · `gpt_param_groups` · Arena 3 green |
| **0.7.0** | 2A | ResNet-50 ImageNet result |
| **0.8.0** | 2B | GPT-2 medium · GLUE · Whisper |
| **0.9.0** | 3 | Auto-γ · zero-config · ≤ 15% overhead |
| **0.95.0** | 4 | arXiv · HuggingFace · W&B public |
| **1.0.0** | 5 | API freeze · 5/7 arenas · 1B flagship · industry release |

---

# How To Track Progress

1. Check off items in this file as they ship.
2. Update the **Scorecard** table at the top after every benchmark run.
3. Tag releases on GitHub matching the Version Map.
4. Mirror headline numbers in `README.md` and `PAPER.md` — never let docs drift from data.
5. Run `pytest tests/ -v` before every release — zero regressions.

---

*Repository: https://github.com/Troxter222/psilogic*
*DOI: https://doi.org/10.5281/zenodo.18739857*
*Author: Ali (Troxter222)*
