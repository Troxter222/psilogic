# ΨLogic — Roadmap to Industry Standard

> *"Fire hard when wrong. Disappear when right."*

**North Star:** When a practitioner starts a new PyTorch training run, they reach for `PsiLogic` before `AdamW` or `Lion` — because it converges faster, is more stable across seeds, and needs less manual tuning.

**Hard target:** Beat **AdamW** and **Lion** (Google, 2023) on a published, reproducible benchmark suite that the community actually trusts — then get integrated into HuggingFace / Lightning / official PyTorch discussions.

**Timeline:** v0.5 (now) → v1.0 (industry-ready) by **Q4 2027**

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

### Quality (FairBench · Jun 2026 · pre-fusion overhead baseline)

| Arena | Task | AdamW | Lion | ΨLogic | Winner | Notes |
|-------|------|-------|------|--------|--------|-------|
| **FairBench NLP** | GPT / TinyStories · 3 seeds · H100 | 8.17 PPL | 21.04 | **7.79** | **ΨLogic** | Per-optimizer LR sweep |
| **FairBench ViT** | ViT-T / CIFAR-100 · 3 seeds · H100 | 0.223 | 0.213 | **0.244** | **ΨLogic** | p=0.015 vs AdamW |
| **FairBench ResNet** | R-18 / Tiny ImageNet · 3 seeds | 0.219 | 0.205 | **0.222** | **Tie AdamW / Beat Adam** | vs Adam p=0.001; vs AdamW p=0.44 |
| **FairBench Diffusion** | DDPM / CelebA · 3 seeds | **0.0199** | 0.0218 | 0.0201 | **Tie AdamW** | Within noise (p=0.49) |
| Legacy (archived) | see [OLD_RESULTS.md](OLD_RESULTS.md) | — | — | — | — | Pre-FairBench runs |

### Performance (wall time ΨLogic / AdamW · same FairBench run)

| Arena | Pre-fusion (Jun 2026) | Target (fused CUDA) | Status |
|-------|----------------------|---------------------|--------|
| NLP | 1.20× | ≤ 1.25× | `[ ]` re-run pending |
| ViT | **1.79×** | **≤ 1.25×** | `[~]` Triton shipped 0.5.0; FairBench re-run pending |
| ResNet | 1.42× | ≤ 1.25× | `[ ]` re-run pending |
| Diffusion | 1.77× | ≤ 1.25× | `[ ]` re-run pending |

**Diagnosis (updated Jul 2026):**

- **FairBench flips ViT green** — ΨLogic 0.244 vs Lion 0.213 under fair LR sweep on H100.
- **NLP from scratch wins** — perplexity 7.79 vs AdamW 8.17 on TinyStories GPT.
- **ResNet** — top-1 0.222, lowest std (±0.001); beats Adam (*p*=0.001), ties AdamW (*p*=0.44).
- **Diffusion ties** — val MSE 0.0201 vs 0.0199 AdamW (*p*=0.49).
- **Triton fusion shipped (0.5.0)** — `psilogic/_cuda/` + `pip install psilogic[cuda]`. Quality scorecard numbers are still from the **pre-fusion** Jun 2026 run; overhead table must be refreshed after FairBench re-run with `[cuda]`.
- **Current focus:** Gate 1C (validate ≤1.25× AdamW step time) → Gate 2 (ImageNet + GPT-2 medium).

**Gates passed:** Gate 0 (v0.4) · Gate 1A (ViT) · Gate 1B (NLP / TinyStories)

---

## Definition of Done — "Industry Standard"

ΨLogic earns the label when **all** of the following are true:

1. **Benchmark dominance** — Wins or statistically ties (p > 0.05) AdamW **and** Lion on ≥ 5 of 7 reference arenas (see Phase 5).
2. **Reproducibility** — Every headline number reproducible with one command + public W&B logs.
3. **Ecosystem** — Listed in HuggingFace `Trainer` docs as supported optimizer; PR merged or officially recommended.
4. **Adoption signal** — ≥ 1,000 PyPI downloads/month **or** cited in ≥ 3 independent papers/repos.
5. **API stability** — Semver v1.0 with no breaking changes for 6 months.
6. **Performance** — ≤ 15% step-time overhead vs AdamW on A100 (documented).

---

# Phase 0 — Engineering Foundation (v0.4) · Jun 2026 · **COMPLETE**

> Package refactor + CI. **Done.**

### Code structure `[x]`

- [x] Split monolith `psilogic.py` → `optimizer.py`, `_chaos.py`, `presets.py`, `param_groups.py`, `convenience.py`
- [x] `py.typed` marker for type checkers
- [x] Ruff lint + format in CI
- [x] GitHub Actions CI (`.github/workflows/ci.yml`)
- [x] GPU test markers (`@pytest.mark.gpu`, auto-skip on CPU)

### v0.4 deliverables `[x]`

- [x] Bump version to `0.4.0` in `pyproject.toml`
- [x] Add `CHANGELOG.md` — document modular refactor, no API breaks
- [x] README: update project structure section (new module layout)
- [x] README: fix Reproduce section — point to `python -m fairbench` (from `benchmark/`)
- [x] Add `psilogic.debug` module for diagnostics
- [x] Run full test suite on GitHub Actions and badge README with CI status

**Gate 0:** `pytest tests/` green on CI · `ruff check` clean · PyPI publish v0.4.0 — **PASSED**

---

# Phase 1 — Close Remaining Gaps (v0.5 – v0.6) · Jul–Sep 2026

> ViT + NLP gates passed (FairBench H100). Phase 1C closes the performance gap.

## 1A — Fix ViT / Vision (beat Lion) · **COMPLETE**

**Root cause (hypothesis):** Triple-decay compounding on patch embeddings + Adam-style update too conservative vs Lion sign-momentum on sparse ViT gradients.

### Experiments to run

| ID | Change | Config | Metric | Target |
|----|--------|--------|--------|--------|
| V1 | `PsiLogicViT` + `vision_defaults()` | γ=0.04, qd=0, τ_scale=2.5 | CIFAR-100 Top-1 | ≥ 0.48 |
| V2 | Lower γ on embeddings via param-group split | embed γ=0.005, blocks γ=0.03 | CIFAR-100 Top-1 | ≥ 0.49 |
| V3 | `lion_blocks=True` in ViT preset | Lion update + chaos damping | CIFAR-100 Top-1 | ≥ 0.50 |
| V4 | Grid `tau_scale` | 1.5, 2.0, 2.5, 3.0, 4.0 | Best of above | ≥ 0.50 |
| V5 | `gamma_T_max = total_steps` | Cosine γ decay | CIFAR-100 Top-1 | ≥ 0.50 |
| V6 | ViT-Base / ImageNet-100 (subset) | 30 epochs, 3 seeds | Top-1 | Beat AdamW |

### Code changes

- [x] **`psilogic/param_groups.py`** — `vit_param_groups(model, ...)`:
  - Separate groups: patch embed, cls token, attention, MLP, norm/bias (no decay)
  - Per-group `gamma`: embed=0.005, attn=0.02, mlp=0.03
  - Optional `lion_blocks=True` — Lion on transformer blocks, Adam on embeddings
- [x] **`psilogic/presets.py`** — `vision_defaults()` aligned with V1–V5 winners
- [x] **`psilogic/convenience.py`** — `PsiLogicViT` auto-calls `vit_param_groups` when passed full model
- [x] **`psilogic/optimizer.py`** — per-group `lion_mode` via param groups (`vit_param_groups(lion_blocks=True)`)
- [x] **`psilogic/debug.py`**:
  - `norm_history(optimizer, model)` → dict of layer-wise weight norms per step
  - `chaos_stats(optimizer)` → fast/slow/spike_rate per param group
- [x] **`benchmark/fairbench/`** — ViT arena uses `PsiLogicViT` (`python -m fairbench --arenas vit`)
- [x] **`tests/test_vit_preset.py`** — smoke test that `vit_param_groups` splits correctly

### Documentation

- [x] Update README FairBench table with H100 numbers (3 seeds)
- [x] Update `PAPER.md` §4.4 — FairBench results with significance tests
- [ ] Add learning curve plot: epoch 1–15, ΨLogic vs Lion vs AdamW

**Gate 1A:** ViT-Small / CIFAR-100 · 3 seeds · ΨLogic Top-1 **≥ Lion mean** — **PASSED**
(FairBench H100: ΨLogic 0.244 vs Lion 0.213, *p* < 0.001)

---

## 1B — Fix GPT / From-Scratch LM (beat AdamW) · **COMPLETE (code)**

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

- [x] **`psilogic/param_groups.py`** — `gpt_param_groups(model, ...)`:
  - Embeddings: γ=0.005, no quantum decay
  - Transformer blocks: γ=0.02
  - LM head: γ=0.01
- [x] **`psilogic/_chaos.py`** — `effective_warmup(step, total_steps, base_warmup)`:
  - Auto-scale warmup as `max(500, total_steps // 20)` when `chaos_warmup=-1`
  - Linear ramp over a quarter of the warmup window
- [x] **`psilogic/presets.py`** — `gpt_scratch_defaults()` from G1–G4 winner
- [x] **`psilogic/convenience.py`** — `PsiLogicGPT` uses `gpt_param_groups` internally
- [x] **`tests/test_chaos_warmup.py`** — verify auto-warmup scaling
- [x] **`tests/test_gpt_preset.py`** — param group split smoke test

### Documentation

- [x] README FairBench NLP table — 3 seeds, updated PPL (supersedes legacy Arena 3)
- [ ] `PAPER.md` — add GPT warmup ablation figure
- [x] Document recommended config block for from-scratch LM in README

**Gate 1B:** NLP / TinyStories · FairBench · ΨLogic PPL **≤ AdamW** — **PASSED**
(FairBench H100: ΨLogic 7.79 vs AdamW 8.17, *p* = 0.049)

---

## 1C — Performance Close (v0.5.x) · **IN PROGRESS**

> Confirm README claim: **≤1.25× AdamW** step time on ViT with `use_fused_cuda=True`.

| ID | Deliverable | Status |
|----|-------------|--------|
| P1 | Triton fused step backend (`psilogic/_cuda/`) | `[x]` 0.5.0 |
| P2 | `tests/test_step_overhead.py` + `scripts/profile_optimizer.py` | `[x]` 0.5.0 |
| P3 | `tests/test_numerical_parity.py` — scalar / foreach / fused match | `[x]` 0.5.0 |
| P4 | Re-run FairBench ViT / ResNet / Diffusion with `psilogic[cuda]` on H100 | `[ ]` |
| P5 | Update README performance table + scorecard above | `[ ]` |
| P6 | CI: optional self-hosted GPU job for `test_gpu_fused_overhead` | `[ ]` |

```bash
# Re-run example (from benchmark/)
pip install -e "../.[cuda,benchmark]"
python -m fairbench --arenas vit resnet diffusion --data-root ./data --output-dir results/fused
python scripts/profile_optimizer.py   # quick local step-time check
```

**Gate 1C:** FairBench ViT wall time ΨLogic/AdamW **≤ 1.25×** (mean over 3 seeds) **or** `profile_optimizer.py` median step ratio ≤ 1.25× on reference hardware.

---

# Phase 2 — Scale Proof (v0.7 – v0.8) · Oct 2026 – Mar 2027

> Prove ΨLogic works at the scale people actually train — not just toy benchmarks.

## 2A — Computer Vision at Scale

| Benchmark | Model | Dataset | Epochs | Seeds | vs | Target |
|-----------|-------|---------|--------|-------|-----|--------|
| **CV-1** | ResNet-50 | ImageNet-1k | 90 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-2** | ViT-Base | ImageNet-1k | 300 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-3** | ConvNeXt-T | CIFAR-100 | 100 | 5 | AdamW, Lion | Beat FairBench ViT protocol |

### Code / infra

- [ ] **`benchmark/imagenet/`** (new directory):
  - `train_imagenet.py` — DDP-ready, AMP bf16, cosine LR
  - Shared config dataclass (match `fairbench` style)
- [x] **`psilogic/optimizer.py`** — `profile_step_time=True` records `last_step_time_ms` / `step_time_ms_ema`
- [x] **`scripts/profile_optimizer.py`** — ViT-like step profiler vs AdamW
- [ ] Mixed precision: verify `torch.amp` path in all scale benchmarks (no deprecated API)
- [x] Gradient accumulation test — `tests/test_grad_accum.py`
- [ ] Multi-GPU DDP benchmark script — 2×GPU scaling efficiency report

## 2B — NLP / LM at Scale

| Benchmark | Model | Dataset | Steps | Seeds | vs | Target |
|-----------|-------|---------|-------|-------|-----|--------|
| **NLP-1** | BERT-large | GLUE (MNLI, QQP, QNLI, SST-2) | 3 ep each | 3 | AdamW | Avg ≥ AdamW |
| **NLP-2** | GPT-2 medium (345M) | OpenWebText | 50k | 3 | AdamW | PPL ≤ AdamW |
| **NLP-3** | Whisper-small | LibriSpeech fine-tune | 10 ep | 3 | AdamW | WER ≤ AdamW |

### Code / infra

- [x] **`psilogic/convenience.py`** — `PsiLogicWhisper` preset (audio/speech)
- [x] **`psilogic/presets.py`** — `whisper_defaults()`, `glue_defaults()`
- [ ] FSDP test with 2 GPUs — extend `tests/test_fsdp.py` to assert loss decreases
- [ ] Memory benchmark: peak VRAM vs AdamW on GPT-2 medium

**Gate 2:** Win or tie on **CV-1 (ResNet-50 ImageNet)** AND **NLP-2 (GPT-2 medium)** — these are the numbers people cite.

---

# Phase 3 — Algorithm Improvements (v0.9) · Apr–Jun 2027

> Research-backed upgrades to the core engine — ablations and remaining research items.

## 3A — Core optimizer enhancements

| Feature | File | Description | Status |
|---------|------|-------------|--------|
| **Auto-γ scheduler** | `_chaos.py` | Reduce γ when slow EMA signals convergence | `[x]` 0.4.0 (`gamma_auto`) |
| **Per-layer chaos sync (DDP)** | `optimizer.py` | All-reduce chaos signal across ranks | `[x]` 0.4.0 (`sync_chaos_ddp`) |
| **Fused CUDA kernel** | `psilogic/_cuda/` | Triton fused step — target ≤ 1.25× AdamW | `[x]` 0.5.0 |
| **Zero-config mode** | `convenience.py` | `PsiLogic.auto(model)` | `[x]` 0.4.0 |
| **State dict versioning** | `optimizer.py` | Schema v2 + v0.3 migration | `[x]` 0.4.0 |
| **Muon compatibility** | `optimizer.py` | Orthogonalized updates for 2D params (hybrid) | `[ ]` explore |
| **ValueError validation** | `optimizer.py` | Replace `assert` for hyperparameter checks | `[x]` 0.4.0 |

### Remaining 3A items

- [x] **`psilogic/_chaos.py`** — `auto_gamma()`, `get_chaos_metrics(state) -> dict`
- [x] **`psilogic/optimizer.py`** — `load_state_dict` v0.3 migration, `sync_chaos_ddp`, `ValueError` validation
- [x] **`psilogic/__init__.py`** — export `PsiLogic.auto`, `get_chaos_metrics`
- [x] **Performance** — foreach vs fused parity tests; graceful fallback when Triton / foreach unavailable
- [ ] **Muon hybrid** — optional orthogonalized updates for 2D weight matrices

## 3B — Ablation program (publish all results)

Run and document in `PAPER.md` + W&B:

- [ ] GC on/off
- [ ] AGC on/off
- [ ] adaptive_tau on/off
- [ ] quantum_decay on/off
- [ ] lion_mode on/off
- [ ] chaos entirely disabled (= AdamW equivalent?) — verify mirror test
- [ ] γ schedule: constant vs cosine vs auto

**Gate 3:** Step overhead ≤ 15% vs AdamW (documented on A100) · Ablation suite published · Auto-config validated on 3 unseen architectures without manual tuning

---

# Phase 4 — Credibility & Visibility (v0.95) · Jul–Sep 2027

> Best algorithm in the world means nothing if nobody knows.

## 4A — Scientific publication

- [x] LaTeX source — [arxiv/paper.tex](arxiv/paper.tex)
- [x] PDF build pipeline — [scripts/build_arxiv_pdf.py](scripts/build_arxiv_pdf.py)
- [ ] Sync paper with v0.5 FairBench results + fused-CUDA overhead numbers
- [ ] Submit **arXiv preprint** (cs.LG)
  - Required sections: method, ablations, 7-arena table, ImageNet result, limitations
- [ ] Update **Zenodo DOI** for each major release
- [ ] Reproduce **one published AdamW baseline** from literature (e.g. original GPT-2 training curve)
- [ ] Reproduce **Lion paper** ViT result and show ΨLogic match/beat
- [ ] Write formal **convergence sketch** (not full proof — honest about assumptions)

## 4B — Ecosystem integrations

| Platform | Deliverable | File | Status |
|----------|-------------|------|--------|
| **HuggingFace** | `create_psilogic_optimizer` / `psilogic_trainer_class` | `psilogic/integrations/hf.py` | `[x]` |
| **Lightning** | `configure_psilogic` + `ChaosMonitorCallback` | `psilogic/integrations/lightning.py` | `[x]` |
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

- [x] Unit test with tiny HF model — `tests/test_integrations.py` (1-step smoke)
- [x] Example script: `examples/hf_sst2_finetune.py`
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
- [x] Security policy (`SECURITY.md`) — no telemetry, no network calls in optimizer

## 5B — Flagship benchmark (the one number that gets cited)

- [ ] **LLaMA-style 1B model · 100k steps · OpenWebText** vs AdamW vs Lion
  - Multi-GPU (4×A100 80GB)
  - Report: final PPL, steps-to-PPL-15, wall-clock time, seed variance
  - Target: PPL ≤ AdamW at same step count **OR** reach target PPL in ≤ 80% steps

## 5C — Reference benchmark suite (7 arenas)

| # | Arena | Must beat | FairBench overlap |
|---|-------|-----------|-------------------|
| 1 | CIFAR-10 / ResNet-18 | AdamW | ResNet arena (partial) |
| 2 | CIFAR-100 / ViT-Small | **Lion** | ViT arena `[x]` |
| 3 | ImageNet-1k / ResNet-50 | AdamW | Phase 2 CV-1 |
| 4 | BERT-base / SST-2 | AdamW | HF example exists |
| 5 | GPT-2 / Wikitext-2 scratch | AdamW | NLP arena (partial) |
| 6 | GPT-2 medium / OpenWebText | AdamW | Phase 2 NLP-2 |
| 7 | nanoGPT / Tiny Shakespeare | AdamW (val loss) | — |

- [ ] One-command reproduction: `cd benchmark && python -m fairbench` (FairBench 4 arenas) + scale suite TBD
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

Gates 0, 1A, and 1B are **passed**. Current focus is Gate 1C, then Gate 2.

```
Now (Jul 2026)     Gate 1C → FairBench re-run with psilogic[cuda], refresh overhead numbers
Aug–Sep 2026       v0.6 polish → learning curve plots, W&B tagging, vision_defaults retune if needed
Oct 2026 – Mar 27  Gate 2  → ImageNet-1k (CV-1) + GPT-2 medium (NLP-2)
Apr–Jun 2027       Phase 3 → ablation program, Muon hybrid explore
Jul–Sep 2027       Phase 4 → arXiv submit (paper.tex ready), HF docs, community push
Q4 2027            Gate 5  → v1.0, 5/7 arenas, 1B flagship
```

---

# Technical Debt Register

Issues to fix alongside feature work:

| Item | Location | Action |
|------|----------|--------|
| FairBench results pre-fusion | `benchmark/results/full/` | Re-run with `psilogic[cuda]`, commit new aggregate |
| `gamma_auto` untested | `tests/` | Add behavioral test for γ reduction at convergence |
| `sync_chaos_ddp` untested | `tests/` | Multi-process DDP test with `sync_chaos_ddp=True` |
| Whisper / GLUE presets untested | `tests/` | Smoke tests for `PsiLogicWhisper`, `glue_defaults` |
| FSDP loss assertion | `tests/test_fsdp.py` | Assert loss decreases over N steps |
| GPU CI gap | `.github/workflows/` | Self-hosted runner or nightly GPU workflow |
| arXiv out of sync | `arxiv/paper.tex` | Sync with FairBench + fusion results before submit |
| External integrations | Axolotl, LLaMA-Factory | Track as external PRs |
| Learning curve plots | docs / W&B | Epoch 1–15 ViT comparison figure |

---

# Test Coverage

**17 test modules** under `tests/`. GPU-marked tests (`@pytest.mark.gpu`, `@pytest.mark.multi_gpu`) auto-skip on CPU CI.

| Area | Covered | Gap |
|------|---------|-----|
| Core optimizer | convergence, Lion mode, γ decay, checkpoints, determinism | — |
| Presets | `test_vit_preset`, `test_gpt_preset`, `test_auto_config` | Whisper, GLUE |
| Chaos | `test_chaos_warmup`, `test_debug` | `gamma_auto` behavior |
| Backends | `test_numerical_parity`, `test_step_overhead` | GPU overhead not enforced in CI |
| Distributed | `test_ddp`, `test_fsdp` (smoke) | `sync_chaos_ddp`, FSDP loss decrease |
| Integrations | `test_integrations` (HF + Lightning) | DistilBERT-specific 1-step test optional |
| Training patterns | `test_grad_accum`, `test_amp`, `test_torch_compile` | — |
| FairBench harness | manual / benchmark-only | No pytest e2e for arenas |

Add tests alongside Phase 2 scale work — regressions at ImageNet scale are expensive to debug.

---

# Version Map

| Version | Phase | Key deliverable | Status |
|---------|-------|-----------------|--------|
| **0.4.0** | 0 | Modular package · CI · ruff · py.typed | Released |
| **0.5.0** | 1A + 1B + 1C (partial) | ViT/NLP presets · Triton fusion · parity tests | **Released (current PyPI)** |
| **0.5.1** | 1C | FairBench perf re-run · updated overhead docs | Planned |
| **0.6.0** | 1 wrap-up | Learning curves · doc polish | Planned |
| **0.7.0** | 2A | ResNet-50 ImageNet result | Planned |
| **0.8.0** | 2B | GPT-2 medium · GLUE · Whisper benchmarks | Planned |
| **0.9.0** | 3 | Ablations · Muon explore · ≤ 15% overhead documented | Planned |
| **0.95.0** | 4 | arXiv · HuggingFace docs · W&B public | Planned |
| **1.0.0** | 5 | API freeze · 5/7 arenas · 1B flagship · industry release | Planned |

---

# How To Track Progress

1. Check off items in this file as they ship.
2. Update the **Scorecard** table at the top after every benchmark run.
3. Tag releases on GitHub matching the Version Map.
4. Mirror headline numbers in `README.md` and `PAPER.md` — never let docs drift from data.
5. Run `pytest tests/ -v` before every release — zero regressions.
6. Keep **Test Coverage** gaps in sync when adding optimizer features.

---

*Repository: https://github.com/Troxter222/psilogic*
*DOI: https://doi.org/10.5281/zenodo.18739857*
*Author: Ali (Troxter222)*
