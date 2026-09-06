# ΨLogic — Roadmap to Industry Standard

> *"Fire hard when wrong. Disappear when right."*

**Last updated:** 2026-09-06 · **Current release:** v0.6.0

**North Star:** When a practitioner starts a new PyTorch training run, they reach for
`PsiLogic` before `AdamW` or `Lion` — because it converges faster, is more stable across
seeds, and needs less manual tuning.

**Hard target:** Beat **AdamW** and **Lion** on a published, reproducible benchmark suite
the community trusts — then earn HuggingFace / Lightning / PyTorch ecosystem mindshare.

**Timeline:** v0.6 (now) → v1.0 (industry-ready) by **Q4 2027**

---

## Contents

- [Now / Next / Later](#now--next--later)
- [Gate dashboard](#gate-dashboard)
- [Scorecard](#scorecard--where-we-stand-today-v060)
- [Definition of Done](#definition-of-done--industry-standard)
- [Non-goals (next 2 quarters)](#non-goals-next-2-quarters)
- [Risk register](#risk-register)
- [Phase 0–1 (shipped)](#phase-0--engineering-foundation-v04--complete)
- [Phase 1C (in progress)](#phase-1c--performance-close-v05x--in-progress)
- [Phase 2–6](#phase-2--scale-proof-v07--v08--oct-2026--mar-2027)
- [Technical debt](#technical-debt-register)
- [Test coverage](#test-coverage)
- [Version map](#version-map)
- [Docs sync](#docs-sync-debt)
- [How to track progress](#how-to-track-progress)

### Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | TODO |
| `[~]` | In progress |
| `[x]` | Done |
| **Gate** | Must pass before the next phase ships |

---

## Now / Next / Later

| Horizon | Focus |
|:--------|:------|
| **Now** | **Gate 1C** — FairBench H100 re-run with `psilogic[cuda]`; refresh wall-time tables in README / PAPER / this file |
| **Next** | Gate 2 prep — ImageNet ResNet-50 + GPT-2 medium budgets; learning-curve figures; W&B tagging |
| **Later** | Ablation suite · arXiv submit sync · ecosystem PRs · v1.0 API freeze · 1B flagship |

```
Now (Sep 2026)     Gate 1C → fused FairBench re-run, refresh overhead numbers
Oct 2026 – Mar 27  Gate 2  → ImageNet-1k (CV-1) + GPT-2 medium (NLP-2)
Apr–Jun 2027       Phase 3 → ablation program, Muon hybrid explore
Jul–Sep 2027       Phase 4 → arXiv submit (paper.tex SoT), HF docs, community
Q4 2027            Gate 5  → v1.0, ≥5/7 arenas, 1B flagship
```

---

## Gate dashboard

| Gate | Type | Status | Evidence |
|:-----|:-----|:-------|:---------|
| **0** | Eng | **PASSED** | CI green, modular package, PyPI v0.4 |
| **1A** | Research | **PASSED** | FairBench ViT 0.244 vs Lion 0.213 ([`aggregate.csv`](benchmark/results/full/aggregate.csv)) |
| **1B** | Research | **PASSED** | FairBench NLP PPL 7.79 vs AdamW 8.17 |
| **1C** | Eng | **IN PROGRESS** | Triton fusion shipped (v0.5); H100 FairBench fused re-run **pending** |
| **2** | Research | Open | ImageNet R50 + GPT-2 medium |
| **3** | Mixed | Open | ≤15% A100 overhead + published ablations |
| **4** | Credibility | Open | arXiv live · W&B · HF mindshare |
| **5** | Release | Open | v1.0 · ≥5/7 arenas · 1B result |

**Protocol note (Gate 1A):** Early experiment tables targeted CIFAR Top-1 ≥ 0.48 under a
different recipe. The **passed** gate uses the FairBench ViT arena (ViT-Tiny / CIFAR-100,
2000 steps, per-optimizer LR sweep) where absolute accuracy is lower but the comparison
is fair. Do not mix those targets.

---

## Scorecard — Where We Stand Today (v0.6.0)

### Quality (FairBench · Jun 2026 · H100 · quality unchanged by fusion)

| Arena | Task | AdamW | Lion | ΨLogic | Winner | Notes |
|-------|------|-------|------|--------|--------|-------|
| **NLP** | GPT / TinyStories · 3 seeds | 8.17 PPL | 21.04 | **7.79** | **ΨLogic** | *p*=0.049 vs AdamW |
| **ViT** | ViT-T / CIFAR-100 · 3 seeds | 0.223 | 0.213 | **0.244** | **ΨLogic** | *p*=0.015 vs AdamW |
| **ResNet** | R-18 / Tiny ImageNet | 0.219 | 0.205 | **0.222** | **Tie AdamW / Beat Adam** | vs Adam *p*=0.001; vs AdamW *p*=0.44 |
| **Diffusion** | DDPM / CelebA | **0.0199** | 0.0218 | 0.0201 | **Tie AdamW** | *p*=0.49 |
| Legacy | [OLD_RESULTS.md](OLD_RESULTS.md) | — | — | — | — | Pre-FairBench only |

### Performance (wall time ΨLogic / AdamW)

| Arena | Pre-fusion H100 (Jun 2026) | Target (fused) | Status |
|-------|----------------------------|----------------|--------|
| NLP | 1.20× | ≤ 1.25× | `[ ]` re-run pending |
| ViT | **1.79×** | **≤ 1.25×** | `[~]` local TinyViTLike **1.96×** on GTX 1650 (Turing); not the Gate hardware |
| ResNet | 1.42× | ≤ 1.25× | `[ ]` re-run pending |
| Diffusion | 1.77× | ≤ 1.25× | `[ ]` re-run pending |

**Current focus (one line):** close Gate 1C on Ampere+ / H100 with `psilogic[cuda]`, then
start Gate 2 scale proof. v0.6 shipped safer bare defaults (AGC/GC off) after NLP follow-ups.

**Gates passed:** Gate 0 · Gate 1A · Gate 1B

---

## Definition of Done — "Industry Standard"

ΨLogic earns the label when **all** of the following are true:

1. **Benchmark dominance** — Wins or statistically ties (*p* > 0.05) AdamW **and** Lion on ≥ 5 of 7 reference arenas (Phase 5).
2. **Reproducibility** — Every headline number reproducible with one command + public W&B logs.
3. **Ecosystem** — Listed in HuggingFace `Trainer` docs as supported optimizer; PR merged or officially recommended.
4. **Adoption signal** — ≥ 1,000 PyPI downloads/month **or** cited in ≥ 3 independent papers/repos.
   - *How we measure:* [pypistats](https://pypistats.org/packages/psilogic) / PePy monthly; Semantic Scholar / Google Scholar alerts for `PsiLogic` / arXiv:2607.16268.
5. **API stability** — Semver v1.0 with no breaking changes for 6 months.
6. **Performance** — ≤ 15% step-time overhead vs AdamW on A100 (documented).

---

## Non-goals (next 2 quarters)

Through Gate 2 (≈ Mar 2027) we are **not** prioritizing:

- New optimizer variants unrelated to chaos cancellation (full Muon rewrite, second-order methods)
- Mobile / edge deployment packaging
- Replacing FairBench with a private closed suite
- Marketing campaigns before fused overhead numbers and Gate 2 evidence exist
- Breaking public API churn without CHANGELOG + README callouts

---

## Risk register

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| No reliable H100 / A100 access for Gate 1C / Gate 2 | Blocks overhead + scale claims | Rent cloud GPU; document exact SKU; accept delayed gates |
| Seed count = 3 underpowered for ties | Over-/under-claim significance | Report *p*-values honestly; plan n≥5 on flagship runs |
| External groups do not replicate | Credibility gap | One-command FairBench; Zenodo DOI; invite reproductions |
| Docs drift (README / PAPER / ROADMAP / tex) | Conflicting public numbers | [Docs sync debt](#docs-sync-debt) checklist after every bench |
| Fusion helps Turing less than Ampere | Gate 1C fails on wrong hardware | Gate hardware = Ampere+; microbench on 1650 is diagnostic only |

---

<details>
<summary><strong>Phase 0 — Engineering Foundation (v0.4) · COMPLETE</strong></summary>

Package refactor + CI. **Done.**

- [x] Split monolith → `optimizer.py`, `_chaos.py`, `presets.py`, `param_groups.py`, `convenience.py`
- [x] `py.typed`, Ruff, GitHub Actions CI, GPU test markers
- [x] `psilogic.debug`, CHANGELOG, README structure, FairBench reproduce path
- **Gate 0:** `pytest` green · `ruff` clean · PyPI v0.4.0 — **PASSED**

</details>

---

<details>
<summary><strong>Phase 1A / 1B — ViT + NLP quality gates · COMPLETE</strong></summary>

### 1A — ViT (beat Lion) · PASSED

Winner path: `PsiLogicViT` + `vision_defaults` / FairBench ViT arena.
FairBench H100: ΨLogic **0.244** vs Lion **0.213** (*p* < 0.001).

Shipped: `vit_param_groups`, `PsiLogicViT`, debug helpers, FairBench ViT wiring,
`tests/test_vit_preset.py`.

Open docs debt: learning-curve plot epoch 1–15 (see Phase 4 / debt register).

### 1B — GPT / from-scratch LM (beat AdamW) · PASSED

Winner path: bare / `gpt_scratch_defaults` aligned with FairBench NLP.
FairBench H100: ΨLogic PPL **7.79** vs AdamW **8.17** (*p* = 0.049).

Shipped: `gpt_param_groups`, auto warmup, `PsiLogicGPT`, chaos warmup tests.

Open docs debt: GPT warmup ablation figure in PAPER (deferred ablations).

### v0.6 wrap-up (shipped)

- [x] **Safer drop-in defaults** — bare `PsiLogic` uses `agc_clip=0.0`,
  `grad_centralize=False` (AGC/GC hurt TinyStories GPT-scratch vs AdamW in follow-ups).
  Task helpers still opt in. See [CHANGELOG.md](CHANGELOG.md) `[0.6.0]`.

</details>

---

## Phase 1C — Performance Close (v0.5.x) · **IN PROGRESS**

> Confirm claim: **≤1.25× AdamW** step time on ViT-like models with `use_fused_cuda=True`
> on **Ampere+** (A100/H100). FairBench wall times in the scorecard are **pre-fusion**.

| ID | Deliverable | Status |
|----|-------------|--------|
| P1 | Triton fused step (`psilogic/_cuda/`) + multi-tensor kernel | `[x]` 0.5.x |
| P2 | `tests/test_step_overhead.py` + `scripts/profile_optimizer.py` | `[x]` |
| P3 | `tests/test_numerical_parity.py` — scalar / foreach / fused | `[x]` 0.5.0 |
| P4 | Re-run FairBench with `psilogic[cuda]` on H100 | `[ ]` pending |
| P5 | Update README + scorecard overhead tables | `[~]` docs call out pending; numbers not replaced |
| P6 | CI: optional self-hosted GPU job for fused overhead | `[ ]` |

```bash
pip install -e ".[cuda,benchmark]"
cd benchmark
python -m fairbench --arenas vit resnet diffusion --data-root ./data --output-dir results/fused
python ../scripts/profile_optimizer.py
```

**Gate 1C:** FairBench ViT wall time ΨLogic/AdamW **≤ 1.25×** (mean over 3 seeds)
**or** `profile_optimizer.py` median step ratio ≤ 1.25× on reference Ampere+ hardware.

---

# Phase 2 — Scale Proof (v0.7 – v0.8) · Oct 2026 – Mar 2027

> Prove ΨLogic works at the scale people actually train — not just toy budgets.

**Rough compute budget (planning):** CV-1 ≈ 3 seeds × 90 epoch ResNet-50 ImageNet
(~few hundred A100-hours depending on nodes); NLP-2 ≈ 3 seeds × 50k GPT-2 medium steps
(same order). Schedule cloud spend before starting.

## 2A — Computer Vision at Scale

| Benchmark | Model | Dataset | Epochs | Seeds | vs | Target |
|-----------|-------|---------|--------|-------|-----|--------|
| **CV-1** | ResNet-50 | ImageNet-1k | 90 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-2** | ViT-Base | ImageNet-1k | 300 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-3** | ConvNeXt-T | CIFAR-100 | 100 | 5 | AdamW, Lion | Beat FairBench ViT protocol |

### Code / infra

- [ ] **`benchmark/imagenet/`** — DDP-ready train script, AMP bf16, cosine LR
- [x] `profile_step_time` + `scripts/profile_optimizer.py`
- [ ] Verify non-deprecated `torch.amp` in all scale benchmarks
- [x] `tests/test_grad_accum.py`
- [ ] Multi-GPU DDP scaling efficiency report (2×GPU)

## 2B — NLP / LM at Scale

| Benchmark | Model | Dataset | Steps | Seeds | vs | Target |
|-----------|-------|---------|-------|-------|-----|--------|
| **NLP-1** | BERT-large | GLUE (MNLI, QQP, QNLI, SST-2) | 3 ep each | 3 | AdamW | Avg ≥ AdamW |
| **NLP-2** | GPT-2 medium (345M) | OpenWebText | 50k | 3 | AdamW | PPL ≤ AdamW |
| **NLP-3** | Whisper-small | LibriSpeech fine-tune | 10 ep | 3 | AdamW | WER ≤ AdamW |

### Code / infra

- [x] `PsiLogicWhisper`, `whisper_defaults()`, `glue_defaults()`
- [ ] FSDP: assert loss decreases (`tests/test_fsdp.py`)
- [ ] Peak VRAM vs AdamW on GPT-2 medium

**Gate 2:** Win or tie on **CV-1 (ResNet-50 ImageNet)** AND **NLP-2 (GPT-2 medium)**.

---

# Phase 3 — Algorithm Improvements (v0.9) · Apr–Jun 2027

## 3A — Core optimizer enhancements

| Feature | Status |
|---------|--------|
| Auto-γ scheduler (`gamma_auto`) | `[x]` 0.4.0 |
| Per-layer chaos sync DDP | `[x]` 0.4.0 |
| Fused CUDA kernel | `[x]` 0.5.0 |
| `PsiLogic.auto(model)` | `[x]` 0.4.0 |
| State dict versioning | `[x]` 0.4.0 |
| ValueError validation | `[x]` 0.4.0 |
| Muon compatibility (hybrid 2D) | `[ ]` explore |

## 3B — Ablation program (publish all results)

Document in `PAPER.md` / `arxiv/paper.tex` + W&B (FairBench-scale; synthetic-only is insufficient):

- [ ] GC on/off
- [ ] AGC on/off
- [ ] adaptive_tau on/off
- [ ] quantum_decay on/off
- [ ] lion_mode on/off
- [x] chaos disabled / mirror test
- [ ] γ schedule: constant vs cosine vs auto
- [ ] GPT `chaos_warmup` ablation (carried from 1B docs debt)

**Gate 3:** Step overhead ≤ 15% vs AdamW on A100 · ablation suite published · auto-config on 3 unseen architectures.

---

# Phase 4 — Credibility & Visibility (v0.95) · Jul–Sep 2027

## 4A — Scientific publication

- [x] LaTeX — [arxiv/paper.tex](arxiv/paper.tex) (**source of truth for PDF**)
- [x] PDF build — [scripts/build_arxiv_pdf.py](scripts/build_arxiv_pdf.py)
- [ ] Sync tex + [PAPER.md](PAPER.md) with fused-CUDA overhead + v0.6 defaults note
- [ ] Submit / update **arXiv** (cs.LG)
- [ ] Update **Zenodo DOI** each major scientific release
- [ ] Reproduce one published AdamW baseline curve; Lion ViT match/beat narrative
- [ ] Formal **convergence sketch** (honest assumptions)

## 4B — Ecosystem integrations

| Platform | Status |
|----------|--------|
| HuggingFace Trainer helpers | `[x]` |
| Lightning helpers | `[x]` |
| torchtune example YAML | `[x]` |
| Axolotl / LLaMA-Factory PRs | `[ ]` external |
| HF example notebook | `[ ]` |
| README “Works with HF” badge | `[ ]` |

## 4C — Community

- [ ] Public W&B project (all benches tagged by version)
- [ ] Blog / RL / X threads after Gate 1C+2 evidence
- [ ] PyPI downloads badge
- [ ] `OPTIMIZER_COMPARISON.md` — honest head-to-head including losses

**Gate 4:** arXiv live · HF path documented · W&B public · ≥ 500 PyPI downloads/month

---

# Phase 5 — v1.0 Industry Release · Q4 2027

## 5A — API & quality

- [ ] Semver **v1.0.0** API freeze
- [ ] mypy strict on `psilogic/`
- [ ] 100% public docstring coverage
- [ ] Sphinx docs on GitHub Pages
- [ ] `torch.optim` compatibility audit
- [x] Deprecation policy in [CONTRIBUTING.md](CONTRIBUTING.md)
- [x] [SECURITY.md](SECURITY.md)

## 5B — Flagship benchmark

- [ ] LLaMA-style **1B** · 100k steps · OpenWebText vs AdamW vs Lion (4×A100-class)
  - Target: PPL ≤ AdamW at same steps **or** target PPL in ≤ 80% steps

## 5C — Reference suite (7 arenas)

| # | Arena | Must beat | Overlap |
|---|-------|-----------|---------|
| 1 | CIFAR-10 / ResNet-18 | AdamW | Partial |
| 2 | CIFAR-100 / ViT-Small | **Lion** | FairBench ViT `[x]` |
| 3 | ImageNet-1k / ResNet-50 | AdamW | Gate 2 CV-1 |
| 4 | BERT-base / SST-2 | AdamW | HF example |
| 5 | GPT-2 / Wikitext-2 scratch | AdamW | FairBench NLP partial |
| 6 | GPT-2 medium / OpenWebText | AdamW | Gate 2 NLP-2 |
| 7 | nanoGPT / Tiny Shakespeare | AdamW | — |

- [ ] One-command reproduction + optional nightly GPU CI
- [ ] Venue target: ICLR / NeurIPS 2028 with 1B result

**Gate 5:** ≥ 5/7 green · v1.0 on PyPI · Sphinx live · HF path · 1B published

---

# Phase 6 — Beyond v1.0 (2028+)

- [ ] First-class HF `Trainer` enum / PyTorch docs mention
- [ ] Chaos dashboard (W&B panel)
- [ ] FSDP global chaos; Shampoo/Muon + chaos hybrids
- [ ] Early-stopping proxy via `slow_EMA`
- [ ] Hardware-partner writeups

---

## Technical debt register

| Item | Action |
|------|--------|
| FairBench results pre-fusion | Re-run with `psilogic[cuda]`, commit new aggregate + SHA |
| `gamma_auto` / `sync_chaos_ddp` lightly tested | Behavioral + multi-process tests |
| Whisper / GLUE presets | Smoke tests |
| FSDP loss assertion | Extend `tests/test_fsdp.py` |
| GPU CI gap | Self-hosted or nightly GPU workflow |
| arXiv / PAPER out of sync with fusion + v0.6 | Sync before next arXiv bump |
| Learning-curve / warmup ablation figures | Close or mark deferred in PAPER |
| Axolotl / LLaMA-Factory | External PRs |

---

## Test coverage

GPU-marked tests (`@pytest.mark.gpu`, `@pytest.mark.multi_gpu`) auto-skip on CPU CI.

| Area | Covered | Gap |
|------|---------|-----|
| Core optimizer | convergence, Lion mode, γ decay, checkpoints | — |
| Presets | ViT, GPT, auto-config | Whisper, GLUE |
| Chaos | warmup, debug | `gamma_auto` behavior |
| Backends | numerical parity, step overhead | GPU overhead not enforced in CI |
| Distributed | DDP, FSDP smoke | `sync_chaos_ddp`, FSDP loss ↓ |
| Integrations | HF + Lightning | — |
| Training patterns | grad accum, AMP, compile | — |
| FairBench | manual | No pytest e2e arenas |

---

## Version map

| Version | Phase | Key deliverable | Status |
|---------|-------|-----------------|--------|
| **0.4.0** | 0 | Modular package · CI | Released |
| **0.5.0** | 1A/1B + 1C partial | Presets · Triton fusion · parity | Released |
| **0.6.0** | 1 wrap-up | Safer bare defaults (AGC/GC off) · docs | **Released (current)** |
| **0.6.x / 0.5.1-perf** | 1C | Fused FairBench re-run · overhead docs | Planned |
| **0.7.0** | 2A | ResNet-50 ImageNet | Planned |
| **0.8.0** | 2B | GPT-2 medium · GLUE · Whisper benches | Planned |
| **0.9.0** | 3 | Ablations · Muon explore · ≤15% overhead | Planned |
| **0.95.0** | 4 | arXiv sync · HF docs · W&B | Planned |
| **1.0.0** | 5 | API freeze · 5/7 arenas · 1B | Planned |

Each shipped gate should appear in [CHANGELOG.md](CHANGELOG.md) with date + version.

---

## Docs sync debt

After any benchmark or default change, update **together**:

1. `benchmark/results/…` (or note “pending”)
2. [README.md](README.md) tables / callouts
3. This scorecard + gate dashboard
4. [PAPER.md](PAPER.md) notes **and** `arxiv/paper.tex` if numbers are claimed in PDF
5. [CHANGELOG.md](CHANGELOG.md) if user-facing

---

## How to track progress

1. Check off items here as they ship; keep **Gate dashboard** truthful.
2. Update the **Scorecard** after every reference benchmark run.
3. Tag GitHub releases to match the Version Map.
4. Mirror headline numbers in README + PAPER — never let docs drift from CSV data.
5. `pytest tests/ -v` before every release.
6. Keep Test Coverage gaps in sync when adding optimizer features.

---

*Repository: https://github.com/Troxter222/psilogic*  
*DOI: https://doi.org/10.5281/zenodo.18739857*  
*Author: Ali (Troxter222)*
