╔══════════════════════════════════════════════════════════════════════════╗
║                    PSILOGIC — BENCHMARK LOGS                             ║
║         FairBench reference run · Jun 2026 · NVIDIA H100 80GB            ║
╚══════════════════════════════════════════════════════════════════════════╝

> Source CSVs: `benchmark/results/full/aggregate.csv`, `summary.csv`, `significance.csv`  
> Config: `benchmark/results/full/config.json`  
> Archived pre-FairBench: [OLD_RESULTS.md](OLD_RESULTS.md)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FairBench · 4 arenas · 3 seeds · per-optimizer LR sweep · bf16 AMP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Hardware : NVIDIA H100 80GB HBM3 (79.2 GB VRAM)
  Software : PyTorch 2.4.1+cu124 · CUDA 12.4
  Protocol : LR sweep 500 steps (7 LRs) → train 2000 steps × 3 seeds
  Settings : batch=64 · grad_clip=1.0 · warmup=100 · cosine LR · foreach=True

  ── AGGREGATE (mean ± std, 3 seeds) ─────────────────────────────────

  NLP — selected LR 3.16e-4 for all optimizers
    Optimizer   Perplexity          Val loss            VRAM MB   Time s
    Adam        13.661 ± 0.222      2.614 ± 0.016       458       46.6 ± 1.4
    AdamW        8.173 ± 0.079      2.101 ± 0.010       458       45.9 ± 1.6
    Lion        21.038 ± 1.408      3.045 ± 0.068       445       38.2 ± 7.4
    PsiLogic     7.790 ± 0.180      2.053 ± 0.023       458       55.2 ± 0.8  ← BEST PPL

  ViT — LR: Adam 3.16e-5 · AdamW/PsiLogic 3.16e-4 · Lion 1e-4
    Optimizer   Val acc             Val loss            VRAM MB   Time s
    Adam         0.079 ± 0.003      4.214 ± 0.002      1229       95.2 ± 8.8
    AdamW        0.223 ± 0.002      3.205 ± 0.005      1229       98.5 ± 5.9
    Lion         0.213 ± 0.002      3.224 ± 0.027      1208       98.6 ± 5.6
    PsiLogic     0.244 ± 0.006      3.078 ± 0.035      1229      176.7 ± 5.4  ← BEST

  ResNet — LR: Adam/Lion 1e-4 · AdamW/PsiLogic 3.16e-4
    Optimizer   Val acc             Val loss            VRAM MB   Time s
    Adam         0.172 ± 0.004      3.816 ± 0.005       823       45.3 ± 1.1
    AdamW        0.219 ± 0.005      3.454 ± 0.017       825       47.6 ± 1.3
    Lion         0.205 ± 0.007      3.514 ± 0.035       777       46.1 ± 1.5
    PsiLogic     0.222 ± 0.001      3.462 ± 0.007       823       67.4 ± 0.6  ← BEST std

  Diffusion — LR: Adam/AdamW/PsiLogic 1e-3 · Lion 1e-4
    Optimizer   Val MSE             VRAM MB             Time s
    Adam         0.01987 ± 0.00006  3780                 94.2 ± 1.0   ← tie
    AdamW        0.01987 ± 0.00006  3780                 95.2 ± 3.3   ← tie
    Lion         0.02175 ± 0.00025  3768                 91.6 ± 1.7
    PsiLogic     0.02009 ± 0.00045  3781                168.3 ± 2.6

  ── PER-SEED DETAIL (summary.csv) ───────────────────────────────────

  NLP perplexity:
    seed 0: Adam 13.89  AdamW 8.18  Lion 19.47  PsiLogic 7.87
    seed 1: Adam 13.64  AdamW 8.09  Lion 21.47  PsiLogic 7.58
    seed 2: Adam 13.45  AdamW 8.25  Lion 22.18  PsiLogic 7.92

  ViT val acc:
    seed 0: Adam 0.078  AdamW 0.226  Lion 0.214  PsiLogic 0.238
    seed 1: Adam 0.083  AdamW 0.222  Lion 0.211  PsiLogic 0.247
    seed 2: Adam 0.076  AdamW 0.221  Lion 0.213  PsiLogic 0.249

  ResNet val acc:
    seed 0: Adam 0.175  AdamW 0.219  Lion 0.202  PsiLogic 0.221
    seed 1: Adam 0.167  AdamW 0.214  Lion 0.199  PsiLogic 0.224
    seed 2: Adam 0.173  AdamW 0.224  Lion 0.212  PsiLogic 0.222

  Diffusion val MSE:
    seed 0: Adam 0.01985  AdamW 0.01985  Lion 0.02169  PsiLogic 0.02004
    seed 1: Adam 0.01982  AdamW 0.01982  Lion 0.02153  PsiLogic 0.01967
    seed 2: Adam 0.01993  AdamW 0.01993  Lion 0.02202  PsiLogic 0.02057

  ── SIGNIFICANCE (PsiLogic vs baseline, Welch t-test) ───────────────

  NLP  PPL:  vs Adam p<10⁻⁵ · vs AdamW p=0.049 · vs Lion p=0.003
  ViT  acc: vs Adam p<10⁻⁴ · vs AdamW p=0.015 · vs Lion p=0.007
  ResNet acc: vs Adam p=0.001 · vs AdamW p=0.44 · vs Lion p=0.044
  Diffusion MSE: vs Adam/AdamW p=0.49 · vs Lion p=0.010

  Full terminal log: benchmark/logs.txt
  Learning curves: benchmark/results/full/plots/


╔══════════════════════════════════════════════════════════════════════════╗
║                              SUMMARY                                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  ΨLogic wins 3/4 quality arenas (NLP PPL, ViT acc, ResNet vs Adam)        ║
║  NLP PPL     7.79 ± 0.18  (AdamW 8.17, p=0.049)                          ║
║  ViT acc     0.244 ± 0.006 (AdamW 0.223, p=0.015)                        ║
║  ResNet acc  0.222 ± 0.001 (Adam 0.172, p=0.001; AdamW tie p=0.44)       ║
║  Diffusion   ~tie Adam/AdamW (p=0.49)                                    ║
║  Overhead    1.2–1.8× wall time vs AdamW (peak 1.79× ViT)              ║
╚══════════════════════════════════════════════════════════════════════════╝
