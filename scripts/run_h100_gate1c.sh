#!/usr/bin/env bash
# Gate 1C — fused FairBench + step-time profile on NVIDIA H100.
# Paste into a Lightning.ai Studio terminal after switching the machine to H100.
#
# Protocol matches benchmark/results/full/config.json (paper reference), with
# psilogic[cuda] so use_fused_cuda=True (default) actually hits Triton.
#
# Usage (inside Studio, repo root):
#   bash scripts/run_h100_gate1c.sh
#   OUTPUT_DIR=results/fused_h100 bash scripts/run_h100_gate1c.sh
#   SKIP_DOWNLOAD=1 bash scripts/run_h100_gate1c.sh   # if data already staged
#   ARENAS="vit" bash scripts/run_h100_gate1c.sh       # smoke one arena first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="${REPO_ROOT}/benchmark"
cd "$REPO_ROOT"

# Paths relative to REPO_ROOT unless absolute.
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/results/fused_h100}"
ARENAS="${ARENAS:-nlp vit resnet diffusion}"
SEEDS="${SEEDS:-0 1 2}"
MAX_STEPS="${MAX_STEPS:-2000}"
SWEEP_STEPS="${SWEEP_STEPS:-500}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PROFILE="${SKIP_PROFILE:-0}"
SKIP_FAIRBENCH="${SKIP_FAIRBENCH:-0}"

# fairbench is not an installed package — run as module from benchmark/.
export PYTHONPATH="${BENCHMARK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=================================================================="
echo " ΨLogic Gate 1C — H100 fused FairBench"
echo " repo=$REPO_ROOT"
echo " data=$DATA_ROOT  out=$OUTPUT_DIR"
echo " arenas=$ARENAS  seeds=$SEEDS  steps=$MAX_STEPS  amp=$AMP_DTYPE"
echo " started: $(date -Is)"
echo "=================================================================="

# --- hardware gate ---------------------------------------------------------
python - <<'PY'
import sys
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Switch Studio machine to H100 / GPU.", file=sys.stderr)
    sys.exit(2)
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
vram = props.total_memory / (1024**3)
print(f"gpu={name!r} vram_gb={vram:.2f} cc={props.major}.{props.minor}")
# Ampere+ = SM 8.0+; Hopper H100 = 9.0
if props.major < 8:
    print(f"WARNING: compute capability {props.major}.{props.minor} < 8.0; Gate 1C target is Ampere+/H100.")
if "H100" not in name and "A100" not in name:
    print(f"WARNING: expected H100/A100, got {name!r}. Continuing anyway.")
PY

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# --- deps ------------------------------------------------------------------
python -m pip install -U pip
python -m pip install -e ".[cuda,benchmark]"
python - <<'PY'
from psilogic._cuda import is_fused_available
import triton
print(f"triton={triton.__version__} fused_available={is_fused_available()}")
if not is_fused_available():
    raise SystemExit("ERROR: Triton fused path not available — Gate 1C requires psilogic[cuda] + CUDA.")
PY

# --- datasets --------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  python -m fairbench.download --data-root "$DATA_ROOT"
else
  python -m fairbench.download --data-root "$DATA_ROOT" --check-only
fi

mkdir -p "$OUTPUT_DIR/logs"
PROOF="$OUTPUT_DIR/PROOF.md"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# --- microbench (speed proof, seconds) -------------------------------------
if [[ "$SKIP_PROFILE" != "1" ]]; then
  echo "[profile] scripts/profile_optimizer.py"
  python scripts/profile_optimizer.py 2>&1 | tee "$OUTPUT_DIR/logs/profile_${TS}.txt"
fi

# --- FairBench (quality + wall time) ---------------------------------------
if [[ "$SKIP_FAIRBENCH" != "1" ]]; then
  echo "[fairbench] paper protocol + fused PsiLogic"
  # shellcheck disable=SC2086
  python -m fairbench \
    --arenas $ARENAS \
    --seeds $SEEDS \
    --max-steps "$MAX_STEPS" \
    --max-epochs 10 \
    --eval-every 200 \
    --sweep-steps "$SWEEP_STEPS" \
    --sweep-epochs 5 \
    --batch-size 64 \
    --amp-dtype "$AMP_DTYPE" \
    --num-workers "$NUM_WORKERS" \
    --data-root "$DATA_ROOT" \
    --offline \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/logs/fairbench_${TS}.txt"

  python -m fairbench.analysis \
    --output-dir "$OUTPUT_DIR" \
    --metric val_loss \
    --out "$OUTPUT_DIR/table_val_loss.tex" || true
fi

# --- write proof summary ---------------------------------------------------
OUTPUT_DIR="$OUTPUT_DIR" PROOF_TS="$TS" python - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

out = Path(os.environ["OUTPUT_DIR"])
ts = os.environ["PROOF_TS"]
agg_path = out / "aggregate.csv"
sig_path = out / "significance.csv"
cfg_path = out / "config.json"
proof = out / "PROOF.md"

lines: list[str] = []
lines.append("# ΨLogic Gate 1C — H100 fused proof")
lines.append("")
lines.append(f"Generated: {ts}")
lines.append("")

if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    hw = cfg.get("runtime_hardware") or {}
    lines.append("## Hardware")
    lines.append("")
    lines.append(f"- GPU: **{hw.get('gpu_name', '?')}** ({hw.get('gpu_vram_gb', '?')} GB)")
    lines.append(f"- Torch / CUDA: {hw.get('torch_version', '?')} / {hw.get('cuda_version', '?')}")
    lines.append(f"- AMP: {cfg.get('hardware', {}).get('amp_dtype', '?')}")
    lines.append(f"- Seeds / steps: {cfg.get('train', {}).get('seeds')} / {cfg.get('train', {}).get('max_steps')}")
    lines.append("")

if agg_path.exists():
    rows = list(csv.DictReader(agg_path.open()))
    by = {(r["arena"], r["optimizer"]): r for r in rows}
    lines.append("## Quality (mean over seeds)")
    lines.append("")
    lines.append("| Arena | Metric | AdamW | ΨLogic | Δ vs AdamW |")
    lines.append("|-------|--------|------:|-------:|-----------:|")
    for arena, metric, _higher in [
        ("nlp", "perplexity", False),
        ("vit", "val_acc", True),
        ("resnet", "val_acc", True),
        ("diffusion", "val_loss", False),
    ]:
        a = by.get((arena, "adamw"))
        p = by.get((arena, "psilogic"))
        if not a or not p:
            continue
        av = a.get(f"{metric}_mean") or ""
        pv = p.get(f"{metric}_mean") or ""
        try:
            af, pf = float(av), float(pv)
            delta = f"{(pf - af) / af * 100:+.1f}%"
            lines.append(f"| {arena} | {metric} | {af:.4g} | **{pf:.4g}** | {delta} |")
        except ValueError:
            lines.append(f"| {arena} | {metric} | {av} | {pv} | — |")
    lines.append("")
    lines.append("## Speed (wall_time ΨLogic / AdamW)")
    lines.append("")
    lines.append("| Arena | AdamW (s) | ΨLogic (s) | Ratio | Target |")
    lines.append("|-------|----------:|-----------:|------:|:------:|")
    for arena in ("nlp", "vit", "resnet", "diffusion"):
        a = by.get((arena, "adamw"))
        p = by.get((arena, "psilogic"))
        if not a or not p:
            continue
        try:
            aw = float(a["wall_time_s_mean"])
            pw = float(p["wall_time_s_mean"])
            ratio = pw / aw
            ok = "PASS" if ratio <= 1.25 else "FAIL"
            lines.append(f"| {arena} | {aw:.1f} | {pw:.1f} | **{ratio:.2f}x** | <=1.25x {ok} |")
        except (KeyError, ValueError, ZeroDivisionError):
            continue
    lines.append("")
    lines.append("Gate 1C (ViT): ratio <= 1.25x **or** `profile_optimizer.py` median step <= 1.25x.")
    lines.append("")

prof = sorted((out / "logs").glob("profile_*.txt"))
if prof:
    lines.append("## Microbench (`profile_optimizer.py`)")
    lines.append("")
    lines.append("```")
    lines.append(prof[-1].read_text()[-2000:])
    lines.append("```")
    lines.append("")

if sig_path.exists():
    lines.append("## Significance file")
    lines.append("")
    lines.append(f"See `{sig_path.name}` for Welch t-tests / Cohen's d.")
    lines.append("")

proof.write_text("\n".join(lines) + "\n")
print(proof.read_text())
print(f"\nWrote {proof}")
PY

echo "=================================================================="
echo " DONE: $(date -Is)"
echo " Proof: $PROOF"
echo " CSVs:  $OUTPUT_DIR/{aggregate,summary,significance,config}.*"
echo "=================================================================="
