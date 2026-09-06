#!/usr/bin/env bash
# fairbench_pipeline.sh -- the actual download -> train -> analysis pipeline.
# Meant to be launched inside tmux by run_fairbench.sh, not run directly
# (though running it directly also works, it just won't survive a closed
# terminal on its own).
#
# Resumable: each stage writes a marker file under $STATE_DIR on success.
# Re-running this script (e.g. after a PC crash/reboot) skips stages that
# already finished and continues from where it left off.

# Use -u/-o pipefail, but NOT -e: the train stage captures exit codes and
# retries on failure. With errexit, a crashed fairbench run would abort the
# script before the retry loop can handle exit_code.
set -uo pipefail

# Path to the directory that CONTAINS the `fairbench` package (i.e. the dir
# where `python -m fairbench...` actually finds the module). Defaults to
# <repo>/benchmark next to this script. Override with BENCHMARK_DIR if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="${BENCHMARK_DIR:-${SCRIPT_DIR}/benchmark}"

if [[ ! -d "${BENCHMARK_DIR}/fairbench" ]]; then
    echo "Error: '${BENCHMARK_DIR}/fairbench' not found." >&2
    echo "Set BENCHMARK_DIR to the folder that contains the fairbench/ package, e.g.:" >&2
    echo "  BENCHMARK_DIR=/path/to/benchmark ./run_fairbench.sh" >&2
    exit 1
fi
cd "${BENCHMARK_DIR}" || exit 1

# NOTE: defaults below are a longer *local* recipe (5 seeds, 5000 steps, fp16).
# The paper / committed reference is results/full/ (3 seeds, 2000 steps, bf16).
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/local_full}"
CONFIG_JSON="${CONFIG_JSON:-}"
LOG_DIR="${OUTPUT_DIR}/logs"
STATE_DIR="${OUTPUT_DIR}/.run_state"
MAX_TRAIN_RETRIES="${MAX_TRAIN_RETRIES:-5}"
TRAIN_RETRY_SLEEP_S="${TRAIN_RETRY_SLEEP_S:-30}"
FINAL_SLEEP_S="${FINAL_SLEEP_S:-60}"

mkdir -p "$LOG_DIR" "$STATE_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/run_${TS}.log"

# Mirror everything to both stdout (visible in tmux) and a log file on disk,
# so logs survive even if the tmux session is later lost.
exec > >(tee -a "$MAIN_LOG") 2>&1

mark_done() { touch "${STATE_DIR}/$1.done"; }
is_done()   { [[ -f "${STATE_DIR}/$1.done" ]]; }

echo "=================================================================="
echo " fairbench run started: $(date)"
echo " data_root=${DATA_ROOT}  output_dir=${OUTPUT_DIR}"
echo "=================================================================="

# --- Stage 1: dataset download ----------------------------------------------
if is_done download; then
    echo "[stage: download] already completed previously, skipping."
else
    echo "[stage: download] starting..."
    if python -m fairbench.download --data-root "$DATA_ROOT"; then
        mark_done download
        echo "[stage: download] done."
    else
        echo "[stage: download] FAILED. Re-run this script to retry."
        exit 1
    fi
fi

# --- Stage 2: training run, with automatic crash-retry ----------------------
if is_done train; then
    echo "[stage: train] already completed previously, skipping."
else
    echo "[stage: train] starting..."
    attempt=0
    while true; do
        attempt=$((attempt + 1))
        if [[ -n "$CONFIG_JSON" ]]; then
            cmd=(python -m fairbench --config "$CONFIG_JSON")
        else
            cmd=(python -m fairbench
                 --data-root "$DATA_ROOT" --offline
                 --output-dir "$OUTPUT_DIR"
                 --seeds 0 1 2 3 4
                 --max-steps 5000 --max-epochs 20 --eval-every 250
                 --batch-size 32
                 --amp-dtype float16
                 --num-workers 4)
        fi
        echo "[stage: train] attempt ${attempt}/${MAX_TRAIN_RETRIES}: ${cmd[*]}"

        # Capture exit status without errexit so the retry loop can continue.
        if "${cmd[@]}"; then
            exit_code=0
        else
            exit_code=$?
        fi
        if [[ $exit_code -eq 0 ]]; then
            mark_done train
            echo "[stage: train] done."
            break
        fi

        if [[ $attempt -ge $MAX_TRAIN_RETRIES ]]; then
            echo "[stage: train] FAILED after ${MAX_TRAIN_RETRIES} attempts (last exit=${exit_code}). Giving up."
            echo "  Partial results (CSV logs are append-only) are already saved under ${OUTPUT_DIR}."
            echo "  Fix the issue, then re-run this script -- it will retry the train stage."
            exit 1
        fi
        echo "[stage: train] crashed (exit=${exit_code}); retrying in ${TRAIN_RETRY_SLEEP_S}s..."
        sleep "$TRAIN_RETRY_SLEEP_S"
    done
fi

# --- Stage 3: analysis / LaTeX table ----------------------------------------
if is_done analysis; then
    echo "[stage: analysis] already completed previously, skipping."
else
    echo "[stage: analysis] starting..."
    if python -m fairbench.analysis --output-dir "$OUTPUT_DIR" --metric val_loss \
        --out "${OUTPUT_DIR}/table_val_loss.tex"; then
        mark_done analysis
        echo "[stage: analysis] done. Table written to ${OUTPUT_DIR}/table_val_loss.tex"
    else
        echo "[stage: analysis] FAILED (non-fatal; results are still in ${OUTPUT_DIR})."
    fi
fi

echo "=================================================================="
echo " fairbench run finished: $(date)"
echo "=================================================================="
echo "This tmux pane will stay open for ${FINAL_SLEEP_S}s so you can read this, then exit."
sleep "$FINAL_SLEEP_S"