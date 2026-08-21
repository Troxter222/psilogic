#!/usr/bin/env bash
# run_fairbench.sh -- launch (or reattach to) the fairbench download+train+
# analysis pipeline inside a detached tmux session, so it survives closing
# the terminal / losing an SSH connection.
#
# Usage:
#   ./run_fairbench.sh              start (or attach if already running)
#   ./run_fairbench.sh --attach     just attach, don't start anything new
#
# Config (env vars, all optional, same defaults as fairbench_pipeline.sh):
#   DATA_ROOT, OUTPUT_DIR, CONFIG_JSON, MAX_TRAIN_RETRIES, TRAIN_RETRY_SLEEP_S
#
# Does NOT survive a full PC crash/reboot by itself (nothing running in
# userspace can). What it does give you: re-running this script after a
# reboot resumes from the last completed stage instead of starting over,
# because fairbench_pipeline.sh checks marker files before each stage.
# For true survive-a-reboot behavior, add a systemd user service or a
# crontab @reboot entry that runs this same script.

set -uo pipefail

SESSION_NAME="fairbench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/fairbench_pipeline.sh"

if [[ ! -f "$PIPELINE" ]]; then
    echo "Error: $PIPELINE not found (expected next to this script)." >&2
    exit 1
fi

if [[ "${1:-}" == "--attach" ]]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        exec tmux attach -t "$SESSION_NAME"
    fi
    echo "No tmux session named '$SESSION_NAME' is currently running."
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' is already running. Attaching..."
    exec tmux attach -t "$SESSION_NAME"
fi

chmod +x "$PIPELINE"
tmux new-session -d -s "$SESSION_NAME" "bash '${PIPELINE}'"

echo "Started in tmux session '${SESSION_NAME}'."
echo "  Attach:    tmux attach -t ${SESSION_NAME}   (or: $0 --attach)"
echo "  Detach:    Ctrl+B then D  (job keeps running)"
echo "  Logs:      \${OUTPUT_DIR:-./results/local_full}/logs/"
echo ""
echo "If the PC crashes or reboots, just run this script again -- the"
echo "pipeline resumes from the last completed stage (download/train/"
echo "analysis) instead of restarting from scratch."