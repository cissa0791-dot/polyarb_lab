#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-research}"
RUN_ID="${RUN_ID:-research-$(date -u +%Y%m%dT%H%M%SZ)}"
CYCLES="${CYCLES:-240}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
MAX_SELECTED_MARKETS="${MAX_SELECTED_MARKETS:-3}"
OUT_DIR="${OUT_DIR:-data/reports}"
PULL_FIRST="${PULL_FIRST:-0}"

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi
if [ -f "$HOME/.polymarket_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.polymarket_env"
fi

if [ "$PULL_FIRST" = "1" ]; then
  git pull origin claude/fix-system-issues-Y8LCp
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" bash -lc "
set -euo pipefail
cd '$PWD'
source venv/bin/activate
source \"\$HOME/.polymarket_env\"
python scripts/run_evidence_research_pipeline.py \
  --run-id '$RUN_ID' \
  --cycles '$CYCLES' \
  --interval-sec '$INTERVAL_SEC' \
  --event-limit 1000 \
  --market-limit 2000 \
  --snapshot-max-markets 80 \
  --snapshot-filtered-max 60 \
  --max-selected-markets '$MAX_SELECTED_MARKETS' \
  --out-dir '$OUT_DIR' \
  --verbose
python scripts/run_autonomous_project_manager.py --mode print-live-command
"

echo "started tmux session: $SESSION_NAME"
echo "run_id: $RUN_ID"
echo "attach: tmux attach -t $SESSION_NAME"
echo "detach: Ctrl+B then d"
