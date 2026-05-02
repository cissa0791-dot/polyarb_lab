#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-research}"
RUN_ID="${RUN_ID:-research-$(date -u +%Y%m%dT%H%M%SZ)}"
CYCLES="${CYCLES:-240}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
MAX_SELECTED_MARKETS="${MAX_SELECTED_MARKETS:-3}"
OUT_DIR="${OUT_DIR:-data/reports}"
PULL_FIRST="${PULL_FIRST:-0}"
MANAGER_MODE="${MANAGER_MODE:-execute-live-canary}"
MAX_LIVE_RISK_USDC="${MAX_LIVE_RISK_USDC:-40}"

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

RUN_DIR="$OUT_DIR/research_runs/$RUN_ID"
mkdir -p "$RUN_DIR"
CONSOLE_LOG="$RUN_DIR/research_console.log"

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
exec > >(tee -a '$CONSOLE_LOG') 2>&1
echo \"started_at_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
echo \"run_id=$RUN_ID\"
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
MANAGER_EXIT=0
python scripts/run_autonomous_project_manager.py \
  --mode '$MANAGER_MODE' \
  --max-live-risk-usdc '$MAX_LIVE_RISK_USDC' || MANAGER_EXIT=\$?
python scripts/build_research_run_report.py \
  --run-dir '$RUN_DIR' \
  --out '$RUN_DIR/research_run_report.md'
echo \"finished_at_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
echo \"manager_exit=\$MANAGER_EXIT\"
exit \"\$MANAGER_EXIT\"
"

echo "started tmux session: $SESSION_NAME"
echo "run_id: $RUN_ID"
echo "attach: tmux attach -t $SESSION_NAME"
echo "detach: Ctrl+B then d"
echo "manager_mode: $MANAGER_MODE"
echo "console_log: $CONSOLE_LOG"
