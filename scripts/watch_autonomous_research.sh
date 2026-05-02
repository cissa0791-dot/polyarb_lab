#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

WATCH_LOG="${WATCH_LOG:-data/reports/research_watch.log}"
SLEEP_SEC="${SLEEP_SEC:-60}"
PULL_FIRST="${PULL_FIRST:-1}"
CYCLES="${CYCLES:-240}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
MAX_SELECTED_MARKETS="${MAX_SELECTED_MARKETS:-3}"
MAX_LIVE_RISK_USDC="${MAX_LIVE_RISK_USDC:-20}"

mkdir -p "$(dirname "$WATCH_LOG")"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WATCH_LOG"
}

has_active_run() {
  pgrep -f "run_evidence_research_pipeline.py|run_auto_trade_profit.py" >/dev/null 2>&1
}

while true; do
  while has_active_run; do
    log "active research/trading process detected; waiting ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
  done

  run_id="research-$(date -u +%Y%m%dT%H%M%SZ)"
  session_name="research_${run_id}"
  log "starting next autonomous research run_id=${run_id} session=${session_name}"
  PULL_FIRST="$PULL_FIRST" \
  RUN_ID="$run_id" \
  SESSION_NAME="$session_name" \
  CYCLES="$CYCLES" \
  INTERVAL_SEC="$INTERVAL_SEC" \
  MAX_SELECTED_MARKETS="$MAX_SELECTED_MARKETS" \
  MAX_LIVE_RISK_USDC="$MAX_LIVE_RISK_USDC" \
    scripts/start_autonomous_research.sh | tee -a "$WATCH_LOG"
  sleep "$SLEEP_SEC"
done
