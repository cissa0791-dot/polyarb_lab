#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

WATCH_LOG="${WATCH_LOG:-data/reports/research_watch.log}"
SLEEP_SEC="${SLEEP_SEC:-60}"
PULL_FIRST="${PULL_FIRST:-1}"
CYCLES="${CYCLES:-}"
INTERVAL_SEC="${INTERVAL_SEC:-}"
MAX_SELECTED_MARKETS="${MAX_SELECTED_MARKETS:-}"
RESEARCH_PER_MARKET_CAP_USDC="${RESEARCH_PER_MARKET_CAP_USDC:-}"
MAX_LIVE_RISK_USDC="${MAX_LIVE_RISK_USDC:-20}"

mkdir -p "$(dirname "$WATCH_LOG")"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WATCH_LOG"
}

has_active_run() {
  ps -eo comm=,args= \
    | awk '
      $1 ~ /python/ && ($0 ~ /run_evidence_research_pipeline.py/ || $0 ~ /run_auto_trade_profit.py/) {
        found=1
      }
      END { exit found ? 0 : 1 }
    '
}

while true; do
  while has_active_run; do
    log "active research/trading process detected; waiting ${SLEEP_SEC}s"
    python scripts/refresh_research_report.py --out-dir data/reports >> "$WATCH_LOG" 2>&1 || true
    sleep "$SLEEP_SEC"
  done

  run_id="research-$(date -u +%Y%m%dT%H%M%SZ)"
  session_name="research_${run_id}"
  next_policy="$(
    python - <<'PY'
import json
from pathlib import Path

path = Path("data/reports/autonomous_decision_latest.json")
default = {"max_selected": 3, "per_market_cap": 40, "cycles": 240, "interval_sec": 30}
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    data = {}
policy = data.get("research_policy") if isinstance(data.get("research_policy"), dict) else {}
max_selected = int(policy.get("recommended_max_selected_markets") or default["max_selected"])
per_market_cap = float(policy.get("recommended_dry_run_per_market_cap_usdc") or default["per_market_cap"])
if max_selected < 1:
    max_selected = default["max_selected"]
if per_market_cap < 1:
    per_market_cap = default["per_market_cap"]
cycles = int(policy.get("recommended_cycles") or default["cycles"])
interval_sec = int(policy.get("recommended_interval_sec") or default["interval_sec"])
if cycles < 1:
    cycles = default["cycles"]
if interval_sec < 1:
    interval_sec = default["interval_sec"]
print(f"{max_selected} {per_market_cap:g} {cycles} {interval_sec}")
PY
  )"
  read -r policy_max_selected policy_per_market_cap policy_cycles policy_interval_sec <<< "$next_policy"
  effective_cycles="${CYCLES:-$policy_cycles}"
  effective_interval_sec="${INTERVAL_SEC:-$policy_interval_sec}"
  effective_max_selected="${MAX_SELECTED_MARKETS:-$policy_max_selected}"
  effective_per_market_cap="${RESEARCH_PER_MARKET_CAP_USDC:-$policy_per_market_cap}"
  log "starting next autonomous research run_id=${run_id} session=${session_name} cycles=${effective_cycles} interval=${effective_interval_sec} max_selected=${effective_max_selected} research_per_market_cap=${effective_per_market_cap}"
  PULL_FIRST="$PULL_FIRST" \
  RUN_ID="$run_id" \
  SESSION_NAME="$session_name" \
  CYCLES="$effective_cycles" \
  INTERVAL_SEC="$effective_interval_sec" \
  MAX_SELECTED_MARKETS="$effective_max_selected" \
  RESEARCH_PER_MARKET_CAP_USDC="$effective_per_market_cap" \
  MAX_LIVE_RISK_USDC="$MAX_LIVE_RISK_USDC" \
    scripts/start_autonomous_research.sh | tee -a "$WATCH_LOG"
  sleep "$SLEEP_SEC"
done
