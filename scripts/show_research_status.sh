#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== tmux =="
tmux ls 2>/dev/null || true

echo
echo "== processes =="
ps aux | grep -E "run_evidence_research_pipeline|run_auto_trade_profit|run_autonomous_project_manager" | grep -v grep || true

echo
echo "== latest research files =="
ls -lh data/reports/research*_latest.* 2>/dev/null || true

echo
echo "== latest full summary =="
python - <<'PY'
import json
from pathlib import Path

path = Path("data/reports/research_pipeline_summary_latest.json")
if not path.exists():
    print("missing data/reports/research_pipeline_summary_latest.json")
else:
    data = json.loads(path.read_text())
    keys = [
        "run_id",
        "partial",
        "partial_reason",
        "scale_recommendation",
        "live_canary_eligible_count",
        "dry_run_focus_count",
        "blacklist_count",
        "simulated_profitable_market_count",
        "replay_confirmed_market_count",
        "actual_reward_confirmed_market_count",
        "live_ready_blockers",
    ]
    for key in keys:
        print(f"{key}: {data.get(key)}")
PY

echo
echo "== autonomous decision =="
python scripts/run_autonomous_project_manager.py || true
