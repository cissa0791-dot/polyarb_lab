"""
Runner integration test for the Hungary PM maker-MM wide-scan path.

Validates that the MAKER_REWARDED_EVENT_MM_V1 candidate can traverse
the full ResearchRunner pipeline:
  wide_scan_maker_mm scanner
    -> _run_maker_mm_scan()
    -> qualify (local G5 config adapter)
    -> rank
    -> size (local G6 config adapter)
    -> risk (local G6 config adapter)
    -> _submit_candidate_orders()
    -> ResearchStore persistence

Isolation:
  - Uses runner_mm_test.db (not paper.db, not bridge_test.db)
  - Only maker_rewarded_event_mm_v1 scan runs (other scans disabled)
  - runner.store and runner.opportunity_store patched to test DB

Usage:
    py -3 scripts/run_runner_maker_mm_integration.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, select, func as sqlfunc, MetaData

from src.runtime.runner import ResearchRunner
from src.storage.event_store import ResearchStore
from src.utils.db import OpportunityStore

TEST_DB_URL = f"sqlite:///{ROOT / 'data' / 'processed' / 'runner_mm_test.db'}"
MAKER_MM_FAMILY = "maker_rewarded_event_mm_v1"


def count_table_rows(db_url: str, table_names: list[str]) -> dict[str, int]:
    engine = create_engine(db_url, future=True)
    meta = MetaData()
    meta.reflect(bind=engine)
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for name in table_names:
            if name in meta.tables:
                counts[name] = conn.execute(
                    select(sqlfunc.count()).select_from(meta.tables[name])
                ).scalar()
            else:
                counts[name] = 0
    return counts


def main() -> None:
    print("=" * 70)
    print("RUNNER MAKER-MM INTEGRATION TEST — Hungary PM neg-risk pair")
    print("=" * 70)
    print(f"  Test DB:  {TEST_DB_URL}")
    print(f"  Strategy: {MAKER_MM_FAMILY}")
    print()

    # Instantiate runner with default config (settings.yaml).
    print("[1] Instantiating ResearchRunner...")
    runner = ResearchRunner()

    # Patch stores to use isolated test DB (paper.db is not touched).
    print(f"[2] Patching stores -> {TEST_DB_URL}")
    runner.store = ResearchStore(TEST_DB_URL)
    runner.opportunity_store = OpportunityStore(TEST_DB_URL)

    # Snapshot row counts before the run.
    tables = [
        "opportunity_candidates",
        "risk_decisions",
        "order_intents",
        "execution_reports",
        "account_snapshots",
        "rejection_events",
        "position_events",
    ]
    before = count_table_rows(TEST_DB_URL, tables)
    print(f"[3] Pre-run row counts: {before}")
    print()

    # Target ONLY maker_mm — disables single_market and cross_market scans.
    print("[4] Running run_once() with campaign_target_strategy_families=[maker_rewarded_event_mm_v1]...")
    print()
    summary = runner.run_once(
        experiment_context={
            "campaign_target_strategy_families": [MAKER_MM_FAMILY],
        }
    )
    print()

    # Snapshot row counts after the run.
    after = count_table_rows(TEST_DB_URL, tables)
    delta = {k: after[k] - before.get(k, 0) for k in after}

    print("=" * 70)
    print("RUNNER INTEGRATION RESULT")
    print("=" * 70)
    meta = summary.metadata or {}
    raw_by_family = meta.get("raw_candidates_by_family", {})
    print(f"  run_id:              {summary.run_id}")
    print(f"  raw_candidates:      {raw_by_family.get(MAKER_MM_FAMILY, 0)}")
    print(f"  candidates_generated:{summary.candidates_generated}")
    print(f"  risk_accepted:       {summary.risk_accepted}")
    print(f"  risk_rejected:       {summary.risk_rejected}")
    print(f"  paper_orders_created:{summary.paper_orders_created}")
    print(f"  fills:               {summary.fills}")
    print(f"  partial_fills:       {summary.partial_fills}")
    print(f"  open_positions:      {summary.open_positions}")
    print(f"  closed_positions:    {summary.closed_positions}")
    print(f"  realized_pnl:        {round(summary.realized_pnl, 4)}")
    print(f"  near_miss_candidates:{summary.near_miss_candidates}")
    print()
    print("  DB rows written this run:")
    for table, count in delta.items():
        if count:
            print(f"    {table}: +{count}")
    print()
    print("  DB totals after run:")
    for table, count in after.items():
        if count:
            print(f"    {table}: {count}")
    print()

    # Derive verdict.
    raw_detected  = (summary.metadata or {}).get("raw_candidates_by_family", {}).get(MAKER_MM_FAMILY, 0)
    qualified     = summary.candidates_generated
    risk_ok       = summary.risk_accepted
    orders        = summary.paper_orders_created
    candidates_ok = delta.get("opportunity_candidates", 0)
    risk_rows_ok  = delta.get("risk_decisions", 0)
    intent_rows   = delta.get("order_intents", 0)
    exec_rows     = delta.get("execution_reports", 0)

    if raw_detected > 0 and qualified > 0 and risk_ok > 0 and orders > 0:
        path_verdict = "RUNNER_PATH_CONFIRMED"
    elif raw_detected > 0 and qualified > 0 and risk_ok == 0:
        path_verdict = "BLOCKED_AT_RISK"
    elif raw_detected > 0 and qualified == 0:
        path_verdict = "BLOCKED_AT_QUALIFICATION"
    elif raw_detected == 0:
        path_verdict = "NO_CANDIDATES_DETECTED"
    else:
        path_verdict = "UNKNOWN"

    persist_verdict = (
        "PERSISTENCE_CONFIRMED"
        if candidates_ok > 0 and risk_rows_ok > 0 and intent_rows > 0 and exec_rows > 0
        else "PERSISTENCE_INCOMPLETE"
    )

    incomplete_fills = summary.paper_orders_created > 0 and any(
        "INCOMPLETE" in str(getattr(summary, "event_counts", {}))
        for _ in [None]
    )

    print(f"  path_verdict:        {path_verdict}")
    print(f"  persist_verdict:     {persist_verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
