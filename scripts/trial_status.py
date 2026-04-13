"""Read-only forward profitability trial monitor.

Trial spec (locked 2026-03-19):
  strategy:    single_market_mispricing
  sizing:      $10/pair
  min_edge:    3 cents
  take_profit: $0.30 / stop_loss: $0.30
  max_holding: 600 s
  N target:    10 round-trips
  PASS:        cumulative_pnl > $0
  Circuit:     TRIGGERED if cumulative_pnl < -$5.00

Trial trades are all trade_summaries whose matching position_closed event
does NOT carry reason_code='MANUAL_EXTERNAL_CLOSE' (that code marks the
2026-03-19 backfill reconciliation, not forward trial entries).

Run:
    python scripts/trial_status.py
    python scripts/trial_status.py --db sqlite:///data/processed/paper.db
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

# Trial spec constants
_TRIAL_N       = 10
_CIRCUIT_LEVEL = -5.00
_BACKFILL_CODE = "MANUAL_EXTERNAL_CLOSE"


def _col(value: float, *, good: str = _GREEN, bad: str = _RED) -> str:
    return (good if value >= 0 else bad) + f"{value:+.4f}" + _RESET


def _load_trial_trades(engine) -> list[dict]:
    """Return all trade_summaries that are part of the forward trial."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                ts.position_id,
                ts.market_slug,
                ts.entry_cost_usd,
                ts.exit_proceeds_usd,
                ts.realized_pnl_usd,
                ts.holding_duration_sec,
                ts.opened_ts,
                ts.closed_ts,
                pe.reason_code
            FROM trade_summaries ts
            JOIN position_events pe
                ON pe.position_id = ts.position_id
               AND pe.event_type = 'position_closed'
            WHERE pe.reason_code != :backfill_code
               OR pe.reason_code IS NULL
            ORDER BY ts.closed_ts
        """), {"backfill_code": _BACKFILL_CODE}).fetchall()
    return [dict(r._mapping) for r in rows]


def _load_open_positions(engine) -> int:
    """Count positions that are opened but not yet closed."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(DISTINCT position_id) as n
            FROM position_events
            WHERE event_type = 'position_opened'
              AND position_id NOT IN (
                SELECT position_id FROM position_events
                WHERE event_type IN ('position_closed','position_expired','position_force_closed')
              )
        """)).fetchone()
    return int(row[0]) if row else 0


def run(db_url: str) -> None:
    engine = create_engine(db_url, future=True)

    trades    = _load_trial_trades(engine)
    n_done    = len(trades)
    cum_pnl   = sum(t["realized_pnl_usd"] for t in trades)
    open_pos  = _load_open_positions(engine)
    now       = datetime.now(timezone.utc)

    # Circuit breaker
    circuit   = cum_pnl < _CIRCUIT_LEVEL
    cb_label  = f"{_RED}TRIGGERED{_RESET}" if circuit else f"{_GREEN}CLEAR{_RESET}"

    # Days elapsed
    if trades:
        first_open = trades[0]["opened_ts"]
        if isinstance(first_open, str):
            first_open = datetime.fromisoformat(first_open)
        if first_open.tzinfo is None:
            first_open = first_open.replace(tzinfo=timezone.utc)
        elapsed_days = (now - first_open).total_seconds() / 86400
        elapsed_str  = f"{elapsed_days:.1f} days"
    else:
        elapsed_str = "—"

    # Pass/fail (only meaningful once N=10)
    if n_done >= _TRIAL_N:
        result = f"{_GREEN}PASS{_RESET}" if cum_pnl > 0 else f"{_RED}FAIL{_RESET}"
        result_line = f"  {'Result':<22} {result}"
    else:
        result_line = None

    print()
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}  Forward Profitability Trial — Status{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print(f"  {'Completed round-trips':<22} {_BOLD}{n_done}{_RESET} / {_TRIAL_N}")
    print(f"  {'Open positions':<22} {open_pos}")
    print(f"  {'Cumulative realized PnL':<22} {_col(cum_pnl)}")
    print(f"  {'Circuit breaker':<22} {cb_label}  (threshold ${_CIRCUIT_LEVEL:.2f})")
    print(f"  {'Days elapsed':<22} {elapsed_str}")
    if result_line:
        print(result_line)
    print()

    if not trades:
        print(f"  {_YELLOW}No trial trades recorded yet.{_RESET}")
        print()
        return

    # Per-trade table
    hdr = f"{'#':<3}  {'market_slug':<36}  {'entry':>8}  {'exit':>8}  {'pnl':>8}  {'hold_s':>7}  {'reason'}"
    print(f"  {_BOLD}{hdr}{_RESET}")
    print(f"  {'-' * len(hdr)}")
    for i, t in enumerate(trades, 1):
        slug      = str(t["market_slug"] or "")[:36]
        entry     = t["entry_cost_usd"]
        exit_p    = t["exit_proceeds_usd"]
        pnl       = t["realized_pnl_usd"]
        hold      = t["holding_duration_sec"]
        reason    = str(t["reason_code"] or "—")[:20]
        pnl_str   = _col(pnl)
        print(f"  {i:<3}  {slug:<36}  {entry:>8.4f}  {exit_p:>8.4f}  {pnl_str}  {hold:>7.0f}  {reason}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only forward trial status")
    parser.add_argument("--db", default="sqlite:///data/processed/paper.db")
    args = parser.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()
