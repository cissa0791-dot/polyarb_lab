"""Live reconciliation audit report.

Reads the SQLite store and prints a concise, auditable summary of:
  - all live order fills (sync + async)
  - position open/close events
  - fee and PnL consistency
  - any open (unreconciled) orders
  - rewards status

Usage:
    python scripts/reconcile_audit.py
    python scripts/reconcile_audit.py --db sqlite:///data/processed/paper.db
    python scripts/reconcile_audit.py --since 2026-03-01

Output is plain text, one section per topic.  Exit code 0 = clean, 1 = mismatches found.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP  = "-" * 64
_WARN = "  WARN "
_OK   = "  OK   "
_INFO = "  INFO "


def _print_section(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def _since_clause(col: str, since: datetime | None) -> str:
    if since is None:
        return ""
    ts = since.strftime("%Y-%m-%d %H:%M:%S")
    return f" AND {col} >= '{ts}'"


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _live_executions(conn, since: datetime | None) -> list[dict[str, Any]]:
    """All execution_report rows for live orders, ordered by ts."""
    since_sql = _since_clause("er.ts", since)
    rows = conn.execute(text(f"""
        SELECT
            oi.intent_id,
            oi.market_slug,
            oi.token_id,
            oi.side,
            oi.size       AS requested_size,
            oi.limit_price,
            er.status,
            er.filled_size,
            er.avg_fill_price,
            er.fee_paid_usd,
            er.payload_json,
            er.ts
        FROM order_intents oi
        JOIN execution_reports er ON oi.intent_id = er.intent_id
        WHERE oi.mode = 'live'
        {since_sql}
        ORDER BY er.ts
    """)).fetchall()
    return [dict(r._mapping) for r in rows]


def _open_live_orders(conn) -> list[dict[str, Any]]:
    """Live orders whose most recent report is still SUBMITTED or PARTIAL."""
    rows = conn.execute(text("""
        SELECT
            oi.intent_id,
            oi.market_slug,
            oi.token_id,
            oi.side,
            oi.size,
            oi.limit_price,
            er.status,
            er.filled_size,
            er.payload_json,
            er.ts
        FROM order_intents oi
        JOIN (
            SELECT intent_id, MAX(id) AS max_id FROM execution_reports GROUP BY intent_id
        ) latest ON oi.intent_id = latest.intent_id
        JOIN execution_reports er ON er.id = latest.max_id
        WHERE oi.mode = 'live'
          AND er.status IN ('submitted', 'partial')
        ORDER BY er.ts
    """)).fetchall()
    return [dict(r._mapping) for r in rows]


def _position_events(conn, since: datetime | None) -> list[dict[str, Any]]:
    since_sql = _since_clause("ts", since)
    rows = conn.execute(text(f"""
        SELECT position_id, candidate_id, event_type, symbol, market_slug, state, reason_code, payload_json, ts
        FROM position_events
        WHERE 1=1 {since_sql}
        ORDER BY ts
    """)).fetchall()
    return [dict(r._mapping) for r in rows]


def _account_snapshots(conn, since: datetime | None) -> list[dict[str, Any]]:
    since_sql = _since_clause("ts", since)
    rows = conn.execute(text(f"""
        SELECT cash, realized_pnl, unrealized_pnl, daily_pnl, open_positions, ts
        FROM account_snapshots
        WHERE 1=1 {since_sql}
        ORDER BY ts
    """)).fetchall()
    return [dict(r._mapping) for r in rows]


def _trade_summaries(conn, since: datetime | None) -> list[dict[str, Any]]:
    since_sql = _since_clause("opened_ts", since)
    rows = conn.execute(text(f"""
        SELECT position_id, symbol, market_slug, state, entry_cost_usd, exit_proceeds_usd,
               fees_paid_usd, realized_pnl_usd, holding_duration_sec, opened_ts, closed_ts
        FROM trade_summaries
        WHERE 1=1 {since_sql}
        ORDER BY opened_ts
    """)).fetchall()
    return [dict(r._mapping) for r in rows]


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_live_fills(conn, since: datetime | None) -> int:
    """Returns number of mismatches found."""
    _print_section("1. LIVE FILLS (all execution_report rows for live orders)")
    rows = _live_executions(conn, since)
    if not rows:
        print(f"{_INFO} No live execution reports found.")
        return 0

    mismatches = 0
    status_counts: dict[str, int] = {}
    total_filled_usd = 0.0
    total_fees_usd   = 0.0

    for r in rows:
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        filled = r["filled_size"] or 0.0
        price  = r["avg_fill_price"] or r["limit_price"] or 0.0
        notional = filled * price
        total_filled_usd += notional
        total_fees_usd   += (r["fee_paid_usd"] or 0.0)

        meta = json.loads(r["payload_json"] or "{}")
        live_id = meta.get("metadata", {}).get("live_order_id") or meta.get("live_order_id")
        terminal = meta.get("metadata", {}).get("terminal") or meta.get("terminal")

        line = (
            f"  {r['ts']}  {st:12s}  filled={filled:.4f}  "
            f"notional=${notional:.4f}  fee=${r['fee_paid_usd']:.4f}  "
            f"market={r['market_slug'][:40]}  live_id={str(live_id)[:12]}  "
            f"{'[terminal]' if terminal else ''}"
        )
        print(line)

    print()
    print(f"  Status breakdown: {dict(sorted(status_counts.items()))}")
    print(f"  Total filled notional:  ${total_filled_usd:.6f}")
    print(f"  Total fees recorded:    ${total_fees_usd:.6f}")
    if total_fees_usd == 0.0 and status_counts.get("filled", 0) + status_counts.get("partial", 0) > 0:
        print(f"{_WARN} fee_paid_usd=0 for all live fills — expected for maker orders (0 bps), verify if taker.")
    return mismatches


def report_open_orders(conn) -> int:
    _print_section("2. OPEN / UNRECONCILED LIVE ORDERS")
    rows = _open_live_orders(conn)
    if not rows:
        print(f"{_OK} No unreconciled live orders (all orders have terminal status).")
        return 0

    print(f"{_WARN} {len(rows)} order(s) still in SUBMITTED/PARTIAL state — will be recovered on next startup:")
    for r in rows:
        meta = json.loads(r["payload_json"] or "{}")
        live_id = meta.get("metadata", {}).get("live_order_id") or meta.get("live_order_id", "?")
        print(f"  {r['ts']}  {r['status']:10s}  filled={r['filled_size']:.4f}/{r['size']:.4f}  "
              f"market={r['market_slug'][:40]}  live_id={str(live_id)[:16]}")
    return len(rows)


def report_positions(conn, since: datetime | None) -> int:
    _print_section("3. POSITION EVENTS (open / close audit trail)")
    rows = _position_events(conn, since)
    if not rows:
        print(f"{_INFO} No position events recorded.")
        return 0

    open_ids:  set[str] = set()
    close_ids: set[str] = set()
    for r in rows:
        et = r["event_type"]
        pid = r["position_id"]
        src = json.loads(r["payload_json"] or "{}").get("source", "")
        src_tag = f"  [{src}]" if src else ""
        print(f"  {r['ts']}  {et:20s}  {pid[:16]}  {r['market_slug'][:40]}{src_tag}")
        if et == "position_opened":
            open_ids.add(pid)
        elif et in ("position_closed", "position_expired", "position_force_closed"):
            close_ids.add(pid)

    unclosed = open_ids - close_ids
    print()
    print(f"  Opened:  {len(open_ids)}   Closed: {len(close_ids)}   Currently open: {len(unclosed)}")
    return 0


def report_pnl(conn, since: datetime | None) -> int:
    _print_section("4. PNL & FEE LEDGER")
    trades = _trade_summaries(conn, since)
    snapshots = _account_snapshots(conn, since)

    mismatches = 0

    if trades:
        total_entry    = sum(t["entry_cost_usd"] or 0 for t in trades)
        total_exit     = sum(t["exit_proceeds_usd"] or 0 for t in trades)
        total_fees     = sum(t["fees_paid_usd"] or 0 for t in trades)
        total_rpnl     = sum(t["realized_pnl_usd"] or 0 for t in trades)
        implied_rpnl   = total_exit - total_entry - total_fees

        print(f"  Trade summaries:    {len(trades)}")
        print(f"  Total entry cost:   ${total_entry:.6f}")
        print(f"  Total exit proceeds:${total_exit:.6f}")
        print(f"  Total fees paid:    ${total_fees:.6f}")
        print(f"  Realized PnL (DB):  ${total_rpnl:.6f}")
        print(f"  Implied PnL (calc): ${implied_rpnl:.6f}")
        if abs(total_rpnl - implied_rpnl) > 0.0001:
            print(f"{_WARN} PnL mismatch: DB={total_rpnl:.6f}  calc={implied_rpnl:.6f}  delta={total_rpnl-implied_rpnl:.6f}")
            mismatches += 1
        else:
            print(f"{_OK} PnL consistent (DB == exit - entry - fees within $0.0001)")
    else:
        print(f"{_INFO} No closed trade summaries.")

    if snapshots:
        latest = snapshots[-1]
        print()
        print(f"  Latest account snapshot ({latest['ts']}):")
        print(f"    cash={latest['cash']:.6f}  realized_pnl={latest['realized_pnl']:.6f}  "
              f"unrealized_pnl={latest['unrealized_pnl']:.6f}  open_positions={latest['open_positions']}")
    else:
        print(f"{_INFO} No account snapshots recorded.")

    return mismatches


def report_rewards() -> int:
    _print_section("5. REWARDS")
    print(f"{_WARN} REWARDS_API_UNVERIFIED — /rewards/epoch returns HTTP 405 (no public endpoint confirmed).")
    print(f"  Rewards are non-blocking and excluded from PnL ledger.")
    print(f"  Maker fee rate is 0 bps for standard accounts — fee_paid_usd=0 is expected.")
    return 0


def report_consistency(conn, since: datetime | None) -> int:
    """Cross-check: every FILLED/PARTIAL execution_report has a position_opened event."""
    _print_section("6. CROSS-CHECK: fills ↔ position_events consistency")
    exec_rows = _live_executions(conn, since)
    pos_rows  = _position_events(conn, since)

    filled_intent_ids = {
        r["intent_id"] for r in exec_rows
        if r["status"] in ("filled", "partial") and (r["filled_size"] or 0) > 1e-9
    }
    pos_opened_intent_ids = set()
    for r in pos_rows:
        if r["event_type"] == "position_opened":
            payload = json.loads(r["payload_json"] or "{}")
            iid = payload.get("intent_id")
            if iid:
                pos_opened_intent_ids.add(iid)

    missing = filled_intent_ids - pos_opened_intent_ids
    if not filled_intent_ids:
        print(f"{_INFO} No filled live orders to cross-check.")
        return 0
    if missing:
        print(f"{_WARN} {len(missing)} filled order(s) have no position_opened event:")
        for iid in sorted(missing):
            print(f"  intent_id={iid}")
        return len(missing)
    print(f"{_OK} All {len(filled_intent_ids)} filled order(s) have a matching position_opened event.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket live reconciliation audit")
    parser.add_argument("--db",    default="sqlite:///data/processed/paper.db",
                        help="SQLAlchemy DB URL (default: sqlite:///data/processed/paper.db)")
    parser.add_argument("--since", default=None,
                        help="Only include records on or after this date (YYYY-MM-DD or ISO-8601)")
    args = parser.parse_args()

    since: datetime | None = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: --since value {args.since!r} is not a valid ISO date.")
            sys.exit(2)

    engine = create_engine(args.db, future=True)

    print(f"\nPolymarket Live Reconciliation Audit")
    print(f"DB:    {args.db}")
    print(f"Since: {since or 'all time'}")

    total_mismatches = 0
    with engine.connect() as conn:
        total_mismatches += report_live_fills(conn, since)
        total_mismatches += report_open_orders(conn)
        total_mismatches += report_positions(conn, since)
        total_mismatches += report_pnl(conn, since)
        total_mismatches += report_rewards()
        total_mismatches += report_consistency(conn, since)

    print(f"\n{_SEP}")
    if total_mismatches == 0:
        print("RECONCILIATION: CLEAN — no mismatches detected.")
    else:
        print(f"RECONCILIATION: {total_mismatches} MISMATCH(ES) — review WARN lines above.")
    print(_SEP)

    sys.exit(0 if total_mismatches == 0 else 1)


if __name__ == "__main__":
    main()
