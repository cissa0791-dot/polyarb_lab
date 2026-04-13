"""Backfill an external manual SELL into the internal accounting system.

The funder wallet (0x8E5C2ABc...) sold 25 shares of the Russia-Ukraine
Ceasefire token in a single on-chain transaction on 2026-03-19 09:14 UTC,
outside of this system.  The 5 open positions in the DB were each 5 shares,
so the 25-share SELL is split evenly (5 shares each) across all 5 positions.

What this script does:
  1. Fetches authoritative BUY and SELL prices from Polymarket data-api.
  2. Corrects avg_fill_price in execution_reports for each BUY fill.
  3. Replays each BUY into a fresh Ledger.
  4. Applies the proportional SELL fill to each position in the Ledger.
  5. Calls build_trade_summary for each position.
  6. Writes trade_summary + position_closed event + account_snapshot to DB.

Run:
    python scripts/backfill_manual_close.py --dry-run   # preview only
    python scripts/backfill_manual_close.py             # commit to DB
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from sqlalchemy import create_engine, text

from src.domain.models import PositionState
from src.paper.ledger import Ledger
from src.storage.event_store import ResearchStore

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _ok(msg):   print(f"  {_GREEN}✓{_RESET} {msg}")
def _fail(msg): print(f"  {_RED}✗{_RESET} {msg}")
def _info(msg): print(f"  {_YELLOW}·{_RESET} {msg}")

DATA_API   = "https://data-api.polymarket.com"
FUNDER     = "0x8E5C2ABc4387cC0c5d06AE34B6d49a1AE68c65C1"
TOKEN_ID   = "8501497159083948713316135768103773293754490207922884688769443031624417212426"

# On-chain SELL: 25 shares @ 0.53, 2026-03-19 09:14:19 UTC
SELL_TS    = datetime(2026, 3, 19, 9, 14, 19, tzinfo=timezone.utc)
SELL_PRICE = 0.53
SELL_TOTAL = 25.0


def _fetch_trades() -> tuple[list[dict], list[dict]]:
    resp = httpx.get(f"{DATA_API}/trades",
                     params={"user": FUNDER, "limit": 100}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    buys  = sorted([t for t in data if t["asset"] == TOKEN_ID and t["side"] == "BUY"],
                   key=lambda x: x["timestamp"])
    sells = sorted([t for t in data if t["asset"] == TOKEN_ID and t["side"] == "SELL"],
                   key=lambda x: x["timestamp"])
    return buys, sells


def _load_db_positions(engine) -> list[dict]:
    """Load the 5 open BUY positions ordered by fill timestamp."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT oi.intent_id, oi.token_id, oi.market_slug, oi.size, oi.limit_price,
                   pe.position_id, pe.candidate_id, pe.ts as opened_ts,
                   pe.payload_json as pe_payload,
                   er.filled_size, er.avg_fill_price, er.ts as fill_ts,
                   er.id as er_row_id
            FROM order_intents oi
            JOIN execution_reports er ON er.intent_id = oi.intent_id
            JOIN position_events pe
                ON json_extract(pe.payload_json, '$.intent_id') = oi.intent_id
            WHERE er.status = 'filled' AND oi.side = 'BUY'
              AND pe.event_type = 'position_opened'
              AND pe.position_id NOT IN (
                SELECT position_id FROM position_events
                WHERE event_type IN ('position_closed','position_expired','position_force_closed')
              )
            ORDER BY er.ts
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def run_backfill(db_url: str, dry_run: bool) -> int:
    print(f"\n{_BOLD}Backfill Manual Close — external SELL reconciliation{_RESET}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Fetch authoritative trade data
    # ------------------------------------------------------------------
    print(f"\n[1] Fetching trade data from Polymarket data-api")
    buys, sells = _fetch_trades()
    _ok(f"data-api: {len(buys)} BUY fills, {len(sells)} SELL fills")
    if not buys:
        _fail("No BUY fills found at data-api — wrong funder address?")
        return 1
    if not sells:
        _fail("No SELL fills found at data-api — external SELL not visible yet?")
        return 1

    real_buy_prices = [t["price"] for t in buys]
    sell = sells[0]  # single 25-share SELL
    actual_sell_price = float(sell["price"])
    actual_sell_ts    = datetime.fromtimestamp(sell["timestamp"], tz=timezone.utc)

    total_buy_cost     = sum(t["size"] * t["price"] for t in buys)
    total_sell_proceed = sell["size"] * sell["price"]
    net_pnl            = total_sell_proceed - total_buy_cost

    _info(f"BUY prices (data-api): {real_buy_prices}")
    _info(f"SELL: {sell['size']} shares @ {actual_sell_price}  ts={actual_sell_ts.isoformat()}")
    _info(f"Total buy cost:      ${total_buy_cost:.4f}")
    _info(f"Total sell proceeds: ${total_sell_proceed:.4f}")
    _info(f"Net realized P&L:    ${net_pnl:.4f}")

    # ------------------------------------------------------------------
    # Step 2: Load DB positions
    # ------------------------------------------------------------------
    print(f"\n[2] Loading open positions from DB")
    engine = create_engine(db_url, future=True)
    positions = _load_db_positions(engine)
    if not positions:
        _fail("No open positions found in DB — already backfilled?")
        return 1
    _ok(f"{len(positions)} open positions found")

    if len(positions) != len(buys):
        _fail(f"DB positions ({len(positions)}) != data-api BUY fills ({len(buys)}) — cannot safely align")
        return 1

    # Each position gets its matching BUY price from data-api (by order index)
    # and a proportional share of the SELL (25 / 5 = 5 shares each at 0.53)
    shares_per_position = SELL_TOTAL / len(positions)
    if abs(shares_per_position - round(shares_per_position)) > 1e-9:
        _fail(f"SELL total {SELL_TOTAL} does not divide evenly into {len(positions)} positions")
        return 1
    shares_per_position = round(shares_per_position, 6)
    _info(f"SELL split: {shares_per_position} shares per position @ {actual_sell_price}")

    if dry_run:
        print(f"\n{_YELLOW}--- DRY-RUN: showing what would be written ---{_RESET}")
        for i, (pos, buy_trade) in enumerate(zip(positions, buys)):
            real_buy_price  = float(buy_trade["price"])
            entry_cost      = pos["filled_size"] * real_buy_price
            exit_proceeds   = shares_per_position * actual_sell_price
            realized_pnl    = exit_proceeds - entry_cost
            holding_sec     = (actual_sell_ts - datetime.fromisoformat(str(pos["fill_ts"])).replace(tzinfo=timezone.utc)).total_seconds()
            _info(f"position {pos['position_id'][:8]}...")
            _info(f"  BUY price DB={pos['avg_fill_price']} → real={real_buy_price}")
            _info(f"  entry_cost=${entry_cost:.4f}  exit_proceeds=${exit_proceeds:.4f}  realized_pnl=${realized_pnl:.4f}")
            _info(f"  holding_sec={holding_sec:.0f}")
        return 0

    # ------------------------------------------------------------------
    # Step 3: Correct BUY avg_fill_price in execution_reports
    # ------------------------------------------------------------------
    print(f"\n[3] Correcting BUY avg_fill_price in execution_reports")
    store = ResearchStore(db_url)
    with engine.begin() as conn:
        for pos, buy_trade in zip(positions, buys):
            real_price = float(buy_trade["price"])
            conn.execute(text("""
                UPDATE execution_reports
                SET avg_fill_price = :price
                WHERE id = :row_id
            """), {"price": real_price, "row_id": pos["er_row_id"]})
            _ok(f"  {pos['position_id'][:8]}... price {pos['avg_fill_price']} → {real_price}")

    # ------------------------------------------------------------------
    # Step 4: Replay BUY + apply SELL into Ledger → build trade_summaries
    # ------------------------------------------------------------------
    print(f"\n[4] Replaying fills into Ledger and building trade_summaries")
    for pos, buy_trade in zip(positions, buys):
        position_id    = pos["position_id"]
        real_buy_price = float(buy_trade["price"])
        filled_size    = float(pos["filled_size"])
        symbol         = pos["token_id"]
        market_slug    = pos["market_slug"]
        candidate_id   = pos["candidate_id"]
        fill_ts_str    = str(pos["fill_ts"])

        # Fresh ledger per position (each is independent)
        ledger = Ledger()

        placed = ledger.place_limit_order(
            order_id=position_id, symbol=symbol, market_slug=market_slug,
            side="BUY", shares=filled_size, limit_price=real_buy_price,
            ts=fill_ts_str, candidate_id=candidate_id, position_id=position_id,
        )
        if not placed:
            _fail(f"place_limit_order failed for {position_id[:8]}")
            store.close(); return 1

        applied = ledger.apply_fill(order_id=position_id, shares=filled_size, price=real_buy_price)
        if not applied:
            _fail(f"apply_fill (BUY) failed for {position_id[:8]}")
            store.close(); return 1

        # Apply SELL fill
        sell_intent_id = str(uuid.uuid4())
        placed_sell = ledger.place_limit_order(
            order_id=sell_intent_id, symbol=symbol, market_slug=market_slug,
            side="SELL", shares=shares_per_position, limit_price=actual_sell_price,
            ts=actual_sell_ts.isoformat(), candidate_id=candidate_id, position_id=position_id,
        )
        if not placed_sell:
            _fail(f"place_limit_order (SELL) failed for {position_id[:8]}")
            store.close(); return 1

        applied_sell = ledger.apply_fill(order_id=sell_intent_id, shares=shares_per_position, price=actual_sell_price)
        if not applied_sell:
            _fail(f"apply_fill (SELL) failed for {position_id[:8]}")
            store.close(); return 1

        closed_pos = ledger.position_records.get(position_id)
        if not closed_pos or closed_pos.is_open:
            _fail(f"position {position_id[:8]} did not close in ledger")
            store.close(); return 1

        trade_summary = ledger.build_trade_summary(position_id)
        if trade_summary is None:
            _fail(f"build_trade_summary returned None for {position_id[:8]}")
            store.close(); return 1

        store.save_trade_summary(trade_summary)
        _ok(f"trade_summary: {position_id[:8]}  entry=${trade_summary.entry_cost_usd:.4f}  "
            f"exit=${trade_summary.exit_proceeds_usd:.4f}  "
            f"pnl=${trade_summary.realized_pnl_usd:.4f}")

        store.save_position_event(
            position_id=position_id,
            candidate_id=candidate_id,
            event_type="position_closed",
            symbol=symbol,
            market_slug=market_slug,
            state=PositionState.CLOSED.value,
            reason_code="MANUAL_EXTERNAL_CLOSE",
            payload={
                "realized_pnl_usd":   trade_summary.realized_pnl_usd,
                "entry_cost_usd":     trade_summary.entry_cost_usd,
                "exit_proceeds_usd":  trade_summary.exit_proceeds_usd,
                "sell_price":         actual_sell_price,
                "sell_ts":            actual_sell_ts.isoformat(),
                "sell_tx":            sell["transactionHash"],
                "source":             "backfill_manual_close",
            },
            ts=actual_sell_ts,
        )
        _ok(f"position_closed event: {position_id[:8]}")

    store.save_account_snapshot(ledger.snapshot())
    _ok("account_snapshot saved")
    store.close()

    # ------------------------------------------------------------------
    # Step 5: Reconcile audit
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"{_BOLD}[5] Running reconcile_audit.py{_RESET}")
    print("=" * 60)
    audit = subprocess.run(
        [sys.executable, "scripts/reconcile_audit.py", "--db", db_url],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return audit.returncode


def main():
    parser = argparse.ArgumentParser(description="Backfill external manual SELL into DB")
    parser.add_argument("--db",      default="sqlite:///data/processed/paper.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.exit(run_backfill(args.db, args.dry_run))


if __name__ == "__main__":
    main()
