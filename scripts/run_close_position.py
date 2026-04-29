"""Minimal live SELL script — proves the close / realized-PnL path end-to-end.

Loads one open live BUY position from the DB, submits a SELL limit order at
the current CLOB best bid, records the terminal ExecutionReport, builds a
TradeSummary, saves a position_closed event, and runs reconcile_audit.

DB evidence after success:
  - trade_summaries:  1 new row (entry_cost_usd, exit_proceeds_usd, realized_pnl_usd)
  - position_events:  position_closed event
  - execution_reports: SELL order with FILLED / SUBMITTED status

WARNING: Existing validation positions were bought at $0.99.  Current bid is
~$0.01.  Closing one position books a realized loss of ~$4.90.  This is
expected — the goal is to prove the pipeline, not to profit from validation
fills.

Usage:
    python scripts/run_close_position.py --dry-run
    python scripts/run_close_position.py
    python scripts/run_close_position.py --position-id <UUID>
    python scripts/run_close_position.py --db sqlite:///data/processed/paper.db
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from src.domain.models import (
    ExecutionReport,
    OrderIntent,
    OrderMode,
    OrderStatus,
    OrderType,
    PositionState,
)
from src.live.auth import build_authenticated_client, load_live_credentials
from src.live.broker import LiveBroker
from src.live.client import LiveWriteClient
from src.live.clob_compat import AssetType, BalanceAllowanceParams
from src.paper.ledger import Ledger
from src.storage.event_store import ResearchStore

_CLOB_HOST = "https://clob.polymarket.com"

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _ok(msg: str)   -> None: print(f"  {_GREEN}✓{_RESET} {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}✗{_RESET} {msg}")
def _info(msg: str) -> None: print(f"  {_YELLOW}·{_RESET} {msg}")


# ---------------------------------------------------------------------------
# Ledger helpers — replay BUY then apply SELL
# ---------------------------------------------------------------------------

def _replay_buy(ledger: Ledger, row: dict) -> bool:
    """Replay the original BUY fill into a fresh Ledger so SELL can follow."""
    placed = ledger.place_limit_order(
        order_id=row["position_id"],
        symbol=row["symbol"],
        market_slug=row["market_slug"],
        side="BUY",
        shares=float(row["filled_size"]),
        limit_price=float(row["avg_fill_price"] or row["limit_price"]),
        ts=str(row["ts"]),
        candidate_id=row["candidate_id"],
        position_id=row["position_id"],
    )
    if not placed:
        return False
    return ledger.apply_fill(
        order_id=row["position_id"],
        shares=float(row["filled_size"]),
        price=float(row["avg_fill_price"] or row["limit_price"]),
    )


def _apply_sell_to_ledger(
    ledger: Ledger,
    sell_intent_id: str,
    position_id: str,
    shares: float,
    price: float,
    ts: datetime,
) -> bool:
    placed = ledger.place_limit_order(
        order_id=sell_intent_id,
        symbol=ledger.position_records[position_id].symbol,
        market_slug=ledger.position_records[position_id].market_slug,
        side="SELL",
        shares=shares,
        limit_price=price,
        ts=ts.isoformat(),
        candidate_id=ledger.position_records[position_id].candidate_id,
        position_id=position_id,
    )
    if not placed:
        return False
    return ledger.apply_fill(order_id=sell_intent_id, shares=shares, price=price)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_REQUIRED_ENV = [
    "POLYMARKET_PRIVATE_KEY",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_API_PASSPHRASE",
    "POLYMARKET_CHAIN_ID",
    "POLYMARKET_SIGNATURE_TYPE",
    "POLYMARKET_FUNDER",
]


def _check_env() -> bool:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k, "").strip()]
    if missing:
        for k in missing:
            _fail(f"missing env var: {k}")
        return False
    return True


def run_close(db_url: str, dry_run: bool, target_position_id: str | None) -> int:
    print(f"\n{_BOLD}Close Position — live SELL proof{_RESET}")
    print("=" * 60)

    if not dry_run:
        if not _check_env():
            return 1

    # ------------------------------------------------------------------
    # Step 1: Load open positions
    # ------------------------------------------------------------------
    print(f"\n[1] Loading open live positions from DB")
    store = ResearchStore(db_url)
    rows = store.load_open_live_positions()
    if not rows:
        _fail("No open live positions found — nothing to close")
        store.close()
        return 1
    _ok(f"{len(rows)} open live position(s) found")

    # Select target
    if target_position_id:
        row = next((r for r in rows if r["position_id"] == target_position_id), None)
        if row is None:
            _fail(f"position_id {target_position_id!r} not found in open positions")
            store.close()
            return 1
    else:
        row = rows[0]  # close the oldest first

    pos_id     = row["position_id"]
    token_id   = row["symbol"]
    market_slug = row["market_slug"]
    entry_price = float(row["avg_fill_price"] or row["limit_price"])
    filled_size = float(row["filled_size"])

    _info(f"target position_id: {pos_id}")
    _info(f"market: {market_slug}")
    _info(f"entry_price: {entry_price}  filled_size: {filled_size}")
    _info(f"entry_cost:  ${entry_price * filled_size:.4f}")

    # ------------------------------------------------------------------
    # Step 2: Fetch current CLOB book — get best bid
    # ------------------------------------------------------------------
    print(f"\n[2] Fetching CLOB book for token")
    resp = httpx.get(f"{_CLOB_HOST}/book", params={"token_id": token_id}, timeout=8)
    resp.raise_for_status()
    book = resp.json()
    bids = book.get("bids", [])
    if not bids:
        _fail("No bids available — cannot price SELL exit")
        store.close()
        return 1
    best_bid = float(bids[0]["price"])
    exit_proceeds = best_bid * filled_size
    realized_pnl  = exit_proceeds - (entry_price * filled_size)
    _ok(f"best_bid={best_bid}  exit_proceeds=${exit_proceeds:.4f}")
    _info(f"expected realized_pnl: ${realized_pnl:.4f}  "
          f"({'LOSS' if realized_pnl < 0 else 'GAIN'})")

    if dry_run:
        _info("DRY-RUN: would submit SELL — not submitting")
        store.close()
        return 0

    # ------------------------------------------------------------------
    # Step 3: Credentials + broker
    # ------------------------------------------------------------------
    print(f"\n[3] Loading credentials + building LiveBroker")
    creds     = load_live_credentials()
    sig_type  = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "0").strip() or 0)
    funder    = os.environ.get("POLYMARKET_FUNDER", "").strip() or None
    clob      = build_authenticated_client(creds, _CLOB_HOST, signature_type=sig_type, funder=funder)
    client    = LiveWriteClient(clob, dry_run=False)
    broker    = LiveBroker(client)
    _ok(f"LiveBroker ready  sig_type={sig_type}  funder={funder}")

    # ------------------------------------------------------------------
    # Step 3b: Ensure CONDITIONAL (ERC1155) allowance is set for SELL
    # ------------------------------------------------------------------
    # BUY uses USDC (ERC20 collateral allowance — set during account setup).
    # SELL uses conditional ERC1155 tokens.  A separate setApprovalForAll must
    # be active on the proxy wallet before the CTF Exchange can transfer them.
    # /balance-allowance/update with asset_type=CONDITIONAL triggers this on-chain.
    # Idempotent: a no-op when the approval is already in place.
    print(f"\n[3b] Checking and updating CONDITIONAL (ERC1155) allowance for SELL")
    try:
        bal_params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL,
            token_id=token_id,
            signature_type=sig_type,
        )
        current = clob.get_balance_allowance(bal_params)
        _info(f"CONDITIONAL balance_allowance: {current}")

        updated = clob.update_balance_allowance(bal_params)
        _ok(f"update_balance_allowance: {updated}")
    except Exception as exc:
        _fail(f"balance_allowance step failed: {exc} — proceeding anyway")

    # ------------------------------------------------------------------
    # Step 4: Build ledger — replay BUY so SELL position check passes
    # ------------------------------------------------------------------
    print(f"\n[4] Replaying BUY into ledger")
    ledger = Ledger()
    if not _replay_buy(ledger, row):
        _fail("Failed to replay BUY into ledger — cannot proceed")
        store.close()
        return 1
    position = ledger.position_records.get(pos_id)
    _ok(f"position reconstructed  remaining_shares={position.remaining_shares if position else '?'}")

    # ------------------------------------------------------------------
    # Step 5: Build and submit SELL intent
    # ------------------------------------------------------------------
    print(f"\n[5] Submitting SELL {filled_size} @ {best_bid}")
    ts_now       = datetime.now(timezone.utc)
    sell_intent_id = str(uuid.uuid4())
    sell_intent  = OrderIntent(
        intent_id=sell_intent_id,
        candidate_id=row["candidate_id"] or "close_proof",
        mode=OrderMode.LIVE,
        market_slug=market_slug,
        token_id=token_id,
        position_id=pos_id,
        side="SELL",
        order_type=OrderType.LIMIT,
        size=filled_size,
        limit_price=best_bid,
        max_notional_usd=exit_proceeds * 1.1,
        ts=ts_now,
    )
    store.save_order_intent(sell_intent)
    report = broker.submit_limit_order(sell_intent)
    _info(f"status={report.status.value}  filled={report.filled_size}  "
          f"live_id={report.metadata.get('live_order_id')}")
    store.save_execution_report(report)
    _ok("initial execution_report saved")

    if report.status == OrderStatus.REJECTED:
        _fail(f"SELL rejected: {report.metadata.get('error')}")
        store.close()
        return 1

    # ------------------------------------------------------------------
    # Step 6: Poll for terminal status
    # ------------------------------------------------------------------
    print(f"\n[6] Polling for terminal state")
    live_order_id   = report.metadata.get("live_order_id")
    TERMINAL        = frozenset({"matched", "canceled", "cancelled", "expired"})
    final_report    = report
    POLL_ATTEMPTS   = 12
    POLL_SLEEP_S    = 5

    if report.status == OrderStatus.FILLED:
        _ok("SELL filled synchronously")
    elif live_order_id:
        for attempt in range(1, POLL_ATTEMPTS + 1):
            time.sleep(POLL_SLEEP_S)
            try:
                status = client.get_order_status(live_order_id)
                _info(f"poll {attempt}/{POLL_ATTEMPTS}  status={status.status!r}  "
                      f"matched={status.size_matched}  remaining={status.size_remaining}")
                is_terminal   = status.status.lower() in TERMINAL
                is_full       = status.size_remaining <= 1e-9 and status.size_matched > 0
                if is_terminal or is_full:
                    order_status = (
                        OrderStatus.FILLED   if status.size_matched >= filled_size - 1e-9
                        else OrderStatus.CANCELED if status.size_matched <= 1e-9
                        else OrderStatus.PARTIAL
                    )
                    final_report = ExecutionReport(
                        intent_id=sell_intent_id,
                        position_id=pos_id,
                        status=order_status,
                        filled_size=round(status.size_matched, 6),
                        avg_fill_price=best_bid if status.size_matched > 1e-9 else None,
                        metadata={"live_order_id": live_order_id,
                                  "live_status": status.status, "terminal": True},
                        ts=datetime.now(timezone.utc),
                    )
                    store.save_execution_report(final_report)
                    _ok(f"terminal report saved  status={order_status.value}")
                    break
            except Exception as exc:
                _info(f"poll {attempt} error: {exc}")

    # ------------------------------------------------------------------
    # Step 7: Apply SELL fill to ledger → build trade_summary
    # ------------------------------------------------------------------
    print(f"\n[7] Applying SELL fill to ledger and building trade_summary")
    sell_filled = final_report.filled_size
    if sell_filled <= 1e-9:
        _info("No shares sold — skipping trade_summary (position remains open)")
        store.save_account_snapshot(ledger.snapshot())
        store.close()
        return 0

    sell_ts = final_report.ts
    filled_ok = _apply_sell_to_ledger(ledger, sell_intent_id, pos_id, sell_filled, best_bid, sell_ts)
    if not filled_ok:
        _fail("apply_fill for SELL failed in ledger")
        store.close()
        return 1

    # Ledger auto-closes position when remaining_shares hits 0
    closed_position = ledger.position_records.get(pos_id)
    if closed_position and not closed_position.is_open:
        _ok(f"position closed in ledger  realized_pnl=${closed_position.realized_pnl_usd:.4f}")

    # Build and persist trade_summary
    trade_summary = ledger.build_trade_summary(pos_id)
    if trade_summary is None:
        _fail("build_trade_summary returned None — position may still be open in ledger")
        store.close()
        return 1

    store.save_trade_summary(trade_summary)
    _ok(f"trade_summary saved  entry=${trade_summary.entry_cost_usd:.4f}  "
        f"exit=${trade_summary.exit_proceeds_usd:.4f}  "
        f"realized_pnl=${trade_summary.realized_pnl_usd:.4f}")

    # Persist position_closed event
    store.save_position_event(
        position_id=pos_id,
        candidate_id=row["candidate_id"],
        event_type="position_closed",
        symbol=token_id,
        market_slug=market_slug,
        state=PositionState.CLOSED.value,
        reason_code="CLOSE_PROOF",
        payload={
            "realized_pnl_usd": trade_summary.realized_pnl_usd,
            "entry_cost_usd": trade_summary.entry_cost_usd,
            "exit_proceeds_usd": trade_summary.exit_proceeds_usd,
            "sell_intent_id": sell_intent_id,
            "source": "run_close_position",
        },
        ts=sell_ts,
    )
    _ok("position_closed event saved")

    store.save_account_snapshot(ledger.snapshot())
    _ok("account_snapshot saved")
    store.close()

    # ------------------------------------------------------------------
    # Step 8: Reconcile audit
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"{_BOLD}[8] Running reconcile_audit.py{_RESET}")
    print("=" * 60)
    audit = subprocess.run(
        [sys.executable, "scripts/reconcile_audit.py", "--db", db_url],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return audit.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SELL — close one open position")
    parser.add_argument("--db",          default="sqlite:///data/processed/paper.db")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show what would happen without submitting")
    parser.add_argument("--position-id", default=None,
                        help="Specific position_id to close (default: oldest open)")
    args = parser.parse_args()
    sys.exit(run_close(args.db, args.dry_run, args.position_id))


if __name__ == "__main__":
    main()
