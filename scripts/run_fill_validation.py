"""Minimal real-fill validation script for final operational acceptance.

Submits one tiny real order priced at the current best ask so it fills
immediately (synchronous fill path).  Records all required rows in the
ResearchStore, then invokes reconcile_audit.py to verify the DB is clean.

Cost: orderMinSize (typically 5) × best_ask_found.  The script searches for
the lowest-ask active token to minimise exposure — typically $0.005–$0.05.

Prerequisites (env vars already set in your PowerShell session):
    POLYMARKET_PRIVATE_KEY
    POLYMARKET_API_KEY / SECRET / PASSPHRASE
    POLYMARKET_CHAIN_ID=137
    POLYMARKET_SIGNATURE_TYPE=2
    POLYMARKET_FUNDER=<your proxy wallet address>

Usage:
    python scripts/run_fill_validation.py
    python scripts/run_fill_validation.py --db sqlite:///data/processed/paper.db
    python scripts/run_fill_validation.py --dry-run   # discover + print plan, no order
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
)
from src.live.auth import build_authenticated_client, load_live_credentials
from src.live.broker import LiveBroker
from src.live.client import LiveWriteClient
from src.paper.ledger import Ledger
from src.storage.event_store import ResearchStore

_CLOB_HOST  = "https://clob.polymarket.com"
_GAMMA_HOST = "https://gamma-api.polymarket.com"

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _ok(msg: str)   -> None: print(f"  {_GREEN}✓{_RESET} {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}✗{_RESET} {msg}")
def _info(msg: str) -> None: print(f"  {_YELLOW}·{_RESET} {msg}")


# ---------------------------------------------------------------------------
# Market discovery — find lowest-ask token to minimise cost
# ---------------------------------------------------------------------------

def _find_cheapest_fillable_token(
    early_exit_cost_usd: float = 0.10,
    market_limit: int = 15,
) -> tuple[str, str, str, float, float, float] | None:
    """Return (token_id, tick_size, market_slug, min_size, best_ask, cost_usd) or None.

    Exits as soon as any token with cost_usd <= early_exit_cost_usd is found,
    so scanning terminates in seconds rather than exhausting all markets.
    Checks at most market_limit markets to bound total scan time.
    """
    resp = httpx.get(
        f"{_GAMMA_HOST}/markets",
        params={"limit": market_limit, "active": "true", "closed": "false"},
        timeout=15,
    )
    resp.raise_for_status()
    markets = resp.json()
    if not isinstance(markets, list):
        markets = markets.get("markets", [])

    best: tuple[float, str, str, str, float, float] | None = None

    for i, market in enumerate(markets):
        raw_ids = market.get("clobTokenIds") or []
        if isinstance(raw_ids, str):
            try:
                import json as _j; raw_ids = _j.loads(raw_ids)
            except Exception:
                continue
        if not raw_ids:
            continue

        min_size = float(market.get("orderMinSize") or 5)
        slug = str(market.get("slug") or market.get("id") or "unknown")
        print(f"    scanning market {i+1}/{len(markets)}: {slug[:50]}", flush=True)

        for token_id in raw_ids[:2]:
            token_id = str(token_id)
            try:
                br = httpx.get(f"{_CLOB_HOST}/book", params={"token_id": token_id}, timeout=6)
                if br.status_code != 200:
                    continue
                bk = br.json()
                asks = bk.get("asks") or []
                if not asks:
                    continue
                best_ask = float(asks[0]["price"])
                if best_ask <= 0:
                    continue
                cost = min_size * best_ask
                tr = httpx.get(f"{_CLOB_HOST}/tick-size", params={"token_id": token_id}, timeout=4)
                tick = str(tr.json().get("minimum_tick_size", "0.01")) if tr.status_code == 200 else "0.01"
                print(f"      token ask={best_ask}  cost=${cost:.4f}", flush=True)

                if best is None or cost < best[0]:
                    best = (cost, token_id, tick, slug, min_size, best_ask)

                if cost <= early_exit_cost_usd:
                    cost, token_id, tick, slug, min_size, best_ask = best
                    return token_id, tick, slug, min_size, best_ask, cost
            except Exception:
                continue

    if best is None:
        return None
    cost, token_id, tick, slug, min_size, best_ask = best
    return token_id, tick, slug, min_size, best_ask, cost


# ---------------------------------------------------------------------------
# DB write helpers (mirrors what runner._poll_live_fills / _submit_* do)
# ---------------------------------------------------------------------------

def _write_position_opened(store: ResearchStore, intent: OrderIntent, report: ExecutionReport) -> None:
    store.save_position_event(
        position_id=intent.position_id or intent.intent_id,
        candidate_id=intent.candidate_id,
        event_type="position_opened",
        symbol=intent.token_id,
        market_slug=intent.market_slug,
        state="opened",
        reason_code=None,
        payload={
            "filled_size": report.filled_size,
            "avg_fill_price": report.avg_fill_price,
            "intent_id": report.intent_id,
            "live_order_id": report.metadata.get("live_order_id"),
            "source": "fill_validation_script",
        },
        ts=report.ts,
    )


def _write_account_snapshot(store: ResearchStore, ledger: Ledger) -> None:
    snapshot = ledger.snapshot()
    store.save_account_snapshot(snapshot)


# ---------------------------------------------------------------------------
# Main validation flow
# ---------------------------------------------------------------------------

def run_validation(
    db_url: str,
    dry_run: bool,
    pinned: tuple[str, str, str, float, float, float] | None = None,
) -> int:
    print(f"\n{_BOLD}Fill Validation — minimal real-fill acceptance test{_RESET}")
    print("=" * 60)

    # --- Step 1: Load credentials ---
    print(f"\n[1] Loading credentials")
    creds = load_live_credentials()
    _ok(f"credentials loaded  {creds!r}")

    sig_type_raw = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    sig_type     = int(sig_type_raw) if sig_type_raw else None
    funder       = os.environ.get("POLYMARKET_FUNDER", "").strip() or None
    _info(f"signature_type={sig_type}  funder={funder}")

    # --- Step 2: Find cheapest fillable token (or use pinned) ---
    if pinned is not None:
        print(f"\n[2] Using pinned token (skipping discovery)")
        result = pinned
    else:
        print(f"\n[2] Finding cheapest fillable token (early-exit at $0.10, max 15 markets)")
        result = _find_cheapest_fillable_token()
    if result is None:
        _fail("No fillable token found — aborting")
        return 1
    token_id, tick_size, market_slug, min_size, best_ask, cost_usd = result
    _ok(f"token  {token_id[:24]}...")
    _info(f"market_slug={market_slug[:50]}")
    _info(f"best_ask={best_ask}  min_size={min_size}  estimated_cost=${cost_usd:.6f}")

    if dry_run:
        _info("DRY-RUN: would submit BUY {min_size} @ {best_ask} — not submitting")
        return 0

    # --- Step 3: Build store and live client ---
    print(f"\n[3] Building ResearchStore + LiveBroker")
    store = ResearchStore(db_url)
    store.meta.create_all(store.engine)   # ensure tables exist
    _ok(f"store connected  {db_url}")

    clob_client = build_authenticated_client(
        creds, _CLOB_HOST, signature_type=sig_type, funder=funder
    )
    write_client = LiveWriteClient(clob_client, dry_run=False)
    broker = LiveBroker(write_client)
    ledger = Ledger()   # uses default cash=10000.0; debit not modelled here, positions only
    _ok("LiveBroker ready")

    # --- Step 4: Build and persist OrderIntent ---
    print(f"\n[4] Creating and persisting OrderIntent")
    ts_now    = datetime.now(timezone.utc)
    intent_id = str(uuid.uuid4())
    pos_id    = str(uuid.uuid4())
    cand_id   = "fill_validation"
    intent    = OrderIntent(
        intent_id=intent_id,
        candidate_id=cand_id,
        mode=OrderMode.LIVE,
        market_slug=market_slug,
        token_id=token_id,
        position_id=pos_id,
        side="BUY",
        order_type=OrderType.LIMIT,
        size=min_size,
        limit_price=best_ask,
        max_notional_usd=round(cost_usd * 1.1, 4),
        ts=ts_now,
    )
    store.save_order_intent(intent)
    _ok(f"intent saved  intent_id={intent_id[:16]}")

    # --- Step 5: Submit live order ---
    print(f"\n[5] Submitting live order  BUY {min_size} @ {best_ask}")
    report = broker.submit_limit_order(intent)
    _info(f"status={report.status.value}  filled={report.filled_size}  "
          f"live_id={report.metadata.get('live_order_id')}")

    store.save_execution_report(report)
    _ok("initial execution_report saved")

    if report.status == OrderStatus.REJECTED:
        _fail(f"Order rejected: {report.metadata.get('error')}")
        store.close()
        return 1

    # --- Step 6: Poll until terminal (handles resting GTC) ---
    print(f"\n[6] Polling for terminal state")
    live_order_id = report.metadata.get("live_order_id")
    terminal_statuses = frozenset({"matched", "canceled", "cancelled", "expired"})
    final_report = report
    POLL_ATTEMPTS = 12
    POLL_SLEEP_S  = 5

    if report.status in (OrderStatus.FILLED,):
        _ok("Order filled synchronously — no polling needed")
    elif live_order_id:
        for attempt in range(1, POLL_ATTEMPTS + 1):
            time.sleep(POLL_SLEEP_S)
            try:
                status = write_client.get_order_status(live_order_id)
                _info(f"poll {attempt}/{POLL_ATTEMPTS}  clob_status={status.status!r}  "
                      f"matched={status.size_matched}  remaining={status.size_remaining}")

                is_terminal  = status.status.lower() in terminal_statuses
                is_fully_filled = status.size_remaining <= 1e-9 and status.size_matched > 0

                if is_terminal or is_fully_filled:
                    order_status = (
                        OrderStatus.FILLED   if status.size_matched >= min_size - 1e-9
                        else OrderStatus.CANCELED if status.size_matched <= 1e-9
                        else OrderStatus.PARTIAL
                    )
                    final_report = ExecutionReport(
                        intent_id=intent_id,
                        position_id=pos_id,
                        status=order_status,
                        filled_size=round(status.size_matched, 6),
                        avg_fill_price=best_ask if status.size_matched > 1e-9 else None,
                        metadata={
                            "live_order_id": live_order_id,
                            "live_status": status.status,
                            "terminal": True,
                        },
                        ts=datetime.now(timezone.utc),
                    )
                    store.save_execution_report(final_report)
                    _ok(f"terminal report saved  status={order_status.value}")
                    break
            except Exception as exc:
                _info(f"poll {attempt} error: {exc}")
        else:
            # Still resting after all polls — cancel it
            _info("Order still resting after polling — cancelling")
            try:
                write_client.cancel_order(live_order_id)
                cancel_report = ExecutionReport(
                    intent_id=intent_id,
                    position_id=pos_id,
                    status=OrderStatus.CANCELED,
                    filled_size=0.0,
                    metadata={"live_order_id": live_order_id, "live_status": "cancelled", "terminal": True},
                    ts=datetime.now(timezone.utc),
                )
                store.save_execution_report(cancel_report)
                final_report = cancel_report
                _ok("cancel sent and terminal report saved")
            except Exception as exc:
                _fail(f"cancel failed: {exc}")

    # --- Step 7: Save position_opened if any shares filled ---
    print(f"\n[7] Recording position event")
    if final_report.filled_size > 1e-9:
        # Mirror into ledger so snapshot is consistent
        ledger.place_limit_order(
            order_id=intent_id,
            symbol=token_id,
            market_slug=market_slug,
            side="BUY",
            shares=min_size,
            limit_price=best_ask,
            ts=ts_now.isoformat(),
            candidate_id=cand_id,
            position_id=pos_id,
        )
        ledger.apply_fill(order_id=intent_id, shares=final_report.filled_size, price=best_ask)
        _write_position_opened(store, intent, final_report)
        _ok(f"position_opened event saved  filled={final_report.filled_size}")
    else:
        _info("No shares filled — position_opened not written (correct)")

    _write_account_snapshot(store, ledger)
    _ok("account_snapshot saved")
    store.close()

    # --- Step 8: Run reconcile_audit ---
    print(f"\n{'=' * 60}")
    print(f"{_BOLD}[8] Running reconcile_audit.py{_RESET}")
    print("=" * 60)
    audit_result = subprocess.run(
        [sys.executable, "scripts/reconcile_audit.py", "--db", db_url],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return audit_result.returncode


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fill validation + reconcile audit")
    parser.add_argument("--db",          default="sqlite:///data/processed/paper.db")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Discover cheapest token and print plan; no order or DB write")
    parser.add_argument("--token",       default=None,
                        help="Skip discovery: use this token_id directly")
    parser.add_argument("--ask",         type=float, default=None,
                        help="Best ask for --token (required when --token is set)")
    parser.add_argument("--market-slug", default="manual",
                        help="Market slug label for --token (default: 'manual')")
    parser.add_argument("--min-size",    type=float, default=5.0,
                        help="Minimum order size for --token (default: 5)")
    args = parser.parse_args()

    # If --token supplied, bypass discovery entirely.
    pinned: tuple[str, str, str, float, float, float] | None = None
    if args.token:
        if args.ask is None:
            print("ERROR: --ask is required when --token is set.")
            sys.exit(2)
        tick = "0.01"   # conservative default; CLOB will reject if wrong
        cost = args.min_size * args.ask
        pinned = (args.token, tick, args.market_slug, args.min_size, args.ask, cost)

    sys.exit(run_validation(args.db, args.dry_run, pinned=pinned))


if __name__ == "__main__":
    main()
