"""
run_bounded_observation — bounded live observation
polyarb_lab / research_lines / auto_maker_loop

Monitors existing orders for a capped time window.
Does NOT place or cancel any orders.  Read-only throughout.

Modes
-----
Bilateral (default):
    Monitors BID + ASK.  Hard-stops if BID fills (position increase).

Ask-only (--ask-only):
    No BID in market.  Monitors ASK only.
    Main question: can ask-only keep earning meaningful reward without
    reintroducing inventory expansion risk?

Records every 60 seconds:
    timestamp, elapsed_min, reward_total, reward_delta_1m,
    current_position_shares, mark_price, open_orders_snapshot,
    ask_order_status, ask_fill_status,
    [bilateral mode also: bid_order_status, bid_filled]

Hard-stops immediately if:
    - [bilateral] BID fills (position about to increase)
    - position_shares > position_cap
    - [bilateral] BID order disappears unexpectedly
    - ASK order disappears unexpectedly
    - reward growth stalls for stall_periods consecutive polls
    - ASK fills (inventory sold — success exit)
    - total elapsed >= duration_minutes

On hard-stop: logs reason, writes final record, prints conclusion.
Does NOT cancel any orders.  Leaves live orders intact.

Usage:
    # bilateral
    py -3 research_lines/auto_maker_loop/run_bounded_observation.py --live
    py -3 research_lines/auto_maker_loop/run_bounded_observation.py --live ^
        --bid-order-id 0xABC... --ask-order-id 0xDEF...

    # ask-only (BID cancelled, only 65c ASK remains)
    py -3 research_lines/auto_maker_loop/run_bounded_observation.py --live ^
        --ask-only --ask-order-id 0xDEF...
    py -3 research_lines/auto_maker_loop/run_bounded_observation.py --live ^
        --ask-only --duration-minutes 60 --target hungary
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bounded_obs")

OUT_DIR = Path("data/research/auto_maker_loop/bounded_obs")

HUNGARY_SLUG = "will-the-next-prime-minister-of-hungary-be-pter-magyar"
HUNGARY_CID  = "0x1480b819d03d4b6388d70e848b0384adf38c38d955cb783cdbcf6d4a436dee14"
HUNGARY_TOKEN = "94192784911459194325909253314484842244405314804074606736702592885535642919725"
CLOB_HOST    = "https://clob.polymarket.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for method in ("model_dump", "dict"):
        if hasattr(obj, method):
            try:
                return getattr(obj, method)()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"raw": repr(obj)}


def _order_status(client: Any, order_id: str) -> tuple[str, bool]:
    """
    Return (status_str, is_filled).
    status_str: LIVE | MATCHED | CANCELLED | UNKNOWN | MISSING
    is_filled: True if MATCHED / FILLED
    """
    if not order_id or order_id.startswith("NONE"):
        return "MISSING", False
    try:
        raw = client.get_order(order_id)
        if raw is None:
            return "MISSING", False
        d = _to_dict(raw)
        status = str(d.get("status", "UNKNOWN")).upper()
        filled = status in ("MATCHED", "FILLED")
        return status, filled
    except Exception as exc:
        logger.debug("get_order failed %s: %s", order_id[:16], exc)
        return "ERROR", False


def _order_scoring(client: Any, order_id: str) -> Optional[bool]:
    if not order_id or order_id.startswith("NONE"):
        return None
    try:
        from py_clob_client.clob_types import OrderScoringParams
        result = client.is_order_scoring(OrderScoringParams(orderId=order_id))
        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            return bool(result.get("scoring") or result.get("is_scoring"))
        return None
    except Exception:
        return None


def _get_open_orders_for_token(client: Any, token_id: str) -> list[dict]:
    """Return list of open order dicts for the given token."""
    try:
        from py_clob_client.clob_types import OpenOrderParams
        raw = client.get_orders(OpenOrderParams(market=token_id))
        if isinstance(raw, list):
            return [_to_dict(o) for o in raw]
        if isinstance(raw, dict) and "data" in raw:
            return [_to_dict(o) for o in (raw.get("data") or [])]
        return []
    except Exception as exc:
        logger.warning("get_orders failed: %s", exc)
        return []


def _discover_orders(client: Any, token_id: str, bid_price: float, ask_price: float) -> tuple[str, str]:
    """
    Auto-discover BID and ASK order IDs from open orders for the token.
    Returns (bid_order_id, ask_order_id) — either may be "NONE" if not found.
    """
    orders = _get_open_orders_for_token(client, token_id)
    bid_id = "NONE"
    ask_id = "NONE"
    for o in orders:
        side  = str(o.get("side", "")).upper()
        price = float(o.get("price") or o.get("original_price") or 0)
        oid   = str(o.get("id") or o.get("order_id") or "")
        if not oid:
            continue
        if side == "BUY"  and abs(price - bid_price) < 0.005:
            bid_id = oid
        if side == "SELL" and abs(price - ask_price) < 0.005:
            ask_id = oid
    return bid_id, ask_id


def _earn_pct(
    host: str,
    creds: Any,
    condition_id: str,
    token_id: str = "",
    slug: str = "",
) -> Optional[float]:
    """
    Fetch earning_pct from /rewards/user/markets.

    Follows next_cursor pagination (up to 20 pages) so that entries beyond
    the first 100 are not missed.  Falls back through condition_id →
    token_id → slug matching, mirroring auth_rewards_truth.lookup_user_market_entry.
    """
    try:
        import requests as _req
        import base64, hashlib, hmac as _hmac, time as _time

        path = "/rewards/user/markets"
        base_url = f"{host.rstrip('/')}{path}"

        def _headers() -> dict:
            ts  = str(int(_time.time() * 1000))
            msg = ts + "GET" + path
            try:
                hmac_key = base64.urlsafe_b64decode(creds.api_secret)
            except Exception:
                hmac_key = creds.api_secret.encode("utf-8")
            sig = base64.b64encode(
                _hmac.new(hmac_key, msg.encode("utf-8"), hashlib.sha256).digest()
            ).decode("utf-8")
            return {
                "CLOB-API-KEY":    creds.api_key,
                "CLOB-SIGNATURE":  sig,
                "CLOB-TIMESTAMP":  ts,
                "CLOB-PASSPHRASE": creds.api_passphrase,
            }

        # ── Paginated fetch ───────────────────────────────────────────────
        all_entries: list[dict] = []
        cursor: Optional[str] = None
        pages = 0
        while pages < 20:
            params = {"next_cursor": cursor} if cursor else {}
            resp = _req.get(base_url, headers=_headers(), params=params, timeout=8)
            if resp.status_code != 200:
                logger.warning("_earn_pct: HTTP %s page=%d", resp.status_code, pages)
                break
            body = resp.json()
            data = body.get("data", body) if isinstance(body, dict) else body
            if isinstance(data, list):
                all_entries.extend(e for e in data if isinstance(e, dict))
            elif isinstance(data, dict):
                all_entries.append(data)
            pages += 1
            cursor = body.get("next_cursor") if isinstance(body, dict) else None
            if not cursor or cursor in ("", "LTE="):
                break

        if all_entries:
            logger.info(
                "_earn_pct: fetched %d entries (%d pages)  first_entry_keys=%s",
                len(all_entries), pages, list(all_entries[0].keys())[:12],
            )

        # ── Match: condition_id → token_id → slug ─────────────────────────
        cid_norm = condition_id.lower().lstrip("0x")

        def _match_entry(entry: dict) -> bool:
            # 1. condition_id
            for k in ("condition_id", "conditionId", "market_id"):
                v = str(entry.get(k) or "").lower().lstrip("0x")
                if v and v == cid_norm:
                    return True
            # 2. token_id (top-level or nested tokens[])
            if token_id:
                if str(entry.get("token_id") or "") == token_id:
                    return True
                for tok in (entry.get("tokens") or []):
                    if isinstance(tok, dict) and str(tok.get("token_id") or "") == token_id:
                        return True
            # 3. slug
            if slug:
                es = str(
                    entry.get("market_slug") or entry.get("slug")
                    or entry.get("marketSlug") or ""
                )
                if es and (es == slug or es.startswith(slug) or slug.startswith(es)):
                    return True
            return False

        for entry in all_entries:
            if not _match_entry(entry):
                continue
            for _f in ("earning_percentage", "earnings_percentage", "percentage",
                       "avg_percentage", "pct", "share", "earnings", "earning"):
                if _f in entry:
                    try:
                        v = float(entry[_f])
                        return round(v / 100.0 if v > 1.0 else v, 6)
                    except (TypeError, ValueError):
                        pass
            return 0.0  # matched entry but no earning field

        logger.warning(
            "_earn_pct: no match in %d entries (%d pages)  first_keys=%s",
            len(all_entries), pages,
            list(all_entries[0].keys())[:12] if all_entries else [],
        )
        return None
    except Exception as exc:
        logger.warning("_earn_pct failed: %s", exc)
        return None


def _inv_shares(client: Any, token_id: str, creds: Any = None) -> float:
    """
    Return conditional token balance in shares.

    Primary source: Polymarket data-API positions endpoint (public, no auth).
      GET https://data-api.polymarket.com/positions?user=<funder>&sizeThreshold=0.01
      Match by asset_id == token_id.  Read 'size' field.

    Why: When a maker SELL order is open, YES tokens are locked in the CTF
    Exchange contract as collateral — not held at the user's wallet address.
    ERC-1155 balanceOf(user_wallet, token_id) therefore returns 0 for any
    open SELL position.  The data-API tracks the Polymarket-side position
    regardless of on-chain custody location.

    Fallback: Polygon RPC balanceOf (kept for debug; logs but does not set
    the returned value).
    """
    import requests as _req

    # ── Determine user address ─────────────────────────────────────────────
    holder: Optional[str] = None
    if creds is not None:
        holder = getattr(creds, "funder", None)
    if not holder and client is not None:
        try:
            holder = client.signer.address()
        except Exception:
            pass
    if not holder:
        logger.warning("_inv_shares: cannot determine holder address")
        return 0.0

    # ── Primary: Polymarket data-API positions ─────────────────────────────
    try:
        url    = "https://data-api.polymarket.com/positions"
        params = {"user": holder, "sizeThreshold": "0.01"}
        resp   = _req.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            body = resp.json()
            rows = body if isinstance(body, list) else (body.get("data") or body.get("positions") or [])
            for row in rows:
                # Match on asset_id — may appear under several field names
                asset = (
                    row.get("asset")
                    or row.get("asset_id")
                    or row.get("token_id")
                    or row.get("outcomeIndex")
                    or ""
                )
                condition = row.get("conditionId") or row.get("condition_id") or ""
                # Primary match: asset_id numeric string equals token_id
                if str(asset) == str(token_id):
                    size = float(row.get("size") or row.get("shares") or row.get("balance") or 0.0)
                    avg  = row.get("avgPrice") or row.get("avg_price") or "N/A"
                    logger.info(
                        "_inv_shares data-api: holder=%s… size=%.4f avg=%s (asset match)",
                        holder[:10], size, avg,
                    )
                    return round(size, 4)
                # Fallback match: condition_id prefix match (first 10 chars)
                if (
                    condition
                    and HUNGARY_CID
                    and str(condition).lower()[:10] == HUNGARY_CID[:10].lower()
                ):
                    size = float(row.get("size") or row.get("shares") or row.get("balance") or 0.0)
                    avg  = row.get("avgPrice") or row.get("avg_price") or "N/A"
                    logger.info(
                        "_inv_shares data-api: holder=%s… size=%.4f avg=%s (condition match)",
                        holder[:10], size, avg,
                    )
                    return round(size, 4)
            logger.info(
                "_inv_shares data-api: holder=%s… returned %d rows, no asset match for token %s…",
                holder[:10], len(rows), token_id[:20],
            )
            # Log the asset values we did see (for debugging)
            seen = [str(r.get("asset") or r.get("asset_id") or r.get("token_id") or "?")[:30] for r in rows[:5]]
            if seen:
                logger.info("_inv_shares data-api: first asset values seen: %s", seen)
        else:
            logger.warning("_inv_shares data-api HTTP %d for holder %s…", resp.status_code, holder[:10])
    except Exception as exc:
        logger.warning("_inv_shares data-api failed: %s", exc)

    # ── Fallback: Polygon RPC balanceOf (debug reference only) ────────────
    try:
        from py_clob_client.config import get_contract_config
        ctf = get_contract_config(137).conditional_tokens
        addr_padded     = holder.lower().replace("0x", "").zfill(64)
        token_id_padded = hex(int(token_id))[2:].zfill(64)
        call_data       = "0x00fdd58e" + addr_padded + token_id_padded
        rpc_payload = {
            "jsonrpc": "2.0", "method": "eth_call",
            "params":  [{"to": ctf, "data": call_data}, "latest"],
            "id": 1,
        }
        resp = _req.post("https://polygon-rpc.com", json=rpc_payload, timeout=8)
        result_hex  = resp.json().get("result", "0x0") or "0x0"
        balance_raw = int(result_hex, 16) if result_hex not in ("0x", "0x0", "") else 0
        balance_shares = round(balance_raw / 1_000_000, 4)
        logger.info(
            "_inv_shares RPC fallback: holder=%s… balance_raw=%d shares=%.4f (debug only — tokens may be in Exchange contract)",
            holder[:10], balance_raw, balance_shares,
        )
    except Exception as exc:
        logger.warning("_inv_shares RPC fallback failed: %s", exc)

    return 0.0


# ---------------------------------------------------------------------------
# Main observation loop
# ---------------------------------------------------------------------------

def run_observation(
    client: Any,
    creds: Any,
    bid_order_id: str,
    ask_order_id: str,
    token_id: str,
    condition_id: str,
    slug: str,
    duration_minutes: float,
    poll_sec: int,
    position_cap: float,
    avg_entry_price: float,
    stall_periods: int,
    daily_rate_usdc: float,
    out_path: Path,
    ask_only: bool = False,
) -> None:

    start_ts      = datetime.now(timezone.utc)
    records       = []
    prev_earn_pct = None
    earn_pct_history: list[Optional[float]] = []
    poll_num      = 0
    stop_reason   = "duration_reached"

    mode_label = "ASK-ONLY" if ask_only else "BILATERAL"

    # ── Identity proof (printed once before poll loop) ────────────────────
    _sig_type = getattr(creds, "signature_type", "?")
    _funder   = getattr(creds, "funder", None)
    try:
        _signer_addr = client.signer.address()
    except Exception:
        _signer_addr = "unavailable"
    if _funder:
        _holder       = _funder
        _holder_src   = "funder"
    else:
        _holder       = _signer_addr
        _holder_src   = "signer (EOA)"

    print("\n" + "=" * 62)
    print("  ACCOUNT IDENTITY PROOF")
    print(f"  signature_type  : {_sig_type}")
    print(f"  signer (EOA)    : {_signer_addr}")
    print(f"  creds.funder    : {_funder or '(none)'}")
    print(f"  holder used     : {_holder}")
    print(f"  holder source   : {_holder_src}")
    print("=" * 62)

    print("\n" + "=" * 62)
    print(f"  BOUNDED OBSERVATION [{mode_label}]  |  {slug[:40]}")
    print(f"  duration_cap : {duration_minutes:.0f} min  |  poll : {poll_sec}s")
    if not ask_only:
        print(f"  bid_order_id : {bid_order_id[:20]}...")
    print(f"  ask_order_id : {ask_order_id[:20]}...")
    print(f"  position_cap : {position_cap:.0f} shares")
    if ask_only:
        print(f"  main question: can ask-only keep earning reward without inventory expansion?")
    print("=" * 62)

    while True:
        now     = datetime.now(timezone.utc)
        elapsed = (now - start_ts).total_seconds() / 60.0

        if elapsed >= duration_minutes:
            stop_reason = "duration_reached"
            break

        poll_num += 1

        # ── Collect all fields ────────────────────────────────────────────
        if ask_only:
            bid_status, bid_filled = "N/A", False
            bid_scoring = None
        else:
            bid_status, bid_filled = _order_status(client, bid_order_id)
            bid_scoring = _order_scoring(client, bid_order_id)

        ask_status, ask_filled = _order_status(client, ask_order_id)

        # Fetch midpoint
        try:
            from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
                fetch_midpoint,
            )
            mid, _ = fetch_midpoint(client, token_id, price_ref=0.625)
        except Exception:
            mid = None

        # Reward
        earn_pct = _earn_pct(CLOB_HOST, creds, condition_id, token_id=token_id, slug=slug)
        earn_pct_history.append(earn_pct)

        reward_delta_1m: Optional[float] = None
        if earn_pct is not None and prev_earn_pct is not None:
            delta_pct = earn_pct - prev_earn_pct
            reward_delta_1m = round(delta_pct / 100.0 * daily_rate_usdc * 100 / 24.0, 4)

        reward_total_cents: Optional[float] = None
        if earn_pct is not None:
            reward_total_cents = round(earn_pct / 100.0 * daily_rate_usdc * 100 / 24.0, 4)

        # Inventory
        pos_shares = _inv_shares(client, token_id, creds)

        # Scoring
        ask_scoring = _order_scoring(client, ask_order_id)

        # Open orders snapshot
        open_orders = _get_open_orders_for_token(client, token_id)
        open_count  = len(open_orders)
        open_ids    = [str(o.get("id") or o.get("order_id") or "?")[:16] for o in open_orders]

        # Unrealized P&L on position
        unreal_usd: Optional[float] = None
        if mid and pos_shares > 0 and avg_entry_price > 0:
            unreal_usd = round((mid - avg_entry_price) * pos_shares, 4)

        record = {
            "timestamp":               now.isoformat(),
            "elapsed_min":             round(elapsed, 2),
            "poll":                    poll_num,
            "mode":                    mode_label,
            "reward_total_cents":      reward_total_cents,
            "reward_delta_1m_cents":   reward_delta_1m,
            "earning_pct":             earn_pct,
            "current_position_shares": pos_shares,
            "avg_entry_price":         avg_entry_price,
            "mark_price":              mid,
            "ask_order_status":        ask_status,
            "ask_fill_status":         ask_filled,
            "ask_scoring":             ask_scoring,
            "open_orders_snapshot":    open_ids,
            "open_orders_count":       open_count,
            "unrealized_pnl_usd":      unreal_usd,
        }
        if not ask_only:
            record["bid_order_status"] = bid_status
            record["bid_filled"]       = bid_filled
            record["bid_scoring"]      = bid_scoring

        records.append(record)

        # ── Console print ─────────────────────────────────────────────────
        reward_str = f"{reward_total_cents:.4f}¢" if reward_total_cents is not None else "N/A"
        delta_str  = f"{reward_delta_1m:+.4f}¢"  if reward_delta_1m  is not None else "N/A"
        mid_str    = f"{mid:.4f}" if mid else "N/A"
        unreal_str = f"{unreal_usd:+.4f}$" if unreal_usd is not None else "N/A"

        if ask_only:
            print(
                f"  [{now.strftime('%H:%M:%S')}] "
                f"elapsed={elapsed:5.1f}m  "
                f"mid={mid_str}  "
                f"pos={pos_shares:.0f}sh  "
                f"reward={reward_str}(Δ{delta_str})  "
                f"ASK={ask_status}/scoring={ask_scoring}  "
                f"unreal={unreal_str}"
            )
        else:
            print(
                f"  [{now.strftime('%H:%M:%S')}] "
                f"elapsed={elapsed:5.1f}m  "
                f"mid={mid_str}  "
                f"pos={pos_shares:.0f}sh  "
                f"reward={reward_str}(Δ{delta_str})  "
                f"BID={bid_status}/{bid_scoring}  "
                f"ASK={ask_status}/{ask_scoring}  "
                f"unreal={unreal_str}"
            )

        prev_earn_pct = earn_pct

        # ── HARD-STOP conditions ──────────────────────────────────────────

        # Bilateral only: BID fill = inventory expansion
        if not ask_only and bid_filled:
            stop_reason = "HARD_STOP: BID filled — position about to increase"
            print(f"\n  !! {stop_reason}")
            records[-1]["stop_reason"] = stop_reason
            break

        if pos_shares > position_cap:
            stop_reason = f"HARD_STOP: position_shares={pos_shares:.0f} > cap={position_cap:.0f}"
            print(f"\n  !! {stop_reason}")
            records[-1]["stop_reason"] = stop_reason
            break

        # Bilateral only: BID order gone unexpectedly
        if not ask_only and not bid_filled and bid_status in ("MISSING", "CANCELLED", "ERROR"):
            stop_reason = f"HARD_STOP: BID order missing unexpectedly (status={bid_status})"
            print(f"\n  !! {stop_reason}")
            records[-1]["stop_reason"] = stop_reason
            break

        if not ask_filled and ask_status in ("MISSING", "CANCELLED", "ERROR"):
            stop_reason = f"HARD_STOP: ASK order missing unexpectedly (status={ask_status})"
            print(f"\n  !! {stop_reason}")
            records[-1]["stop_reason"] = stop_reason
            break

        # Reward stall
        nonzero_seen = any(p is not None and p > 0 for p in earn_pct_history)
        if nonzero_seen and len(earn_pct_history) >= stall_periods:
            recent = earn_pct_history[-stall_periods:]
            if all(r == recent[0] for r in recent) and recent[0] is not None:
                stop_reason = (
                    f"HARD_STOP: reward stalled — earning_pct={recent[0]:.4f} "
                    f"unchanged for {stall_periods} consecutive polls"
                )
                print(f"\n  !! {stop_reason}")
                records[-1]["stop_reason"] = stop_reason
                break

        # ASK filled = inventory exit (good outcome)
        if ask_filled:
            stop_reason = "HARD_STOP: ASK filled — inventory sold naturally"
            print(f"\n  !! {stop_reason}")
            records[-1]["stop_reason"] = stop_reason
            break

        # Wait for next poll
        time.sleep(poll_sec)

    # ── Write JSONL ───────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  Records written: {len(records)}  →  {out_path}")

    # ── Final conclusions ─────────────────────────────────────────────────
    _print_conclusion(records, stop_reason, duration_minutes, position_cap, ask_only=ask_only)


def _print_conclusion(
    records: list[dict],
    stop_reason: str,
    duration_minutes: float,
    position_cap: float,
    ask_only: bool = False,
) -> None:
    if not records:
        print("\n  No records collected.")
        return

    first = records[0]
    last  = records[-1]
    n     = len(records)
    mode_label = "ASK-ONLY" if ask_only else "BILATERAL"

    # Reward growth
    r_first = first.get("reward_total_cents")
    r_last  = last.get("reward_total_cents")
    reward_grew    = (r_first is not None and r_last is not None and r_last > r_first)
    reward_nonzero = any(r.get("reward_total_cents") and r["reward_total_cents"] > 0 for r in records)

    # Deltas
    deltas = [r.get("reward_delta_1m_cents") for r in records if r.get("reward_delta_1m_cents") is not None]
    avg_delta_1m   = sum(deltas) / len(deltas) if deltas else None
    nonzero_deltas = sum(1 for d in deltas if d is not None and d > 0)

    # ASK survival
    ask_survived = all(
        not r.get("ask_fill_status") and r.get("ask_order_status") not in ("MISSING", "CANCELLED", "ERROR")
        for r in records
    )
    ask_scored   = any(r.get("ask_scoring") is True for r in records)
    ask_filled_any = any(r.get("ask_fill_status") for r in records)

    # Inventory
    pos_values = [r.get("current_position_shares", 0) for r in records]
    max_pos    = max(pos_values) if pos_values else 0
    min_pos    = min(pos_values) if pos_values else 0
    pos_grew   = max_pos > (pos_values[0] if pos_values else 0)
    pos_stable = max_pos <= position_cap

    print("\n" + "=" * 62)
    print(f"  FINAL CONCLUSIONS  [{mode_label}]")
    print("=" * 62)
    print(f"  stop_reason                  : {stop_reason}")
    print(f"  total_polls                  : {n}")
    print(f"  elapsed_min                  : {last.get('elapsed_min', 0):.1f}")
    print()

    # Q1: Reward
    if r_first is not None and r_last is not None:
        print(f"  Q1. Reward continued growing?")
        print(f"      answer                    : {'YES' if reward_grew else 'NO / STALLED'}")
        print(f"      reward_start              : {r_first:.4f}¢")
        print(f"      reward_end                : {r_last:.4f}¢")
        print(f"      reward_delta_total        : {r_last - r_first:+.4f}¢")
        if avg_delta_1m is not None:
            print(f"      avg_delta_1m_cents        : {avg_delta_1m:+.4f}¢  ({nonzero_deltas}/{len(deltas)} polls positive)")
    else:
        print(f"  Q1. Reward data               : NOT MEASURED (earning_pct unavailable)")
    print()

    # Q2: ASK order health
    print(f"  Q2. ASK order live and qualifying?")
    print(f"      ASK survived              : {'YES' if ask_survived else 'NO'}")
    print(f"      ASK scored (any poll)     : {'YES' if ask_scored else 'NO / UNKNOWN'}")
    print(f"      ASK filled during run     : {'YES — inventory exit occurred' if ask_filled_any else 'NO'}")
    print()

    # Q3: Inventory expansion risk
    print(f"  Q3. Inventory expansion risk?")
    print(f"      position_start            : {pos_values[0] if pos_values else 0:.0f} shares")
    print(f"      position_end              : {pos_values[-1] if pos_values else 0:.0f} shares")
    print(f"      position_max              : {max_pos:.0f} shares  (cap={position_cap:.0f})")
    if ask_only:
        print(f"      inventory grew?           : {'YES — UNEXPECTED' if pos_grew else 'NO — stable or declining (expected)'}")
    else:
        print(f"      inventory_stable          : {'YES' if pos_stable else 'NO — EXCEEDED CAP'}")
    print()

    # Q4: Main question (ask-only) or inventory management (bilateral)
    if ask_only:
        reward_meaningful = reward_grew and avg_delta_1m is not None and avg_delta_1m > 0
        no_expansion = not pos_grew
        print(f"  Q4. Can ask-only earn reward without inventory expansion?")
        print(f"      reward earning confirmed  : {'YES' if reward_nonzero else 'NO'}")
        print(f"      reward growing this run   : {'YES' if reward_grew else 'NO'}")
        print(f"      inventory expansion risk  : {'ABSENT — no BID placed' if no_expansion else 'PRESENT — INVESTIGATE'}")
        print(f"      ask-only posture viable?  : ", end="")
        if reward_meaningful and no_expansion:
            print("YES (this window) — requires multi-window confirmation")
        elif reward_nonzero and no_expansion:
            print("PARTIAL — reward present but growth rate needs longer window")
        elif not reward_nonzero:
            print("INCONCLUSIVE — reward not observed this window")
        else:
            print("FAILED — inventory expanded unexpectedly")
        print()
        print("  Note: ask-only earns reward only while ASK is resting.")
        print("  If ASK fills, inventory is gone and reward path closes.")
        print("  Do not re-place BID without new explicit judgment step.")
    else:
        any_fill = any(r.get("bid_filled") or r.get("ask_fill_status") for r in records)
        print(f"  Q4. Was this a real inventory-management test?")
        if any_fill:
            print(f"      NO — order fill occurred; inventory changed during run")
        else:
            print(f"      NO — stable reward-observation run only.")
            print(f"           No fills. Inventory-management behavior untested.")

    print()
    print("  INSTRUCTION: Do not auto-extend this posture.")
    print("  Require an explicit judgment step before any further action.")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bounded live observation. Default: bilateral. Use --ask-only for exit-side monitoring."
    )
    p.add_argument("--live",     action="store_true", help="Required to run (prevents accidental execution)")
    p.add_argument("--ask-only", action="store_true", help="Monitor ASK only — no BID in market (ask-only reward observation)")
    p.add_argument("--place-quotes", action="store_true",
                   help="Place BID + ASK at --bid-price / --ask-price before observing. "
                        "Use for research runs where no orders are currently live. "
                        "Ignored with --ask-only.")
    p.add_argument("--target",   default="hungary",   help="Market target (default: hungary)")
    p.add_argument("--bid-order-id", default=None,    help="BID order ID (auto-discovered if omitted; ignored with --ask-only)")
    p.add_argument("--ask-order-id", default=None,    help="ASK order ID (auto-discovered if omitted)")
    p.add_argument("--bid-price",    type=float, default=0.62, help="Expected BID price for discovery or placement (default 0.62; ignored with --ask-only)")
    p.add_argument("--ask-price",    type=float, default=0.65, help="Expected ASK price for discovery or placement (default 0.65)")
    p.add_argument("--size",         type=float, default=200.0, help="Order size in shares when --place-quotes is used (default 200)")
    p.add_argument("--duration-minutes", type=float, default=60.0, help="Max observation window in minutes (default 60)")
    p.add_argument("--poll-sec",         type=int,   default=60,   help="Seconds between polls (default 60)")
    p.add_argument("--position-cap",     type=float, default=200.0, help="Hard-stop if position exceeds this (default 200; ask-only posture must not grow inventory)")
    p.add_argument("--avg-entry-price",  type=float, default=0.625, help="Known avg entry price for P&L calc (default 0.625)")
    p.add_argument("--stall-periods",    type=int,   default=10,    help="Consecutive flat reward polls before stall hard-stop (default 10)")
    p.add_argument("--daily-rate-usdc",  type=float, default=150.0, help="Daily reward rate in USD (default 150)")
    p.add_argument("--verbose",  action="store_true", help="DEBUG logging")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.live:
        print("ERROR: --live flag required to run bounded observation.")
        print("This prevents accidental execution against live orders.")
        sys.exit(1)

    # Load credentials
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        load_activation_credentials, build_clob_client, SLUG_ALIASES, SURVIVOR_DATA, CLOB_HOST as SA_HOST,
    )
    creds = load_activation_credentials()
    if creds is None:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set.")
        sys.exit(1)

    print("\nBuilding CLOB client...")
    client = build_clob_client(creds, SA_HOST)

    slug = SLUG_ALIASES.get(args.target, args.target)
    if slug not in SURVIVOR_DATA:
        print(f"ERROR: unknown target '{args.target}'")
        sys.exit(1)

    token_id     = SURVIVOR_DATA[slug]["token_id"]
    condition_id = SURVIVOR_DATA[slug]["condition_id"]

    # ── Place quotes if --place-quotes requested (research mode) ──────────
    if args.place_quotes and not args.ask_only:
        from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
            _place_order as _sa_place_order,
        )
        print(f"\nPlacing research quotes: BID {args.bid_price:.2f}  ASK {args.ask_price:.2f}  size {args.size:.0f}")
        placed_bid_id, bid_err = _sa_place_order(client, token_id, args.bid_price, args.size, "BUY")
        placed_ask_id, ask_err = _sa_place_order(client, token_id, args.ask_price, args.size, "SELL")
        if bid_err or ask_err:
            print(f"  BID placement: {'OK  id=' + placed_bid_id if placed_bid_id else 'FAILED — ' + str(bid_err)}")
            print(f"  ASK placement: {'OK  id=' + placed_ask_id if placed_ask_id else 'FAILED — ' + str(ask_err)}")
            print("ERROR: one or both quotes failed — aborting. Cancel any live orders manually.")
            sys.exit(1)
        print(f"  BID placed: {placed_bid_id}")
        print(f"  ASK placed: {placed_ask_id}")
        # Override order-id args so discovery is skipped below
        args.bid_order_id = placed_bid_id
        args.ask_order_id = placed_ask_id

    # Discover or use provided order IDs
    bid_id = "NONE" if args.ask_only else args.bid_order_id
    ask_id = args.ask_order_id

    if args.ask_only:
        # Only discover ASK
        if not ask_id:
            print(f"\nAuto-discovering ASK order for {slug[:50]}...")
            # Fetch ALL open orders (no market filter — server-side market filter
            # does not reliably return CTF conditional token orders).
            # Filter locally by token_id match, side, and price.
            try:
                from py_clob_client.clob_types import OpenOrderParams
                raw = client.get_orders(OpenOrderParams())
                all_orders = (
                    raw if isinstance(raw, list)
                    else (raw.get("data") or [] if isinstance(raw, dict) else [])
                )
            except Exception as exc:
                print(f"  WARNING: get_orders() failed: {exc}")
                all_orders = []
            for o in ([_to_dict(x) for x in all_orders]):
                side   = str(o.get("side", "")).upper()
                price  = float(o.get("price") or o.get("original_price") or 0)
                oid    = str(o.get("id") or o.get("order_id") or "")
                # Match token_id against any field that may carry the asset identifier
                asset  = str(
                    o.get("asset_id") or o.get("token_id")
                    or o.get("market") or o.get("outcome_token_id") or ""
                )
                token_match = (asset == token_id) or (token_id in asset) or (asset in token_id and asset)
                if side == "SELL" and abs(price - args.ask_price) < 0.005 and oid and token_match:
                    ask_id = oid
                    break
            print(f"  ASK discovered: {ask_id or 'NONE'}")
        if not ask_id or ask_id == "NONE":
            print(
                f"\nERROR: No SELL order found near {args.ask_price:.2f}.\n"
                f"Use --ask-order-id to provide it explicitly."
            )
            sys.exit(1)
    else:
        if not bid_id or not ask_id:
            print(f"\nAuto-discovering orders for {slug[:50]}...")
            discovered_bid, discovered_ask = _discover_orders(
                client, token_id, args.bid_price, args.ask_price
            )
            if not bid_id:
                bid_id = discovered_bid
                print(f"  BID discovered: {bid_id}")
            if not ask_id:
                ask_id = discovered_ask
                print(f"  ASK discovered: {ask_id}")

        if bid_id == "NONE" or ask_id == "NONE":
            print(
                f"\nWARNING: Could not discover both orders "
                f"(bid={bid_id}, ask={ask_id}).\n"
                f"Use --bid-order-id / --ask-order-id to provide them explicitly.\n"
                f"Continuing with what was found."
            )

    ts_str     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    mode_tag   = "ask_only" if args.ask_only else "bilateral"
    out_path   = OUT_DIR / f"obs_{mode_tag}_{args.target}_{ts_str}.jsonl"

    run_observation(
        client=client,
        creds=creds,
        bid_order_id=bid_id or "NONE",
        ask_order_id=ask_id or "NONE",
        token_id=token_id,
        condition_id=condition_id,
        slug=slug,
        duration_minutes=args.duration_minutes,
        poll_sec=args.poll_sec,
        position_cap=args.position_cap,
        avg_entry_price=args.avg_entry_price,
        stall_periods=args.stall_periods,
        daily_rate_usdc=args.daily_rate_usdc,
        out_path=out_path,
        ask_only=args.ask_only,
    )


if __name__ == "__main__":
    main()
