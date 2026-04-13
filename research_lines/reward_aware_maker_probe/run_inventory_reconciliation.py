"""
reward_aware_inventory_reconciliation — read-only
polyarb_lab / research_line / diagnostic

Proves exactly where the 200 Hungary YES tokens went between:
  - SCORING_ACTIVE run end  : 2026-03-26T09:59:57 UTC
  - PREFLIGHT_FAILED start  : 2026-03-26T11:26:38 UTC

Evidence table goal:
  timestamp | order_id | side | price | size | status | filled_size | inv_impact | source

Queries (all read-only, no order submission):
  1. get_order()  for each known order_id — full field dump
  2. get_trades() windowed — Hungary YES, after=09:29, before=11:26
  3. get_trades() unfiltered — Hungary YES, full linkage, no truncation
  4. get_balance_allowance(CONDITIONAL) — current live balance
  5. get_order() raw dump — SCORING_ACTIVE ASK (safe serialization)
  6. get_order() raw dump — SCORING_ACTIVE BID (safe serialization)
  7. get_orders() open/all — Hungary YES token (any surviving open order?)
  8. get_trades(maker_address=EOA) — all trades for this account, full linkage

Patch notes (v3):
  - ALL_KNOWN_IDS_LOWER: flat dict for O(1) attribution matching against all order IDs
  - _match_time_utc(): converts match_time/timestamp to readable UTC; handles Unix
    seconds, Unix ms, and ISO strings; always shows raw value alongside
  - _attr_check(): checks maker_order_id, taker_order_id, maker_orders[].order_id
    against ALL known order IDs (not just ASK); returns labelled match strings
  - _print_trade_full(): now prints match_time_utc as derived field; adds
    match_time/matchTime and maker_orders/makerOrders to priority field list
  - Step 3: attribution check via _attr_check() against all known IDs
  - Step 8: PRIMARY query uses proxy/funder address (0x8e5c…) — previous run used
    EOA which returned empty because fills are recorded against proxy address;
    EOA query runs as crosscheck; both results attribution-checked

Usage (from repo root, Windows PowerShell):
  py -3 research_lines/reward_aware_maker_probe/run_inventory_reconciliation.py
"""
from __future__ import annotations

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
    load_activation_credentials,
    get_missing_credential_vars,
    build_clob_client,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUNGARY_TOKEN_ID = (
    "94192784911459194325909253314484842244405314804074606736702592885535642919725"
)
CLOB_HOST    = "https://clob.polymarket.com"
CTF_DECIMALS = 1_000_000

SCORING_ACTIVE_ASK_ID = "0xe3c3ba2b8c45f55db09c36fd8771f458016f7b7f3915c2e03836d2d3ff9b88ce"
SCORING_ACTIVE_BID_ID = "0xfb8dc9ecf0a9d194bb22d60a4e82e5944f36cc18ea71874768c7a3f4c2a84a5c"

# All order IDs placed during live runs on 2026-03-26
# Source: JSON artifacts in data/research/reward_aware_maker_probe/scoring_activation/
KNOWN_ORDERS = [
    {
        "run_ts":     "2026-03-26T09:04:12Z",
        "order_id":   "0x41ad7730b91448cb338a950ed1ccb0f51c78d23009be2986fdc61270b418dfcd",
        "side":       "BUY",
        "price":      0.62,
        "size":       200.0,
        "run_verdict":"PREFLIGHT_FAILED/ALLOWANCE",
    },
    {
        "run_ts":     "2026-03-26T09:08:17Z",
        "order_id":   "0x2c4b27eee57c532578f3d12b22f92b5a7c1bb48d76f0b8ac1299f01512fc46e4",
        "side":       "BUY",
        "price":      0.62,
        "size":       200.0,
        "run_verdict":"PREFLIGHT_FAILED/ALLOWANCE",
    },
    {
        "run_ts":     "2026-03-26T09:29:45Z",
        "order_id":   SCORING_ACTIVE_BID_ID,
        "side":       "BUY",
        "price":      0.62,
        "size":       200.0,
        "run_verdict":"SCORING_ACTIVE/BID",
    },
    {
        "run_ts":     "2026-03-26T09:29:45Z",
        "order_id":   SCORING_ACTIVE_ASK_ID,
        "side":       "SELL",
        "price":      0.65,
        "size":       200.0,
        "run_verdict":"SCORING_ACTIVE/ASK ← KEY",
    },
    {
        "run_ts":     "2026-03-26T11:26:38Z",
        "order_id":   "0x46ca7535a7d4225485a1b821db120b739729c17f6df2fd084244eb863192ba03",
        "side":       "BUY",
        "price":      0.63,
        "size":       200.0,
        "run_verdict":"PREFLIGHT_FAILED/INV_MISSING",
    },
]

WINDOW_START_ISO = "2026-03-26T09:29:45Z"
WINDOW_END_ISO   = "2026-03-26T11:26:38Z"


def _iso_to_unix_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


WINDOW_AFTER_MS  = _iso_to_unix_ms(WINDOW_START_ISO)
WINDOW_BEFORE_MS = _iso_to_unix_ms(WINDOW_END_ISO)

# Flat lookup: order_id_lower → run_verdict label, for attribution matching
ALL_KNOWN_IDS_LOWER: dict[str, str] = {
    meta["order_id"].lower(): meta["run_verdict"]
    for meta in KNOWN_ORDERS
}


# ---------------------------------------------------------------------------
# Safe serializer  ← PATCH: replaces bare vars() / hasattr(__dict__) pattern
# ---------------------------------------------------------------------------

def _to_dict(obj) -> dict:
    """
    Convert any py_clob_client response object to a plain dict without raising.

    py_clob_client returns several different object types depending on the
    endpoint and version:
      - plain dict (most GET responses after httpx deserialization)
      - Pydantic v2 BaseModel  → .model_dump()
      - Pydantic v1 BaseModel  → .dict()
      - dataclass / plain obj  → vars() / __dict__
      - something else         → repr() fallback (never crashes)

    The previous code used `vars(obj)` unconditionally which raises
    TypeError("vars() argument must have __dict__ attribute") on Pydantic v2
    models (which store fields in __slots__ / __pydantic_fields_set__).
    """
    if obj is None:
        return {"_note": "response was None"}
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1 / other .dict()
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    # Standard dataclass / object with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return vars(obj)
        except Exception:
            pass
    # Last resort — never raises, always gives something printable
    return {"_type": type(obj).__name__, "_repr": repr(obj)}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 110) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


def _ts_from_dict(d: dict) -> str:
    for key in ("created_at", "timestamp", "transacted_at", "updatedAt",
                "createdAt", "time", "trade_time"):
        val = d.get(key)
        if val:
            return str(val)
    return "N/A"


def _safe_float(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _get_field(d: dict, *keys) -> str:
    """Try each key in order, return first non-None value as str, else ''."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return str(v)
    return ""


def _match_time_utc(t: dict) -> str:
    """
    Extract and render match_time / timestamp as a readable UTC string.

    Polymarket CLOB uses several timestamp field names and formats:
      - match_time  : Unix seconds (float) — most common on trade records
      - timestamp   : may be Unix ms (int > 1e12) or seconds
      - created_at  : ISO string

    Heuristic: if numeric value > 1e11 treat as milliseconds, else seconds.
    Returns "(raw=<original>)" suffix so the unconverted value is always visible.
    """
    for key in ("match_time", "matchTime", "timestamp", "transacted_at",
                "created_at", "time", "trade_time"):
        raw = t.get(key)
        if raw is None:
            continue
        # Numeric path
        try:
            val = float(raw)
            if val > 1e11:      # likely milliseconds
                val /= 1000.0
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ") + f"  (raw={raw}, key={key})"
        except (ValueError, TypeError, OSError):
            pass
        # ISO string path
        try:
            s = str(raw).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s).astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ") + f"  (raw={raw}, key={key})"
        except Exception:
            pass
        return f"{raw}  (unconverted, key={key})"
    return "N/A"


def _attr_check(t: dict) -> list[str]:
    """
    Check all linkage ID fields in a trade dict against ALL_KNOWN_IDS_LOWER.

    Fields checked (all aliases):
      maker_order_id / makerOrderId
      taker_order_id / takerOrderId
      maker_orders   (list — each item may have order_id / orderId)

    Returns a list of match strings (empty list = no match).
    The caller should print these before the full trade detail.
    """
    matches: list[str] = []

    def _check_id(raw_id: str, field_name: str) -> None:
        oid = raw_id.strip().lower()
        if oid and oid in ALL_KNOWN_IDS_LOWER:
            label = ALL_KNOWN_IDS_LOWER[oid]
            matches.append(
                f"  *** ATTRIBUTION MATCH: {field_name} = {raw_id}  →  {label} ***"
            )

    for field in ("maker_order_id", "makerOrderId"):
        v = t.get(field)
        if v is not None:
            _check_id(str(v), field)

    for field in ("taker_order_id", "takerOrderId"):
        v = t.get(field)
        if v is not None:
            _check_id(str(v), field)

    # maker_orders is a list of dicts — Polymarket sometimes nests order IDs here
    mo = t.get("maker_orders") or t.get("makerOrders") or []
    if isinstance(mo, list):
        for idx, item in enumerate(mo):
            if not isinstance(item, dict):
                continue
            for id_key in ("order_id", "orderId", "id"):
                v = item.get(id_key)
                if v is not None:
                    _check_id(str(v), f"maker_orders[{idx}].{id_key}")

    return matches


# ---------------------------------------------------------------------------
# Per-order detail printer  ← PATCH: prints all attribution-grade fields
# ---------------------------------------------------------------------------

def _print_order_detail(label: str, order_id: str, raw: dict) -> None:
    """
    Print all fields relevant to fill attribution.
    Uses _get_field() with multiple key aliases to handle both camelCase
    and snake_case response shapes.
    """
    status       = _get_field(raw, "status", "order_status", "orderStatus") or "UNKNOWN"
    size_matched = _get_field(raw, "size_matched", "sizeMatched", "matched_size",
                               "matchedSize", "filled_size", "filledSize") or "0"
    maker_addr   = _get_field(raw, "maker_address", "makerAddress", "maker", "owner")
    asset_id     = _get_field(raw, "asset_id", "assetId", "token_id", "tokenId")
    order_id_srv = _get_field(raw, "id", "order_id", "orderId", "orderID")
    created_at   = _ts_from_dict(raw)
    price        = _get_field(raw, "price", "limit_price", "limitPrice")
    size         = _get_field(raw, "size", "original_size", "originalSize",
                               "size_original", "quantity")

    print(f"  [{label}]")
    print(f"    order_id (known)   : {order_id}")
    print(f"    order_id (server)  : {order_id_srv or '(not in response)'}")
    print(f"    status             : {status}")
    print(f"    size (original)    : {size or '?'}")
    print(f"    size_matched       : {size_matched}")
    print(f"    price              : {price or '?'}")
    print(f"    maker_address      : {maker_addr or '(not in response)'}")
    print(f"    asset_id / token   : {asset_id or '(not in response)'}")
    print(f"    created_at         : {created_at}")
    print(f"    all_keys           : {sorted(raw.keys())}")
    print(f"    full_raw           :")
    # Print the full raw dict — no truncation
    for k, v in sorted(raw.items()):
        print(f"      {k:30s} : {v}")
    print()


# ---------------------------------------------------------------------------
# Trade detail printer  ← PATCH: no truncation on linkage fields
# ---------------------------------------------------------------------------

def _print_trade_full(idx: int, t: dict) -> None:
    """
    Print all linkage fields for one trade dict — no truncation anywhere.
    These are the fields needed to attribute a fill to an order_id.

    match_time_utc is always printed as a derived field so the caller
    can immediately read a human timestamp without post-processing.
    """
    print(f"  trade[{idx}]")

    # ── Derived: match_time in readable UTC ──────────────────────────────
    print(f"    {'match_time_utc':30s} : {_match_time_utc(t)}")

    # ── Primary linkage fields — printed individually, full values ───────
    for key in (
        "id", "trade_id", "tradeId",
        "match_time", "matchTime",
        "timestamp", "created_at", "transacted_at", "time",
        "maker_order_id", "makerOrderId",
        "taker_order_id", "takerOrderId",
        "maker_orders",   "makerOrders",       # may be a list of {order_id, ...}
        "asset_id", "assetId", "market", "token_id",
        "side", "maker_side", "taker_side",
        "price", "match_price",
        "size", "size_matched", "sizeMatched",
        "status", "outcome",
    ):
        v = t.get(key)
        if v is not None:
            print(f"    {key:30s} : {v}")

    # ── Every remaining key (sorted, no truncation) ───────────────────────
    printed = {
        "id", "trade_id", "tradeId",
        "match_time", "matchTime",
        "timestamp", "created_at", "transacted_at", "time",
        "maker_order_id", "makerOrderId",
        "taker_order_id", "takerOrderId",
        "maker_orders",   "makerOrders",
        "asset_id", "assetId", "market", "token_id",
        "side", "maker_side", "taker_side",
        "price", "match_price",
        "size", "size_matched", "sizeMatched",
        "status", "outcome",
    }
    for k, v in sorted(t.items()):
        if k not in printed:
            print(f"    {k:30s} : {v}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    _section("reward_aware_inventory_reconciliation v2 — read-only")
    print(f"  Window start : {WINDOW_START_ISO}  (SCORING_ACTIVE run started)")
    print(f"  Window end   : {WINDOW_END_ISO}  (PREFLIGHT_FAILED run, balance=0 confirmed)")
    print(f"  Token        : Hungary YES  {HUNGARY_TOKEN_ID}")
    print(f"  Key order    : SCORING_ACTIVE ASK  {SCORING_ACTIVE_ASK_ID}")
    print()

    # ── Credentials ───────────────────────────────────────────────────────
    # Only POLYMARKET_PRIVATE_KEY is required.  API trio is auto-derived.
    creds = load_activation_credentials()
    if creds is None:
        missing = get_missing_credential_vars()
        print("  PRIVATE KEY NOT AVAILABLE — hard stop.")
        print(f"  Missing: {missing}")
        print()
        print("  Set: $env:POLYMARKET_PRIVATE_KEY = '<your_evm_private_key>'")
        sys.exit(1)
    print(f"  private_key    : SET  (needs_api_derivation={creds.needs_api_derivation})")
    print(f"  signature_type : {creds.signature_type}")
    print(f"  funder         : {creds.funder!r}")
    print()

    # ── Build client — derives API creds automatically if not configured ──
    # build_clob_client() prints: credential_source / effective_api_key /
    # signer EOA / funder proxy.  creds is updated in-place with effective values.
    try:
        client = build_clob_client(creds, CLOB_HOST)
        eoa_address = client.get_address()
    except Exception as exc:
        print(f"  ERROR: build_clob_client failed: {exc}")
        sys.exit(1)
    print()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Per-order status — all known order IDs
    # PATCH: uses _to_dict(); prints status/size_matched/maker_address/asset_id
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 1: Per-order status — get_order() for every known order_id")
    print("  Fields: status | size_matched | maker_address | asset_id | orderID | created_at")
    print()

    for meta in KNOWN_ORDERS:
        oid = meta["order_id"]
        try:
            resp = client.get_order(oid)
            raw  = _to_dict(resp)     # ← PATCH: was bare vars()/hasattr branch
            _print_order_detail(meta["run_verdict"], oid, raw)
        except Exception as exc:
            print(f"  [{meta['run_verdict']}]")
            print(f"    order_id : {oid}")
            print(f"    ERROR    : {exc}")
            print()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Trade log — windowed (09:29 → 11:26 UTC)
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 2: Trade log — Hungary YES, windowed 09:29→11:26 UTC")
    print(f"  after_ms  = {WINDOW_AFTER_MS}  ({WINDOW_START_ISO})")
    print(f"  before_ms = {WINDOW_BEFORE_MS}  ({WINDOW_END_ISO})")
    print()

    windowed_trades: list[dict] = []
    try:
        from py_clob_client.clob_types import TradeParams
        cursor = "MA=="
        page   = 0
        while True:
            result = client.get_trades(
                TradeParams(
                    asset_id=HUNGARY_TOKEN_ID,
                    after=WINDOW_AFTER_MS,
                    before=WINDOW_BEFORE_MS,
                ),
                next_cursor=cursor,
            )
            if result is None:
                print("  result=None from windowed get_trades")
                break
            if isinstance(result, dict):
                data   = result.get("data") or []
                cursor = result.get("next_cursor", "LTE=")
            elif isinstance(result, list):
                data   = result
                cursor = "LTE="
            else:
                data   = []
                cursor = "LTE="

            for t in data:
                windowed_trades.append(_to_dict(t))

            page += 1
            if cursor in ("LTE=", None) or not data:
                break
            if page > 20:
                print("  [WARN] pagination > 20 pages — stopped")
                break

    except Exception as exc:
        print(f"  ERROR: {exc}")

    if not windowed_trades:
        print("  NO TRADES in window — windowed query returned empty.")
        print()
    else:
        print(f"  {len(windowed_trades)} trade(s) in window:")
        print()
        for i, t in enumerate(windowed_trades):
            _print_trade_full(i, t)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Broader trade scan — no time filter, full linkage, no truncation
    # PATCH: removed _fmt_order_id() clipping; removed 200-char json cap
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 3: Recent trades — Hungary YES, no time filter, full linkage")
    print("  get_trades(asset_id=<hungary_yes>)  — unfiltered, full field dump")
    print()

    recent_trades: list[dict] = []
    try:
        from py_clob_client.clob_types import TradeParams
        result = client.get_trades(TradeParams(asset_id=HUNGARY_TOKEN_ID))
        if result is None:
            print("  result=None")
        elif isinstance(result, dict):
            recent_trades = [_to_dict(t) for t in (result.get("data") or [])]
            print(f"  next_cursor : {result.get('next_cursor')}")
            print(f"  count       : {len(recent_trades)}")
        elif isinstance(result, list):
            recent_trades = [_to_dict(t) for t in result]
            print(f"  count       : {len(recent_trades)}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print()
    if not recent_trades:
        print("  NO TRADES returned.")
        print()
    else:
        # Attribution check: compare all linkage fields against ALL known order IDs
        print(f"  Checking linkage fields against all known order_ids:")
        for label, oid in [(m["run_verdict"], m["order_id"]) for m in KNOWN_ORDERS]:
            print(f"    {label:35s} : {oid}")
        print()
        for i, t in enumerate(recent_trades):
            matches = _attr_check(t)
            for m in matches:
                print(m)
            _print_trade_full(i, t)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Current conditional balance
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 4: Current conditional balance — Hungary YES (live)")
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams
        try:
            client.update_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=HUNGARY_TOKEN_ID,
                    signature_type=-1,
                )
            )
        except Exception as upd_exc:
            print(f"  [WARN] update_balance_allowance: {upd_exc}")

        resp     = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type="CONDITIONAL",
                token_id=HUNGARY_TOKEN_ID,
                signature_type=-1,
            )
        )
        raw_bal  = _to_dict(resp)
        bal_raw  = int(float(raw_bal.get("balance", 0) or 0))
        print(f"  balance_raw    : {bal_raw}")
        print(f"  balance_shares : {bal_raw / CTF_DECIMALS:.6f}")
        print(f"  full_raw       : {json.dumps(raw_bal, default=str)}")
    except Exception as exc:
        print(f"  ERROR: {exc}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Full raw dump — SCORING_ACTIVE ASK
    # PATCH: uses _to_dict() instead of vars()
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 5: Full raw response — SCORING_ACTIVE ASK order")
    print(f"  order_id : {SCORING_ACTIVE_ASK_ID}")
    print()
    try:
        resp = client.get_order(SCORING_ACTIVE_ASK_ID)
        raw  = _to_dict(resp)    # ← PATCH: was vars(resp) which crashed
        print(f"  type(response) : {type(resp).__name__}")
        print(f"  repr (first 200): {repr(resp)[:200]}")
        print()
        _print_order_detail("SCORING_ACTIVE/ASK", SCORING_ACTIVE_ASK_ID, raw)
    except Exception as exc:
        print(f"  ERROR: {exc}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Full raw dump — SCORING_ACTIVE BID
    # PATCH: uses _to_dict() instead of vars()
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 6: Full raw response — SCORING_ACTIVE BID order")
    print(f"  order_id : {SCORING_ACTIVE_BID_ID}")
    print()
    try:
        resp = client.get_order(SCORING_ACTIVE_BID_ID)
        raw  = _to_dict(resp)    # ← PATCH: was vars(resp) which crashed
        print(f"  type(response) : {type(resp).__name__}")
        print(f"  repr (first 200): {repr(resp)[:200]}")
        print()
        _print_order_detail("SCORING_ACTIVE/BID", SCORING_ACTIVE_BID_ID, raw)
    except Exception as exc:
        print(f"  ERROR: {exc}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7: Open-order scan — Hungary YES token
    # Does the SCORING_ACTIVE ASK still exist as an open order?
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 7: Open-order scan — Hungary YES token (get_orders)")
    print("  get_orders(OpenOrderParams(asset_id=<hungary_yes>))")
    print()
    try:
        from py_clob_client.clob_types import OpenOrderParams
        result = client.get_orders(OpenOrderParams(asset_id=HUNGARY_TOKEN_ID))
        if result is None:
            print("  result=None")
        elif isinstance(result, dict):
            orders = result.get("data") or []
            print(f"  next_cursor : {result.get('next_cursor')}")
            print(f"  count       : {len(orders)}")
        elif isinstance(result, list):
            orders = result
            print(f"  count       : {len(orders)}")
        else:
            orders = []
            print(f"  unexpected type: {type(result).__name__}  repr={repr(result)[:100]}")

        if not orders:
            print("  NO open orders for Hungary YES.")
            print("  → SCORING_ACTIVE ASK is NOT currently open.")
        else:
            ask_id_lower = SCORING_ACTIVE_ASK_ID.lower()
            for i, o in enumerate(orders):
                od = _to_dict(o)
                srv_id = str(od.get("id", od.get("order_id", od.get("orderId", "")))).lower()
                match  = "  *** MATCHES SCORING_ACTIVE ASK ***" if srv_id == ask_id_lower else ""
                print(f"  open_order[{i}]{match}")
                for k, v in sorted(od.items()):
                    print(f"    {k:30s} : {v}")
                print()
    except Exception as exc:
        print(f"  ERROR: {exc}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 8: Maker-address trade scan — proxy/funder address PRIMARY
    #
    # Patch: previous run used EOA (0x1B0E…) which returned no trades.
    # Evidence shows funder/proxy (0x8e5c…) is the maker_address recorded
    # on fills.  Query proxy first; also query EOA as a crosscheck.
    # Both results are attribution-checked against ALL known order IDs.
    # ─────────────────────────────────────────────────────────────────────
    _section("Step 8: Maker-address trade scan — proxy/funder (primary) + EOA (crosscheck)")

    proxy_address = creds.funder   # 0x8e5c… — address recorded as maker on fills
    addresses_to_scan: list[tuple[str, str]] = []
    if proxy_address:
        addresses_to_scan.append(("proxy/funder", proxy_address))
    addresses_to_scan.append(("signer/EOA", eoa_address))

    print(f"  Addresses to scan:")
    for role, addr in addresses_to_scan:
        print(f"    {role:15s} : {addr}")
    print(f"  Token        : Hungary YES  {HUNGARY_TOKEN_ID[:24]}...")
    print()

    try:
        from py_clob_client.clob_types import TradeParams

        for role, addr in addresses_to_scan:
            print(f"  ── {role} ({addr}) ──")
            try:
                result = client.get_trades(
                    TradeParams(
                        maker_address=addr,
                        asset_id=HUNGARY_TOKEN_ID,
                    )
                )
                if result is None:
                    print(f"    result=None")
                    print()
                    continue
                elif isinstance(result, dict):
                    addr_trades = [_to_dict(t) for t in (result.get("data") or [])]
                    print(f"    next_cursor : {result.get('next_cursor')}")
                    print(f"    count       : {len(addr_trades)}")
                elif isinstance(result, list):
                    addr_trades = [_to_dict(t) for t in result]
                    print(f"    count       : {len(addr_trades)}")
                else:
                    print(f"    unexpected type: {type(result).__name__}")
                    print()
                    continue

                print()
                if not addr_trades:
                    print(f"    NO fills for {role}.")
                    print()
                    continue

                # Attribution check against ALL known order IDs
                print(f"    Checking linkage fields against all known order_ids:")
                for meta in KNOWN_ORDERS:
                    print(f"      {meta['run_verdict']:35s} : {meta['order_id']}")
                print()

                for i, t in enumerate(addr_trades):
                    matches = _attr_check(t)
                    for m in matches:
                        print(m)
                    _print_trade_full(i, t)

            except Exception as addr_exc:
                print(f"    ERROR for {role}: {addr_exc}")
                print()

    except Exception as exc:
        print(f"  ERROR: {exc}")
    print()

    _sep("=")
    print()
    print("reward_aware_maker_probe")


if __name__ == "__main__":
    main()
