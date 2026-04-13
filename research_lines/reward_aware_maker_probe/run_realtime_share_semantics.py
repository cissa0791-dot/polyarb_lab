"""
realtime_share_semantics_reconciliation
polyarb_lab / research_line / diagnostic

Sub-goal of reward_share_read_path_audit_line.

Synchronized read-only snapshot during a live scoring window:
  1. GET open orders for Hungary YES token
  2. is_order_scoring for each live order found
  3. /rewards/user/percentages  (maker_address=funder)
  4. /rewards/user/markets      (maker_address=funder, paginated to Hungary)
  All four calls made in the same run, timestamps recorded per call.

Questions answered:
  Q1: When order-scoring=True, does /rewards/user/percentages remain {}?
  Q2: When order-scoring=True, does user/markets.earning_percentage remain 0?
  Q3: If both empty/zero during confirmed scoring=True, is this a latency/
      coverage issue on the official endpoint rather than an auth problem?

Credential diagnostic:
  Prints which env vars are resolving and whether the derived-key fallback
  (_try_wallet_as_key via POLY_WALLET_ADDRESS) is in use.  The user must
  set POLYMARKET_PRIVATE_KEY explicitly if this fallback is active.

No orders placed. No state changes. Exits immediately.

Usage (from repo root):
  py -3 research_lines/reward_aware_maker_probe/run_realtime_share_semantics.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
    load_activation_credentials,
    get_missing_credential_vars,
    build_clob_client,
    _try_wallet_as_key,
    SURVIVOR_DATA,
)

CLOB_HOST      = "https://clob.polymarket.com"
HUNGARY_SLUG   = "will-the-next-prime-minister-of-hungary-be-pter-magyar"
HUNGARY_DATA   = SURVIVOR_DATA[HUNGARY_SLUG]
HUNGARY_CID    = HUNGARY_DATA["condition_id"]
HUNGARY_TID    = HUNGARY_DATA["token_id"]

_CURSOR_END = "LTE="


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


def _l2_headers(client, path: str) -> dict:
    from py_clob_client.clob_types import RequestArgs
    from py_clob_client.headers.headers import create_level_2_headers
    return create_level_2_headers(
        client.signer, client.creds, RequestArgs(method="GET", request_path=path)
    )


def _get(client, signed_path: str, url_suffix: str = "") -> tuple[int, object]:
    import httpx
    headers = _l2_headers(client, signed_path)
    url = f"{CLOB_HOST}{signed_path}{url_suffix}"
    try:
        with httpx.Client() as h:
            resp = h.get(url, headers=headers, timeout=10)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return resp.status_code, body
    except Exception as exc:
        return -1, str(exc)


# ---------------------------------------------------------------------------
# Credential diagnostic
# ---------------------------------------------------------------------------

def _print_credential_diagnostic() -> None:
    _section("Credential diagnostic")

    # Check each source independently so we can warn on fallback usage
    has_poly_pk          = bool(os.environ.get("POLYMARKET_PRIVATE_KEY"))
    has_poly_pk_short    = bool(os.environ.get("POLY_PRIVATE_KEY"))
    has_wallet_fallback  = bool(_try_wallet_as_key(os.environ.get("POLY_WALLET_ADDRESS")))
    has_poly_api_key     = bool(os.environ.get("POLYMARKET_API_KEY") or os.environ.get("POLY_API_KEY"))
    has_poly_secret      = bool(os.environ.get("POLYMARKET_API_SECRET") or os.environ.get("POLY_API_SECRET"))
    has_poly_pass        = bool(os.environ.get("POLYMARKET_API_PASSPHRASE") or os.environ.get("POLY_PASSPHRASE"))
    has_sig_type         = bool(os.environ.get("POLYMARKET_SIGNATURE_TYPE") or os.environ.get("POLY_SIGNATURE_TYPE"))
    has_funder           = bool(os.environ.get("POLYMARKET_FUNDER") or os.environ.get("POLY_FUNDER"))

    sig_type_raw = (
        os.environ.get("POLYMARKET_SIGNATURE_TYPE")
        or os.environ.get("POLY_SIGNATURE_TYPE")
        or "0 (default — not explicitly set)"
    )
    funder_val = (
        os.environ.get("POLYMARKET_FUNDER")
        or os.environ.get("POLY_FUNDER")
        or "(not set)"
    )

    print(f"  POLYMARKET_PRIVATE_KEY         : {'SET' if has_poly_pk else 'not set'}")
    print(f"  POLY_PRIVATE_KEY               : {'SET' if has_poly_pk_short else 'not set'}")
    print(f"  POLY_WALLET_ADDRESS fallback    : {'ACTIVE ← WARNING' if has_wallet_fallback and not has_poly_pk and not has_poly_pk_short else 'not in use'}")
    print(f"  API key                        : {'SET' if has_poly_api_key else 'NOT SET'}")
    print(f"  API secret                     : {'SET' if has_poly_secret else 'NOT SET'}")
    print(f"  API passphrase                 : {'SET' if has_poly_pass else 'NOT SET'}")
    print(f"  POLYMARKET_SIGNATURE_TYPE      : {sig_type_raw}")
    print(f"  POLYMARKET_FUNDER              : {funder_val}")
    print()

    if has_wallet_fallback and not has_poly_pk and not has_poly_pk_short:
        print("  WARNING: Private key is being derived from POLY_WALLET_ADDRESS.")
        print("  This fallback must NOT be used for any automated trading path.")
        print("  Set POLYMARKET_PRIVATE_KEY explicitly to the raw hex private key.")
        print()

    sig_type_int = 0
    try:
        sig_type_int = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE") or os.environ.get("POLY_SIGNATURE_TYPE") or "0")
    except ValueError:
        pass

    if sig_type_int in (1, 2) and not has_funder:
        print(f"  WARNING: signature_type={sig_type_int} requires POLYMARKET_FUNDER to be set.")
        print("  Without funder, reward queries will target the EOA, not the proxy wallet.")
        print("  Set POLYMARKET_FUNDER to the proxy wallet address shown in Polymarket Settings.")
        print()

    if sig_type_int == 0 and not has_sig_type:
        print("  NOTE: signature_type defaulting to 0 (EOA). If your API key was created")
        print("  via a Polymarket proxy/GNOSIS_SAFE wallet, set POLYMARKET_SIGNATURE_TYPE=2")
        print("  and POLYMARKET_FUNDER=<proxy wallet address>.")
        print()


# ---------------------------------------------------------------------------
# Step 1 — live open orders for Hungary YES token
# ---------------------------------------------------------------------------

def _fetch_open_orders(client) -> list[dict]:
    """
    Fetch all live open orders for Hungary YES token_id.
    Uses client.get_orders() which paginates internally.
    Returns a list of order dicts.
    """
    from py_clob_client.clob_types import OpenOrderParams
    try:
        orders = client.get_orders(OpenOrderParams(asset_id=HUNGARY_TID))
        return orders if isinstance(orders, list) else []
    except Exception as exc:
        print(f"  get_orders error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Step 2 — is_order_scoring per order
# ---------------------------------------------------------------------------

def _check_scoring(client, order_id: str) -> tuple[bool | None, str]:
    """Returns (scoring_bool_or_None, raw_response_repr)."""
    from py_clob_client.clob_types import OrderScoringParams
    try:
        result = client.is_order_scoring(OrderScoringParams(orderId=order_id))
        if result is None:
            return None, "None"
        if isinstance(result, bool):
            return result, repr(result)
        if isinstance(result, dict):
            scoring = bool(result.get("scoring") or result.get("is_scoring"))
            return scoring, repr(result)
        return None, repr(result)
    except Exception as exc:
        return None, f"error: {exc}"


# ---------------------------------------------------------------------------
# Step 3 — /rewards/user/percentages
# ---------------------------------------------------------------------------

def _fetch_percentages(client, sig_type_val: int, maker_address: str | None) -> tuple[int, object, str]:
    signed_path = "/rewards/user/percentages"
    url_suffix  = f"?signature_type={sig_type_val}"
    if maker_address:
        url_suffix += f"&maker_address={maker_address}"
    ts = _now_utc()
    status, body = _get(client, signed_path, url_suffix)
    return status, body, ts


# ---------------------------------------------------------------------------
# Step 4 — /rewards/user/markets  (paginated to Hungary)
# ---------------------------------------------------------------------------

def _fetch_markets_hungary(
    client, sig_type_val: int, maker_address: str | None
) -> tuple[dict | None, int, bool, str]:
    """
    Paginate /rewards/user/markets until Hungary found or exhausted.
    Returns (entry_or_None, pages_fetched, fetch_error, first_call_timestamp).
    """
    SIGNED_PATH = "/rewards/user/markets"
    cursor: str | None = None
    pages = 0
    first_ts: str | None = None
    fetch_error = False

    while True:
        url_suffix = f"?signature_type={sig_type_val}"
        if maker_address:
            url_suffix += f"&maker_address={maker_address}"
        if cursor:
            url_suffix += f"&next_cursor={cursor}"

        ts = _now_utc()
        if first_ts is None:
            first_ts = ts

        status, body = _get(client, SIGNED_PATH, url_suffix)
        pages += 1

        print(f"    [page {pages}] HTTP {status}  ts={ts}")

        if status != 200:
            print(f"    body: {body!r}")
            fetch_error = True
            return None, pages, True, first_ts or ts

        entries = []
        next_cursor = None
        if isinstance(body, dict):
            raw = body.get("data", body)
            entries = raw if isinstance(raw, list) else []
            next_cursor = body.get("next_cursor") or body.get("nextCursor")
            total = body.get("total_count") or body.get("totalCount") or "?"
            print(f"    [page {pages}] entries={len(entries)}  total={total}  next_cursor={next_cursor!r}")
        elif isinstance(body, list):
            entries = body

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cid = str(
                entry.get("condition_id") or entry.get("conditionId")
                or entry.get("market_id") or entry.get("marketId") or ""
            )
            if cid == HUNGARY_CID:
                print(f"    Hungary found on page {pages}")
                return entry, pages, False, first_ts

        if not next_cursor or next_cursor == _CURSOR_END:
            print(f"    Pagination exhausted after {pages} page(s) — Hungary not found")
            return None, pages, False, first_ts or ts

        cursor = next_cursor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    _section("realtime_share_semantics_reconciliation")
    print("  Read-only. No orders. No state changes.")
    print(f"  Hungary condition_id : {HUNGARY_CID}")
    print(f"  Hungary token_id     : {HUNGARY_TID}")
    print()

    # ── Credential diagnostic ─────────────────────────────────────────────
    _print_credential_diagnostic()

    creds = load_activation_credentials()
    if creds is None:
        missing = get_missing_credential_vars()
        print(f"  CREDENTIALS NOT AVAILABLE  missing={missing}")
        sys.exit(1)

    client = build_clob_client(creds, CLOB_HOST)
    sig_type_val = client.builder.sig_type

    try:
        from eth_account import Account as _Acct
        eoa = _Acct.from_key(creds.private_key).address
    except Exception:
        eoa = "(eth_account unavailable)"
    maker_addr = creds.funder

    _section("Identity context  (used for all reward calls)")
    print(f"  EOA/signer         : {eoa}")
    print(f"  funder/proxy       : {maker_addr!r}")
    print(f"  api_key            : {creds.api_key[:8]}...")
    print(f"  signature_type     : {sig_type_val}")
    print(f"  maker_address used : {maker_addr!r}  (all reward calls)")
    print()

    # ── Step 1: Open orders ───────────────────────────────────────────────
    _section("Step 1 — live open orders  (Hungary YES token)")
    t_orders = _now_utc()
    orders = _fetch_open_orders(client)
    print(f"  ts={t_orders}")
    print(f"  open orders found  : {len(orders)}")
    if orders:
        for o in orders:
            oid   = o.get("id") or o.get("order_id") or "?"
            side  = o.get("side") or "?"
            price = o.get("price") or "?"
            size  = o.get("size_matched") or o.get("original_size") or "?"
            print(f"    order_id={oid}  side={side}  price={price}  original_size={size}")
    else:
        print("  No live orders found for Hungary YES token.")
        print("  is_order_scoring will be skipped (no order IDs).")
    print()

    # ── Step 2: is_order_scoring ──────────────────────────────────────────
    _section("Step 2 — is_order_scoring per live order")
    scoring_results: list[tuple[str, bool | None, str]] = []
    if orders:
        for o in orders:
            oid = o.get("id") or o.get("order_id") or ""
            if not oid:
                continue
            t_sc = _now_utc()
            sc, raw = _check_scoring(client, oid)
            scoring_results.append((oid, sc, t_sc))
            tag = "SCORING=TRUE" if sc is True else ("SCORING=FALSE" if sc is False else "SCORING=None")
            print(f"  {tag}  order_id={oid}  ts={t_sc}  raw={raw}")
    else:
        print("  (skipped — no live orders)")
    any_scoring_true = any(sc is True for _, sc, _ in scoring_results)
    any_scoring_false = any(sc is False for _, sc, _ in scoring_results)
    print()

    # ── Step 3: /rewards/user/percentages ────────────────────────────────
    _section("Step 3 — /rewards/user/percentages")
    pct_status, pct_body, pct_ts = _fetch_percentages(client, sig_type_val, maker_addr)
    print(f"  ts={pct_ts}  HTTP {pct_status}")
    if pct_status == 200:
        pct_is_empty = isinstance(pct_body, dict) and len(pct_body) == 0
        print(f"  response_type={type(pct_body).__name__}  "
              f"keys={len(pct_body) if isinstance(pct_body, dict) else 'N/A'}")
        if pct_is_empty:
            print("  RESULT: {} — empty map returned")
        else:
            print("  raw body:")
            print(json.dumps(pct_body, indent=4))
            # Check if Hungary is in the map
            hungary_pct = None
            if isinstance(pct_body, dict):
                hungary_pct = pct_body.get(HUNGARY_CID)
            print(f"  Hungary in map    : {hungary_pct is not None}")
            if hungary_pct is not None:
                print(f"  Hungary value     : {hungary_pct}")
    else:
        pct_is_empty = True
        print(f"  error body: {pct_body!r}")
    print()

    # ── Step 4: /rewards/user/markets ────────────────────────────────────
    _section("Step 4 — /rewards/user/markets  (paginated to Hungary)")
    markets_entry, markets_pages, markets_err, markets_ts = _fetch_markets_hungary(
        client, sig_type_val, maker_addr
    )
    print()
    if markets_entry is not None:
        print("  Hungary entry (full):")
        print(json.dumps(markets_entry, indent=4))
        ep = markets_entry.get("earning_percentage")
        earnings_arr = markets_entry.get("earnings")
        print()
        print(f"  earning_percentage : {ep}")
        if isinstance(earnings_arr, list):
            total_e = sum(
                float(item.get("earnings", 0) or 0)
                for item in earnings_arr
                if isinstance(item, dict)
            )
            print(f"  earnings (array)   : {earnings_arr}")
            print(f"  earnings total     : {total_e:.6f}")
        elif earnings_arr is not None:
            print(f"  earnings           : {earnings_arr}")
    else:
        print("  Hungary NOT found in user/markets")
    print()

    # ── Collect raw evidence ──────────────────────────────────────────────
    ep_val = markets_entry.get("earning_percentage") if markets_entry else None
    try:
        ep_float = float(ep_val) if ep_val is not None else None
    except (ValueError, TypeError):
        ep_float = None
    ep_zero = ep_float is None or ep_float == 0.0

    earnings_arr = markets_entry.get("earnings") if markets_entry else None
    earnings_nonzero = False
    if isinstance(earnings_arr, list):
        earnings_nonzero = any(
            float(i.get("earnings", 0) or 0) > 0
            for i in earnings_arr if isinstance(i, dict)
        )
    elif earnings_arr is not None:
        try:
            earnings_nonzero = float(earnings_arr) > 0
        except (ValueError, TypeError):
            pass

    q1_premise = any_scoring_true

    # ── Synchronized snapshot summary ────────────────────────────────────
    _sep("=")
    print()
    print("  SYNCHRONIZED SNAPSHOT SUMMARY")
    print()

    # Direct evidence only — no interpretation
    print("  [DIRECT EVIDENCE]")
    print()
    print(f"  open_orders_found      : {len(orders)}")
    if scoring_results:
        for oid, sc, ts in scoring_results:
            print(f"    order_id={oid[:16]}...  scoring={sc}  ts={ts}")
    else:
        print("    (no live orders — scoring not checked)")
    print()
    print(f"  /rewards/user/percentages")
    print(f"    HTTP {pct_status}  ts={pct_ts}")
    print(f"    body                 : {'{}  (empty map)' if pct_is_empty else 'non-empty'}")
    print()
    print(f"  /rewards/user/markets")
    print(f"    first_page_ts        : {markets_ts}")
    print(f"    hungary_found        : {markets_entry is not None}")
    if markets_entry is not None:
        print(f"    earning_percentage   : {ep_val}")
        if isinstance(earnings_arr, list):
            _total = sum(float(i.get('earnings', 0) or 0) for i in earnings_arr if isinstance(i, dict))
            print(f"    earnings_array_sum   : {_total:.6f}")
        elif earnings_arr is not None:
            print(f"    earnings             : {earnings_arr}")
    print()

    # ── Q1 / Q2 ──────────────────────────────────────────────────────────
    _sep("-")
    print()
    print("  QUESTIONS (observation only — no causal claim):")
    print()

    # Q1
    if not scoring_results:
        print("  Q1  [NO BASELINE] Cannot answer — no live orders to check scoring state.")
        print("      Place qualifying bilateral Hungary quotes first, then rerun.")
    elif q1_premise:
        if pct_is_empty:
            print("  Q1  [OBSERVED] order-scoring=True AND /rewards/user/percentages returns {}.")
        else:
            print("  Q1  [OBSERVED] order-scoring=True AND percentages returned non-empty.")
    elif any_scoring_false:
        print("  Q1  [PREMISE NOT MET] All live orders show scoring=False.")
    else:
        print("  Q1  [PREMISE NOT MET] Scoring state indeterminate (None result).")
    print()

    # Q2
    if not scoring_results:
        print("  Q2  [NO BASELINE] Cannot answer — no live orders to confirm scoring state.")
    elif q1_premise:
        if ep_zero:
            print(f"  Q2  [OBSERVED] order-scoring=True AND earning_percentage={ep_val} (zero or absent).")
        else:
            print(f"  Q2  [OBSERVED] order-scoring=True AND earning_percentage={ep_val} (non-zero).")
    else:
        print("  Q2  [PREMISE NOT MET] scoring=True not confirmed (see Q1).")
    print()

    # ── Q3 verdict ───────────────────────────────────────────────────────
    _sep("-")
    print()
    print("  Q3  VERDICT (evidence-bounded):")
    print()

    if not scoring_results:
        verdict_q3 = "AUDIT_INCONCLUSIVE"
        print(f"  {verdict_q3}")
        print()
        print("  [DIRECT EVIDENCE]")
        print("    No live orders found — scoring baseline absent.")
        print("    Cannot distinguish auth problem from no-order state.")
        print()
        print("  [INTERPRETATION]  — none; insufficient evidence.")
        print()
        print("  [EXTRAPOLATION]   — none; do not extrapolate without scoring baseline.")
    elif markets_err or markets_entry is None:
        verdict_q3 = "AUDIT_INCONCLUSIVE"
        print(f"  {verdict_q3}")
        print()
        print("  [DIRECT EVIDENCE]")
        print("    markets endpoint failed or Hungary not found after pagination.")
        print()
        print("  [INTERPRETATION]  — none; markets fetch unreliable this run.")
        print()
        print("  [EXTRAPOLATION]   — none.")
    elif q1_premise and pct_is_empty and ep_zero and earnings_nonzero:
        verdict_q3 = "CURRENT_REALTIME_SHARE_NOT_OBSERVED"
        print(f"  {verdict_q3}")
        print()
        print("  [DIRECT EVIDENCE]")
        print(f"    order-scoring=True       for {sum(1 for _,s,_ in scoring_results if s is True)} order(s)")
        print(f"    percentages endpoint     returned {{}}")
        print(f"    earning_percentage       = {ep_val}")
        print(f"    dated earnings           non-zero (reward did accrue at some prior point)")
        print()
        print("  [INTERPRETATION]  — candidate, not confirmed:")
        print("    The percentages map being empty while dated earnings are non-zero")
        print("    may indicate the endpoint reflects a current scoring window only,")
        print("    not cumulative history.  This is one possible explanation.")
        print("    Alternative: remaining auth/identity mismatch not yet surfaced.")
        print("    Alternative: account has no share in the current competitive window.")
        print()
        print("  [EXTRAPOLATION]   — not supported by this single snapshot:")
        print("    Do not conclude 'epoch-latency boundary' or 'not an auth bug'")
        print("    from this run alone.  A second run with different timing is required.")
    elif q1_premise and pct_is_empty and ep_zero and not earnings_nonzero:
        verdict_q3 = "CURRENT_REALTIME_SHARE_NOT_OBSERVED"
        print(f"  {verdict_q3}")
        print()
        print("  [DIRECT EVIDENCE]")
        print(f"    order-scoring=True       for {sum(1 for _,s,_ in scoring_results if s is True)} order(s)")
        print(f"    percentages endpoint     returned {{}}")
        print(f"    earning_percentage       = {ep_val}")
        print(f"    dated earnings           = 0 or absent")
        print()
        print("  [INTERPRETATION]  — candidate, not confirmed:")
        print("    No reward data observed from any source during this window.")
        print("    Cannot distinguish: too short a window / competitive exclusion /")
        print("    identity mismatch / endpoint coverage gap.")
        print()
        print("  [EXTRAPOLATION]   — not supported; more runs needed.")
    elif q1_premise and not pct_is_empty and ep_zero and earnings_nonzero:
        verdict_q3 = "FIELD_MISMATCH"
        print(f"  {verdict_q3}")
        print()
        print("  [DIRECT EVIDENCE]")
        print(f"    order-scoring=True")
        print(f"    percentages endpoint     returned non-empty but Hungary not present or zero")
        print(f"    dated earnings           non-zero")
        print()
        print("  [INTERPRETATION]  — candidate, not confirmed:")
        print("    Earnings and percentages disagree on Hungary presence/value.")
        print("    Possible identity mismatch between the two endpoints,")
        print("    or different epoch windows being read.")
        print()
        print("  [EXTRAPOLATION]   — do not conclude cause without further endpoint inspection.")
    elif q1_premise and not pct_is_empty and not ep_zero:
        verdict_q3 = "CURRENT_REALTIME_SHARE_NOT_OBSERVED"
        # Actually this is a positive case — earning_pct is non-zero
        verdict_q3 = "CONFIRMED_EARNINGS_NONZERO"
        print(f"  {verdict_q3}")
        print()
        print("  [DIRECT EVIDENCE]")
        print(f"    order-scoring=True")
        print(f"    earning_percentage       = {ep_val}  (non-zero)")
        print()
        print("  [INTERPRETATION]  — supported by direct evidence:")
        print("    Realtime earning_percentage is updating.  Read path is unblocked.")
        print()
        print("  [EXTRAPOLATION]   — confirm with additional runs before enabling automation.")
    else:
        verdict_q3 = "AUDIT_INCONCLUSIVE"
        print(f"  {verdict_q3}")
        print()
        print("  [DIRECT EVIDENCE]")
        print(f"    scoring_any_true={q1_premise}  pct_empty={pct_is_empty}  "
              f"ep_zero={ep_zero}  earnings_nonzero={earnings_nonzero}")
        print()
        print("  [INTERPRETATION]  — cannot classify with current evidence.")
        print()
        print("  [EXTRAPOLATION]   — none.")

    print()
    _sep("=")
    print()
    print("reward_aware_maker_probe")


if __name__ == "__main__":
    main()
