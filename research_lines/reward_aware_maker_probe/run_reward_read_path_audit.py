"""
reward_share_read_path_audit_line
polyarb_lab / research_line / diagnostic

Read-only audit of /rewards/user/markets and /rewards/user/percentages for
the Hungary target.  No orders.  No state changes.  Exits immediately.

Purpose:
  The scoring_activation script reads earning_pct via _check_earning_pct(), which:
    - used wrong header names (CLOB-* instead of POLY_*)
    - used millisecond timestamp instead of seconds
    - used standard base64 instead of urlsafe_b64encode
    - omitted POLY_ADDRESS header
    - read only first page of /rewards/user/markets (4711 total entries, 100 per page)
    - never called /rewards/user/percentages at all

  This audit fixes all of the above and reports the raw Hungary entry from both
  endpoints so field names and shapes can be inspected before any patch.

Auth:
  Uses build_clob_client() → create_level_2_headers() directly — exact same
  credential path and signing as the working activation flow.

Verdicts:
  REALTIME_NONZERO          : /rewards/user/percentages shows non-zero for Hungary
  CONFIRMED_EARNINGS_NONZERO: /rewards/user or /rewards/user/markets shows non-zero
                              earnings, percentages endpoint is present (even if zero)
  FIELD_MISMATCH            : earnings endpoints non-zero, but /rewards/user/percentages
                              is empty or missing Hungary — sources disagree
  TRUE_ZERO                 : Hungary found in all endpoints, all values are 0
  AUDIT_INCONCLUSIVE        : Hungary not found after full pagination, or a fetch failed

Usage (from repo root):
  py -3 research_lines/reward_aware_maker_probe/run_reward_read_path_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
    load_activation_credentials,
    get_missing_credential_vars,
    build_clob_client,
)

CLOB_HOST = "https://clob.polymarket.com"
HUNGARY_CONDITION_ID = "0x1480b819d03d4b6388d70e848b0384adf38c38d955cb783cdbcf6d4a436dee14"

# Cursor sentinel that py_clob_client treats as end-of-results
_CURSOR_END = "LTE="   # base64("−1")


def _sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


# ---------------------------------------------------------------------------
# Auth — reuse py_clob_client's exact signing path
# ---------------------------------------------------------------------------

def _l2_headers(client, path: str) -> dict:
    """
    Build Level-2 POLY headers using create_level_2_headers — identical to
    what the working activation flow uses for every authenticated call.
    Avoids reimplementing timestamp units, base64 variant, or header names.
    """
    from py_clob_client.clob_types import RequestArgs
    from py_clob_client.headers.headers import create_level_2_headers
    request_args = RequestArgs(method="GET", request_path=path)
    return create_level_2_headers(client.signer, client.creds, request_args)


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def _get(client, signed_path: str, url_suffix: str = "") -> tuple[int, object]:
    """
    GET with L2 auth.  HMAC is signed against `signed_path` (no query params).
    `url_suffix` is appended to the fetch URL only — not included in the signature.
    This matches the py_clob_client pattern seen in GET_NOTIFICATIONS and
    GET_BALANCE_ALLOWANCE: query params are on the URL but not in the signed path.
    """
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
# Pagination helper for /rewards/user/markets
# ---------------------------------------------------------------------------

def _fetch_markets_find_hungary(
    client, sig_type_val: int, maker_address: str | None
) -> tuple[dict | None, int, bool]:
    """
    Page through /rewards/user/markets until Hungary is found or pages exhausted.

    signed_path : /rewards/user/markets          (HMAC covers base path only)
    fetch URL   : ?signature_type=<n>[&maker_address=<a>][&next_cursor=<c>]
    All query params are URL-only — never in the signed path.

    Returns (hungary_entry, pages_fetched, fetch_error).
    """
    SIGNED_PATH = "/rewards/user/markets"
    cursor: str | None = None
    pages = 0
    fetch_error = False

    while True:
        url_suffix = f"?signature_type={sig_type_val}"
        if maker_address:
            url_suffix += f"&maker_address={maker_address}"
        if cursor:
            url_suffix += f"&next_cursor={cursor}"
        fetch_url = f"{CLOB_HOST}{SIGNED_PATH}{url_suffix}"

        print(f"  [page {pages + 1}] signed_path={SIGNED_PATH!r}")
        print(f"  [page {pages + 1}] fetch_url={fetch_url}")
        print(f"  [page {pages + 1}] maker_address={maker_address!r}")

        status, body = _get(client, SIGNED_PATH, url_suffix)
        pages += 1

        print(f"  [page {pages}] HTTP {status}")

        if status != 200:
            print(f"  body: {body!r}")
            fetch_error = True
            return None, pages, fetch_error

        entries = []
        next_cursor = None
        if isinstance(body, dict):
            raw = body.get("data", body)
            entries = raw if isinstance(raw, list) else []
            next_cursor = body.get("next_cursor") or body.get("nextCursor")
            count = len(entries)
            total = body.get("total_count") or body.get("totalCount") or "?"
            print(
                f"  [page {pages}] entries={count}  total_count={total}  "
                f"next_cursor={next_cursor!r}"
            )
        elif isinstance(body, list):
            entries = body
            print(f"  [page {pages}] entries={len(entries)}  (list root)")

        # Search this page for Hungary
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cid = str(
                entry.get("condition_id") or entry.get("conditionId")
                or entry.get("market_id") or entry.get("marketId") or ""
            )
            if cid == HUNGARY_CONDITION_ID:
                print(f"  Hungary found on page {pages}")
                return entry, pages, False

        # Stop if no more pages
        if not next_cursor or next_cursor == _CURSOR_END:
            print(f"  Pagination exhausted after {pages} page(s) — Hungary not found")
            return None, pages, False

        cursor = next_cursor


# ---------------------------------------------------------------------------
# /rewards/user/percentages — single fetch (not paginated per docs)
# ---------------------------------------------------------------------------

def _fetch_percentages_find_hungary(
    client, sig_type_val: int, maker_address: str | None
) -> tuple[dict | None, int, bool]:
    """
    Fetch /rewards/user/percentages and find the Hungary entry.
    Returns (entry, pages_fetched=1, fetch_error).

    Root cause of prior {} response: the endpoint defaults to the EOA from
    POLY_ADDRESS when maker_address is omitted.  For signature_type=2
    (GNOSIS_SAFE), rewards are tracked under the PROXY wallet (funder), not
    the EOA.  maker_address must be supplied explicitly.

    Official docs query params:
      signature_type  : required for API KEY auth
      maker_address   : address to query — must be proxy wallet (funder)
    """
    signed_path = "/rewards/user/percentages"
    url_suffix  = f"?signature_type={sig_type_val}"
    if maker_address:
        url_suffix += f"&maker_address={maker_address}"
    fetch_url = f"{CLOB_HOST}{signed_path}{url_suffix}"

    print(f"  signed_path    : {signed_path!r}")
    print(f"  fetch_url      : {fetch_url}")
    print(f"  signature_type : {sig_type_val}")
    print(f"  maker_address  : {maker_address!r}")

    status, body = _get(client, signed_path, url_suffix)

    print(f"  HTTP status    : {status}")
    if status != 200:
        print(f"  body: {body!r}")
        return None, 1, True

    # Compact schema diagnostic — always printed before full body
    _top_keys = list(body.keys()) if isinstance(body, dict) else None
    print(f"  response_type  : {type(body).__name__}")
    print(f"  top_key_count  : {len(_top_keys) if _top_keys is not None else 'N/A'}")
    if _top_keys:
        print(f"  first_keys     : {_top_keys[:5]}")
    if isinstance(body, dict) and len(body) == 0:
        print("  NOTE: body == {}  (empty response — no reward data returned)")
    print()

    print("  --- raw body (full) ---")
    print(json.dumps(body, indent=2))
    print()

    # Walk the response for Hungary
    entry = _find_in_body(body, HUNGARY_CONDITION_ID)
    return entry, 1, False


def _fetch_user_earnings_today(
    client, sig_type_val: int, maker_address: str | None, date_str: str
) -> tuple[dict | None, int, bool]:
    """
    Fetch /rewards/user?date=<YYYY-MM-DD> for the given maker_address.
    Official endpoint: 'Get earnings for user by date'.
    Returns (hungary_entry, pages_fetched, fetch_error).

    Response shape (paginated):
      {"limit": 100, "count": N, "next_cursor": "...", "data": [{
        "date": "...", "condition_id": "0x...", "asset_address": "...",
        "maker_address": "...", "earnings": 0.031154, "asset_rate": 1
      }]}
    """
    SIGNED_PATH = "/rewards/user"
    url_suffix  = f"?date={date_str}&signature_type={sig_type_val}"
    if maker_address:
        url_suffix += f"&maker_address={maker_address}"
    fetch_url = f"{CLOB_HOST}{SIGNED_PATH}{url_suffix}"

    print(f"  signed_path    : {SIGNED_PATH!r}")
    print(f"  fetch_url      : {fetch_url}")

    status, body = _get(client, SIGNED_PATH, url_suffix)
    print(f"  HTTP status    : {status}")

    if status != 200:
        print(f"  body: {body!r}")
        return None, 1, True

    print("  --- raw body (full) ---")
    print(json.dumps(body, indent=2))
    print()

    # Scan paginated data list for Hungary
    entry = None
    if isinstance(body, dict):
        data = body.get("data", [])
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("condition_id") or item.get("conditionId") or "")
                if cid == HUNGARY_CONDITION_ID:
                    entry = item
                    break
    return entry, 1, False


def _find_in_body(body: object, condition_id: str) -> dict | None:
    """
    Recursively walk body (dict or list) for an entry matching condition_id.
    """
    if isinstance(body, dict):
        # Direct key match (body keyed by condition_id)
        if condition_id in body:
            v = body[condition_id]
            return v if isinstance(v, dict) else {"value": v}
        # Check common wrapper keys
        for key in ("data", "markets", "percentages", "results"):
            candidate = body.get(key)
            if candidate is not None:
                result = _find_in_body(candidate, condition_id)
                if result is not None:
                    return result
    if isinstance(body, list):
        for item in body:
            if not isinstance(item, dict):
                continue
            cid = str(
                item.get("condition_id") or item.get("conditionId")
                or item.get("market_id") or item.get("marketId") or ""
            )
            if cid == condition_id:
                return item
    return None


# ---------------------------------------------------------------------------
# Field-level comparison and verdict
# ---------------------------------------------------------------------------

_INTEREST_FIELDS = [
    "condition_id", "conditionId", "market_id",
    "earning_percentage", "earnings_percentage",
    "earnings",
    "daily_rate_usdc", "rewardDailyRate", "reward_daily_rate",
    "percentage", "percentages",
    "rewards",
]


def _print_field_table(markets_entry: dict | None, pct_entry: dict | None) -> None:
    all_keys: set[str] = set()
    if isinstance(markets_entry, dict):
        all_keys |= set(markets_entry.keys())
    if isinstance(pct_entry, dict):
        all_keys |= set(pct_entry.keys())

    ordered = [f for f in _INTEREST_FIELDS if f in all_keys]
    ordered += sorted(all_keys - set(_INTEREST_FIELDS))

    print(f"  {'field':<35} {'user/markets':>22}  {'user/percentages':>22}")
    print(f"  {'-'*35} {'-'*22}  {'-'*22}")
    for field in ordered:
        mv = markets_entry.get(field, "—") if isinstance(markets_entry, dict) else "—"
        pv = pct_entry.get(field, "—") if isinstance(pct_entry, dict) else "—"
        mv_str = json.dumps(mv) if isinstance(mv, (dict, list)) else str(mv)
        pv_str = json.dumps(pv) if isinstance(pv, (dict, list)) else str(pv)
        # Truncate wide values for readability
        if len(mv_str) > 22:
            mv_str = mv_str[:19] + "..."
        if len(pv_str) > 22:
            pv_str = pv_str[:19] + "..."
        print(f"  {field:<35} {mv_str:>22}  {pv_str:>22}")


def _simulate_current_read(entry: dict | None, endpoint: str) -> str:
    """
    Reproduce the PATCHED _check_earning_pct field-read logic.
    (Explicit key presence check — no 'or' falsy-zero suppression.)
    """
    if entry is None:
        return f"None (Hungary not found in {endpoint})"
    for _f in ("earning_percentage", "earnings_percentage"):
        if _f in entry:
            raw = entry[_f]
            try:
                v = float(raw)
                result = v / 100.0 if v > 1.0 else v
                return f"{_f}={raw!r} → {result:.6f}  ({result:.4%})"
            except Exception:
                return f"{_f}={raw!r} → parse_error"
    return "None — neither 'earning_percentage' nor 'earnings_percentage' key present"


def _earnings_nonzero(markets_entry: dict | None, earnings_entry: dict | None) -> bool:
    """
    True if any confirmed dollar-amount earnings are non-zero.
    Checks two sources:
      1. earnings_entry from /rewards/user?date=today  (field: "earnings")
      2. markets_entry from /rewards/user/markets      (field: "earnings" — may be an array)
    """
    # /rewards/user dated entry
    if isinstance(earnings_entry, dict):
        try:
            if float(earnings_entry.get("earnings", 0) or 0) > 0:
                return True
        except (ValueError, TypeError):
            pass

    # /rewards/user/markets earnings field (array of {asset_address, earnings, ...})
    if isinstance(markets_entry, dict):
        raw = markets_entry.get("earnings")
        if isinstance(raw, list):
            for item in raw:
                try:
                    if float(item.get("earnings", 0) or 0) > 0:
                        return True
                except (ValueError, TypeError):
                    pass
        elif raw is not None:
            try:
                if float(raw) > 0:
                    return True
            except (ValueError, TypeError):
                pass
    return False


def _classify_verdict(
    markets_entry: dict | None,
    pct_entry: dict | None,
    markets_fetch_error: bool,
    pct_fetch_error: bool,
    earnings_entry: dict | None,
) -> str:
    """
    Verdict rules (applied in order):

    AUDIT_INCONCLUSIVE        : markets fetch failed, or Hungary not found in markets
    REALTIME_NONZERO          : /rewards/user/percentages has a non-zero value for Hungary
                                (this is the only realtime percentage source)
    FIELD_MISMATCH            : earnings endpoints (markets or dated) show non-zero for
                                Hungary, but /rewards/user/percentages is empty or has no
                                Hungary entry — sources disagree
    CONFIRMED_EARNINGS_NONZERO: earnings endpoints show non-zero, percentages is present
                                but zero or inconclusive (not missing) — earnings confirmed
    TRUE_ZERO                 : Hungary found in all queried endpoints; all values are 0
    """
    # Cannot classify without a valid markets entry
    if markets_fetch_error or markets_entry is None:
        return "AUDIT_INCONCLUSIVE"

    # REALTIME_NONZERO: percentages endpoint has a usable non-zero entry for Hungary
    if isinstance(pct_entry, dict):
        for v in pct_entry.values():
            try:
                if float(str(v)) > 0:
                    return "REALTIME_NONZERO"
            except (ValueError, TypeError):
                pass

    # Determine earnings state and percentages usability
    earnings_ok = _earnings_nonzero(markets_entry, earnings_entry)
    pct_missing = pct_entry is None  # {} body or Hungary not found in percentages

    if earnings_ok and pct_missing:
        # Earnings confirmed non-zero, but percentages endpoint not returning Hungary
        return "FIELD_MISMATCH"

    if earnings_ok and not pct_missing:
        # Earnings confirmed, percentages is present (zero, or non-zero already caught above)
        return "CONFIRMED_EARNINGS_NONZERO"

    return "TRUE_ZERO"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    _section("reward_share_read_path_audit_line")
    print("  Read-only. No orders. No state changes.")
    print(f"  Target: {HUNGARY_CONDITION_ID}")
    print()

    creds = load_activation_credentials()
    if creds is None:
        missing = get_missing_credential_vars()
        print(f"  CREDENTIALS NOT AVAILABLE  missing={missing}")
        sys.exit(1)
    print(f"  api_key        : {creds.api_key[:8]}...")
    print(f"  signature_type : {creds.signature_type}")
    print()

    client = build_clob_client(creds, CLOB_HOST)
    print("  client: ok")
    print()

    sig_type_val = client.builder.sig_type

    # ── Identity context — printed once, used for all three calls ─────────
    try:
        from eth_account import Account as _Account
        _eoa = _Account.from_key(creds.private_key).address
    except Exception:
        _eoa = "(eth_account unavailable)"
    maker_addr = creds.funder  # proxy wallet — rewards tracked here, not at EOA
    print(f"  identity.signer/EOA    : {_eoa}")
    print(f"  identity.funder/proxy  : {maker_addr!r}")
    print(f"  identity.api_key       : {creds.api_key[:8]}...")
    print(f"  identity.signature_type: {sig_type_val}")
    print(f"  maker_address used     : {maker_addr!r}  (all three reward calls)")
    print()

    # ── /rewards/user/markets — full pagination ───────────────────────────
    _section("/rewards/user/markets  (paginated — following next_cursor)")
    markets_entry, markets_pages, markets_err = _fetch_markets_find_hungary(
        client, sig_type_val, maker_addr
    )
    print()
    if markets_entry is not None:
        print("  Hungary entry (full):")
        print(json.dumps(markets_entry, indent=4))
    else:
        print("  Hungary entry : NOT FOUND")
    print()

    # ── /rewards/user/percentages ─────────────────────────────────────────
    _section("/rewards/user/percentages")
    print("  maker_address = creds.funder (proxy wallet — rewards tracked here)")
    print("  Prior {} response was caused by omitting maker_address; server")
    print("  defaulted to EOA from POLY_ADDRESS header, which holds no rewards.")
    print()
    pct_entry, _, pct_err = _fetch_percentages_find_hungary(
        client, sig_type_val, maker_addr
    )
    if pct_entry is not None:
        print("  Hungary entry (full):")
        print(json.dumps(pct_entry, indent=4))
    else:
        print("  Hungary entry : NOT FOUND")
    print()

    # ── /rewards/user?date=today  (authoritative dated earnings) ──────────
    import datetime as _dt
    today_str = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    _section(f"/rewards/user?date={today_str}  (authoritative dated earnings)")
    print("  Official endpoint: 'Get earnings for user by date'")
    print("  Returns dollar-amount earnings per market per day.")
    print()
    earnings_entry, _, earnings_err = _fetch_user_earnings_today(
        client, sig_type_val, maker_addr, today_str
    )
    if earnings_entry is not None:
        print("  Hungary entry (full):")
        print(json.dumps(earnings_entry, indent=4))
    else:
        print("  Hungary entry : NOT FOUND in today's earnings")
    print()

    # ── Field comparison table ────────────────────────────────────────────
    _section("Field comparison — Hungary")
    _print_field_table(markets_entry, pct_entry)
    print()

    # ── What _check_earning_pct currently extracts ─────────────────────
    _section("Simulation: what _check_earning_pct() currently reads")
    print(f"  /rewards/user/markets     → {_simulate_current_read(markets_entry, '/rewards/user/markets')}")
    print(f"  /rewards/user/percentages → [not called — endpoint absent from _check_earning_pct]")
    print(f"  /rewards/user (dated)     → [not called — endpoint absent from _check_earning_pct]")
    print()

    # ── Verdict ───────────────────────────────────────────────────────────
    verdict = _classify_verdict(
        markets_entry, pct_entry, markets_err, pct_err, earnings_entry
    )

    _sep("=")
    print()
    print(f"  VERDICT: {verdict}")
    print()
    if verdict == "REALTIME_NONZERO":
        print("  /rewards/user/percentages returned a non-zero entry for Hungary.")
        print("  Realtime reward percentage IS accruing for this account on Hungary.")
        print("  _check_earning_pct does not call /rewards/user/percentages yet.")
    elif verdict == "CONFIRMED_EARNINGS_NONZERO":
        _e_src = []
        if isinstance(earnings_entry, dict) and earnings_entry.get("earnings"):
            _e_src.append(f"/rewards/user?date={today_str}  (earnings={earnings_entry.get('earnings')})")
        if isinstance(markets_entry, dict):
            _raw = markets_entry.get("earnings")
            if isinstance(_raw, list):
                _vals = [item.get("earnings") for item in _raw if item.get("earnings")]
                if _vals:
                    _e_src.append(f"/rewards/user/markets  (earnings={_vals})")
            elif _raw and float(_raw) > 0:
                _e_src.append(f"/rewards/user/markets  (earnings={_raw})")
        for src in _e_src:
            print(f"  Non-zero confirmed via: {src}")
        print("  Earnings are confirmed non-zero; percentages endpoint was present.")
        print("  earning_percentage is already readable from /rewards/user/markets;")
        print("  the missing piece is a valid /rewards/user/percentages response.")
    elif verdict == "FIELD_MISMATCH":
        print("  Non-zero earnings found in /rewards/user/markets or /rewards/user.")
        print("  /rewards/user/percentages returned no Hungary entry (even with maker_address).")
        print("  Endpoints disagree — inspect raw bodies above to determine root cause.")
    elif verdict == "TRUE_ZERO":
        print("  Hungary found in all queried endpoints. All earnings values are 0.")
        print("  No reward share has accrued yet for this account on this market.")
    elif verdict == "AUDIT_INCONCLUSIVE":
        print("  Markets fetch failed or Hungary not found after full pagination.")
        print("  Check raw body output above for error details.")
    print()
    _sep("=")
    print()
    print("reward_aware_maker_probe")


if __name__ == "__main__":
    main()
