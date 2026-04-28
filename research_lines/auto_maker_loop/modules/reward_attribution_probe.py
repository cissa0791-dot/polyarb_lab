"""
auto_maker_loop — reward_attribution_probe
polyarb_lab / research_lines / auto_maker_loop / modules

Temporal reward-row probe for a single market.

Fetches the FULL matched /rewards/user/markets row (all raw fields) at T_before
and at T_after_0s / T_after_60s / T_after_180s.

Key focus: the `earnings` field is a nested array of per-asset objects:
    [{"asset_address": "0x...", "earnings": "0.000123", "asset_rate": "100.0"}, ...]

sum_earnings(row) sums row["earnings"][i]["earnings"] across all entries.
This is the dollar-denominated accumulator the UI reward panel reflects.

earning_percentage is a competition-share scalar (0-100 or 0-1), NOT the dollar
amount.  It is still captured for completeness but is NOT the primary target.

Zero-handling rule: all numeric extraction uses key-presence checks (if k in d),
never `or`-chaining on values.  `0` is a valid, meaningful reward value.

Public interface
----------------
    fetch_row(host, creds, condition_id) -> Optional[dict]
    probe_after(host, creds, condition_id, delays_sec=(0, 60, 180)) -> list[ProbeSnapshot]
    compute_deltas(before_row, probe_snapshots) -> dict
    sum_earnings(row) -> Optional[float]
    print_probe_table(before_row, probe_snapshots, deltas) -> None

Output schema — compute_deltas return dict
------------------------------------------
    "T_after_Xs": {
        "delay_sec":              int,
        "ts":                     str,
        "error":                  str|None,
        "numeric_deltas":         {field: delta},   # top-level scalars only
        "total_earnings_before":  float|None,
        "total_earnings_after":   float|None,
        "earnings_delta":         float|None,
        "earnings_entries_before": list[dict],       # raw per-asset entries
        "earnings_entries_after":  list[dict],
    },
    "max_earning_pct_delta":      float,
    "max_earning_pct_delay_sec":  int|None,
    "max_total_earnings_delta":   float,
    "max_earnings_delta_at_sec":  int|None,
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PATH            = "/rewards/user/markets"
_PATH_USER_TOTAL = "/rewards/user/total"


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class ProbeSnapshot:
    delay_sec:  int
    ts:         str
    row:        Optional[dict]
    error:      Optional[str]


# ---------------------------------------------------------------------------
# Public: earnings array helper
# ---------------------------------------------------------------------------

def sum_earnings(row: Optional[dict]) -> Optional[float]:
    """
    Sum row["earnings"][i]["earnings"] across all asset entries.

    Returns None if the field is absent or empty (distinguishable from 0.0
    which means the array is present but all entries are zero).
    """
    if not row:
        return None
    raw = row.get("earnings")
    if not isinstance(raw, list) or not raw:
        return None
    total = 0.0
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        v = entry.get("earnings")
        if v is None:
            continue
        try:
            total += float(v)
        except (TypeError, ValueError):
            pass
    return round(total, 8)


# ---------------------------------------------------------------------------
# Public: fetch / probe
# ---------------------------------------------------------------------------

def fetch_row(
    host: str,
    creds: Any,
    condition_id: str,
) -> Optional[dict]:
    """
    Fetch the FULL matched /rewards/user/markets row.
    Paginates up to 20 pages.  Returns raw entry dict on match, None otherwise.
    """
    try:
        import httpx
        url          = f"{host.rstrip('/')}{_PATH}"
        _norm_target = _norm_cid(condition_id)

        with httpx.Client() as hc:
            resp = hc.get(url, headers=_make_headers(creds), timeout=8)
            if resp.status_code != 200:
                logger.warning(
                    "reward_probe: HTTP %d fetching %s — body: %.300s",
                    resp.status_code, url, resp.text,
                )
                return None
            body    = resp.json()
            entries = body.get("data", body) if isinstance(body, dict) else []
            if isinstance(entries, list):
                hit = _find_row(entries, _norm_target, condition_id)
                if hit is not None:
                    return hit

            cursor = body.get("next_cursor", "") if isinstance(body, dict) else ""
            pages  = 1
            while cursor and cursor not in ("", "LTE=") and pages < 20:
                resp2 = hc.get(
                    url,
                    params={"next_cursor": cursor},
                    headers=_make_headers(creds),
                    timeout=8,
                )
                if resp2.status_code != 200:
                    logger.warning(
                        "reward_probe: pagination HTTP %d page=%d", resp2.status_code, pages + 1
                    )
                    break
                body2    = resp2.json()
                entries2 = body2.get("data", body2) if isinstance(body2, dict) else []
                if isinstance(entries2, list):
                    hit = _find_row(entries2, _norm_target, condition_id)
                    if hit is not None:
                        logger.debug(
                            "reward_probe: found condition_id %s on page %d",
                            condition_id[:24], pages + 1,
                        )
                        return hit
                cursor = body2.get("next_cursor", "") if isinstance(body2, dict) else ""
                pages += 1

        logger.warning(
            "reward_probe: condition_id %s not found after %d page(s)",
            condition_id[:24], pages,
        )
        return None

    except Exception as exc:
        logger.warning("reward_probe: fetch_row failed: %s", exc)
        return None


def probe_after(
    host: str,
    creds: Any,
    condition_id: str,
    delays_sec: tuple[int, ...] = (0, 60, 180),
) -> list[ProbeSnapshot]:
    """
    Take timed snapshots after cycle end.  First probe fires immediately (delay=0).
    Each subsequent probe sleeps (this_delay - prev_delay) seconds before firing.
    Total elapsed ≈ max(delays_sec) seconds.
    """
    snapshots: list[ProbeSnapshot] = []
    prev_delay = 0

    for delay in delays_sec:
        wait = delay - prev_delay
        if wait > 0:
            logger.info(
                "reward_probe: sleeping %ds before T_after_%ds probe...", wait, delay
            )
            time.sleep(wait)
        prev_delay = delay

        ts  = datetime.now(timezone.utc).isoformat()
        row = None
        err = None
        try:
            row = fetch_row(host, creds, condition_id)
            if row is None:
                err = "not_found"
        except Exception as exc:
            err = str(exc)

        snapshots.append(ProbeSnapshot(
            delay_sec=delay, ts=ts, row=row, error=err,
        ))

    return snapshots


# ---------------------------------------------------------------------------
# Public: delta computation
# ---------------------------------------------------------------------------

def compute_deltas(
    before_row: Optional[dict],
    probe_snapshots: list[ProbeSnapshot],
) -> dict:
    """
    Compute (after - before) for:
      - every top-level numeric scalar field
      - sum(earnings[].earnings) — the dollar accumulator

    Returns a structured dict safe to store in runs.jsonl.
    """
    result: dict = {}
    max_ep_delta    = 0.0
    max_ep_delay    = None
    max_earn_delta  = 0.0
    max_earn_delay  = None

    earn_before = sum_earnings(before_row)

    for snap in probe_snapshots:
        key = f"T_after_{snap.delay_sec}s"

        # ── earnings array ────────────────────────────────────────────────
        earn_after = sum_earnings(snap.row)
        earn_delta: Optional[float] = None
        if earn_before is not None and earn_after is not None:
            earn_delta = round(earn_after - earn_before, 8)
            if abs(earn_delta) > abs(max_earn_delta):
                max_earn_delta = earn_delta
                max_earn_delay = snap.delay_sec

        # ── top-level numeric scalars ─────────────────────────────────────
        numeric_deltas: dict = {}
        if before_row is not None and snap.row is not None:
            for field_name, after_val in snap.row.items():
                if isinstance(after_val, (list, dict)):
                    continue   # skip nested structures — handled separately
                if field_name not in before_row:
                    continue
                before_val = before_row[field_name]
                if isinstance(before_val, (list, dict)):
                    continue
                try:
                    delta = float(after_val) - float(before_val)
                    numeric_deltas[field_name] = round(delta, 8)
                    if field_name in ("earning_percentage", "earnings_percentage"):
                        if abs(delta) > abs(max_ep_delta):
                            max_ep_delta = delta
                            max_ep_delay = snap.delay_sec
                except (TypeError, ValueError):
                    pass

        result[key] = {
            "delay_sec":               snap.delay_sec,
            "ts":                      snap.ts,
            "error":                   snap.error,
            "numeric_deltas":          numeric_deltas,
            "total_earnings_before":   earn_before,
            "total_earnings_after":    earn_after,
            "earnings_delta":          earn_delta,
            "earnings_entries_before": _get_earnings_entries(before_row),
            "earnings_entries_after":  _get_earnings_entries(snap.row),
        }

    result["max_earning_pct_delta"]     = round(max_ep_delta, 8)
    result["max_earning_pct_delay_sec"] = max_ep_delay
    result["max_total_earnings_delta"]  = round(max_earn_delta, 8)
    result["max_earnings_delta_at_sec"] = max_earn_delay
    return result


# ---------------------------------------------------------------------------
# Public: print
# ---------------------------------------------------------------------------

def print_probe_table(
    before_row: Optional[dict],
    probe_snapshots: list[ProbeSnapshot],
    deltas: dict,
) -> None:
    """
    Print temporal probe results.

    Section 1 — scalar fields table (earning_percentage, spread, etc.)
    Section 2 — earnings[] array expansion (asset_address, earnings, asset_rate)
    Section 3 — delta summary
    """
    col_labels = ["T_before"] + [f"T+{s.delay_sec}s" for s in probe_snapshots]
    n_cols     = len(col_labels)
    W_FIELD    = 30
    W_VAL      = 16

    print("\n" + "=" * (W_FIELD + W_VAL * n_cols + 2))
    print("  REWARD ATTRIBUTION PROBE — earnings[] expansion")
    print("=" * (W_FIELD + W_VAL * n_cols + 2))

    # ── Section 1: scalar fields ──────────────────────────────────────────
    print(f"\n  {'Field':<{W_FIELD}}" + "".join(f"{lbl:>{W_VAL}}" for lbl in col_labels))
    print("  " + "-" * (W_FIELD + W_VAL * n_cols))

    # Collect scalar field names from any snapshot
    scalar_fields: list[str] = []
    seen: set[str] = set()
    for src in ([before_row] + [s.row for s in probe_snapshots if s.row]):
        if not src:
            continue
        for k, v in src.items():
            if k not in seen and not isinstance(v, (list, dict)):
                scalar_fields.append(k)
                seen.add(k)

    for fn in scalar_fields:
        before_val = _get_scalar(before_row, fn)
        row_vals   = [
            (_get_scalar(s.row, fn) if s.row else f"<{s.error}>")
            for s in probe_snapshots
        ]
        all_vals = [before_val] + row_vals
        row_str  = f"  {fn:<{W_FIELD}}" + "".join(f"{_fmt_scalar(v):>{W_VAL}}" for v in all_vals)
        print(row_str)

    # Mark earnings[] as array in scalar table
    if before_row and "earnings" in before_row and isinstance(before_row["earnings"], list):
        n = len(before_row["earnings"])
        array_marker = f"[array:{n}]"
        print(f"  {'earnings':<{W_FIELD}}" + "".join(f"{array_marker:>{W_VAL}}" for _ in col_labels))

    print("  " + "-" * (W_FIELD + W_VAL * n_cols))

    # ── Section 2: earnings[] expansion ──────────────────────────────────
    print("\n  EARNINGS ARRAY EXPANSION")
    print("  " + "-" * (W_FIELD + W_VAL * n_cols))

    all_sources = [before_row] + [s.row for s in probe_snapshots]
    all_asset_addrs: list[str] = []
    seen_addr: set[str] = set()
    for src in all_sources:
        for entry in _get_earnings_entries(src):
            addr = str(entry.get("asset_address") or "")
            if addr and addr not in seen_addr:
                all_asset_addrs.append(addr)
                seen_addr.add(addr)

    if not all_asset_addrs:
        print("  (no earnings[] entries found in any snapshot)")
    else:
        for addr in all_asset_addrs:
            short_addr = addr[:10] + "…" + addr[-6:] if len(addr) > 18 else addr
            for sub_field in ("earnings", "asset_rate"):
                label = f"  {short_addr}.{sub_field}"
                vals  = []
                for src in all_sources:
                    entry = _find_earnings_entry(src, addr)
                    if entry is None:
                        vals.append("—")
                    else:
                        v = entry.get(sub_field)
                        vals.append(_fmt_scalar(v))
                print(f"  {label:<{W_FIELD}}" + "".join(f"{v:>{W_VAL}}" for v in vals))

    # Totals row
    total_vals: list[str] = []
    for src in all_sources:
        t = sum_earnings(src)
        total_vals.append(f"{t:.8f}" if t is not None else "—")
    print(f"  {'TOTAL earnings':<{W_FIELD}}" + "".join(f"{v:>{W_VAL}}" for v in total_vals))

    print("  " + "-" * (W_FIELD + W_VAL * n_cols))

    # ── Section 3: delta summary ──────────────────────────────────────────
    print("\n  DELTA SUMMARY (vs T_before)")
    for snap in probe_snapshots:
        key  = f"T_after_{snap.delay_sec}s"
        d    = deltas.get(key, {})
        ed   = d.get("earnings_delta")
        eb   = d.get("total_earnings_before")
        ea   = d.get("total_earnings_after")
        nd   = {k: v for k, v in d.get("numeric_deltas", {}).items() if v != 0.0}

        ed_str = f"{ed:+.8f}" if ed is not None else "N/A"
        ea_str = f"{ea:.8f}" if ea is not None else "N/A"
        print(
            f"    T+{snap.delay_sec:3d}s  "
            f"total_earnings_before={eb!r}  "
            f"total_earnings_after={ea_str}  "
            f"earnings_delta={ed_str}"
        )
        if nd:
            for fn, dv in nd.items():
                print(f"             scalar: {fn}={dv:+.8f}")

    max_ed   = deltas.get("max_total_earnings_delta", 0.0)
    max_at   = deltas.get("max_earnings_delta_at_sec")
    max_ep   = deltas.get("max_earning_pct_delta", 0.0)
    max_ep_t = deltas.get("max_earning_pct_delay_sec")
    print(f"\n  max_total_earnings_delta   : {max_ed:+.8f}  "
          f"(at T+{max_at}s)" if max_at is not None
          else f"\n  max_total_earnings_delta   : {max_ed:+.8f}  (no change)")
    print(f"  max_earning_pct_delta      : {max_ep:+.8f}  "
          f"(at T+{max_ep_t}s)" if max_ep_t is not None
          else f"  max_earning_pct_delta      : {max_ep:+.8f}  (no change)")
    print("=" * (W_FIELD + W_VAL * n_cols + 2) + "\n")


# ---------------------------------------------------------------------------
# Account-wide (dual-scope) probe — new section
# ---------------------------------------------------------------------------

@dataclass
class CombinedSnapshot:
    """Holds one time-point sample of both Hungary row and account-wide totals."""
    delay_sec:        int
    ts:               str
    hungary_row:      Optional[dict]   # matched Hungary row (or None)
    account_total:    Optional[float]  # sum of earnings[] across ALL market rows
    account_rows_n:   int              # number of market rows fetched
    error:            Optional[str]    # concatenated errors from both fetches


def fetch_all_rows(
    host: str,
    creds: Any,
) -> tuple[list, Optional[str]]:
    """
    Fetch ALL rows from /rewards/user/markets — no early exit.

    Returns (all_rows, error_str).  Paginates up to 20 pages.
    Zero-handling: returns an empty list on error, never None.
    """
    try:
        import httpx
        url      = f"{host.rstrip('/')}{_PATH}"
        all_rows: list = []

        with httpx.Client() as hc:
            resp = hc.get(url, headers=_make_headers(creds), timeout=8)
            if resp.status_code != 200:
                return [], f"HTTP_{resp.status_code}"
            body    = resp.json()
            entries = body.get("data", body) if isinstance(body, dict) else []
            if isinstance(entries, list):
                all_rows.extend(e for e in entries if isinstance(e, dict))

            cursor = body.get("next_cursor", "") if isinstance(body, dict) else ""
            pages  = 1
            while cursor and cursor not in ("", "LTE=") and pages < 20:
                resp2 = hc.get(
                    url,
                    params={"next_cursor": cursor},
                    headers=_make_headers(creds),
                    timeout=8,
                )
                if resp2.status_code != 200:
                    return all_rows, f"pagination_HTTP_{resp2.status_code}_page{pages+1}"
                body2    = resp2.json()
                entries2 = body2.get("data", body2) if isinstance(body2, dict) else []
                if isinstance(entries2, list):
                    all_rows.extend(e for e in entries2 if isinstance(e, dict))
                cursor = body2.get("next_cursor", "") if isinstance(body2, dict) else ""
                pages += 1

        return all_rows, None
    except Exception as exc:
        return [], str(exc)


def sum_earnings_all(all_rows: list) -> Optional[float]:
    """
    Sum earnings across ALL market rows.

    Returns None  — list is empty (no rows fetched at all).
    Returns 0.0   — rows fetched but no earnings[] arrays present OR all zero.
    Returns float — actual sum.

    Zero-handling: 0.0 is meaningful (present-and-zero), never coerced to None.
    """
    if not all_rows:
        return None
    total    = 0.0
    any_data = False
    for row in all_rows:
        v = sum_earnings(row)
        if v is not None:   # v=0.0 means earnings[] was present but zero — valid
            total    += v
            any_data  = True
    # If rows were present but none had an earnings[] list, return 0.0 (not None)
    # so downstream can still compute a meaningful delta.
    return round(total, 8)


def probe_combined_after(
    host: str,
    creds: Any,
    condition_id: str,
    delays_sec: tuple = (0, 60, 180),
) -> list:
    """
    At each time-point: fetch Hungary row (early-exit) + fetch ALL rows (account total).
    Replaces separate probe_after + account fetch calls so there is only one sleep schedule.
    Returns list[CombinedSnapshot].
    """
    snapshots: list = []
    prev_delay = 0

    for delay in delays_sec:
        wait = delay - prev_delay
        if wait > 0:
            logger.info(
                "reward_probe: sleeping %ds before T_after_%ds combined probe...", wait, delay
            )
            time.sleep(wait)
        prev_delay = delay

        ts       = datetime.now(timezone.utc).isoformat()
        h_row    = None
        a_total  = None
        a_rows_n = 0
        err_parts: list = []

        # Hungary row (fast — early exit)
        try:
            h_row = fetch_row(host, creds, condition_id)
            if h_row is None:
                err_parts.append("hungary_not_found")
        except Exception as exc:
            err_parts.append(f"hungary:{exc}")

        # Account-wide (full pagination)
        try:
            a_rows, a_err = fetch_all_rows(host, creds)
            a_rows_n = len(a_rows)
            a_total  = sum_earnings_all(a_rows)
            if a_err:
                err_parts.append(f"account:{a_err}")
        except Exception as exc:
            err_parts.append(f"account:{exc}")

        snapshots.append(CombinedSnapshot(
            delay_sec=delay,
            ts=ts,
            hungary_row=h_row,
            account_total=a_total,
            account_rows_n=a_rows_n,
            error="; ".join(err_parts) if err_parts else None,
        ))

    return snapshots


def compute_combined_deltas(
    hungary_before_row: Optional[dict],
    account_total_before: Optional[float],
    combined_probes: list,
) -> dict:
    """
    Compute per-probe deltas for both Hungary and account scope.
    Classifies the final outcome into one of 4 states.
    """
    result: dict = {}
    max_h_delta = 0.0;  max_h_delay = None
    max_a_delta = 0.0;  max_a_delay = None

    h_earn_before = sum_earnings(hungary_before_row)

    h_earn_afters: list = []
    a_earn_afters: list = []

    for snap in combined_probes:
        key = f"T_after_{snap.delay_sec}s"

        h_earn_after = sum_earnings(snap.hungary_row)
        h_earn_delta: Optional[float] = None
        if h_earn_before is not None and h_earn_after is not None:
            h_earn_delta = round(h_earn_after - h_earn_before, 8)
            if abs(h_earn_delta) > abs(max_h_delta):
                max_h_delta = h_earn_delta
                max_h_delay = snap.delay_sec
        if h_earn_after is not None:
            h_earn_afters.append(h_earn_after)

        a_earn_delta: Optional[float] = None
        if account_total_before is not None and snap.account_total is not None:
            a_earn_delta = round(snap.account_total - account_total_before, 8)
            if abs(a_earn_delta) > abs(max_a_delta):
                max_a_delta = a_earn_delta
                max_a_delay = snap.delay_sec
        if snap.account_total is not None:
            a_earn_afters.append(snap.account_total)

        result[key] = {
            "delay_sec":           snap.delay_sec,
            "ts":                  snap.ts,
            "error":               snap.error,
            "account_rows_n":      snap.account_rows_n,
            "hungary_earn_before": h_earn_before,
            "hungary_earn_after":  h_earn_after,
            "hungary_earn_delta":  h_earn_delta,
            "account_earn_before": account_total_before,
            "account_earn_after":  snap.account_total,
            "account_earn_delta":  a_earn_delta,
            # Keep legacy field names for result record backwards compat
            "total_earnings_before": h_earn_before,
            "total_earnings_after":  h_earn_after,
            "earnings_delta":        h_earn_delta,
        }

    # Classification uses max observed values (best-case across all probes)
    h_max_after = max(h_earn_afters) if h_earn_afters else h_earn_before
    a_max_after = max(a_earn_afters) if a_earn_afters else account_total_before
    outcome = classify_outcome(h_earn_before, h_max_after, account_total_before, a_max_after)

    result["max_hungary_earn_delta"]     = round(max_h_delta, 8)
    result["max_hungary_earn_delay_sec"] = max_h_delay
    result["max_account_earn_delta"]     = round(max_a_delta, 8)
    result["max_account_earn_delay_sec"] = max_a_delay
    result["attribution_outcome"]        = outcome
    # Legacy keys for backwards compat with existing result record reads
    result["max_total_earnings_delta"]   = round(max_h_delta, 8)
    result["max_earnings_delta_at_sec"]  = max_h_delay
    result["max_earning_pct_delta"]      = 0.0
    result["max_earning_pct_delay_sec"]  = None
    return result


def classify_outcome(
    h_before:    Optional[float],
    h_max_after: Optional[float],
    a_before:    Optional[float],
    a_max_after: Optional[float],
) -> str:
    """
    Classify attribution result into one of four outcomes.

    HUNGARY_UP_ACCOUNT_UP     — both moved: Hungary is contributing to account rewards
    HUNGARY_ZERO_ACCOUNT_UP   — account moved but Hungary did not: other markets driving
    HUNGARY_ZERO_ACCOUNT_ZERO — nothing moved within observation window: lag > 3 min, or
                                  rewards settled before cycle started
    FIELD_OR_ENDPOINT_STILL_WRONG — one or both scopes returned None (data missing)

    Threshold: 1e-6 USDC (sub-micro-cent) to filter float noise.
    Zero-handling: 0.0 is treated as "did not move", not as "data missing".
    """
    _THRESH = 1e-6

    if h_before is None or a_before is None:
        return "FIELD_OR_ENDPOINT_STILL_WRONG"
    if h_max_after is None or a_max_after is None:
        return "FIELD_OR_ENDPOINT_STILL_WRONG"

    h_delta = h_max_after - h_before
    a_delta = a_max_after - a_before

    h_up = h_delta > _THRESH
    a_up = a_delta > _THRESH

    if h_up and a_up:
        return "HUNGARY_UP_ACCOUNT_UP"
    if not h_up and a_up:
        return "HUNGARY_ZERO_ACCOUNT_UP"
    # h_up and not a_up: impossible in practice (Hungary is a subset of account)
    return "HUNGARY_ZERO_ACCOUNT_ZERO"


def print_combined_table(
    hungary_before_row: Optional[dict],
    account_total_before: Optional[float],
    combined_probes: list,
    combined_deltas: dict,
) -> None:
    """
    Print dual-scope attribution table.

    Section A — MARKET PROBE: Hungary earnings[] entries + totals across time
    Section B — ACCOUNT PROBE: account-wide total + deltas across time
    Section C — Classification verdict
    """
    W = 74
    col_labels = ["T_before"] + [f"T+{s.delay_sec}s" for s in combined_probes]
    W_F, W_V   = 30, 14

    print("\n" + "=" * W)
    print("  REWARD ATTRIBUTION PROBE — dual scope")
    print("=" * W)

    # ── Section A: Hungary market probe ───────────────────────────────────
    print("\n  A. MARKET PROBE — Hungary only")
    print("  " + "-" * (W - 2))

    h_sources = [hungary_before_row] + [s.hungary_row for s in combined_probes]

    all_addrs: list = []
    seen_addr: set  = set()
    for src in h_sources:
        for e in _get_earnings_entries(src):
            addr = str(e.get("asset_address") or "")
            if addr and addr not in seen_addr:
                all_addrs.append(addr)
                seen_addr.add(addr)

    if not all_addrs:
        print("  (no earnings[] entries in Hungary row at any time point)")
    else:
        for addr in all_addrs:
            short = addr[:10] + "…" + addr[-6:] if len(addr) > 18 else addr
            for sub in ("earnings", "asset_rate"):
                vals = []
                for src in h_sources:
                    e = _find_earnings_entry(src, addr)
                    vals.append(_fmt_scalar(e.get(sub) if e else None))
                lbl = f"{short}.{sub}"
                print(f"  {lbl:<{W_F}}" + "".join(f"{v:>{W_V}}" for v in vals))

    h_totals = []
    for src in h_sources:
        t = sum_earnings(src)
        h_totals.append(f"{t:.8f}" if t is not None else "—")
    print(f"  {'TOTAL earnings':<{W_F}}" + "".join(f"{v:>{W_V}}" for v in h_totals))

    print("\n  Hungary earnings deltas:")
    for snap in combined_probes:
        key = f"T_after_{snap.delay_sec}s"
        d   = combined_deltas.get(key, {})
        hd  = d.get("hungary_earn_delta")
        print(f"    T+{snap.delay_sec:3d}s  hungary_earnings_delta={hd!r}")

    # ── Section B: Account-wide probe ─────────────────────────────────────
    print(f"\n  B. ACCOUNT PROBE — all markets aggregated")
    print("  " + "-" * (W - 2))

    print(f"  {'Metric':<{W_F}}" + "".join(f"{lbl:>{W_V}}" for lbl in col_labels))
    print("  " + "-" * (W_F + W_V * len(col_labels)))

    a_vals = [f"{account_total_before:.8f}" if account_total_before is not None else "—"]
    n_vals = ["—"]
    for snap in combined_probes:
        a_vals.append(f"{snap.account_total:.8f}" if snap.account_total is not None else "—")
        n_vals.append(str(snap.account_rows_n))
    print(f"  {'account_total_earnings':<{W_F}}" + "".join(f"{v:>{W_V}}" for v in a_vals))
    print(f"  {'rows_fetched':<{W_F}}" + "".join(f"{v:>{W_V}}" for v in n_vals))

    print("\n  Account earnings deltas:")
    for snap in combined_probes:
        key  = f"T_after_{snap.delay_sec}s"
        d    = combined_deltas.get(key, {})
        ad   = d.get("account_earn_delta")
        rows = d.get("account_rows_n", 0)
        err  = d.get("error")
        err_s = f"  err={err}" if err else ""
        print(f"    T+{snap.delay_sec:3d}s  account_earnings_delta={ad!r}  rows={rows}{err_s}")

    # ── Section C: Classification ─────────────────────────────────────────
    print("\n  C. ATTRIBUTION CLASSIFICATION")
    print("  " + "-" * (W - 2))
    outcome   = combined_deltas.get("attribution_outcome", "UNKNOWN")
    max_h     = combined_deltas.get("max_hungary_earn_delta", 0.0)
    max_h_t   = combined_deltas.get("max_hungary_earn_delay_sec")
    max_a     = combined_deltas.get("max_account_earn_delta", 0.0)
    max_a_t   = combined_deltas.get("max_account_earn_delay_sec")
    print(f"\n  ▶ OUTCOME                   : {outcome}")
    print(f"  max_hungary_earn_delta    : {max_h:+.8f}  (at T+{max_h_t}s)" if max_h_t is not None
          else f"  max_hungary_earn_delta    : {max_h:+.8f}  (no change in 3min window)")
    print(f"  max_account_earn_delta    : {max_a:+.8f}  (at T+{max_a_t}s)" if max_a_t is not None
          else f"  max_account_earn_delta    : {max_a:+.8f}  (no change in 3min window)")

    _VERDICT = {
        "HUNGARY_UP_ACCOUNT_UP":
            "Hungary contributing — reward IS accruing to this market within 3min",
        "HUNGARY_ZERO_ACCOUNT_UP":
            "Other markets driving account reward — Hungary lag > 3min OR not current earner",
        "HUNGARY_ZERO_ACCOUNT_ZERO":
            "No reward movement in 3min window — rewards may settle on epoch boundary (daily)",
        "FIELD_OR_ENDPOINT_STILL_WRONG":
            "Data missing — endpoint or field still not returning usable numeric values",
    }
    print(f"  interpretation            : {_VERDICT.get(outcome, 'unknown')}")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_headers_for(creds: Any, path: str) -> dict[str, str]:
    """Build HMAC L2 auth headers for any GET path.

    HMAC message: timestamp + "GET" + path (no query params — matches proven
    pattern from /rewards/user/markets where next_cursor param is NOT in msg).
    Includes CLOB-SIGNATURE-TYPE when creds exposes signature_type.
    """
    import base64
    import hashlib
    import hmac as _hmac

    ts  = str(int(time.time() * 1000))
    msg = ts + "GET" + path
    try:
        hmac_key = base64.urlsafe_b64decode(creds.api_secret)
    except Exception:
        hmac_key = creds.api_secret.encode("utf-8")
    sig = base64.b64encode(
        _hmac.new(hmac_key, msg.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    headers = {
        "CLOB-API-KEY":    creds.api_key,
        "CLOB-SIGNATURE":  sig,
        "CLOB-TIMESTAMP":  ts,
        "CLOB-PASSPHRASE": creds.api_passphrase,
    }
    return headers


# ---------------------------------------------------------------------------
# Public: /rewards/user/total — account-level UI Daily Rewards surface
# ---------------------------------------------------------------------------

def fetch_user_total(
    host: str,
    creds: Any,
    date: Optional[str] = None,
) -> tuple[Optional[float], Optional[str]]:
    """
    Fetch GET /rewards/user/total.

    Uses base-path HMAC auth only. Query params and extra identity headers are
    intentionally omitted because this endpoint follows the same proven shape
    as /rewards/user/markets.

    Zero-safe: key-presence checks only; 0.0 stays 0.0, never coerced to None.
    """
    try:
        import httpx

        path     = _PATH_USER_TOTAL
        url      = f"{host.rstrip('/')}{path}"

        with httpx.Client() as hc:
            resp = hc.get(url, headers=_make_headers_for(creds, path), timeout=8)
            if resp.status_code != 200:
                return None, f"HTTP_{resp.status_code}:{resp.text[:120]}"
            raw = resp.json()
            return _sum_user_total_response(raw), str(raw)[:400]
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _sum_user_total_response(raw: Any) -> Optional[float]:
    """
    Sum earnings from /rewards/user/total response.

    Handles shapes:
      - list of {earnings: ...} dicts          (most likely)
      - {"data": [{earnings: ...}, ...]}       (wrapped)
      - flat {"earnings": ...}                 (single-entry)
      - {"total_earnings": ...}                (aggregate field)

    Zero-safe: key-presence only. Returns None only if no numeric field found.
    """
    if raw is None:
        return None

    # Unwrap envelope
    if isinstance(raw, dict):
        # Try aggregate field first
        for agg_key in ("total_earnings", "totalEarnings", "total"):
            if agg_key in raw:
                try:
                    return round(float(raw[agg_key]), 8)
                except (TypeError, ValueError):
                    pass
        entries: Any = raw.get("data", [raw])
    elif isinstance(raw, list):
        entries = raw
    else:
        return None

    if not isinstance(entries, list):
        entries = [entries]

    total     = 0.0
    found_any = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field_name in ("earnings", "total_earnings", "totalEarnings", "amount", "reward"):
            if field_name in entry:
                try:
                    total    += float(entry[field_name])
                    found_any = True
                except (TypeError, ValueError):
                    pass
                break  # one field per entry
    return round(total, 8) if found_any else None


@dataclass
class UserTotalSnapshot:
    delay_sec: int
    ts:        str
    total_usd: Optional[float]
    raw_str:   Optional[str]
    error:     Optional[str]


def probe_user_total_after(
    host: str,
    creds: Any,
    delays_sec: tuple[int, ...] = (0, 60, 180),
    date: Optional[str] = None,
) -> list[UserTotalSnapshot]:
    """
    Take timed /rewards/user/total snapshots after cycle end.

    First probe fires immediately (delay=0).  Each subsequent probe sleeps
    (this_delay - prev_delay) seconds.  Total elapsed ≈ max(delays_sec) seconds.
    date: YYYY-MM-DD; defaults to today UTC at call time if not given.
    """
    # Pin the date at probe start so all snapshots use the same date param.
    probe_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshots: list[UserTotalSnapshot] = []
    prev_delay = 0

    for delay in delays_sec:
        wait = delay - prev_delay
        if wait > 0:
            logger.info(
                "user_total_probe: sleeping %ds before T_after_%ds probe...", wait, delay
            )
            time.sleep(wait)
        prev_delay = delay

        ts               = datetime.now(timezone.utc).isoformat()
        total_usd, raw_s = fetch_user_total(host, creds, date=probe_date)
        error            = raw_s if total_usd is None else None
        raw_out          = raw_s if total_usd is not None else None

        snapshots.append(UserTotalSnapshot(
            delay_sec=delay,
            ts=ts,
            total_usd=total_usd,
            raw_str=raw_out,
            error=error,
        ))
        logger.info(
            "user_total_probe: T_after_%ds total_usd=%r error=%s",
            delay, total_usd, error,
        )

    return snapshots


def _make_headers(creds: Any) -> dict[str, str]:
    import base64
    import hashlib
    import hmac as _hmac

    ts  = str(int(time.time() * 1000))
    msg = ts + "GET" + _PATH
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


def _norm_cid(cid: str) -> str:
    return cid.lower().lstrip("0x") if cid else ""


def _find_row(entries: list, norm_target: str, raw_cid: str) -> Optional[dict]:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cid = str(
            entry.get("condition_id") or entry.get("conditionId")
            or entry.get("market_id") or ""
        )
        if _norm_cid(cid) == norm_target or cid == raw_cid:
            return entry
    return None


def _get_earnings_entries(row: Optional[dict]) -> list[dict]:
    """Return row['earnings'] as a list of dicts.  Empty list if absent or wrong type."""
    if not row:
        return []
    raw = row.get("earnings")
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict)]


def _find_earnings_entry(row: Optional[dict], asset_address: str) -> Optional[dict]:
    """Return the earnings entry matching asset_address, or None."""
    for entry in _get_earnings_entries(row):
        if str(entry.get("asset_address") or "") == asset_address:
            return entry
    return None


def _get_scalar(row: Optional[dict], field: str) -> Any:
    if not row:
        return ""
    v = row.get(field, "")
    if isinstance(v, (list, dict)):
        return "[array]"
    return v


def _fmt_scalar(v: Any) -> str:
    if v == "" or v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6f}"
    s = str(v)
    return s[:14] if len(s) > 14 else s
