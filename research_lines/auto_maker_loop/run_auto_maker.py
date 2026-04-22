"""
auto_maker_loop — run_auto_maker
polyarb_lab / research_lines / auto_maker_loop

Continuous automated bilateral market-making loop for reward-eligible markets.

What it does each cycle
-----------------------
1.  Select best survivor market (hungary > rubio > vance by reward/competition score).
2.  Preflight: check YES inventory >= min_size (SELL leg requirement).
3.  Fetch live reward config + midpoint; compute qualifying 3-cent bilateral quotes.
4.  Place BID (buy YES) and ASK (sell YES) as GTC limit orders.
5.  Run position manager until position is DONE:
      - NATURAL:        ASK filled at full price — best outcome.
      - CHASE_FILL:     ASK filled after price was lowered.
      - TIME_LIMIT:     Held too long → taker-sell to force exit.
      - STOP_LOSS:      Midpoint dropped too far → taker-sell.
      - NO_FILL_TIME_LIMIT: Neither leg filled → both cancelled, cycle skipped.
6.  Record outcome to data/research/auto_maker_loop/runs.jsonl.
7.  Sleep cycle_sleep_sec, then repeat.

Edge improvements built in
--------------------------
- Live competitiveness refresh each cycle (updates market ranking dynamically)
- Auto-rotation: if preferred market inventory is zero, tries next-best market
- Chase logic: lowers ASK 1 tick after chase_after_minutes if still unfilled
- Stop-loss: forces exit if YES price drops stop_loss_cents below entry

Modes
-----
  --dry-run (default): compute quotes + print plan, no orders submitted
  --live              : submit real orders (requires POLYMARKET_PRIVATE_KEY)

Usage (PowerShell from repo root)
----------------------------------
  # Dry-run — safe, no orders
  py -3 research_lines/auto_maker_loop/run_auto_maker.py

  # Live — continuous loop targeting best market
  $env:POLYMARKET_PRIVATE_KEY = "<key>"
  py -3 research_lines/auto_maker_loop/run_auto_maker.py --live

  # Lock to one market, custom hold limit
  py -3 research_lines/auto_maker_loop/run_auto_maker.py --live --target hungary --max-hold-minutes 120

Required env vars
-----------------
  POLYMARKET_PRIVATE_KEY        EVM private key (only hard requirement)
  POLYMARKET_CHAIN_ID           137 (default)
  POLYMARKET_SIGNATURE_TYPE     0=EOA, 2=GNOSIS_SAFE (default 0)
  POLYMARKET_FUNDER             proxy wallet address (required for sig_type=2)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Optional

# Add repo root to sys.path so `research_lines.*` and `src.*` imports resolve
# regardless of working directory (same pattern as run_scoring_activation.py).
_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto_maker_loop")


# ---------------------------------------------------------------------------
# Imports from existing modules (no duplication)
# ---------------------------------------------------------------------------

def _import_sa():
    """Import scoring_activation from the reward_aware_maker_probe research line."""
    import importlib
    return importlib.import_module(
        "research_lines.reward_aware_maker_probe.modules.scoring_activation"
    )


# ---------------------------------------------------------------------------
# Dynamic market discovery
# ---------------------------------------------------------------------------

_LATEST_PROBE_PATH = Path("data/research/reward_aware_maker_probe/latest_probe.json")
_PROBE_MAX_AGE_HOURS = 24.0     # treat probe as stale after this many hours
_PROBE_AUTO_REFRESH_HOURS = 12.0  # re-run universe_refresh every N hours during session
_DEFAULT_FALLBACK_MAX_SPREAD_CENTS = 3.5


def _auto_refresh_universe(min_rate: float = 100.0) -> bool:
    """Run universe_refresh subprocess to regenerate latest_probe.json. Returns True on success."""
    try:
        result = subprocess.run(
            [sys.executable, str(_FILE_DIR / "run_universe_refresh.py"), "--min-rate", str(min_rate)],
            timeout=180,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("auto_refresh: universe refresh completed successfully")
            return True
        logger.warning("auto_refresh: universe refresh failed (rc=%d): %s", result.returncode, result.stderr[:300])
        return False
    except Exception as exc:
        logger.warning("auto_refresh: universe refresh exception: %s", exc)
        return False
_DEFAULT_FALLBACK_MIN_SIZE = 200.0


def _load_dynamic_survivor_data(
    min_daily_rate_usdc: float = 100.0,
    max_markets: int = 10,
) -> dict | None:
    """
    Load SURVIVOR_DATA from the latest universe-refresh probe file.

    Returns a dict in the same schema as scoring_activation.SURVIVOR_DATA, or
    None if the file is missing, stale, or contains no qualifying markets.

    Parameters
    ----------
    min_daily_rate_usdc : float
        Only include markets with reward_daily_rate_usdc >= this value.
    max_markets : int
        Cap on number of markets returned (sorted by daily rate descending).
    """
    if not _LATEST_PROBE_PATH.exists():
        logger.info("dynamic_survivor: %s not found — using hardcoded fallback", _LATEST_PROBE_PATH)
        return None

    try:
        with _LATEST_PROBE_PATH.open(encoding="utf-8") as fh:
            probe = json.load(fh)
    except Exception as exc:
        logger.warning("dynamic_survivor: failed to read probe file: %s", exc)
        return None

    # Staleness check
    ts_str = probe.get("probe_timestamp") or ""
    if ts_str:
        try:
            probe_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - probe_ts).total_seconds() / 3600.0
            if age_hours > _PROBE_MAX_AGE_HOURS:
                logger.warning(
                    "dynamic_survivor: probe is %.1fh old (max %sh) — using hardcoded fallback",
                    age_hours, _PROBE_MAX_AGE_HOURS,
                )
                return None
        except Exception:
            pass

    markets = probe.get("markets") or []
    qualifying = [
        m for m in markets
        if m.get("condition_id")
        and m.get("token_id")
        and float(m.get("reward_daily_rate_usdc") or 0) >= min_daily_rate_usdc
    ]

    if not qualifying:
        logger.info(
            "dynamic_survivor: no markets with rate >= $%.0f/day in probe — using hardcoded fallback",
            min_daily_rate_usdc,
        )
        return None

    # Sort by daily rate descending, cap at max_markets
    qualifying.sort(key=lambda m: float(m.get("reward_daily_rate_usdc") or 0), reverse=True)
    qualifying = qualifying[:max_markets]

    survivor_data: dict = {}
    for m in qualifying:
        slug = m["market_slug"]
        max_spread = m.get("reward_max_spread_cents") or _DEFAULT_FALLBACK_MAX_SPREAD_CENTS
        min_size = float(m.get("reward_min_size_shares") or _DEFAULT_FALLBACK_MIN_SIZE)
        survivor_data[slug] = {
            "condition_id":             m["condition_id"],
            "token_id":                 m["token_id"],
            "daily_rate_usdc":          float(m["reward_daily_rate_usdc"]),
            "fallback_max_spread_cents": float(max_spread),
            "fallback_min_size":        min_size,
            "yes_price_ref":            float(m.get("yes_price_ref") or m.get("yes_price_clob") or m.get("midpoint") or 0.5),
            "competitiveness_ref":      1.0,   # unknown from probe; live fetch will update
            "market_end_date":          m.get("market_end_date") or "",
            "reward_end_date":          m.get("reward_end_date") or "",
        }

    logger.info(
        "dynamic_survivor: loaded %d markets from probe (rate >= $%.0f/day, probe_ts=%s)",
        len(survivor_data), min_daily_rate_usdc, ts_str[:19],
    )
    for slug, data in survivor_data.items():
        logger.info(
            "  %s  rate=$%.0f/day  token_id=%s…",
            slug[:50], data["daily_rate_usdc"], data["token_id"][:12],
        )
    return survivor_data


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

RUNS_JSONL = Path("data/research/auto_maker_loop/runs.jsonl")


def _append_run(record: dict) -> None:
    RUNS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("recorded → %s", RUNS_JSONL)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# One cycle
# ---------------------------------------------------------------------------

def _is_kill_switch_error(error_str: Optional[str]) -> bool:
    """Return True if the error string indicates an auth/signature failure."""
    if not error_str:
        return False
    s = error_str.lower()
    return any(k in s for k in ("signature", "invalid", "unauthorized", "401", "403"))


def _check_heartbeat(client: Any) -> bool:
    """
    Verify the CLOB client connection is alive.
    Calls client.get_ok() — returns True on success, False on any error.
    Never raises.
    """
    try:
        result = client.get_ok()
        ok = (
            result is True
            or result == "OK"
            or (isinstance(result, dict) and result.get("ok") is True)
        )
        if ok:
            logger.info("heartbeat: get_ok returned OK")
        else:
            logger.warning("heartbeat: get_ok returned unexpected value: %s", result)
        return ok
    except Exception as exc:
        logger.error("heartbeat: get_ok failed: %s", exc)
        return False


def _reconcile_open_orders(client: Any, token_id: str) -> int:
    """
    Pre-placement reconciliation.
    If stale open orders exist for this token, cancel them before placing new ones.
    Returns the count of orders cancelled (0 if none found).
    Never raises.
    """
    from research_lines.auto_maker_loop.modules.restart_ops import (
        cancel_and_verify, _count_open_orders,
    )
    count, err = _count_open_orders(client, token_id)
    if err:
        logger.warning("reconcile: could not count open orders: %s", err)
        return 0
    if count > 0:
        logger.warning(
            "reconcile: found %d stale open order(s) for token %s… — clearing before placement",
            count, token_id[:20],
        )
        result = cancel_and_verify(client, token_id=token_id, reason="pre_cycle_reconciliation")
        logger.info(
            "reconcile: cancel_and_verify result: cancelled=%d verified=%s remaining=%d",
            result["cancelled_count"], result["verification_passed"], result["orders_remaining"],
        )
        return count
    return 0


def _cancel_all_open_orders(client: Any, token_id: str = "", reason: str = "shutdown") -> None:
    """
    Cancel all open orders via restart_ops.cancel_and_verify().
    Handles 425 rate-limit backoff and verifies open order count reaches 0.
    Never raises.
    """
    from research_lines.auto_maker_loop.modules.restart_ops import cancel_and_verify
    result = cancel_and_verify(client, token_id=token_id, reason=reason)
    logger.info(
        "cancel_and_verify: cancelled=%d  verified=%s  remaining=%d  rate_limited=%s",
        result["cancelled_count"], result["verification_passed"],
        result["orders_remaining"], result["rate_limited"],
    )
    if not result["verification_passed"]:
        logger.warning(
            "cancel_and_verify: %d order(s) still open — manual check required",
            result["orders_remaining"],
        )


def _earning_pct_to_cents_per_hour(pct: Optional[float], daily_rate_usdc: float) -> Optional[float]:
    """Convert earning_percentage to cents/hour for comparison with pnl_cents."""
    if pct is None:
        return None
    return round((pct / 100.0) * daily_rate_usdc * 100 / 24.0, 4)


def _inventory_tier(
    total_shares: float,
    base: float,
    soft_cap: float,
    hard_cap: float,
    reward_cover_hours: Optional[float],
) -> str:
    """
    Return inventory tier string.

    FLATTEN   : total > hard_cap OR reward_cover_hours > 48
    AGGRESSIVE: total > soft_cap
    REDUCE_ONLY: total > base  (or reward_cover_hours > 24 with any excess)
    NORMAL    : total <= base
    """
    excess = total_shares - base
    if total_shares > hard_cap:
        return "FLATTEN"
    if reward_cover_hours is not None and reward_cover_hours > 48 and excess > 0:
        return "FLATTEN"
    if total_shares > soft_cap:
        return "AGGRESSIVE"
    if total_shares > base:
        return "REDUCE_ONLY"
    if reward_cover_hours is not None and reward_cover_hours > 24 and excess > 0:
        return "REDUCE_ONLY"
    return "NORMAL"


def _reward_cover_hours(
    excess_shares: float,
    avg_entry_price: float,
    current_mid: float,
    daily_rate_usdc: float,
) -> Optional[float]:
    """
    Hours of reward required to recover the unrealized loss on excess inventory.

    Formula:
        unrealized_loss_usd = excess_shares * (avg_entry_price - current_mid)
        reward_rate_usd_per_hour = daily_rate_usdc / 24
        reward_cover_time_hours = unrealized_loss_usd / reward_rate_usd_per_hour

    Returns None if no excess or no loss.
    """
    if excess_shares <= 0 or daily_rate_usdc <= 0:
        return None
    loss_usd = excess_shares * (avg_entry_price - current_mid)
    if loss_usd <= 0:
        return None
    rate_per_hour = daily_rate_usdc / 24.0
    return round(loss_usd / rate_per_hour, 2)


def _run_cycle(
    sa: Any,
    ms: Any,
    pm: Any,
    slug: str,
    client: Any,
    creds: Any,
    args: argparse.Namespace,
    cycle_num: int,
    survivor_data: dict | None = None,
    reduce_only: bool = False,
    close_buffer: bool = False,
    inventory_tier: str = "NORMAL",
    excess_shares: float = 0.0,
    reward_cover_hours: Optional[float] = None,
) -> dict:
    """
    Execute one full bilateral maker cycle for `slug`.

    Returns a result dict for recording.
    """
    from research_lines.auto_maker_loop.modules.position_manager import (
        PositionConfig, PositionState,
    )

    _sd        = survivor_data if survivor_data is not None else sa.SURVIVOR_DATA
    data       = _sd[slug]
    token_id   = data["token_id"]
    condition_id = data["condition_id"]
    host       = sa.CLOB_HOST

    _section(f"Cycle {cycle_num}  |  {slug[:50]}")
    print(f"  inventory_tier      : {inventory_tier}")
    if excess_shares > 0:
        print(f"  excess_shares       : {excess_shares:.0f}")
    if reward_cover_hours is not None:
        print(f"  reward_cover_hours  : {reward_cover_hours:.1f}h  ({'SELL NOW' if reward_cover_hours > 24 else 'acceptable'})")
    if reduce_only:
        print("  [REDUCE-ONLY] inventory > base — SELL excess only, no new BID")
    if close_buffer:
        print("  [CLOSE-BUFFER] near resolution — no new BIDs")

    # ── 1. Preflight: YES inventory + earning snapshot ───────────────────
    inv = sa._check_sell_inventory(client, token_id, required_shares=data["fallback_min_size"])
    inventory_before = inv.get("balance_shares", 0.0)
    print(f"  inventory verdict   : {inv['verdict']}")
    print(f"  balance_shares      : {inventory_before:.2f}")

    earning_pct_before = sa._check_earning_pct(host, creds, condition_id)

    # ── Reward attribution probe — T_before snapshot (full row) ──────────
    from research_lines.auto_maker_loop.modules import reward_attribution_probe as _rap
    import json as _json
    _reward_row_before = None
    try:
        _reward_row_before = _rap.fetch_row(host, creds, condition_id)
    except Exception as _rap_exc:
        print(f"  reward_probe_fetch  : ERROR — {_rap_exc}")

    # ── UI DAILY REWARDS — T_before (account-level: /rewards/user/total) ─
    # GET /rewards/user/total is the current public endpoint for the
    # account-level UI Daily Rewards total.  /rewards/epoch/... returns 405.
    _ui_rewards_before: Optional[float] = None
    _ui_rewards_before_raw: Optional[str] = None
    print(f"\n  ── UI DAILY REWARDS T_BEFORE (/rewards/user/total) ─────────")
    _today_utc   = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y-%m-%d")
    _sig_type_v  = getattr(creds, "signature_type", "?")
    _maker_addr  = getattr(creds, "funder", None) or "(will derive EOA)"
    print(f"  sig_type      : {_sig_type_v}")
    print(f"  maker_address : {_maker_addr}")
    print(f"  request_url   : {host.rstrip('/')}/rewards/user/total"
          f"?date={_today_utc}&signature_type={_sig_type_v}&maker_address=<addr>")
    try:
        _ui_rewards_before, _ui_rewards_before_raw = _rap.fetch_user_total(host, creds, date=_today_utc)
        if _ui_rewards_before is None:
            print(f"  account_rewards_total_before : FETCH_ERROR — {_ui_rewards_before_raw}")
        else:
            print(f"  account_rewards_total_before : {_ui_rewards_before!r}")
            print(f"  raw_response                 : {_ui_rewards_before_raw}")
    except Exception as _ute:
        print(f"  account_rewards_total_before : ERROR — {type(_ute).__name__}: {_ute}")
    print(f"  ─────────────────────────────────────────────────────────────\n")

    print("\n  ── REWARD ROW T_BEFORE (MARKET TELEMETRY ONLY — NOT UI DAILY REWARDS) ──")
    if _reward_row_before:
        # Dump FULL raw row as JSON so field names are visible
        _row_json = _json.dumps(_reward_row_before, default=str)
        print(f"  raw_row             : {_row_json[:1200]}")
        print(f"  all_fields          : {list(_reward_row_before.keys())}")

        # Key-presence check — do NOT use `or`; 0 is a valid reward value
        _ep_before = None
        for _ep_f in ("earning_percentage", "earnings_percentage"):
            if _ep_f in _reward_row_before:
                _ep_before = _reward_row_before[_ep_f]
                break
        print(f"  earning_pct_before  : {_ep_before!r}")

        # earnings[] presence check with explicit type diagnosis
        _earnings_raw = _reward_row_before.get("earnings")
        if _earnings_raw is None:
            print("  earnings_key        : !! NOT PRESENT in row — check all_fields above for actual name")
            _earn_tot = None
        elif not isinstance(_earnings_raw, list):
            print(f"  earnings_key        : PRESENT but type={type(_earnings_raw).__name__} (expected list)")
            _earn_tot = None
        elif len(_earnings_raw) == 0:
            print("  earnings_key        : PRESENT but EMPTY list")
            _earn_tot = 0.0
        else:
            # Expand every entry
            for _ei, _ee in enumerate(_earnings_raw):
                if isinstance(_ee, dict):
                    print(
                        f"  earnings[{_ei}]"
                        f"  asset={str(_ee.get('asset_address') or '?')[:20]}"
                        f"  earnings={_ee.get('earnings')!r}"
                        f"  rate={_ee.get('asset_rate')!r}"
                    )
            _earn_tot = _rap.sum_earnings(_reward_row_before)

        print(f"  total_earnings_before: {_earn_tot!r}  "
              f"(entries={len(_earnings_raw) if isinstance(_earnings_raw, list) else 'N/A'})")
    else:
        print("  reward_row_before   : NOT FOUND on /rewards/user/markets")
    # Account-wide T_before aggregate (all market rows) — MARKET TELEMETRY ONLY
    _account_total_before: Optional[float] = None
    try:
        _acct_rows_b, _acct_err_b = _rap.fetch_all_rows(host, creds)
        _account_total_before = _rap.sum_earnings_all(_acct_rows_b)
        if _acct_err_b:
            print(f"  account_fetch_err    : {_acct_err_b}")
        print(f"  mkt_telemetry_account_total_before : {_account_total_before!r}  (rows={len(_acct_rows_b)})  [MARKET TELEMETRY — NOT UI LEDGER]")
    except Exception as _ae:
        print(f"  mkt_telemetry_account_total_before : ERROR — {_ae}")
    print("  ─────────────────────────────────────────────────────────────\n")

    if inv["verdict"] not in ("SELL_INVENTORY_READY",):
        logger.warning("inventory not ready: %s — skipping cycle", inv["verdict"])
        return {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "cycle":       cycle_num,
            "slug":        slug,
            "outcome":     "SKIPPED",
            "skip_reason": inv["verdict"],
        }

    # ── 1b. User-activity reconciliation ─────────────────────────────────
    # Provides same_side_pending_shares and global_total_shares for
    # inventory_governor.  Replaces the previous hardcoded 0s.
    from research_lines.auto_maker_loop.modules.user_activity_reconciler import (
        reconcile as _uar_reconcile,
    )
    _runs_path = str(RUNS_JSONL) if RUNS_JSONL.exists() else None
    recon = _uar_reconcile(
        client=client,
        creds=creds,
        survivor_data=_sd,
        token_id=token_id,
        runs_jsonl_path=_runs_path,
    )
    print(f"  pending_bid_shares  : {recon.same_side_pending_shares:.2f}  "
          f"(open_bid_orders={recon.open_bid_count})")
    print(f"  global_total_shares : {recon.global_total_shares:.2f}")
    if not recon.reconcile_ok:
        logger.warning("reconciler errors: %s", recon.errors)
    if abs(recon.unrecorded_inventory_delta) > 1.0:
        logger.warning(
            "unrecorded inventory delta=%.2f — unexpected position change since last cycle",
            recon.unrecorded_inventory_delta,
        )
        print(f"  !! unrecorded delta : {recon.unrecorded_inventory_delta:+.2f} shares")

    # ── 2. Fetch reward config ────────────────────────────────────────────
    from research_lines.auto_maker_loop.modules.reward_config_digest import (
        digest as _rcd_digest, reward_zone as _rcd_zone,
    )
    reward_cfg = sa.fetch_reward_config(
        host=host,
        condition_id=condition_id,
        fallback_max_spread_cents=data["fallback_max_spread_cents"],
        fallback_min_size=data["fallback_min_size"],
        fallback_daily_rate=data["daily_rate_usdc"],
    )
    rcd = _rcd_digest(reward_cfg)   # normalized flat dict — stored in result record (never mutated)
    # _effective_max_spread: local override for planner only; rcd["max_spread_cents"] stays clean
    _effective_max_spread = (
        args.spread_target_cents
        if getattr(args, "spread_target_cents", None) is not None
        else rcd["max_spread_cents"]
    )
    print(f"  reward config src   : {rcd['source']}  (ok={rcd['ok']})")
    print(f"  min_size            : {rcd['min_size_shares']:.0f} shares")
    print(f"  max_spread          : {rcd['max_spread_cents']:.1f}¢"
          + (f"  [planner override → {_effective_max_spread:.1f}¢]"
             if _effective_max_spread != rcd["max_spread_cents"] else ""))
    print(f"  daily_rate          : ${rcd['daily_rate_usdc']:.0f}/day")
    if rcd["competitiveness"] is not None:
        print(f"  competitiveness     : {rcd['competitiveness']:.1f}")

    # ── 3. Fetch midpoint ─────────────────────────────────────────────────
    price_ref = getattr(reward_cfg, "yes_price_live", None) or data["yes_price_ref"]
    midpoint, mid_src = sa.fetch_midpoint(client, token_id, price_ref)

    if midpoint is None:
        logger.warning("midpoint unavailable — skipping cycle")
        return {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "cycle":       cycle_num,
            "slug":        slug,
            "outcome":     "SKIPPED",
            "skip_reason": "midpoint_unavailable",
        }

    print(f"  midpoint            : {midpoint:.4f}  ({mid_src})")
    # Recompute reward zone bounds now that live midpoint is known
    rcd["reward_zone_bid"], rcd["reward_zone_ask"] = _rcd_zone(rcd, midpoint)

    # ── 4. Inventory governor ─────────────────────────────────────────────
    from research_lines.auto_maker_loop.modules.inventory_governor import (
        InventoryParams, assess as _inv_assess,
    )
    inv_params = InventoryParams(
        current_shares=inventory_before,
        market_cap=args.hard_cap_shares,
        global_cap=getattr(args, "global_cap_shares", args.hard_cap_shares * 2),
        global_total_shares=recon.global_total_shares,     # live from data-API
        normal_threshold=getattr(args, "normal_threshold", 0.80),
        same_side_pending_shares=recon.same_side_pending_shares,  # live from CLOB
        reward_cover_hours=reward_cover_hours,
        reduce_only_rch_threshold=24.0,
        consecutive_errors=0,
    )
    inv_decision = _inv_assess(inv_params)
    print(f"  inventory_state     : {inv_decision.inventory_state}")
    print(f"  bid_ok              : {inv_decision.bid_ok}  ask_ok={inv_decision.ask_ok}")
    print(f"  size_factor         : {inv_decision.size_factor:.2f}")
    if inv_decision.excess_shares > 0:
        print(f"  excess_shares       : {inv_decision.excess_shares:.0f}")
    print(f"  inv_reason          : {inv_decision.reason}")

    if inv_decision.hard_stop:
        logger.error("inventory_governor: STOP state — %s", inv_decision.reason)
        return {
            "ts": datetime.now(timezone.utc).isoformat(), "cycle": cycle_num, "slug": slug,
            "outcome": "SKIPPED", "skip_reason": f"INVENTORY_STOP:{inv_decision.reason}",
            "inventory_state": inv_decision.inventory_state,
        }

    # ── 4b. Quote planner v2 ──────────────────────────────────────────────
    from research_lines.auto_maker_loop.modules.reward_aware_quote_planner_v2 import (
        PlannerParams, plan as _qplan,
    )
    # Initial posture guess for planner (posture_selector runs below after suitability)
    # Use REDUCE_ONLY → EXIT_ONLY mapping here so planner gets correct posture
    _initial_posture = (
        "EXIT_ONLY" if not inv_decision.bid_ok
        else ("ASYMMETRIC_ASK_LEAN" if inv_decision.inventory_state == "SOFT_CAP"
              else "BILATERAL")
    )
    plan_params = PlannerParams(
        midpoint=midpoint,
        posture=_initial_posture,
        max_spread_cents=_effective_max_spread,
        min_size=rcd["min_size_shares"],
        reward_zone_bid=rcd["reward_zone_bid"],
        reward_zone_ask=rcd["reward_zone_ask"],
        size_factor=inv_decision.size_factor,
        base_size=data["fallback_min_size"],
    )
    plan_result = _qplan(plan_params)
    # Direct price overrides — applied after planner; bypass spread math for canary testing
    if getattr(args, "bid_price_override", None) is not None:
        plan_result.bid_price = args.bid_price_override
    if getattr(args, "ask_price_override", None) is not None:
        plan_result.ask_price = args.ask_price_override
    print(f"  bid (planned)       : {plan_result.bid_price}")
    print(f"  ask (planned)       : {plan_result.ask_price:.4f}")
    if plan_result.bid_price is not None:
        print(f"  spread              : {round((plan_result.ask_price - plan_result.bid_price) * 100, 2):.2f}¢")
    print(f"  size (bid/ask)      : {plan_result.bid_size}/{plan_result.ask_size:.0f}")
    if plan_result.planner_notes:
        for n in plan_result.planner_notes:
            print(f"  planner_note        : {n}")

    if not plan_result.qualifying:
        logger.warning("quote planner: not qualifying — skipping cycle")
        return {
            "ts": datetime.now(timezone.utc).isoformat(), "cycle": cycle_num, "slug": slug,
            "outcome": "SKIPPED", "skip_reason": "quotes_not_qualifying",
            "reward_config_ok": rcd["ok"],
            "inventory_state": inv_decision.inventory_state,
            "planner_notes": plan_result.planner_notes,
        }

    # ── 4c. Market suitability snapshot (hard gate) ───────────────────────
    from research_lines.auto_maker_loop.modules.market_suitability_snapshot import (
        SuitabilityParams, check as _suit_check,
    )
    _bid_for_gate = plan_result.bid_price if plan_result.bid_price is not None else plan_result.ask_price
    suit_params = SuitabilityParams(
        midpoint=midpoint,
        bid_price=_bid_for_gate,
        ask_price=plan_result.ask_price,
        reward_zone_bid=rcd["reward_zone_bid"],
        reward_zone_ask=rcd["reward_zone_ask"],
        reward_config_ok=rcd["ok"],
        current_shares=inventory_before,
        position_cap=args.hard_cap_shares,
        allow_fallback_config=True,
        reduce_only=(inv_decision.inventory_state == "REDUCE_ONLY"),
    )
    suit_result = _suit_check(suit_params)
    quotes_in_zone = suit_result.quotes_in_zone
    print(f"  suitability         : {'OK' if suit_result.suitable else 'FAIL: ' + suit_result.reason}")
    print(f"  quotes_in_zone      : {quotes_in_zone}")

    if not suit_result.suitable:
        logger.warning("market_suitability gate: FAIL — %s", suit_result.reason)
        return {
            "ts": datetime.now(timezone.utc).isoformat(), "cycle": cycle_num, "slug": slug,
            "outcome": "SKIPPED", "skip_reason": f"SUITABILITY:{suit_result.reason}",
            "reward_config_ok": rcd["ok"],
            "quotes_in_zone": quotes_in_zone,
            "inventory_state": inv_decision.inventory_state,
        }

    # ── 4d. Posture selector ──────────────────────────────────────────────
    from research_lines.auto_maker_loop.modules.posture_selector import (
        PostureParams, select as _posture_select,
    )
    posture_params = PostureParams(
        inventory_state=inv_decision.inventory_state,
        bid_ok=inv_decision.bid_ok,
        ask_ok=inv_decision.ask_ok,
        suitable=suit_result.suitable,
        reward_config_ok=rcd["ok"],
        quotes_in_zone=quotes_in_zone,
        competitiveness=rcd["competitiveness"],
    )
    posture_decision = _posture_select(posture_params)
    print(f"  posture             : {posture_decision.posture}  ({posture_decision.reason})")

    if posture_decision.posture == "SKIP":
        return {
            "ts": datetime.now(timezone.utc).isoformat(), "cycle": cycle_num, "slug": slug,
            "outcome": "SKIPPED", "skip_reason": f"POSTURE_SKIP:{posture_decision.reason}",
            "reward_config_ok": rcd["ok"],
            "inventory_state": inv_decision.inventory_state,
        }

    # Replan with confirmed posture if it changed from the initial guess
    if posture_decision.posture != _initial_posture:
        plan_params.posture = posture_decision.posture
        plan_result = _qplan(plan_params)
        if getattr(args, "bid_price_override", None) is not None:
            plan_result.bid_price = args.bid_price_override
        if getattr(args, "ask_price_override", None) is not None:
            plan_result.ask_price = args.ask_price_override

    # Override legacy reduce_only / close_buffer flags from posture decision
    reduce_only   = not posture_decision.place_bid
    close_buffer  = close_buffer  # unchanged — still driven by time-to-resolution

    if args.dry_run:
        print("\n  [DRY-RUN] Would place:")
        print(f"    BID {plan_result.ask_size:.0f} YES @ {plan_result.bid_price:.4f}")
        print(f"    ASK {plan_result.ask_size:.0f} YES @ {plan_result.ask_price:.4f}")
        return {
            "ts":             datetime.now(timezone.utc).isoformat(),
            "cycle":          cycle_num,
            "slug":           slug,
            "outcome":        "DRY_RUN",
            "posture":        posture_decision.posture,
            "bid_price":      plan_result.bid_price,
            "ask_price":      plan_result.ask_price,
            "size":           plan_result.ask_size,
            "inventory_state": inv_decision.inventory_state,
            "quotes_in_zone": quotes_in_zone,
            "reward_config_ok": rcd["ok"],
        }

    # ── 5. Place orders — prices and sizes from quote planner v2 ─────────
    placed_at = datetime.now(timezone.utc)
    bid_id: Optional[str] = None

    if posture_decision.place_bid and not close_buffer:
        _bid_price = plan_result.bid_price
        _bid_size  = plan_result.bid_size or rcd["min_size_shares"]
        bid_id, bid_err = sa._place_order(client, token_id, _bid_price, _bid_size, "BUY")
        if not bid_id:
            logger.error("BID placement failed: %s", bid_err)
            return {
                "ts":                    placed_at.isoformat(),
                "cycle":                 cycle_num,
                "slug":                  slug,
                "outcome":               "PLACEMENT_FAILED",
                "failed_leg":            "BID",
                "error":                 bid_err,
                "kill_switch_triggered": _is_kill_switch_error(bid_err),
                "posture":               posture_decision.posture,
                "inventory_state":       inv_decision.inventory_state,
            }
        print(f"  BID placed          : {bid_id[:20]}...")
    else:
        print(f"  BID skipped         : posture={posture_decision.posture}"
              + (" close_buffer" if close_buffer else ""))

    # ASK size: use planner result; in EXIT_ONLY use excess_shares if available
    if not posture_decision.place_bid and excess_shares > 0:
        ask_size  = excess_shares
        ask_price = plan_result.ask_price   # planner already set to mid+tick for EXIT_ONLY
        print(f"  exit-only ASK       : {ask_size:.0f} shares @ {ask_price:.4f}")
    else:
        ask_size  = plan_result.ask_size
        ask_price = plan_result.ask_price

    ask_id, ask_err = sa._place_order(client, token_id, ask_price, ask_size, "SELL")
    if not ask_id:
        logger.error("ASK placement failed: %s", ask_err)
        if bid_id:
            sa._cancel_order(client, bid_id)
        return {
            "ts":                    placed_at.isoformat(),
            "cycle":                 cycle_num,
            "slug":                  slug,
            "outcome":               "PLACEMENT_FAILED",
            "failed_leg":            "ASK",
            "error":                 ask_err,
            "kill_switch_triggered": _is_kill_switch_error(ask_err),
            "reduce_only_activated": reduce_only,
            "inventory_tier":        inventory_tier,
        }
    print(f"  ASK placed          : {ask_id[:20]}...")

    # ── 5b. Verify both orders are scoring ───────────────────────────────
    # Scoring may lag placement by a few seconds.  One 30-second retry is
    # allowed before treating non-scoring as a hard signal.
    # If orders are still not scoring after retry: cancel both and skip cycle.
    # Taking inventory risk without reward credit is not acceptable.
    from research_lines.auto_maker_loop.modules.order_scoring_verifier import (
        verify as _osv_verify, all_scoring as _osv_all, format_summary as _osv_fmt,
    )
    _orders_to_verify = [o for o in [bid_id, ask_id] if o]
    scoring_results   = _osv_verify(client, _orders_to_verify)
    print(f"  scoring check       : {_osv_fmt(scoring_results)}")

    if not _osv_all(scoring_results):
        # Single retry after 30s to allow scoring propagation
        logger.info("scoring not confirmed — waiting 30s and re-checking")
        time.sleep(30)
        scoring_results = _osv_verify(client, _orders_to_verify)
        print(f"  scoring re-check    : {_osv_fmt(scoring_results)}")

    scoring_verified = _osv_all(scoring_results)

    if not scoring_verified:
        # Cancel both legs — inventory risk without reward is not justified
        logger.warning(
            "orders not scoring after retry — cancelling both legs and skipping cycle"
        )
        if bid_id:
            sa._cancel_order(client, bid_id)
        sa._cancel_order(client, ask_id)
        return {
            "ts":               placed_at.isoformat(),
            "cycle":            cycle_num,
            "slug":             slug,
            "outcome":          "NOT_SCORING",
            "skip_reason":      "orders_not_scoring",
            "bid_order_id":     bid_id,
            "ask_order_id_final": ask_id,
            "scoring_verified": False,
            "quotes_in_zone":   quotes_in_zone,
            "reward_config_ok": rcd["ok"],
            "competitiveness":  rcd["competitiveness"],
            # Diagnostic fields — already computed, do not discard on early exit
            "reward_zone_bid":    rcd["reward_zone_bid"],
            "reward_zone_ask":    rcd["reward_zone_ask"],
            "last_midpoint":      midpoint,
            "earning_pct_before": earning_pct_before,
            "notes":            [_osv_fmt(scoring_results)],
        }

    # ── 5c. Canary snapshot — printed for every live cycle ───────────────
    if args.live:
        _section("CANARY — Live Order Snapshot")
        print(f"  market              : {slug}")
        print(f"  token_id            : {token_id}")
        print(f"  bid_order_id        : {bid_id or 'NOT_PLACED'}")
        print(f"  ask_order_id        : {ask_id}")
        _csz = plan_result.bid_size or rcd["min_size_shares"]
        print(f"  bid  side/px/size   : BUY  {plan_result.bid_price} x {_csz:.0f}")
        print(f"  ask  side/px/size   : SELL {ask_price:.4f} x {ask_size:.0f}")
        print(f"  placement_accepted  : BID={'YES' if bid_id else 'SKIPPED'}  ASK=YES")
        print(f"  scoring_result      : {_osv_fmt(scoring_results)}")
        print(f"  scoring_verified    : {scoring_verified}")
        print(f"  live_for_scoring    : {'YES — scored within 30s retry' if scoring_verified else 'NO — cancelled (did not score)'}")
        print(f"  inventory_before    : {inventory_before:.2f} shares")

    # ── 6. Run position manager — tier-adjusted config ────────────────────
    # Tier overrides: AGGRESSIVE and FLATTEN get tighter exit parameters.
    # NORMAL and REDUCE_ONLY use the CLI defaults.
    if inventory_tier == "FLATTEN":
        effective_max_hold     = 20.0
        effective_chase_after  = 5.0
        effective_ask_cancel   = 0.5   # very tight — reprice immediately
    elif inventory_tier == "AGGRESSIVE":
        effective_max_hold     = args.max_hold_minutes
        effective_chase_after  = max(5.0, args.chase_after_minutes / 2.0)
        effective_ask_cancel   = max(0.5, args.ask_cancel_distance_cents / 2.0)
    else:
        effective_max_hold     = args.max_hold_minutes
        effective_chase_after  = args.chase_after_minutes
        effective_ask_cancel   = args.ask_cancel_distance_cents

    cfg = PositionConfig(
        poll_interval_sec=args.poll_interval_sec,
        max_hold_minutes=effective_max_hold,
        chase_after_minutes=effective_chase_after,
        max_chases=args.max_chases,
        chase_tick=0.01,
        stop_loss_cents=args.stop_loss_cents,
        drift_threshold_cents=args.drift_threshold_cents,
        ask_cancel_distance_cents=effective_ask_cancel,
        reduce_only=reduce_only,
        close_buffer=close_buffer,
        dry_run=False,
    )
    print(f"  effective_max_hold  : {effective_max_hold:.0f} min")
    print(f"  effective_chase_after: {effective_chase_after:.0f} min")

    # In reduce_only: start in BID_FILLED (already holding inventory).
    # entry_price = midpoint (reference for pnl; real avg cost unknown).
    # size = excess_shares (only selling the excess, not the base 200).
    initial_state   = "BID_FILLED" if reduce_only else "QUOTING"
    pos_entry_price = midpoint if (reduce_only and midpoint) else plan_result.bid_price
    pos_size        = excess_shares if (reduce_only and excess_shares > 0) else plan_result.ask_size
    pos_ask_price   = ask_price  # actual placed price (may differ from plan_result.ask_price in reduce_only)

    pos = PositionState(
        slug=slug,
        token_id=token_id,
        bid_order_id=bid_id or "reduce_only_no_bid",
        ask_order_id=ask_id,
        entry_price=pos_entry_price,
        ask_price=pos_ask_price,
        size=pos_size,
        placed_at=placed_at,
        state=initial_state,
        bid_filled=reduce_only,
    )

    _section("Position Manager — running")
    _loop_end_at = datetime.now(timezone.utc)   # capture before position loop (for hold approx)
    pm.run_position_loop(client, pos, cfg, price_ref=price_ref)
    _loop_end_at = datetime.now(timezone.utc)   # update to actual end time

    # ── 6c. Post-cycle fill attribution ──────────────────────────────────
    # Query the exchange directly to get ground-truth fill state.
    # The polling loop checks fills every poll_interval_sec — it can miss a fill
    # that occurred between polls, or misclassify if get_order returns unexpected
    # status (e.g., after cancel of an already-matched order).
    bid_fill_info: Optional[dict] = sa._get_order_fill_info(client, bid_id) if bid_id else None
    ask_fill_info: Optional[dict] = sa._get_order_fill_info(client, pos.ask_order_id)

    if bid_fill_info:
        logger.info(
            "fill_attribution BID: status=%s size_matched=%.0f/%.0f price=%.4f",
            bid_fill_info["status"], bid_fill_info["size_matched"],
            bid_fill_info["size"], bid_fill_info["price"],
        )
        if bid_fill_info["size_matched"] > 0 and not pos.bid_filled:
            # Position manager missed this fill (order filled between polls or
            # status not detected).  Reconcile from exchange data.
            pos.bid_filled = True
            if pos.hold_minutes == 0.0:
                # Approximate hold: time from placement to loop end (upper bound)
                pos.hold_minutes = round(
                    (_loop_end_at - placed_at).total_seconds() / 60.0, 2
                )
            pos.notes.append(
                f"fill_reconciled BID: exchange size_matched={bid_fill_info['size_matched']:.0f}"
                f" status={bid_fill_info['status']} (missed by poll loop)"
            )

    if ask_fill_info:
        logger.info(
            "fill_attribution ASK: status=%s size_matched=%.0f/%.0f price=%.4f",
            ask_fill_info["status"], ask_fill_info["size_matched"],
            ask_fill_info["size"], ask_fill_info["price"],
        )
        if ask_fill_info["size_matched"] > 0 and not pos.ask_filled:
            pos.ask_filled = True
            if ask_fill_info["price"] and ask_fill_info["price"] != pos.exit_price:
                pos.exit_price = ask_fill_info["price"]
                pos.pnl_cents  = round((pos.exit_price - pos.entry_price) * 100, 4)
            pos.notes.append(
                f"fill_reconciled ASK: exchange size_matched={ask_fill_info['size_matched']:.0f}"
                f" price={ask_fill_info['price']:.4f} status={ask_fill_info['status']}"
                f" (missed by poll loop)"
            )

    # ── 6d. Hold-time floor ──────────────────────────────────────────────────
    # Record how long orders were live even on no-fill cycles.  When the
    # position_manager tracked a fill-to-exit duration it is already non-zero,
    # so this only activates for QUOTING-only cycles (NO_FILL_TIME_LIMIT, DRIFT).
    if pos.hold_minutes == 0.0:
        pos.hold_minutes = round(
            (_loop_end_at - placed_at).total_seconds() / 60.0, 2
        )

    # ── 7. Post-cycle snapshots ───────────────────────────────────────────
    # 7a. UI Daily Rewards probe: T+0s / T+60s / T+180s
    # Source: GET /rewards/user/total — account-level public endpoint
    _ui_rewards_T0:   Optional[float] = None
    _ui_rewards_T60:  Optional[float] = None
    _ui_rewards_T180: Optional[float] = None
    _ui_rewards_classification: str = "FETCH_ERROR"

    _ui_total_probes = _rap.probe_user_total_after(host, creds, delays_sec=(0, 60, 180))

    _ui_total_vals = [s.total_usd for s in _ui_total_probes]
    if len(_ui_total_vals) >= 1:
        _ui_rewards_T0   = _ui_total_vals[0]
    if len(_ui_total_vals) >= 2:
        _ui_rewards_T60  = _ui_total_vals[1]
    if len(_ui_total_vals) >= 3:
        _ui_rewards_T180 = _ui_total_vals[2]

    _ui_after_values = [v for v in (_ui_rewards_T0, _ui_rewards_T60, _ui_rewards_T180) if v is not None]
    _ui_max_after = max(_ui_after_values) if _ui_after_values else None

    if _ui_max_after is not None and _ui_rewards_before is not None and _ui_max_after > _ui_rewards_before:
        _ui_rewards_classification = "UI_LEDGER_CONFIRMED"
    elif _ui_max_after is not None:
        _ui_rewards_classification = "UI_LEDGER_STILL_UNCHANGED_WITHIN_180S"
    else:
        _ui_rewards_classification = "FETCH_ERROR"

    _ui_max_delta = (
        round(_ui_max_after - _ui_rewards_before, 6)
        if (_ui_max_after is not None and _ui_rewards_before is not None)
        else None
    )

    print(f"\n  ── UI DAILY REWARDS POST-CYCLE (/rewards/user/total) ────────")
    print(f"  account_rewards_total_before : {_ui_rewards_before!r}")
    print(f"  account_rewards_total_T0     : {_ui_rewards_T0!r}")
    print(f"  account_rewards_total_T60    : {_ui_rewards_T60!r}")
    print(f"  account_rewards_total_T180   : {_ui_rewards_T180!r}")
    print(f"  max_account_rewards_delta    : {_ui_max_delta!r}")
    print(f"  classification               : {_ui_rewards_classification}")
    for _s in _ui_total_probes:
        if _s.error:
            print(f"  T{_s.delay_sec:>4}s_error           : {_s.error[:120]}")
    print(f"  ─────────────────────────────────────────────────────────────\n")

    # 7b. Market-telemetry probe: T_after_0s / T_after_60s / T_after_180s
    # MARKET TELEMETRY ONLY — NOT UI DAILY REWARDS.
    # earnings[] from /rewards/user/markets is NOT the UI Daily Rewards ledger.
    _reward_deltas: dict = {}
    try:
        _combined_probes = _rap.probe_combined_after(
            host, creds, condition_id, delays_sec=(0, 60, 180)
        )
        _reward_deltas = _rap.compute_combined_deltas(
            _reward_row_before, _account_total_before, _combined_probes
        )
        _rap.print_combined_table(
            _reward_row_before, _account_total_before, _combined_probes, _reward_deltas
        )
    except Exception as _probe_exc:
        print(f"\n  !! reward_probe ERROR in post-cycle section: {_probe_exc}")
        import traceback as _tb
        _tb.print_exc()

    # earning_pct_after: extract from T_after_0s for backwards compat with
    # existing reward_delta_cents formula and runs.jsonl field
    earning_pct_after = sa._check_earning_pct(host, creds, condition_id)

    inv_after = sa._check_sell_inventory(client, token_id, required_shares=1.0)
    inventory_after = inv_after.get("balance_shares", 0.0)

    # Reward delta: earning_pct change * daily_rate / 24h, expressed in cents
    reward_delta_cents: Optional[float] = None
    if earning_pct_before is not None and earning_pct_after is not None:
        delta_pct = earning_pct_after - earning_pct_before
        reward_delta_cents = round(
            (delta_pct / 100.0) * reward_cfg.daily_rate_usdc * 100 / 24.0, 4
        )

    _bid_ref = plan_result.bid_price if plan_result.bid_price is not None else plan_result.ask_price
    gross_edge_cents = round((plan_result.ask_price - _bid_ref) * 100, 4)
    net_edge_cents: Optional[float] = None
    if reward_delta_cents is not None:
        net_edge_cents = round(pos.pnl_cents + reward_delta_cents, 4)

    # ── 8. Build result record ────────────────────────────────────────────
    result = {
        "ts":                    placed_at.isoformat(),
        "cycle":                 cycle_num,
        "slug":                  slug,
        "outcome":               pos.exit_reason or pos.state,
        # Cycle origin — distinguishes pre-owned disposal from new market entries.
        # PREOWNED_INVENTORY_DISPOSAL: reduce_only path; bid_filled=True is synthetic.
        # NEW_ENTRY: normal bilateral entry; bid_filled=True means a real BID was hit.
        "cycle_origin":          "PREOWNED_INVENTORY_DISPOSAL" if reduce_only else "NEW_ENTRY",
        "fill_origin_type":      (
            "PREOWNED_DISPOSAL_FILL" if reduce_only
            else "NEW_ENTRY_FILL"    if pos.bid_filled
            else "UNKNOWN_FILL"
        ),
        # Quote & fill
        "entry_price":           pos.entry_price,
        "ask_price_initial":     plan_result.ask_price,
        "exit_price":            pos.exit_price,
        "size":                  pos.size,
        "bid_filled":            pos.bid_filled,
        "ask_filled":            pos.ask_filled,
        "ask_chases":            pos.ask_chases,
        # Inventory
        "inventory_before_shares": inventory_before,
        "inventory_after_shares":  inventory_after,
        # Execution
        "hold_minutes":          pos.hold_minutes,
        "min_midpoint_hold":     pos.min_midpoint_hold or None,
        "max_midpoint_hold":     pos.max_midpoint_hold or None,
        "last_midpoint":         pos.last_midpoint,
        # Economics
        "pnl_cents":             pos.pnl_cents,
        "gross_edge_cents":      gross_edge_cents,
        "reward_delta_cents":    reward_delta_cents,
        "net_edge_cents":        net_edge_cents,
        "earning_pct_before":    earning_pct_before,
        "earning_pct_after":     earning_pct_after,
        # Reward config snapshot (normalized via reward_config_digest)
        "daily_rate_usdc":       rcd["daily_rate_usdc"],
        "max_spread_cents":      rcd["max_spread_cents"],
        "min_size":              rcd["min_size_shares"],
        "reward_config_source":  rcd["source"],
        "reward_config_ok":      rcd["ok"],
        "competitiveness":       rcd["competitiveness"],
        "reward_zone_bid":       rcd["reward_zone_bid"],
        "reward_zone_ask":       rcd["reward_zone_ask"],
        "quotes_in_zone":        quotes_in_zone,
        "scoring_verified":      scoring_verified,
        # Inventory governor + reconciliation
        "inventory_state":              inv_decision.inventory_state,
        "inv_bid_ok":                   inv_decision.bid_ok,
        "inv_size_factor":              inv_decision.size_factor,
        "same_side_pending_shares":     recon.same_side_pending_shares,
        "global_total_shares":          recon.global_total_shares,
        "unrecorded_inventory_delta":   recon.unrecorded_inventory_delta,
        "reconcile_ok":                 recon.reconcile_ok,
        # Posture selector
        "posture":               posture_decision.posture,
        "posture_reason":        posture_decision.reason,
        # Quote planner v2
        "planner_notes":         plan_result.planner_notes,
        # Orders
        "bid_order_id":            bid_id,
        "ask_order_id_final":      pos.ask_order_id,
        # Fill attribution — ground-truth from post-cycle exchange query
        "bid_order_status":        bid_fill_info["status"]       if bid_fill_info else None,
        "ask_order_status":        ask_fill_info["status"]       if ask_fill_info else None,
        "bid_size_matched":        bid_fill_info["size_matched"] if bid_fill_info else None,
        "ask_size_matched":        ask_fill_info["size_matched"] if ask_fill_info else None,
        "bid_fill_price":          bid_fill_info["price"]        if bid_fill_info else None,
        "ask_fill_price":          ask_fill_info["price"]        if ask_fill_info else None,
        "bid_partial_fill":        bid_fill_info["partial_fill"] if bid_fill_info else None,
        "ask_partial_fill":        ask_fill_info["partial_fill"] if ask_fill_info else None,
        # Inventory tier and economics
        "inventory_tier":          inventory_tier,
        "excess_shares":           excess_shares,
        "reward_cover_hours":      reward_cover_hours,
        "reward_cover_check_passed": (reward_cover_hours is None or reward_cover_hours <= 24),
        # Control state
        "reduce_only_activated":   reduce_only,
        "close_buffer_activated":  close_buffer,
        "kill_switch_triggered":   pos.kill_switch,
        "drift_triggered":         pos.drift_triggered,
        "ask_repriced_up":         pos.ask_repriced_up,
        "consecutive_failures":    pos.consecutive_failures,
        "notes":                   pos.notes,
        # Market WS snapshot — live top-of-book at cycle end (None if WS not running)
        "market_ws_trades":        _mkt_ws.trade_count         if _mkt_ws else None,
        "market_ws_last_bid":      _mkt_ws.last_best_bid       if _mkt_ws else None,
        "market_ws_last_ask":      _mkt_ws.last_best_ask       if _mkt_ws else None,
        # Reward attribution probe — Hungary market scope
        "reward_row_before":             _reward_row_before,
        "total_earnings_before":         _reward_deltas.get("T_after_0s",  {}).get("hungary_earn_before"),
        "total_earnings_T0s":            _reward_deltas.get("T_after_0s",  {}).get("hungary_earn_after"),
        "total_earnings_T60s":           _reward_deltas.get("T_after_60s", {}).get("hungary_earn_after"),
        "total_earnings_T180s":          _reward_deltas.get("T_after_180s",{}).get("hungary_earn_after"),
        "earnings_delta_T0s":            _reward_deltas.get("T_after_0s",  {}).get("hungary_earn_delta"),
        "earnings_delta_T60s":           _reward_deltas.get("T_after_60s", {}).get("hungary_earn_delta"),
        "earnings_delta_T180s":          _reward_deltas.get("T_after_180s",{}).get("hungary_earn_delta"),
        "max_hungary_earn_delta":        _reward_deltas.get("max_hungary_earn_delta"),
        "max_hungary_earn_delay_sec":    _reward_deltas.get("max_hungary_earn_delay_sec"),
        # Reward attribution probe — account-wide scope
        "account_total_earnings_before": _account_total_before,
        "account_total_earnings_T0s":    _reward_deltas.get("T_after_0s",  {}).get("account_earn_after"),
        "account_total_earnings_T60s":   _reward_deltas.get("T_after_60s", {}).get("account_earn_after"),
        "account_total_earnings_T180s":  _reward_deltas.get("T_after_180s",{}).get("account_earn_after"),
        "account_earnings_delta_T0s":    _reward_deltas.get("T_after_0s",  {}).get("account_earn_delta"),
        "account_earnings_delta_T60s":   _reward_deltas.get("T_after_60s", {}).get("account_earn_delta"),
        "account_earnings_delta_T180s":  _reward_deltas.get("T_after_180s",{}).get("account_earn_delta"),
        "max_account_earn_delta":        _reward_deltas.get("max_account_earn_delta"),
        "max_account_earn_delay_sec":    _reward_deltas.get("max_account_earn_delay_sec"),
        # Classification — market telemetry surface (earnings[] from /rewards/user/markets)
        "attribution_outcome":           _reward_deltas.get("attribution_outcome"),
        # UI Daily Rewards — account-level total
        # Source: GET /rewards/user/total
        "account_rewards_total_before":  _ui_rewards_before,
        "account_rewards_total_T0":      _ui_rewards_T0,
        "account_rewards_total_T60":     _ui_rewards_T60,
        "account_rewards_total_T180":    _ui_rewards_T180,
        "max_account_rewards_delta":     _ui_max_delta,
        "ui_rewards_classification":     _ui_rewards_classification,
    }

    print(f"\n  exit_reason         : {pos.exit_reason}")
    print(f"  pnl_cents           : {pos.pnl_cents:.2f}¢")
    print(f"  reward_delta_cents  : {f'{reward_delta_cents:.4f}¢' if reward_delta_cents is not None else 'not measured'}")
    print(f"  net_edge_cents      : {f'{net_edge_cents:.4f}¢' if net_edge_cents is not None else 'not measurable'}")
    print(f"  hold_minutes        : {pos.hold_minutes:.1f}")
    print(f"  inventory_after     : {inventory_after:.2f} shares")
    if args.live:
        print(f"  cancel_safety       : order lifecycle handled by position_manager  "
              f"(exit={pos.exit_reason or pos.state})")

    return result


# ---------------------------------------------------------------------------
# Credential + client setup
# ---------------------------------------------------------------------------

def _setup_client(sa: Any) -> tuple[Any, Any]:
    """Load credentials and build CLOB client. Returns (creds, client)."""
    creds = sa.load_activation_credentials()
    if creds is None:
        print("\nERROR: POLYMARKET_PRIVATE_KEY not set in environment.")
        print("Set it with:  $env:POLYMARKET_PRIVATE_KEY = \"<key>\"")
        sys.exit(1)

    _section("Step 1: Credentials")
    client = sa.build_clob_client(creds, sa.CLOB_HOST)
    return creds, client


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated bilateral market-maker loop for reward-eligible markets."
    )
    p.add_argument("--live",    action="store_true", help="Submit real orders (default: dry-run)")
    p.add_argument("--dry-run", action="store_true", help="Compute quotes only, no orders")
    p.add_argument("--target",  default=None,
                   help="Lock to one market: hungary | rubio | vance (default: auto-select)")
    p.add_argument("--cycles",  type=int, default=0,
                   help="Number of cycles to run (0 = infinite, default 0)")
    p.add_argument("--cycle-sleep",        type=int,   default=30,
                   help="Seconds between cycles (default 30 — minimise reward gaps)")
    p.add_argument("--poll-interval-sec",  type=int,   default=30,
                   help="Seconds between fill checks inside position manager (default 30)")
    p.add_argument("--max-hold-minutes",   type=float, default=90.0,
                   help="TIME_LIMIT exit after this many minutes holding YES (default 90)")
    p.add_argument("--chase-after-minutes", type=float, default=45.0,
                   help="Start lowering ASK after this many minutes unfilled (default 45)")
    p.add_argument("--max-chases",         type=int,   default=2,
                   help="Max number of ASK re-quotes downward (default 2)")
    p.add_argument("--stop-loss-cents",       type=float, default=3.0,
                   help="STOP_LOSS if midpoint drops this many cents below entry (default 3.0)")
    p.add_argument("--drift-threshold-cents", type=float, default=2.0,
                   help="Re-quote if midpoint drifts this many cents from our bid in QUOTING (default 2.0)")
    p.add_argument("--ask-cancel-distance-cents", type=float, default=2.0,
                   help="Cancel hanging ASK if midpoint rises this many cents above it (default 2.0)")
    p.add_argument("--base-inventory-shares",  type=float, default=200.0,
                   help="Working base inventory (default 200). Above this → reduce-only.")
    p.add_argument("--soft-cap-shares",       type=float, default=280.0,
                   help="Soft inventory cap (default 280 = base + 40%%). Above → AGGRESSIVE tier.")
    p.add_argument("--hard-cap-shares",       type=float, default=360.0,
                   help="Hard inventory cap (default 360). Above → FLATTEN tier.")
    p.add_argument("--max-inventory-shares",  type=float, default=200.0,
                   help="Alias for base-inventory-shares (default 200). Kept for compatibility.")
    p.add_argument("--fill-delay-sec",        type=int,   default=300,
                   help="Seconds to delay next BID after a BUY fill (default 300)")
    p.add_argument("--close-buffer-minutes",  type=float, default=60.0,
                   help="Minutes before known resolution: disable new BIDs (default 60)")
    p.add_argument("--time-to-resolution-minutes", type=float, default=None,
                   help="Known minutes to market resolution (optional; enables close-buffer logic)")
    p.add_argument("--global-cap-shares",    type=float, default=800.0,
                   help="Global YES share cap across all markets (default 800)")
    p.add_argument("--normal-threshold",     type=float, default=0.80,
                   help="Fraction of market_cap below which state=NORMAL (default 0.80)")
    p.add_argument("--spread-target-cents", type=float, default=None,
                   help="Override planner max_spread (does NOT pollute telemetry max_spread field)")
    p.add_argument("--bid-price-override", type=float, default=None,
                   help="Hard-set BID price after planner (canary testing only, e.g. 0.62)")
    p.add_argument("--ask-price-override", type=float, default=None,
                   help="Hard-set ASK price after planner (canary testing only, e.g. 0.64)")
    p.add_argument("--market-daily-loss-limit-usd", type=float, default=5.0,
                   help="Skip market rest of session if cumulative realized loss exceeds this USD (default 5.0)")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # dry_run is True unless --live is explicitly passed
    args.dry_run = not args.live

    sa = _import_sa()

    import importlib
    ms = importlib.import_module("research_lines.auto_maker_loop.modules.market_selector")
    pm = importlib.import_module("research_lines.auto_maker_loop.modules.position_manager")

    # ── Dynamic market discovery ─────────────────────────────────────────────
    # Try to load live reward-eligible markets from the universe-refresh probe.
    # Falls back to the hardcoded SURVIVOR_DATA if probe is missing or stale.
    _dynamic = _load_dynamic_survivor_data(min_daily_rate_usdc=100.0, max_markets=10)
    if _dynamic:
        SURVIVOR_DATA = _dynamic
        _market_source = f"dynamic ({len(SURVIVOR_DATA)} markets from latest_probe.json)"
    else:
        SURVIVOR_DATA = sa.SURVIVOR_DATA
        _market_source = f"hardcoded fallback ({len(SURVIVOR_DATA)} markets)"

    # Resolve target slug
    target_slug: Optional[str] = None
    if args.target:
        target_slug = sa.SLUG_ALIASES.get(args.target, args.target)
        if target_slug not in SURVIVOR_DATA:
            # Allow exact slug match even if not in aliases
            if args.target in SURVIVOR_DATA:
                target_slug = args.target
            else:
                valid = list(sa.SLUG_ALIASES.keys()) + list(SURVIVOR_DATA.keys())
                print(f"ERROR: unknown target '{args.target}'.")
                print(f"  Valid aliases : {list(sa.SLUG_ALIASES.keys())}")
                print(f"  Dynamic slugs : {list(SURVIVOR_DATA.keys())[:5]} …")
                sys.exit(1)

    print("=" * 60)
    print("  AUTO MAKER LOOP")
    print(f"  mode        : {'LIVE' if args.live else 'DRY-RUN'}")
    print(f"  markets     : {_market_source}")
    print(f"  target      : {args.target or 'auto-select'}")
    print(f"  cycles      : {args.cycles or 'infinite'}")
    print(f"  max_hold    : {args.max_hold_minutes:.0f} min")
    print(f"  chase_after : {args.chase_after_minutes:.0f} min")
    print(f"  stop_loss   : {args.stop_loss_cents:.1f}¢")
    # ── Wiring proof — unconditional, before any cycle ────────────────
    # If this line does not appear in output, the file being run is NOT
    # the edited version (check path and Python environment).
    print(f"  REWARDCLIENT_WIRING_ACTIVE = YES")
    print("=" * 60)

    # Credentials (always needed — even dry-run fetches live data)
    creds, client = _setup_client(sa)

    # ── Graceful shutdown: SIGTERM / SIGHUP ──────────────────────────────────
    # Sets a flag so the loop exits cleanly after the current cycle completes,
    # cancelling all open orders before stopping.
    _shutdown: list[bool] = [False]

    def _shutdown_handler(signum: int, frame: Any) -> None:
        logger.warning("signal %d received — will shut down after this cycle", signum)
        _shutdown[0] = True

    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        signal.signal(signal.SIGHUP, _shutdown_handler)   # POSIX only
    except (AttributeError, OSError):
        pass

    # ── Startup state sync — read-only diagnostic before first cycle ──────
    from research_lines.auto_maker_loop.modules.startup_state_sync import (
        sync as _startup_sync,
    )
    _startup_sync(client, creds, SURVIVOR_DATA)

    # ── Background WS truth clients (market data only; non-blocking) ────────
    # Start once for the full loop session.  Logs all top-of-book and trade
    # events to data/research/auto_maker_loop/ for post-session analysis.
    _mkt_ws = None
    try:
        from research_lines.auto_maker_loop.modules.market_ws_client import MarketWsClient as _MktWsClient
        _all_token_ids = [v["token_id"] for v in SURVIVOR_DATA.values()]
        _mkt_ws = _MktWsClient(
            token_ids=_all_token_ids,
            log_path=Path("data/research/auto_maker_loop/market_ws_events.jsonl"),
        )
        _mkt_ws.start()
    except Exception as _ws_exc:
        logger.warning("market_ws_client: failed to start (non-fatal): %s", _ws_exc)

    cycle_num        = 0
    skip_slugs: set  = set()
    last_bid_fill_ts: Optional[datetime] = None   # for fill delay logic
    _mid_history: dict  = {}   # slug → last known midpoint (volatility circuit breaker)
    _market_daily_loss: dict = {}  # slug → cumulative loss USD today (per-market loss limit)
    _last_probe_refresh: datetime = datetime.now(timezone.utc)  # for auto-refresh tracking

    while True:
        cycle_num += 1
        if args.cycles and cycle_num > args.cycles:
            print(f"\nCompleted {args.cycles} cycles — exiting.")
            break

        # ── Shutdown check (SIGTERM / SIGHUP) ────────────────────────────────
        if _shutdown[0]:
            print("\n  [SHUTDOWN] Signal received — cancelling open orders and exiting.")
            _cancel_all_open_orders(client, reason="signal_shutdown")
            break

        # ── Auto-refresh probe every 12 hours ────────────────────────────────
        _hours_since_refresh = (datetime.now(timezone.utc) - _last_probe_refresh).total_seconds() / 3600.0
        if _hours_since_refresh >= _PROBE_AUTO_REFRESH_HOURS:
            print(f"\n  [AUTO-REFRESH] Probe is {_hours_since_refresh:.1f}h old — refreshing universe...")
            if _auto_refresh_universe(min_rate=100.0):
                _fresh = _load_dynamic_survivor_data(min_daily_rate_usdc=100.0, max_markets=10)
                if _fresh:
                    SURVIVOR_DATA = _fresh
                    skip_slugs.clear()   # markets may have changed
                    print(f"  [AUTO-REFRESH] Loaded {len(SURVIVOR_DATA)} fresh markets — skip_slugs cleared")
                else:
                    logger.warning("auto_refresh: 0 qualifying markets — keeping current SURVIVOR_DATA")
            _last_probe_refresh = datetime.now(timezone.utc)

        # ── Fill delay: pause BID placement after recent fill ─────────────
        if last_bid_fill_ts is not None:
            secs_since_fill = (datetime.now(timezone.utc) - last_bid_fill_ts).total_seconds()
            if secs_since_fill < args.fill_delay_sec:
                wait = int(args.fill_delay_sec - secs_since_fill)
                print(f"\n  [FILL-DELAY] waiting {wait}s before next BID (fill recency)")
                time.sleep(wait)

        _section(f"Cycle {cycle_num} — market selection")

        # Select market
        if target_slug:
            slug = target_slug
            print(f"  locked target       : {slug[:50]}")
        else:
            # Fetch live reward configs for ranking update
            live_cfgs: dict = {}
            for s, d in SURVIVOR_DATA.items():
                try:
                    cfg = sa.fetch_reward_config(
                        host=sa.CLOB_HOST,
                        condition_id=d["condition_id"],
                        fallback_max_spread_cents=d["fallback_max_spread_cents"],
                        fallback_min_size=d["fallback_min_size"],
                        fallback_daily_rate=d["daily_rate_usdc"],
                    )
                    live_cfgs[s] = cfg
                except Exception:
                    pass

            ranked = ms.rank_all(SURVIVOR_DATA, live_cfgs)
            print("  market ranking:")
            for row in ranked:
                skip_note = " [SKIP]" if row["slug"] in skip_slugs else ""
                print(
                    f"    {row['slug'][:42]:<42}  "
                    f"score={row['score']:.3f}  "
                    f"rate=${row['daily_rate_usdc']:.0f}/day  "
                    f"comp={row['competitiveness']:.1f}{skip_note}"
                )

            slug = ms.pick_best(SURVIVOR_DATA, live_cfgs, skip_slugs=skip_slugs)
            if slug is None:
                print("  all markets skipped — sleeping 60s then retrying")
                skip_slugs.clear()   # reset skips and try again
                time.sleep(60)
                continue

        # ── Determine control flags for this cycle ───────────────────────
        inv_check = sa._check_sell_inventory(
            client,
            SURVIVOR_DATA[slug]["token_id"],
            required_shares=1.0,
        )
        current_inventory = inv_check.get("balance_shares", 0.0)
        base_inv  = args.base_inventory_shares
        cur_excess = max(0.0, current_inventory - base_inv)

        # Fetch current midpoint for reward_cover calculation
        slug_data = SURVIVOR_DATA[slug]
        _mid_now, _ = sa.fetch_midpoint(
            client, slug_data["token_id"], slug_data["yes_price_ref"]
        )
        # ── Volatility circuit breaker ───────────────────────────────────
        # Track midpoint history to detect sudden price movements.
        # If mid moved > 3¢ in the last 2 cycles, skip placement this cycle.
        _prev_mid = _mid_history.get(slug)
        _mid_history[slug] = _mid_now
        if _mid_now and _prev_mid and abs(_mid_now - _prev_mid) * 100 > 3.0:
            print(
                f"  [VOLATILITY BREAKER] mid moved {abs(_mid_now - _prev_mid)*100:.2f}¢ "
                f"({_prev_mid:.4f} → {_mid_now:.4f}) — skipping cycle"
            )
            time.sleep(30)
            continue

        # Reward cover: use midpoint as proxy for avg_entry_price
        # (conservative — assumes entry was at current mid; actual loss may differ)
        _daily_rate = slug_data["daily_rate_usdc"]
        rch = _reward_cover_hours(
            excess_shares=cur_excess,
            avg_entry_price=_mid_now or slug_data["yes_price_ref"],
            current_mid=_mid_now or slug_data["yes_price_ref"],
            daily_rate_usdc=_daily_rate,
        )

        cycle_tier = _inventory_tier(
            total_shares=current_inventory,
            base=base_inv,
            soft_cap=args.soft_cap_shares,
            hard_cap=args.hard_cap_shares,
            reward_cover_hours=rch,
        )
        cycle_reduce_only = cycle_tier != "NORMAL"

        # Close buffer: auto-detect from market_end_date; manual arg overrides
        cycle_close_buffer = False
        _ttl_min: Optional[float] = args.time_to_resolution_minutes
        if _ttl_min is None:
            _med = SURVIVOR_DATA[slug].get("market_end_date") or ""
            if _med and _med not in ("2500-12-31", ""):
                try:
                    _days_left = (date.fromisoformat(_med) - date.today()).days
                    _ttl_min = max(0.0, _days_left * 24 * 60)
                except Exception:
                    pass
        if _ttl_min is not None:
            cycle_close_buffer = _ttl_min < args.close_buffer_minutes

        print(f"  inventory           : {current_inventory:.0f} shares  "
              f"(base={base_inv:.0f}  excess={cur_excess:.0f})")
        print(f"  inventory_tier      : {cycle_tier}")
        if rch is not None:
            print(f"  reward_cover_hours  : {rch:.1f}h  "
                  f"({'SELL NOW — exceeds 24h threshold' if rch > 24 else 'within 24h, acceptable'})")
        if cycle_close_buffer:
            _src = SURVIVOR_DATA[slug].get("market_end_date") or "manual"
            print(f"  [CLOSE-BUFFER] ttl≈{_ttl_min:.0f}min  end_date={_src}")

        # ── Inventory bootstrap ──────────────────────────────────────────
        # Buy YES token shortfall before entering the cycle.
        # Skipped in dry_run.  Skips cycle (continue) only on COLLATERAL_INSUFFICIENT.
        # All other non-ALREADY_READY verdicts proceed — in-cycle gate is the backstop.
        if args.live and current_inventory < base_inv:
            from research_lines.auto_maker_loop.modules.inventory_bootstrap import (
                run_bootstrap,
            )
            _boot = run_bootstrap(
                client=client,
                token_id=SURVIVOR_DATA[slug]["token_id"],
                creds=creds,
                required_shares=base_inv,
                dry_run=False,
                price_ref=SURVIVOR_DATA[slug].get("yes_price_ref"),
            )
            print(f"  inventory bootstrap : {_boot['verdict']}"
                  f"  balance={_boot.get('balance_shares', 0.0):.1f}"
                  f"  shortfall={_boot.get('shortfall', 0.0):.1f}")
            if _boot["verdict"] in ("COLLATERAL_INSUFFICIENT", "PRICE_UNAVAILABLE"):
                print(f"  SKIP CYCLE: bootstrap aborted ({_boot['verdict']})")
                continue
            # Refresh inventory count so tier/reduce_only use post-bootstrap balance
            current_inventory = _boot.get("balance_shares", current_inventory)

        # ── Heartbeat check ──────────────────────────────────────────────
        if not _check_heartbeat(client):
            if args.live:
                logger.error("heartbeat failed (live) — cancelling open orders and stopping")
                _cancel_all_open_orders(
                    client,
                    token_id=SURVIVOR_DATA[slug]["token_id"],
                    reason="heartbeat_failure",
                )
                break
            logger.error("heartbeat failed — sleeping 60s then retrying")
            time.sleep(60)
            continue

        # ── Pre-placement reconciliation: clear stale orders ─────────────
        _stale_token_id = SURVIVOR_DATA[slug]["token_id"]
        _stale_count = _reconcile_open_orders(client, _stale_token_id)
        if _stale_count > 0:
            print(f"  [RECONCILE] cleared {_stale_count} stale order(s) before placement")

        # ── Run cycle ────────────────────────────────────────────────────
        try:
            result = _run_cycle(
                sa, ms, pm, slug, client, creds, args, cycle_num,
                survivor_data=SURVIVOR_DATA,
                reduce_only=cycle_reduce_only,
                close_buffer=cycle_close_buffer,
                inventory_tier=cycle_tier,
                excess_shares=cur_excess,
                reward_cover_hours=rch,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user — cancelling open orders and exiting.")
            _cancel_all_open_orders(client, token_id=SURVIVOR_DATA[slug]["token_id"], reason="keyboard_interrupt")
            break
        except Exception as exc:
            logger.exception("cycle %d failed: %s", cycle_num, exc)
            if args.live:
                logger.error("exception in live cycle — cancelling open orders (failsafe)")
                _cancel_all_open_orders(
                    client,
                    token_id=SURVIVOR_DATA.get(slug, {}).get("token_id", ""),
                    reason="exception_failsafe",
                )
            result = {
                "ts":      datetime.now(timezone.utc).isoformat(),
                "cycle":   cycle_num,
                "slug":    slug,
                "outcome": "ERROR",
                "error":   str(exc),
                "kill_switch_triggered": False,
            }

        # ── Kill switch: signature / auth error → cancelAll + stop ───────
        if result.get("kill_switch_triggered"):
            logger.error(
                "KILL_SWITCH triggered (cycle %d): %s — cancelling all open orders and stopping.",
                cycle_num, result.get("error", result.get("outcome")),
            )
            _cancel_all_open_orders(
                client,
                token_id=SURVIVOR_DATA.get(slug, {}).get("token_id", ""),
                reason="kill_switch",
            )
            result["outcome_code"] = "infrastructure_failure"
            _append_run(result)
            print("\n  KILL SWITCH TRIGGERED — see log for details.")
            break

        # ── Update fill recency for delay logic ──────────────────────────
        if result.get("bid_filled"):
            last_bid_fill_ts = datetime.now(timezone.utc)

        # ── Classify and record ──────────────────────────────────────────
        from research_lines.auto_maker_loop.modules.cycle_analyst import (
            classify_outcome, print_cycle_summary, print_study_summary, load_runs,
        )
        result["outcome_code"] = classify_outcome(result)
        _append_run(result)
        print_cycle_summary(result)

        # ── Per-market daily loss limit ──────────────────────────────────
        _pnl_cents = result.get("pnl_cents", 0.0) or 0.0
        _pos_size  = result.get("size", 0.0) or 0.0
        _cycle_pnl_usd = (_pnl_cents / 100.0) * _pos_size
        if _cycle_pnl_usd < 0:
            _market_daily_loss[slug] = _market_daily_loss.get(slug, 0.0) + abs(_cycle_pnl_usd)
            _loss_limit = args.market_daily_loss_limit_usd
            print(f"\n  [DAILY LOSS] {slug[:40]}: ${_market_daily_loss[slug]:.2f} cumulative loss this session")
            if _market_daily_loss[slug] >= _loss_limit:
                print(f"  [DAILY LOSS LIMIT] ${_loss_limit:.2f} limit reached — "
                      f"skipping {slug[:40]} for rest of session")
                if not target_slug:
                    skip_slugs.add(slug)

        # Print rolling study summary every 3 cycles or on final cycle
        all_records = load_runs(RUNS_JSONL)
        if len(all_records) >= 3 or (args.cycles and cycle_num == args.cycles):
            _section("Rolling Study Summary")
            print_study_summary(all_records)

        # If inventory missing, skip this market next cycle
        if result.get("skip_reason") in ("SELL_INVENTORY_MISSING", "SELL_ALLOWANCE_INSUFFICIENT"):
            if not target_slug:
                skip_slugs.add(slug)
                logger.info("added %s to skip_slugs (no inventory)", slug[:40])

        # Re-quote immediately without sleep if drift triggered (price moved, need fresh quotes)
        if result.get("drift_triggered"):
            logger.info("drift_triggered — re-quoting immediately, no cycle sleep")
            continue

        # Sleep
        if args.cycles == 0 or cycle_num < args.cycles:
            print(f"\n  sleeping {args.cycle_sleep}s before next cycle...")
            time.sleep(args.cycle_sleep)

    if _mkt_ws:
        _mkt_ws.stop()

    print("\nAuto maker loop finished.")
    print(f"Results recorded to: {RUNS_JSONL}")


if __name__ == "__main__":
    main()
