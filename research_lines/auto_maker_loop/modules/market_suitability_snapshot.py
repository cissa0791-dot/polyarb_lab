"""
auto_maker_loop — market_suitability_snapshot
polyarb_lab / research_lines / auto_maker_loop / modules

Hard gate before any order placement.

Previously used only as an observation/logging helper.  This version is the
mandatory pre-trade check in run_auto_maker.py.  If suitable=False, the cycle
skips placement entirely.

Checks performed (all must pass for suitable=True)
---------------------------------------------------
1. Midpoint available and in valid range (0.02 – 0.98)
2. Reward config loaded (ok=True; fallback permitted with a flag)
3. Intended bid/ask prices are inside the reward zone (if zone known)
4. Current inventory does not already exceed position_cap

The check is deliberately narrow:
  - Does NOT decide which posture to use (that is posture_selector's job)
  - Does NOT compute quotes (that is reward_aware_quote_planner_v2's job)
  - Does NOT block on competitiveness threshold (caller decides)

Public interface
----------------
    check(params: SuitabilityParams) -> SuitabilityResult

Output schema
-------------
    suitable            bool
    reason              str        first failing reason, or "OK"
    midpoint_ok         bool
    reward_config_ok    bool
    quotes_in_zone      bool|None  None when zone unknown
    inventory_ok        bool
    spread_cents        float|None computed spread from bid/ask
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SuitabilityParams:
    """Inputs for one suitability check."""
    midpoint:           Optional[float]
    bid_price:          float
    ask_price:          float
    reward_zone_bid:    Optional[float]   # from reward_config_digest
    reward_zone_ask:    Optional[float]
    reward_config_ok:   bool
    current_shares:     float
    position_cap:       float             # from args / inventory_governor
    allow_fallback_config: bool = True    # if False, fallback config → not suitable
    reduce_only: bool = False             # if True, skip inventory cap check (sell/exit-only path)


@dataclass
class SuitabilityResult:
    """Output of the suitability check."""
    suitable:          bool
    reason:            str
    midpoint_ok:       bool
    reward_config_ok:  bool
    quotes_in_zone:    Optional[bool]
    inventory_ok:      bool
    spread_cents:      Optional[float]


def check(params: SuitabilityParams) -> SuitabilityResult:
    """
    Run the pre-trade hard gate.

    Returns SuitabilityResult with suitable=False on the first failing check.
    """
    mid      = params.midpoint
    bid      = params.bid_price
    ask      = params.ask_price
    zone_bid = params.reward_zone_bid
    zone_ask = params.reward_zone_ask
    rok      = params.reward_config_ok
    cur      = params.current_shares
    cap      = params.position_cap

    # ── 1. Midpoint ───────────────────────────────────────────────────────
    mid_ok = mid is not None and 0.02 <= mid <= 0.98
    if not mid_ok:
        return SuitabilityResult(
            suitable=False,
            reason=f"midpoint_invalid(mid={mid})",
            midpoint_ok=False,
            reward_config_ok=rok,
            quotes_in_zone=None,
            inventory_ok=(cur < cap),
            spread_cents=None,
        )

    # ── 2. Reward config ─────────────────────────────────────────────────
    if not rok and not params.allow_fallback_config:
        return SuitabilityResult(
            suitable=False,
            reason="reward_config_ok=False AND allow_fallback_config=False",
            midpoint_ok=True,
            reward_config_ok=False,
            quotes_in_zone=None,
            inventory_ok=(cur < cap),
            spread_cents=None,
        )

    # ── 3. Quotes in zone ────────────────────────────────────────────────
    quotes_in_zone: Optional[bool] = None
    if zone_bid is not None and zone_ask is not None:
        bid_in = bid >= zone_bid
        ask_in = ask <= zone_ask
        quotes_in_zone = bid_in and ask_in
        if not quotes_in_zone:
            return SuitabilityResult(
                suitable=False,
                reason=(
                    f"quotes_outside_zone("
                    f"bid={bid:.4f}<zone_bid={zone_bid:.4f}={not bid_in}, "
                    f"ask={ask:.4f}>zone_ask={zone_ask:.4f}={not ask_in})"
                ),
                midpoint_ok=True,
                reward_config_ok=rok,
                quotes_in_zone=False,
                inventory_ok=(cur < cap),
                spread_cents=round((ask - bid) * 100, 4),
            )

    # ── 4. Inventory cap ─────────────────────────────────────────────────
    # Skipped in reduce_only / exit-only mode: an ASK-only order does not
    # increase inventory, so cap compliance is irrelevant for sell cycles.
    inv_ok = cur < cap or params.reduce_only
    if not inv_ok:
        return SuitabilityResult(
            suitable=False,
            reason=f"inventory_cap_exceeded(cur={cur:.0f}>=cap={cap:.0f})",
            midpoint_ok=True,
            reward_config_ok=rok,
            quotes_in_zone=quotes_in_zone,
            inventory_ok=False,
            spread_cents=round((ask - bid) * 100, 4),
        )

    spread_cents = round((ask - bid) * 100, 4)
    return SuitabilityResult(
        suitable=True,
        reason="OK",
        midpoint_ok=True,
        reward_config_ok=rok,
        quotes_in_zone=quotes_in_zone,
        inventory_ok=True,
        spread_cents=spread_cents,
    )
