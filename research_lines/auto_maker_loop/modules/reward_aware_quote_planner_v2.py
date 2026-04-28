"""
auto_maker_loop — reward_aware_quote_planner_v2
polyarb_lab / research_lines / auto_maker_loop / modules

Compute bid/ask prices and size for one cycle, incorporating:
  - reward band bounds (from reward_config_digest)
  - live midpoint
  - posture (from posture_selector)
  - inventory skew (pulls ASK closer to mid when inventory is elevated)

Replaces the inline compute_qualifying_quotes() call in run_auto_maker.py
for postures that need asymmetric or skewed pricing.
BILATERAL posture without skew falls through to the existing sa.compute_qualifying_quotes().

Pricing rules by posture
------------------------
BILATERAL
    bid = midpoint - half_spread
    ask = midpoint + half_spread
    half_spread = min(max_spread_cents/2, tick) rounded to tick
    Both legs clamped inside reward zone.

ASYMMETRIC_ASK_LEAN  (inventory in SOFT_CAP)
    ask = midpoint + tick              (one tick above mid — more aggressive)
    bid = midpoint - max_spread_cents + tick   (pushed to outer edge of zone)
    Size on BID reduced by inventory_governor's size_factor.

EXIT_ONLY
    ask = midpoint + tick              (one tick above mid)
    bid = None (not placed)

Tick
    Always 0.01 (Polymarket minimum price increment).

Hard constraints (applied after posture pricing)
-------------------------------------------------
  - bid must be >= reward_zone_bid  (or >= midpoint - max_spread_cents/100 if zone unknown)
  - ask must be <= reward_zone_ask  (or <= midpoint + max_spread_cents/100 if zone unknown)
  - ask > bid  (always)
  - 0.01 <= price <= 0.99

Public interface
----------------
    plan(params: PlannerParams) -> PlannerResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_TICK = 0.01


@dataclass
class PlannerParams:
    """Inputs to the planner for one cycle."""
    midpoint:        float
    posture:         str            # from posture_selector
    max_spread_cents: float         # from reward_config_digest
    min_size:        float          # from reward_config_digest
    reward_zone_bid: Optional[float]  # from reward_config_digest (may be None if fallback)
    reward_zone_ask: Optional[float]
    size_factor:     float = 1.0    # from inventory_governor (1.0 = full size)
    base_size:       float = 200.0  # intended lot size before size_factor


@dataclass
class PlannerResult:
    """Output of the planner."""
    bid_price:    Optional[float]   # None when posture=EXIT_ONLY
    ask_price:    float
    bid_size:     Optional[float]   # None when no bid
    ask_size:     float
    qualifying:   bool              # False if constraint violated
    planner_notes: list[str] = field(default_factory=list)


def plan(params: PlannerParams) -> PlannerResult:
    """
    Compute bid/ask prices for one cycle.

    Parameters
    ----------
    params : PlannerParams

    Returns
    -------
    PlannerResult
    """
    mid     = params.midpoint
    ms_frac = params.max_spread_cents / 100.0
    half    = ms_frac / 2.0
    posture = params.posture
    notes: list[str] = []

    # Effective size after inventory governor's size_factor
    raw_size = max(params.min_size, params.base_size)
    bid_size = round(raw_size * params.size_factor)
    ask_size = round(raw_size)      # ASK always at full size

    if posture == "EXIT_ONLY":
        ask = _clamp(_round_tick(mid + _TICK))
        ask = _apply_zone_ask(ask, params.reward_zone_ask, notes)
        return PlannerResult(
            bid_price=None, ask_price=ask,
            bid_size=None,  ask_size=ask_size,
            qualifying=True,
            planner_notes=notes,
        )

    if posture == "ASYMMETRIC_ASK_LEAN":
        # ASK: one tick above midpoint (aggressive, high fill probability)
        ask = _clamp(_round_tick(mid + _TICK))
        # BID: pushed to outer edge of reward zone (lower fill probability)
        bid = _clamp(_round_tick(mid - ms_frac + _TICK))
        ask = _apply_zone_ask(ask, params.reward_zone_ask, notes)
        bid = _apply_zone_bid(bid, params.reward_zone_bid, notes)
        if bid_size < params.min_size:
            bid_size = round(params.min_size)
            notes.append(f"bid_size raised to min_size={params.min_size:.0f} after size_factor")
        ok = _check_qualifying(bid, ask, params, notes)
        return PlannerResult(
            bid_price=bid, ask_price=ask,
            bid_size=bid_size, ask_size=ask_size,
            qualifying=ok,
            planner_notes=notes,
        )

    # BILATERAL (default)
    bid = _clamp(_round_tick(mid - half))
    ask = _clamp(_round_tick(mid + half))
    ask = _apply_zone_ask(ask, params.reward_zone_ask, notes)
    bid = _apply_zone_bid(bid, params.reward_zone_bid, notes)
    if bid_size < params.min_size:
        bid_size = round(params.min_size)
        notes.append(f"bid_size raised to min_size={params.min_size:.0f} after size_factor")
    ok = _check_qualifying(bid, ask, params, notes)
    return PlannerResult(
        bid_price=bid, ask_price=ask,
        bid_size=bid_size, ask_size=ask_size,
        qualifying=ok,
        planner_notes=notes,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_tick(price: float) -> float:
    return round(round(price / _TICK) * _TICK, 4)


def _round_tick_up(price: float) -> float:
    import math

    return round(math.ceil((price - 1e-12) / _TICK) * _TICK, 4)


def _round_tick_down(price: float) -> float:
    import math

    return round(math.floor((price + 1e-12) / _TICK) * _TICK, 4)


def _clamp(price: float) -> float:
    return max(0.01, min(0.99, price))


def _apply_zone_bid(bid: float, zone_bid: Optional[float], notes: list[str]) -> float:
    if zone_bid is not None and bid < zone_bid:
        notes.append(f"bid clamped from {bid:.4f} to zone_bid={zone_bid:.4f}")
        return _round_tick_up(zone_bid)
    return bid


def _apply_zone_ask(ask: float, zone_ask: Optional[float], notes: list[str]) -> float:
    if zone_ask is not None and ask > zone_ask:
        notes.append(f"ask clamped from {ask:.4f} to zone_ask={zone_ask:.4f}")
        return _round_tick_down(zone_ask)
    return ask


def _check_qualifying(
    bid: float,
    ask: float,
    params: PlannerParams,
    notes: list[str],
) -> bool:
    spread_cents = round((ask - bid) * 100, 4)
    if ask <= bid:
        notes.append(f"DISQUALIFIED: ask={ask:.4f} <= bid={bid:.4f}")
        return False
    if spread_cents > params.max_spread_cents:
        notes.append(
            f"DISQUALIFIED: spread={spread_cents:.2f}¢ > max={params.max_spread_cents:.2f}¢"
        )
        return False
    return True
