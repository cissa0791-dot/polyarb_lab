"""
auto_maker_loop — reward_config_digest
polyarb_lab / research_lines / auto_maker_loop / modules

Normalize a RewardConfig (from scoring_activation.fetch_reward_config) into a
flat dict with a fixed schema.  Every field is always present; missing values
are None, not absent.

This module owns the canonical field names for reward config data.
Callers do not need to know whether data came from a live fetch or a fallback.

Public interface
----------------
    digest(reward_cfg: RewardConfig) -> dict
        Convert one RewardConfig object into the canonical flat dict.

    reward_zone(digest_dict: dict, midpoint: float) -> tuple[float, float]
        Derive (zone_bid, zone_ask) from a digest dict and a live midpoint.
        Use when midpoint is known at quote-compute time.

Output schema (all keys always present)
----------------------------------------
    max_spread_cents   float       reward zone full width in cents
    min_size_shares    float       minimum qualifying order size in shares
    daily_rate_usdc    float       pool rate per day in USD
    competitiveness    float|None  market competitiveness score (live only)
    yes_price_live     float|None  live YES token price (live only)
    reward_zone_bid    float|None  lower bound of reward zone (needs midpoint)
    reward_zone_ask    float|None  upper bound of reward zone (needs midpoint)
    fetched_at         str         ISO-8601 UTC timestamp of this digest
    ok                 bool        True = live fetch succeeded; False = fallback
    source             str         "live" or "fallback"
    error              str|None    error detail when ok=False, else None
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def digest(reward_cfg: Any) -> dict:
    """
    Convert a RewardConfig object into the canonical flat dict.

    Parameters
    ----------
    reward_cfg : RewardConfig
        Return value of scoring_activation.fetch_reward_config().

    Returns
    -------
    dict with fixed schema (see module docstring).
    """
    ok     = bool(getattr(reward_cfg, "fetch_ok", False))
    source = str(getattr(reward_cfg, "source", "fallback"))

    max_spread_cents = float(getattr(reward_cfg, "max_spread_cents", 0.0))
    min_size_shares  = float(getattr(reward_cfg, "min_size", 0.0))
    daily_rate_usdc  = float(getattr(reward_cfg, "daily_rate_usdc", 0.0))
    competitiveness  = _safe_float(getattr(reward_cfg, "competitiveness", None))
    yes_price_live   = _safe_float(getattr(reward_cfg, "yes_price_live", None))

    # Derive reward zone using yes_price_live as midpoint proxy when available.
    # Caller can recompute with a fresher midpoint via reward_zone().
    zone_bid: Optional[float] = None
    zone_ask: Optional[float] = None
    if yes_price_live is not None and max_spread_cents > 0:
        half = max_spread_cents / 2.0 / 100.0
        zone_bid = round(yes_price_live - half, 4)
        zone_ask = round(yes_price_live + half, 4)

    return {
        "max_spread_cents": max_spread_cents,
        "min_size_shares":  min_size_shares,
        "daily_rate_usdc":  daily_rate_usdc,
        "competitiveness":  competitiveness,
        "yes_price_live":   yes_price_live,
        "reward_zone_bid":  zone_bid,
        "reward_zone_ask":  zone_ask,
        "fetched_at":       datetime.now(timezone.utc).isoformat(),
        "ok":               ok,
        "source":           source,
        "error":            None if ok else f"fetch_failed:source={source}",
    }


def reward_zone(digest_dict: dict, midpoint: float) -> tuple[float, float]:
    """
    Recompute reward zone bounds using a live midpoint.

    Use this when the midpoint is known after the digest was created, e.g.
    at quote-compute time.

    Returns (zone_bid, zone_ask).
    """
    half = digest_dict["max_spread_cents"] / 2.0 / 100.0
    return round(midpoint - half, 4), round(midpoint + half, 4)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
