from __future__ import annotations

from typing import Any


def buy_cost_from_asks(asks, target_notional_usd: float):
    analysis = analyze_buy_cost_from_asks(asks, target_notional_usd)
    if not analysis["filled"] or analysis["vwap"] is None:
        return None
    return {
        "spent": analysis["spent"],
        "shares": analysis["shares"],
        "vwap": analysis["vwap"],
    }


def analyze_buy_cost_from_asks(asks, target_notional_usd: float) -> dict[str, Any]:
    remaining = float(target_notional_usd)
    spent = 0.0
    shares = 0.0
    levels_consumed = 0

    for level in asks:
        px = float(level.price)
        sz = float(level.size)
        if px <= 0 or sz <= 0:
            continue
        level_notional = px * sz
        take = min(remaining, level_notional)
        if take <= 0:
            break
        take_shares = take / px
        if take_shares > 0:
            levels_consumed += 1
        spent += take
        shares += take_shares
        remaining -= take

    vwap = (spent / shares) if shares > 0 else None
    return {
        "spent": spent,
        "shares": shares,
        "vwap": vwap,
        "filled": remaining <= 1e-9 and shares > 0,
        "remaining_notional": max(remaining, 0.0),
        "levels_consumed": levels_consumed,
    }


def sell_value_from_bids(bids, target_shares: float):
    remaining = float(target_shares)
    received = 0.0
    sold = 0.0

    for level in bids:
        px = float(level.price)
        sz = float(level.size)
        if px <= 0 or sz <= 0:
            continue
        take_shares = min(remaining, sz)
        if take_shares <= 0:
            break
        received += take_shares * px
        sold += take_shares
        remaining -= take_shares

    if remaining > 1e-9 or sold == 0:
        return None
    return {"received": received, "shares": sold, "vwap": received / sold}
