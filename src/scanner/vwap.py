from __future__ import annotations

from typing import Iterable, Optional


def buy_cost_from_asks(asks, target_notional_usd: float):
    remaining = float(target_notional_usd)
    spent = 0.0
    shares = 0.0

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
        spent += take
        shares += take_shares
        remaining -= take

    if remaining > 1e-9 or shares == 0:
        return None
    return {"spent": spent, "shares": shares, "vwap": spent / shares}


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
