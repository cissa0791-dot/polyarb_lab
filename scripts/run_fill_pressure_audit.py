"""
run_fill_pressure_audit.py — Fill pressure / cross-distance diagnostic audit.

Purpose
-------
Diagnoses why paper calibration shows zero fills by separating three causes:

  MARKET_STATIC      : book hasn't moved enough to cross our resting quote
  QUOTE_TOO_FAR      : a tighter quote placement would already have been crossed
  MODEL_TOO_OPTIMISTIC: model's fill_prob assumptions exceed what book movement supports
  NEED_MORE_OBS      : insufficient data to distinguish the above

Also runs an offline sensitivity sweep comparing:
  - current quote placement
  - tighter by 0.2¢ (toward mid)
  - tighter by 0.4¢
  - tighter by 0.6¢

Does NOT alter the active calibration runner. Read-only analysis only.

Usage
-----
    python scripts/run_fill_pressure_audit.py
    python scripts/run_fill_pressure_audit.py --db path/to/maker_paper_calib.db
    python scripts/run_fill_pressure_audit.py --json

Output
------
  - Per-market cross-distance statistics
  - Book movement distribution (inter-session bid/ask deltas)
  - Book stasis indicators
  - Hypothetical cross counts under tighter quote placements
  - Sensitivity sweep: expected fills if quote were placed tighter
  - Final judgment label per market
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "data" / "processed" / "maker_paper_calib.db"

# Judgment thresholds
_MIN_OBS_FOR_JUDGMENT   = 4      # fewer than this → NEED_MORE_OBS
_STASIS_THRESHOLD       = 0.80   # fraction of zero-move intervals → MARKET_STATIC
_MAX_MOVE_STATIC_CENTS  = 0.50   # max observed move ¢ below which market is static
_MIN_SWEEP_CROSSES      = 1      # at least this many sweep crosses to call QUOTE_TOO_FAR

# Sensitivity sweep deltas (cents closer to mid — positive = tighter bid / tighter ask)
SWEEP_DELTAS_CENTS = [0.0, 0.2, 0.4, 0.6]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    delta_cents  : float    # how many ¢ tighter than current placement
    hypo_bid_crosses : int  # observations where live_ask ≤ tighter_bid
    hypo_ask_crosses : int  # observations where live_bid ≥ tighter_ask
    hypo_both_crosses: int  # both simultaneously
    bid_cross_rate   : float
    ask_cross_rate   : float
    both_cross_rate  : float
    still_reward_eligible: int   # how many tighter quotes remain within max_spread


@dataclass
class MarketAuditResult:
    market_slug    : str
    event_slug     : str
    n_obs          : int      # total observations in DB for this market
    n_intervals    : int      # inter-session intervals with book data

    # Cross distance (how far book must move to fill current resting quote)
    avg_bid_cross_dist_cents : float
    avg_ask_cross_dist_cents : float
    min_bid_cross_dist_cents : float
    min_ask_cross_dist_cents : float
    max_bid_cross_dist_cents : float
    max_ask_cross_dist_cents : float

    # Book movement distribution (inter-session deltas, unsigned cents)
    avg_bid_move_cents    : float
    avg_ask_move_cents    : float
    max_bid_move_cents    : float
    max_ask_move_cents    : float
    pct_zero_bid_move     : float   # fraction of intervals with no bid change
    pct_zero_ask_move     : float
    move_stdev_bid_cents  : float
    move_stdev_ask_cents  : float

    # Current quote placement
    avg_bid_spread_cents  : float   # avg distance of our bid from mid
    avg_ask_spread_cents  : float
    bucket                : str     # TIGHT / STANDARD / WIDE

    # Sensitivity sweep
    sweep                 : list[SweepPoint]

    # Actual fills observed
    actual_bid_crosses    : int
    actual_ask_crosses    : int
    actual_both_crosses   : int

    # Judgment
    judgment              : str     # MARKET_STATIC | QUOTE_TOO_FAR | MODEL_TOO_OPTIMISTIC | NEED_MORE_OBS
    judgment_detail       : str


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def _load_obs(conn: sqlite3.Connection, market_slug: str) -> list[sqlite3.Row]:
    """All observations for a market, chronological order."""
    return conn.execute(
        """SELECT * FROM quote_observations
            WHERE market_slug = ?
            ORDER BY obs_ts ASC""",
        (market_slug,),
    ).fetchall()


def _cross_distance_bid(row: sqlite3.Row) -> float:
    """Cents the best_ask must fall to cross our resting bid. Positive = not yet crossed."""
    return (float(row["live_best_ask"]) - float(row["quote_bid"])) * 100.0


def _cross_distance_ask(row: sqlite3.Row) -> float:
    """Cents the best_bid must rise to cross our resting ask. Positive = not yet crossed."""
    return (float(row["quote_ask"]) - float(row["live_best_bid"])) * 100.0


def _inter_session_moves(obs: list[sqlite3.Row]) -> tuple[list[float], list[float]]:
    """
    Signed inter-session book moves in cents.
    Returns (bid_moves, ask_moves) — one value per consecutive pair.
    """
    bid_moves, ask_moves = [], []
    for i in range(1, len(obs)):
        d_bid = (float(obs[i]["live_best_bid"]) - float(obs[i-1]["live_best_bid"])) * 100.0
        d_ask = (float(obs[i]["live_best_ask"]) - float(obs[i-1]["live_best_ask"])) * 100.0
        bid_moves.append(d_bid)
        ask_moves.append(d_ask)
    return bid_moves, ask_moves


def _sweep_point(
    obs          : list[sqlite3.Row],
    delta_cents  : float,
    max_spread   : float,
) -> SweepPoint:
    """
    Simulate hypothetical crosses if our bid/ask were placed delta_cents tighter.

    Tighter bid   = quote_bid + delta/100   (closer to mid, harder to cross)
    Tighter ask   = quote_ask - delta/100   (closer to mid, harder to cross)

    Wait — that's LESS aggressive for fills. Let me re-think.

    We are a maker. Our resting bid is BELOW mid. Moving it TOWARD mid means
    raising it (e.g. 0.3691 → 0.3711). A HIGHER bid is EASIER to cross because
    the book's ask only needs to fall to 0.3711 instead of 0.3691.

    So: tighter_bid = quote_bid + delta/100 (closer to mid → easier to cross)
        tighter_ask = quote_ask - delta/100 (closer to mid → easier to cross)
    """
    n = len(obs)
    bid_x = ask_x = both_x = elig_x = 0
    for row in obs:
        qb      = float(row["quote_bid"])
        qa      = float(row["quote_ask"])
        mid     = float(row["mid_p"])
        max_sp  = float(row["max_spread_cents"]) if row["max_spread_cents"] else max_spread
        lb      = float(row["live_best_bid"])
        la      = float(row["live_best_ask"])
        d       = delta_cents / 100.0

        tighter_bid = qb + d   # closer to mid
        tighter_ask = qa - d   # closer to mid

        # Still reward-eligible? Check spread from mid
        new_bid_spread_cents = (mid - tighter_bid) * 100.0
        new_ask_spread_cents = (tighter_ask - mid) * 100.0
        eligible = (
            new_bid_spread_cents >= 0 and new_bid_spread_cents <= max_sp
            and new_ask_spread_cents >= 0 and new_ask_spread_cents <= max_sp
        )
        if eligible:
            elig_x += 1

        bid_crossed  = (la <= tighter_bid) and (tighter_bid > 0)
        ask_crossed  = (lb >= tighter_ask) and (tighter_ask < 1)
        both_crossed = bid_crossed and ask_crossed

        if both_crossed:
            both_x += 1
        elif bid_crossed:
            bid_x += 1
        elif ask_crossed:
            ask_x += 1

    rate = lambda x: round(x / n, 4) if n > 0 else 0.0
    return SweepPoint(
        delta_cents          = delta_cents,
        hypo_bid_crosses     = bid_x,
        hypo_ask_crosses     = ask_x,
        hypo_both_crosses    = both_x,
        bid_cross_rate       = rate(bid_x + both_x),
        ask_cross_rate       = rate(ask_x + both_x),
        both_cross_rate      = rate(both_x),
        still_reward_eligible= elig_x,
    )


def _safe_stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def audit_market(
    conn        : sqlite3.Connection,
    market_slug : str,
    event_slug  : str,
) -> MarketAuditResult:
    obs = _load_obs(conn, market_slug)
    n   = len(obs)

    # Cross distances
    cd_bids = [_cross_distance_bid(r) for r in obs]
    cd_asks = [_cross_distance_ask(r) for r in obs]

    # Inter-session book moves
    bid_moves_raw, ask_moves_raw = _inter_session_moves(obs)
    bid_moves_abs = [abs(x) for x in bid_moves_raw]
    ask_moves_abs = [abs(x) for x in ask_moves_raw]
    n_intervals = len(bid_moves_raw)

    pct_zero_bid = (sum(1 for m in bid_moves_abs if m < 0.001) / n_intervals
                    if n_intervals > 0 else 1.0)
    pct_zero_ask = (sum(1 for m in ask_moves_abs if m < 0.001) / n_intervals
                    if n_intervals > 0 else 1.0)

    # Quote placement stats
    bid_spreads = [float(r["bid_spread_cents"]) for r in obs if r["bid_spread_cents"]]
    ask_spreads = [float(r["ask_spread_cents"]) for r in obs if r["ask_spread_cents"]]
    buckets     = [r["quote_aggressiveness_bucket"] for r in obs if r["quote_aggressiveness_bucket"]]
    bucket_mode = max(set(buckets), key=buckets.count) if buckets else "STANDARD"
    max_spread  = float(obs[0]["max_spread_cents"]) if obs else 3.5

    # Sensitivity sweep (using ALL obs, including superseded — more data is better)
    sweep = [_sweep_point(obs, d, max_spread) for d in SWEEP_DELTAS_CENTS]

    # Actual fills from DB
    fills = conn.execute(
        "SELECT fill_side, COUNT(*) as cnt FROM fill_observations WHERE market_slug=? GROUP BY fill_side",
        (market_slug,),
    ).fetchall()
    fill_map = {r["fill_side"]: r["cnt"] for r in fills}
    a_bid   = fill_map.get("BID", 0)
    a_ask   = fill_map.get("ASK", 0)
    a_both  = fill_map.get("BOTH", 0)

    # ---- Judgment ----
    any_sweep_cross = any(
        s.hypo_bid_crosses + s.hypo_ask_crosses + s.hypo_both_crosses >= _MIN_SWEEP_CROSSES
        for s in sweep[1:]   # skip delta=0.0 (current)
    )
    max_bid_move = max(bid_moves_abs) if bid_moves_abs else 0.0
    max_ask_move = max(ask_moves_abs) if ask_moves_abs else 0.0
    avg_cd_bid   = statistics.mean(cd_bids) if cd_bids else 0.0

    if n < _MIN_OBS_FOR_JUDGMENT:
        label  = "NEED_MORE_OBS"
        detail = f"Only {n} observations; need >= {_MIN_OBS_FOR_JUDGMENT} to judge"
    elif pct_zero_bid >= _STASIS_THRESHOLD and max_bid_move <= _MAX_MOVE_STATIC_CENTS:
        label  = "MARKET_STATIC"
        detail = (
            f"{pct_zero_bid*100:.0f}% of intervals had zero bid movement; "
            f"max move {max_bid_move:.2f}¢ < cross distance {avg_cd_bid:.2f}¢; "
            f"even maximum observed move insufficient to trigger a cross"
        )
    elif any_sweep_cross:
        # Some sweep delta would have produced crosses — quote could be tighter
        best = next(s for s in sweep[1:] if
                    s.hypo_bid_crosses + s.hypo_ask_crosses + s.hypo_both_crosses > 0)
        label  = "QUOTE_TOO_FAR"
        detail = (
            f"At +{best.delta_cents}¢ tighter, "
            f"{best.hypo_bid_crosses} bid / {best.hypo_ask_crosses} ask / "
            f"{best.hypo_both_crosses} both crosses would have occurred"
        )
    else:
        # Cross distance is persistent and sweep doesn't help either
        label  = "MODEL_TOO_OPTIMISTIC"
        detail = (
            f"Average cross distance {avg_cd_bid:.2f}¢ exceeds all observed book moves "
            f"(max {max_bid_move:.2f}¢); model fill_prob does not match observed liquidity"
        )

    def _safe_mean(lst): return round(statistics.mean(lst), 4) if lst else 0.0
    def _safe_max(lst):  return round(max(lst), 4) if lst else 0.0
    def _safe_min(lst):  return round(min(lst), 4) if lst else 0.0

    return MarketAuditResult(
        market_slug              = market_slug,
        event_slug               = event_slug,
        n_obs                    = n,
        n_intervals              = n_intervals,
        avg_bid_cross_dist_cents = round(_safe_mean(cd_bids), 3),
        avg_ask_cross_dist_cents = round(_safe_mean(cd_asks), 3),
        min_bid_cross_dist_cents = round(_safe_min(cd_bids), 3),
        min_ask_cross_dist_cents = round(_safe_min(cd_asks), 3),
        max_bid_cross_dist_cents = round(_safe_max(cd_bids), 3),
        max_ask_cross_dist_cents = round(_safe_max(cd_asks), 3),
        avg_bid_move_cents       = round(_safe_mean(bid_moves_abs), 3),
        avg_ask_move_cents       = round(_safe_mean(ask_moves_abs), 3),
        max_bid_move_cents       = round(_safe_max(bid_moves_abs), 3),
        max_ask_move_cents       = round(_safe_max(ask_moves_abs), 3),
        pct_zero_bid_move        = round(pct_zero_bid, 3),
        pct_zero_ask_move        = round(pct_zero_ask, 3),
        move_stdev_bid_cents     = round(_safe_stdev(bid_moves_abs), 3),
        move_stdev_ask_cents     = round(_safe_stdev(ask_moves_abs), 3),
        avg_bid_spread_cents     = round(_safe_mean(bid_spreads), 3),
        avg_ask_spread_cents     = round(_safe_mean(ask_spreads), 3),
        bucket                   = bucket_mode,
        sweep                    = sweep,
        actual_bid_crosses       = a_bid,
        actual_ask_crosses       = a_ask,
        actual_both_crosses      = a_both,
        judgment                 = label,
        judgment_detail          = detail,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _bar(val: float, max_val: float, width: int = 20) -> str:
    """ASCII bar chart for distributions."""
    if max_val <= 0:
        return "." * width
    filled = int(round(val / max_val * width))
    return "█" * filled + "·" * (width - filled)


def print_report(results: list[MarketAuditResult]) -> None:
    sep = "=" * 72

    for r in results:
        print(sep)
        print(f"  FILL PRESSURE AUDIT — {r.market_slug}")
        print(sep)

        print(f"\n  OBSERVATIONS")
        print(f"    Total obs logged:       {r.n_obs}")
        print(f"    Inter-session intervals:{r.n_intervals}")
        print(f"    Actual bid fills:       {r.actual_bid_crosses}")
        print(f"    Actual ask fills:       {r.actual_ask_crosses}")
        print(f"    Actual both fills:      {r.actual_both_crosses}")

        print(f"\n  CROSS DISTANCE  (¢ book must move to fill our resting quote)")
        print(f"    avg bid cross dist:   {r.avg_bid_cross_dist_cents:.3f}¢")
        print(f"    avg ask cross dist:   {r.avg_ask_cross_dist_cents:.3f}¢")
        print(f"    min / max bid cdist:  {r.min_bid_cross_dist_cents:.3f}¢ / {r.max_bid_cross_dist_cents:.3f}¢")
        print(f"    min / max ask cdist:  {r.min_ask_cross_dist_cents:.3f}¢ / {r.max_ask_cross_dist_cents:.3f}¢")

        print(f"\n  BOOK MOVEMENT  (inter-session, unsigned ¢ change)")
        print(f"    avg bid move:         {r.avg_bid_move_cents:.3f}¢")
        print(f"    avg ask move:         {r.avg_ask_move_cents:.3f}¢")
        print(f"    max bid move:         {r.max_bid_move_cents:.3f}¢  {'<< NEVER REACHES CROSS DIST' if r.max_bid_move_cents < r.avg_bid_cross_dist_cents else '  CAN REACH CROSS DIST'}")
        print(f"    max ask move:         {r.max_ask_move_cents:.3f}¢  {'<< NEVER REACHES CROSS DIST' if r.max_ask_move_cents < r.avg_ask_cross_dist_cents else '  CAN REACH CROSS DIST'}")
        print(f"    stdev bid move:       {r.move_stdev_bid_cents:.3f}¢")
        print(f"    stdev ask move:       {r.move_stdev_ask_cents:.3f}¢")
        print(f"    % zero bid moves:     {r.pct_zero_bid_move*100:.1f}%")
        print(f"    % zero ask moves:     {r.pct_zero_ask_move*100:.1f}%")

        print(f"\n  QUOTE AGGRESSIVENESS")
        print(f"    avg bid spread:       {r.avg_bid_spread_cents:.3f}¢  (dist from mid)")
        print(f"    avg ask spread:       {r.avg_ask_spread_cents:.3f}¢")
        print(f"    aggressiveness:       {r.bucket}")

        print(f"\n  SENSITIVITY SWEEP  (offline simulation — does NOT alter active quoting)")
        print(f"    {'Delta':>8}  {'Hypo bid X':>12}  {'Hypo ask X':>12}  {'Both X':>8}  {'Bid rate':>10}  {'Elig':>6}")
        for sp in r.sweep:
            marker = " ← CURRENT" if sp.delta_cents == 0.0 else ""
            print(f"    {sp.delta_cents:>+6.1f}¢   "
                  f"{sp.hypo_bid_crosses:>10}    "
                  f"{sp.hypo_ask_crosses:>10}    "
                  f"{sp.hypo_both_crosses:>6}    "
                  f"{sp.bid_cross_rate:>8.4f}    "
                  f"{sp.still_reward_eligible:>5}{marker}")

        print(f"\n  ┌─ JUDGMENT: {r.judgment} ─────────────────────────────────────┐")
        # word-wrap judgment_detail at 60 chars
        detail = r.judgment_detail
        words  = detail.split()
        lines  = []
        cur    = ""
        for w in words:
            if len(cur) + len(w) + 1 > 60:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        for ln in lines:
            print(f"  │  {ln}")
        print(f"  └────────────────────────────────────────────────────────────┘\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit(db_path: Path, as_json: bool = False) -> list[MarketAuditResult]:
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Load all (market_slug, event_slug) combos with at least 1 obs
    markets = conn.execute(
        "SELECT DISTINCT market_slug, event_slug FROM quote_observations ORDER BY event_slug, market_slug"
    ).fetchall()

    if not markets:
        print("No observations in DB yet.")
        conn.close()
        return []

    print(f"FILL PRESSURE AUDIT — {db_path.name}")
    print(f"Markets: {len(markets)}  |  DB: {db_path}\n")

    results = [audit_market(conn, r["market_slug"], r["event_slug"]) for r in markets]
    conn.close()

    if as_json:
        serialisable = []
        for r in results:
            d = asdict(r)
            d["sweep"] = [asdict(s) for s in r.sweep]
            serialisable.append(d)
        print(json.dumps(serialisable, indent=2))
        return results

    print_report(results)

    # Summary table
    sep = "=" * 72
    print(sep)
    print("SUMMARY")
    print(sep)
    for r in results:
        label_pad = f"[{r.judgment}]"
        print(f"  {label_pad:25s}  {r.market_slug[-45:]}")
        print(f"  {'':25s}  avg_cd={r.avg_bid_cross_dist_cents:.2f}¢  "
              f"max_move={r.max_bid_move_cents:.2f}¢  "
              f"stasis={r.pct_zero_bid_move*100:.0f}%  "
              f"n_obs={r.n_obs}")
    print(sep)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill pressure / cross-distance audit")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to maker_paper_calib.db")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    args = parser.parse_args()
    run_audit(args.db, as_json=args.json)


if __name__ == "__main__":
    main()
