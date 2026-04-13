"""Read-side reward income estimator for maker_rewarded_event_mm_v1.

Reads persisted opportunity_candidates and position_events from a SQLite DB
and reports estimated daily reward income using the wide_scan_ev already
computed and stored at scan time.  No schema changes, no runtime changes,
no network calls.

KEY LIMITATION: reward_ev uses a competition_factor heuristic (10–30 estimated
competitors).  The actual competitive landscape is unknown.  These figures are
model estimates, not observed Polymarket reward credits.  They are useful for:
  - sanity-checking that the EV model predicts positive reward income
  - estimating break-even hold hours before a real-money probe
  - identifying which markets have the highest expected reward per dollar held

They do NOT substitute for observing actual reward credits from Polymarket.

Usage:
    py -3 scripts/run_reward_estimate.py [--db PATH] [--settings PATH]
        [--min-edge CENTS] [--out PATH]

DB path resolution (first match wins):
    1. --db argument
    2. config.storage.sqlite_url from --settings file
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MIN_EDGE_CENTS = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate reward income for maker_rewarded_event_mm_v1 candidates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=None, metavar="PATH",
                        help="Path to SQLite DB. Defaults to config sqlite_url.")
    parser.add_argument("--settings", default="config/settings.yaml", metavar="PATH",
                        help="Runtime settings YAML. Default: config/settings.yaml")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE_CENTS, metavar="CENTS",
                        help=f"Minimum net_edge_cents to include. Default: {DEFAULT_MIN_EDGE_CENTS}")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Optional path to write full estimate as JSON.")
    return parser.parse_args()


def _resolve_db(args: argparse.Namespace) -> Path:
    if args.db:
        return Path(args.db)
    try:
        from src.config_runtime.loader import load_runtime_config
        from src.reporting.analytics import resolve_sqlite_path
        config = load_runtime_config(args.settings)
        return resolve_sqlite_path(config.storage.sqlite_url)
    except Exception as exc:
        print(f"ERROR: could not load settings from {args.settings!r}: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_candidates(conn: sqlite3.Connection, min_edge: float) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT candidate_id, net_edge_cents, metadata_json
        FROM opportunity_candidates
        WHERE net_edge_cents >= ?
        """,
        (min_edge,),
    )
    rows = []
    for cid, edge, mj in cur.fetchall():
        meta = json.loads(mj)
        inner = meta.get("metadata", {})
        ev = inner.get("wide_scan_ev", {})
        if not ev or ev.get("reward_ev") is None:
            continue
        rows.append({
            "candidate_id":       cid,
            "net_edge_cents":     edge,
            "market_slugs":       meta.get("market_slugs", []),
            "reward_daily_rate":  inner.get("reward_daily_rate", 0.0),
            "rewards_min_size":   inner.get("rewards_min_size", 0.0),
            "rewards_max_spread": inner.get("rewards_max_spread", 0.0),
            "quote_size":         ev.get("quote_size", 0.0),
            "reward_ev":          ev.get("reward_ev", 0.0),
            "spread_capture_ev":  ev.get("spread_capture_ev", 0.0),
            "adverse_cost":       ev.get("adverse_cost", 0.0),
            "inventory_cost":     ev.get("inventory_cost", 0.0),
            "cancel_cost":        ev.get("cancel_cost", 0.0),
            "total_ev":           ev.get("total_ev", 0.0),
        })
    return rows


def _load_positions(conn: sqlite3.Connection, candidate_ids: set[str]) -> dict[str, list[dict]]:
    """Return position_opened events grouped by candidate_id."""
    if not candidate_ids:
        return {}
    cur = conn.cursor()
    placeholders = ",".join("?" * len(candidate_ids))
    cur.execute(
        f"""
        SELECT candidate_id, symbol, payload_json, ts
        FROM position_events
        WHERE event_type = 'position_opened'
          AND candidate_id IN ({placeholders})
        """,
        list(candidate_ids),
    )
    result: dict[str, list[dict]] = {}
    for cid, sym, pj, ts in cur.fetchall():
        pe = json.loads(pj)
        result.setdefault(cid, []).append({
            "symbol":         sym,
            "filled_size":    float(pe.get("filled_size") or 0.0),
            "avg_fill_price": float(pe.get("avg_fill_price") or 0.0),
            "ts":             ts,
        })
    return result


def _build_estimate(candidate: dict, positions: list[dict]) -> dict:
    """Combine one candidate's EV snapshot with its actual fill data."""
    reward_ev    = candidate["reward_ev"]
    total_ev     = candidate["total_ev"]
    quote_size   = candidate["quote_size"]
    min_size     = candidate["rewards_min_size"]

    # Actual notional filled for this candidate across all legs.
    total_notional = sum(p["filled_size"] * p["avg_fill_price"] for p in positions)

    # Fill coverage: how much of rewards_min_size was actually filled.
    # Reward scales linearly with fill coverage (standard Polymarket model).
    fill_coverage = min(total_notional / min_size, 1.0) if min_size > 0 else 0.0

    # Scale the model reward estimate by actual fill coverage.
    est_daily_reward = round(reward_ev * fill_coverage, 6)
    est_daily_total  = round(total_ev  * fill_coverage, 6)

    # Annual yield on notional held (informational — assumes continuous rewarding).
    est_annual_yield_pct = (
        round((est_daily_reward / total_notional) * 365 * 100, 2)
        if total_notional > 1e-9 else None
    )

    # Break-even hold (hours) for total_ev to cover 1 USD of theoretical cost.
    # With fee=0 for makers, this is the hold needed to earn $1 net.
    break_even_hours_per_dollar = (
        round(24.0 / est_daily_total, 2) if est_daily_total > 1e-9 else None
    )

    return {
        "candidate_id":          candidate["candidate_id"],
        "market_slugs":          candidate["market_slugs"],
        "net_edge_cents":        candidate["net_edge_cents"],
        "reward_daily_rate":     candidate["reward_daily_rate"],
        "rewards_min_size":      candidate["rewards_min_size"],
        "quote_size_usd":        quote_size,
        "total_notional_held":   round(total_notional, 4),
        "fill_coverage":         round(fill_coverage, 4),
        "est_daily_reward_usd":  est_daily_reward,
        "est_daily_total_ev_usd": est_daily_total,
        "est_annual_yield_pct":  est_annual_yield_pct,
        "break_even_hours_per_dollar": break_even_hours_per_dollar,
        "positions_found":       len(positions),
        "ev_components": {
            "reward_ev":         candidate["reward_ev"],
            "spread_capture_ev": candidate["spread_capture_ev"],
            "adverse_cost":      candidate["adverse_cost"],
            "inventory_cost":    candidate["inventory_cost"],
            "cancel_cost":       candidate["cancel_cost"],
        },
    }


def _print_summary(estimates: list[dict], min_edge: float, db_path: str) -> None:
    filled   = [e for e in estimates if e["positions_found"] > 0]
    unfilled = [e for e in estimates if e["positions_found"] == 0]

    print(f"db:               {db_path}")
    print(f"min_edge_cents:   {min_edge}")
    print(f"qualifying_candidates: {len(estimates)}")
    print(f"  with_fills:     {len(filled)}")
    print(f"  no_fills:       {len(unfilled)}")
    print()

    if filled:
        total_notional   = sum(e["total_notional_held"] for e in filled)
        total_daily_rew  = sum(e["est_daily_reward_usd"] for e in filled)
        total_daily_ev   = sum(e["est_daily_total_ev_usd"] for e in filled)
        avg_yield        = (
            round((total_daily_rew / total_notional) * 365 * 100, 2)
            if total_notional > 0 else None
        )
        print("=== filled positions ===")
        print(f"  total_notional_held:       ${total_notional:.2f}")
        print(f"  est_daily_reward_usd:      ${total_daily_rew:.4f}   [MODEL — see caveat below]")
        print(f"  est_daily_total_ev_usd:    ${total_daily_ev:.4f}")
        print(f"  est_annual_yield_pct:      {avg_yield}%   [NOT observed — see caveat]")
        print()

        print("  per-candidate breakdown:")
        for e in sorted(filled, key=lambda x: -x["est_daily_reward_usd"])[:10]:
            slug = (e["market_slugs"] or ["unknown"])[0][:50]
            print(
                f"    {slug:<50}"
                f"  notional=${e['total_notional_held']:>7.2f}"
                f"  rew/day=${e['est_daily_reward_usd']:>7.4f}"
                f"  fill_cov={e['fill_coverage']:.2f}"
                f"  break_even={e['break_even_hours_per_dollar']}h/$"
            )
        print()

    print("NOTE — CAVEAT:")
    print("  reward_ev uses a competition_factor heuristic (10–30 estimated competitors).")
    print("  Actual Polymarket reward distribution depends on the true competitive landscape.")
    print("  These figures are MODEL ESTIMATES only, not observed reward credits.")
    print("  The only way to validate reward income is a real-money hold of >= 24h.")


def main() -> None:
    args = parse_args()

    db_path = _resolve_db(args)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    try:
        candidates = _load_candidates(conn, args.min_edge)
        if not candidates:
            print(f"No qualifying candidates found (min_edge_cents={args.min_edge}).")
            return

        candidate_ids = {c["candidate_id"] for c in candidates}
        positions_by_cand = _load_positions(conn, candidate_ids)

        estimates = [
            _build_estimate(c, positions_by_cand.get(c["candidate_id"], []))
            for c in candidates
        ]
    finally:
        conn.close()

    _print_summary(estimates, args.min_edge, str(db_path))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"db_path": str(db_path), "min_edge_cents": args.min_edge, "estimates": estimates},
                ensure_ascii=False, indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nfull estimate written to: {out_path}")


if __name__ == "__main__":
    main()
