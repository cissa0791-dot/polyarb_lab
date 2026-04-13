"""
run_reward_aware_viability_screen
polyarb_lab / research_lines / auto_maker_loop

Dry-only active-mainline front door for reward-aware maker viability.

Purpose
-------
Screen the current reward-positive shortlist and answer:
  "Is this market worth bilateral reward-aware quoting now,
   before any inventory/bootstrap logic?"

Per candidate output
--------------------
  - reward_positive
  - maker_viable
  - bootstrap_viable
  - best_bid
  - best_ask
  - market_trade_count
  - sparse_flow
  - dead_book_label
  - current_gate_result

Gate outcomes
-------------
  QUOTEABLE_NOW
      reward-positive, active now, and not in the current dead-book regime

  DEAD_BOOK_NOW
      current runtime view matches the dead-book bootstrap block:
      best_bid <= 0.01, best_ask >= 0.99, zero recent trades

  INACTIVE_NOW
      market is no longer active / accepting orders

  SCREEN_ERROR
      current public metadata/runtime view could not be resolved

Strict rules
------------
  - Dry-only. No order submission.
  - No inventory/bootstrap logic.
  - No inactive-inventory governance work.
  - No optimizer logic.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

from research_lines.auto_maker_loop.modules.market_ws_client import MarketWsClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reward_aware_viability_screen")

DATA_DIR = Path("data/research/auto_maker_loop")
PROBE_PATH = Path("data/research/reward_aware_maker_probe/latest_probe.json")
RUNS_JSONL = DATA_DIR / "unbalanced_leg_pilot_runs.jsonl"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
INACTIVE_TERMINAL_LABELS = {
    "LEGACY_UNGOVERNED_INVENTORY",
    "PARKED_STRANDED_POSITION",
}
PROBE_TO_RUNTIME_SLUG = {
    "will-jd-vance-win-the-2028-republican-presidential-nomination":
        "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "will-marco-rubio-win-the-2028-republican-presidential-nomination":
        "will-marco-rubio-win-the-2028-republican-presidential-n",
}


def _load_positive_shortlist() -> list[dict[str, Any]]:
    obj = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for market in obj.get("markets", []):
        slug = market.get("market_slug")
        raw_ev = market.get("reward_adjusted_raw_ev")
        reward_cfg = market.get("reward_config_summary") or {}
        if not slug or not isinstance(raw_ev, (int, float)) or raw_ev <= 0:
            continue
        rows.append(
            {
                "slug": slug,
                "raw_ev": float(raw_ev),
                "reward_rate_daily_usdc": float(
                    reward_cfg.get("reward_daily_rate_usdc") or 0.0
                ),
                "rewards_min_size_shares": float(
                    reward_cfg.get("rewards_min_size_shares") or 0.0
                ),
                "rewards_max_spread_cents": float(
                    reward_cfg.get("rewards_max_spread_cents") or 0.0
                ),
                "paper_best_bid": market.get("best_bid"),
                "paper_best_ask": market.get("best_ask"),
                "paper_midpoint": market.get("midpoint"),
                "modeled_quote_spread": market.get("quoted_spread"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["raw_ev"],
            row["reward_rate_daily_usdc"],
        ),
        reverse=True,
    )
    return rows


def _load_latest_runtime_by_slug() -> dict[str, dict[str, Any]]:
    if not RUNS_JSONL.exists():
        return {}
    rows = [
        json.loads(line)
        for line in RUNS_JSONL.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = row.get("target_slug")
        if slug:
            latest[str(slug)] = row
    return latest


def _runtime_slug(slug: str) -> str:
    return PROBE_TO_RUNTIME_SLUG.get(slug, slug)


def _extract_yes_token_id(gamma_row: dict[str, Any]) -> str:
    raw_token_ids = gamma_row.get("clobTokenIds") or "[]"
    raw_outcomes = gamma_row.get("outcomes") or "[]"
    token_ids = json.loads(raw_token_ids) if isinstance(raw_token_ids, str) else raw_token_ids
    outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    if isinstance(token_ids, list) and isinstance(outcomes, list):
        for outcome, token_id in zip(outcomes, token_ids):
            if str(outcome).strip().lower() == "yes":
                return str(token_id)
    if isinstance(token_ids, list) and token_ids:
        return str(token_ids[0])
    return ""


def _fetch_gamma_market(slug: str) -> dict[str, Any]:
    response = requests.get(
        GAMMA_MARKETS_URL,
        params={"slug": slug, "limit": 5},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and row.get("slug") == slug:
                return row
        if payload and isinstance(payload[0], dict):
            return payload[0]
    if isinstance(payload, dict) and payload.get("slug") == slug:
        return payload
    raise ValueError(f"Gamma market lookup failed for slug={slug}")


def _observe_market(token_id: str, slug: str, observe_seconds: float) -> dict[str, Any]:
    safe_slug = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in slug.lower()
    ).strip("_")[:64] or "target"
    ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    log_path = DATA_DIR / f"reward_viability_market_{safe_slug}_{ts_label}.jsonl"
    ws = MarketWsClient(token_ids=[token_id], log_path=log_path)
    ws.start()
    time.sleep(max(observe_seconds, 0.0))
    ws.stop()
    return {
        "market_log": str(log_path),
        "market_trade_count": int(ws.trade_count),
        "events_received": int(ws.events_received),
        "best_bid": ws.last_best_bid,
        "best_ask": ws.last_best_ask,
    }


def _dead_book_label(
    best_bid: Optional[float],
    best_ask: Optional[float],
    market_trade_count: int,
    sparse_flow: bool,
) -> Optional[str]:
    if best_bid is None or best_ask is None:
        return None
    if market_trade_count != 0 or not sparse_flow:
        return None
    if float(best_bid) <= 0.001 and float(best_ask) >= 0.999:
        return "PINNED_001_999_NO_FLOW"
    if float(best_bid) <= 0.01 and float(best_ask) >= 0.99:
        return "PINNED_01_99_NO_FLOW"
    return None


def _screen_candidate(
    candidate: dict[str, Any],
    latest_runtime_by_slug: dict[str, dict[str, Any]],
    observe_seconds: float,
) -> dict[str, Any]:
    slug = str(candidate["slug"])
    runtime_slug = _runtime_slug(slug)
    latest_runtime = latest_runtime_by_slug.get(runtime_slug) or {}
    latest_terminal = latest_runtime.get("terminal_audit_label")

    gamma_row = _fetch_gamma_market(slug)
    token_id = _extract_yes_token_id(gamma_row)
    if not token_id:
        return {
            "slug": slug,
            "runtime_slug": runtime_slug,
            "reward_positive": True,
            "maker_viable": False,
            "bootstrap_viable": False,
            "best_bid": None,
            "best_ask": None,
            "market_trade_count": 0,
            "sparse_flow": True,
            "dead_book_label": "METADATA_UNRESOLVED",
            "current_gate_result": "SCREEN_ERROR",
            "raw_ev": candidate["raw_ev"],
            "reward_rate_daily_usdc": candidate["reward_rate_daily_usdc"],
            "volume_24hr": gamma_row.get("volume24hr"),
            "latest_terminal_audit_label": latest_terminal,
            "market_log": None,
        }

    observed = _observe_market(token_id, slug, observe_seconds)
    best_bid = (
        observed["best_bid"]
        if observed["best_bid"] is not None
        else gamma_row.get("bestBid")
    )
    best_ask = (
        observed["best_ask"]
        if observed["best_ask"] is not None
        else gamma_row.get("bestAsk")
    )
    market_trade_count = int(observed["market_trade_count"])
    sparse_flow = market_trade_count == 0
    dead_book = _dead_book_label(best_bid, best_ask, market_trade_count, sparse_flow)

    active_now = bool(gamma_row.get("active")) and not bool(gamma_row.get("closed"))
    accepting_orders = bool(gamma_row.get("acceptingOrders"))
    maker_viable = bool(candidate["raw_ev"] > 0) and active_now and accepting_orders and dead_book is None
    bootstrap_viable = active_now and accepting_orders and dead_book is None

    if dead_book is not None:
        gate_result = "DEAD_BOOK_NOW"
    elif not active_now or not accepting_orders:
        gate_result = "INACTIVE_NOW"
    else:
        gate_result = "QUOTEABLE_NOW"

    return {
        "slug": slug,
        "runtime_slug": runtime_slug,
        "reward_positive": True,
        "maker_viable": maker_viable,
        "bootstrap_viable": bootstrap_viable,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "market_trade_count": market_trade_count,
        "sparse_flow": sparse_flow,
        "dead_book_label": dead_book,
        "current_gate_result": gate_result,
        "raw_ev": candidate["raw_ev"],
        "reward_rate_daily_usdc": candidate["reward_rate_daily_usdc"],
        "volume_24hr": gamma_row.get("volume24hr"),
        "latest_terminal_audit_label": latest_terminal,
        "market_log": observed["market_log"],
    }


def _print_table(results: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    print()
    print("=" * 88)
    print("  REWARD-AWARE MAKER VIABILITY SCREEN")
    print("=" * 88)
    print(
        f"  {'slug':<44}  {'raw_ev':>6}  {'rate':>5}  {'bid':>6}  {'ask':>6}  "
        f"{'trades':>6}  {'sparse':>6}  {'gate':>16}"
    )
    print(
        f"  {'-'*44}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  "
        f"{'-'*6}  {'-'*6}  {'-'*16}"
    )
    for row in results:
        bid = "N/A" if row["best_bid"] is None else f"{float(row['best_bid']):.3f}"
        ask = "N/A" if row["best_ask"] is None else f"{float(row['best_ask']):.3f}"
        print(
            f"  {row['slug'][:44]:<44}  {row['raw_ev']:>6.3f}  "
            f"{int(row['reward_rate_daily_usdc']):>5}  {bid:>6}  {ask:>6}  "
            f"{row['market_trade_count']:>6}  {str(row['sparse_flow']):>6}  "
            f"{row['current_gate_result']:>16}"
        )
    if skipped:
        print()
        print("  Skipped inactive-governed targets:")
        for row in skipped:
            print(
                f"    - {row['slug']}  ({row['latest_terminal_audit_label']})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-only reward-aware maker viability screen.\n"
            "Screens reward-positive candidates before any inventory/bootstrap logic."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--observe-seconds",
        type=float,
        default=10.0,
        help="Public market WS observe window per candidate (default: 10)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Limit number of screened candidates after filtering (default: all)",
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        default=None,
        help="Optional explicit candidate slugs to screen",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include targets already governed inactive in runs_jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    shortlist = _load_positive_shortlist()
    if args.slugs:
        requested = set(args.slugs)
        shortlist = [row for row in shortlist if row["slug"] in requested]

    latest_runtime_by_slug = _load_latest_runtime_by_slug()
    skipped_inactive: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in shortlist:
        latest_runtime = latest_runtime_by_slug.get(_runtime_slug(row["slug"])) or {}
        latest_terminal = latest_runtime.get("terminal_audit_label")
        if (
            not args.include_inactive
            and latest_terminal in INACTIVE_TERMINAL_LABELS
        ):
            skipped_inactive.append(
                {
                    "slug": row["slug"],
                    "latest_terminal_audit_label": latest_terminal,
                }
            )
            continue
        filtered.append(row)

    if args.max_candidates and args.max_candidates > 0:
        filtered = filtered[: args.max_candidates]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        _screen_candidate(
            candidate=row,
            latest_runtime_by_slug=latest_runtime_by_slug,
            observe_seconds=args.observe_seconds,
        )
        for row in filtered
    ]
    _print_table(results, skipped_inactive)

    quoteable = [row for row in results if row["current_gate_result"] == "QUOTEABLE_NOW"]
    dead = [row for row in results if row["current_gate_result"] == "DEAD_BOOK_NOW"]
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "screen_type": "reward_aware_maker_viability",
        "observe_seconds": args.observe_seconds,
        "screened_count": len(results),
        "skipped_inactive_count": len(skipped_inactive),
        "quoteable_now_count": len(quoteable),
        "dead_book_now_count": len(dead),
        "first_quoteable_slug": quoteable[0]["slug"] if quoteable else None,
    }

    output = args.output
    if output is None:
        ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = DATA_DIR / f"reward_aware_viability_screen_{ts_label}.json"
    payload = {
        "summary": summary,
        "skipped_inactive": skipped_inactive,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"  Output : {output}")
    print(
        f"  Summary: screened={summary['screened_count']}  "
        f"quoteable_now={summary['quoteable_now_count']}  "
        f"dead_book_now={summary['dead_book_now_count']}"
    )
    if summary["first_quoteable_slug"]:
        print(f"  First quoteable slug: {summary['first_quoteable_slug']}")
    else:
        print("  First quoteable slug: NONE")


if __name__ == "__main__":
    main()
