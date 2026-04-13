"""
auto_maker_loop — cycle_analyst
polyarb_lab / research_lines / auto_maker_loop / modules

Self-study layer. Classifies each cycle into one of 9 outcome codes,
prints a per-cycle readable summary, and aggregates study summaries
across multiple cycle records from runs.jsonl.

No external dependencies. No self-modifying code. Read-only analysis.

Outcome codes
-------------
reward_positive_pnl_positive   Best case. Reward nonzero, trading P&L positive.
reward_positive_pnl_negative   Reward earned but trading P&L negative (inventory cost > reward).
no_fill                        Neither BID nor ASK filled within hold window.
fill_but_bad_exit              BID filled but exit price <= entry price (taker-sell at loss).
stranded_inventory             Exit path failed — taker-sell did not execute.
weak_market_selection          Market skipped due to inventory missing every attempt.
weak_quote_quality             Quote did not qualify or placement failed.
reward_too_small_to_matter     Reward nonzero but < 1¢ net improvement over neutral.
internal_error                 Unhandled exception inside the cycle (NameError, AttributeError, etc.).
preowned_inventory_disposal    Pre-owned inventory sold via REDUCE_ONLY/EXIT_ONLY path — not a new entry.

Public interface
----------------
    classify_outcome(record: dict) -> str
    print_cycle_summary(record: dict) -> None
    print_study_summary(records: list[dict]) -> None
    load_runs(path: str | Path) -> list[dict]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Outcome classifier
# ---------------------------------------------------------------------------

def classify_outcome(record: dict) -> str:
    """
    Map one cycle record to one of 9 outcome codes.

    Priority order: origin check first, then structural failures, then economics.
    """
    outcome    = record.get("outcome", "")
    bid_filled = record.get("bid_filled", False)
    ask_filled = record.get("ask_filled", False)
    pnl_cents  = record.get("pnl_cents", 0.0) or 0.0
    exit_price = record.get("exit_price", 0.0) or 0.0
    entry_price = record.get("entry_price", 0.0) or 0.0
    notes      = record.get("notes", [])
    reward_delta_cents = record.get("reward_delta_cents", None)
    skip_reason = record.get("skip_reason", "")

    # ── Pre-owned inventory disposal — must be checked before any bid_filled logic.
    # These cycles set bid_filled=True synthetically (no real BID was placed).
    # Letting them fall through to fill_but_bad_exit would pollute study summaries.
    # Exception: if exit_price=0 after a disposal attempt, that is still stranded_inventory.
    cycle_origin = record.get("cycle_origin", "NEW_ENTRY")
    if cycle_origin == "PREOWNED_INVENTORY_DISPOSAL":
        if exit_price == 0.0 and bid_filled and not ask_filled:
            return "stranded_inventory"
        return "preowned_inventory_disposal"

    # Structural failures take priority
    if outcome == "ERROR":
        return "internal_error"
    if outcome == "PLACEMENT_FAILED":
        return "weak_quote_quality"

    if outcome == "NOT_SCORING" or skip_reason == "orders_not_scoring":
        return "weak_quote_quality"

    if outcome == "SKIPPED" and skip_reason in (
        "SELL_INVENTORY_MISSING", "SELL_ALLOWANCE_INSUFFICIENT"
    ):
        return "weak_market_selection"

    if outcome in ("SKIPPED",) and skip_reason == "quotes_not_qualifying":
        return "weak_quote_quality"

    # Stranded inventory: taker_sell failed or exit_price=0 after BID filled
    if bid_filled and exit_price == 0.0 and not ask_filled:
        return "stranded_inventory"
    if any("taker_sell failed" in str(n) for n in notes):
        return "stranded_inventory"

    # No fill: neither leg filled
    if not bid_filled and outcome in ("NO_FILL_TIME_LIMIT", "DRY_RUN", "SKIPPED"):
        return "no_fill"
    if not bid_filled and not ask_filled:
        return "no_fill"

    # Bad exit: BID filled but exit was below entry (net trading loss)
    if bid_filled and exit_price > 0.0 and exit_price < entry_price:
        return "fill_but_bad_exit"

    # Economics classification (both filled or natural exit)
    reward_nonzero = reward_delta_cents is not None and reward_delta_cents > 0.0
    pnl_positive   = pnl_cents > 0.0

    if reward_nonzero and pnl_positive:
        return "reward_positive_pnl_positive"

    if reward_nonzero and not pnl_positive:
        # Check if reward is too small to offset pnl loss
        if reward_delta_cents is not None and reward_delta_cents < 1.0 and pnl_cents < 0.0:
            return "reward_too_small_to_matter"
        return "reward_positive_pnl_negative"

    if not reward_nonzero and pnl_positive:
        # Spread captured but reward not observed (may lag)
        return "reward_positive_pnl_positive"  # treat as positive; reward may lag

    return "fill_but_bad_exit"  # fallback: filled but no clear positive signal


# ---------------------------------------------------------------------------
# Per-cycle summary printer
# ---------------------------------------------------------------------------

def print_cycle_summary(record: dict) -> None:
    """Print a readable self-study summary for one cycle."""
    code = classify_outcome(record)

    print("\n" + "=" * 60)
    print("  CYCLE SELF-STUDY SUMMARY")
    print("=" * 60)
    print(f"  outcome_code         : {code}")
    print(f"  market               : {record.get('slug', 'N/A')[:50]}")
    print(f"  cycle                : {record.get('cycle', '?')}")
    print(f"  ts                   : {record.get('ts', '?')}")
    print()

    print("  — Quote & Fill —")
    print(f"  bid_price            : {record.get('entry_price', 'N/A')}")
    print(f"  ask_price_initial    : {record.get('ask_price_initial', 'N/A')}")
    print(f"  size                 : {record.get('size', 'N/A')} shares")
    print(f"  bid_filled           : {record.get('bid_filled', False)}")
    print(f"  ask_filled           : {record.get('ask_filled', False)}")
    print(f"  ask_chases           : {record.get('ask_chases', 0)}")
    print()

    print("  — Inventory —")
    print(f"  inventory_before     : {record.get('inventory_before_shares', 'N/A')} shares")
    print(f"  inventory_after      : {record.get('inventory_after_shares', 'N/A')} shares")
    print()

    print("  — Economics —")
    print(f"  pnl_cents            : {record.get('pnl_cents', 0.0):.4f}¢")
    rdelta = record.get("reward_delta_cents")
    print(f"  reward_delta_cents   : {f'{rdelta:.4f}¢' if rdelta is not None else 'not measured'}")
    net = _net_cents(record)
    print(f"  net_edge_cents       : {f'{net:.4f}¢' if net is not None else 'not measurable yet'}")
    print(f"  daily_rate_usdc      : ${record.get('daily_rate_usdc', 0):.0f}/day")
    print()

    print("  — Execution —")
    print(f"  exit_path            : {record.get('outcome', 'N/A')}")
    print(f"  exit_price           : {record.get('exit_price', 0.0)}")
    print(f"  hold_minutes         : {record.get('hold_minutes', 0.0):.1f} min")
    print(f"  min_midpoint_hold    : {record.get('min_midpoint_hold', 'N/A')}")
    print(f"  max_midpoint_hold    : {record.get('max_midpoint_hold', 'N/A')}")
    print()

    print("  — Reward-Aware Fields —")
    reward_ok  = record.get("reward_config_ok")
    comp       = record.get("competitiveness")
    in_zone    = record.get("quotes_in_zone")
    scoring_v  = record.get("scoring_verified")
    zone_bid   = record.get("reward_zone_bid")
    zone_ask   = record.get("reward_zone_ask")
    print(f"  reward_config_ok     : {reward_ok}")
    print(f"  scoring_verified     : {scoring_v}")
    print(f"  quotes_in_zone       : {in_zone}  "
          f"(zone {zone_bid}–{zone_ask})" if zone_bid is not None else f"  quotes_in_zone       : {in_zone}")
    if comp is not None:
        high = comp > 200
        print(f"  competitiveness      : {comp:.1f}{'  ← HIGH (>200)' if high else ''}")
    print()

    print("  — Diagnosis —")
    _print_diagnosis(code, record)

    notes = record.get("notes", [])
    if notes:
        print("\n  — Notes —")
        for n in notes:
            print(f"    {n}")

    print("=" * 60)


def _net_cents(record: dict) -> Optional[float]:
    pnl = record.get("pnl_cents")
    rdelta = record.get("reward_delta_cents")
    if pnl is None:
        return None
    if rdelta is not None:
        return round(pnl + rdelta, 4)
    return round(float(pnl), 4)


def _print_diagnosis(code: str, record: dict) -> None:
    diag = {
        "reward_positive_pnl_positive": (
            "Both reward and spread captured. Best case confirmed for this cycle. "
            "Repeat to check consistency."
        ),
        "reward_positive_pnl_negative": (
            "Reward earned but trading P&L negative. "
            "Exit was at or below entry. "
            "Check: is TIME_LIMIT the dominant exit? "
            "Consider: lower max_hold_minutes to reduce directional exposure."
        ),
        "no_fill": (
            "Neither BID nor ASK filled within hold window. "
            "Market did not trade at our price level. "
            "Check: is midpoint stable or drifting away from our quotes? "
            "Consider: widen quote placement or check competitiveness."
        ),
        "fill_but_bad_exit": (
            "BID filled but exit price was below entry. "
            "Taker-sell fired at a loss. "
            "Check: did midpoint drop significantly after BID fill? "
            "Consider: tighten stop_loss_cents to cut losses earlier."
        ),
        "stranded_inventory": (
            "EXIT PATH FAILED. Inventory may be stranded on-chain. "
            "Check notes[] for taker_sell error. "
            "Manual intervention required before next cycle."
        ),
        "weak_market_selection": (
            "Market skipped due to inventory missing. "
            "Inventory management is the binding constraint, not strategy economics. "
            "Restore inventory before continuing."
        ),
        "weak_quote_quality": (
            "Quote placement failed, quotes did not qualify, or orders were not scoring. "
            "Check: is midpoint at an extreme? Is reward config stale (ok=False)? "
            "Did scoring_verified=False trigger a cancel? "
            "No economics signal from this cycle."
        ),
        "reward_too_small_to_matter": (
            "Reward was nonzero but below 1 cent net improvement over a neutral hold. "
            "Trading P&L was negative. "
            "Net edge is negative for this cycle. "
            "Requires consistent data across 5+ cycles before downgrade judgment."
        ),
        "preowned_inventory_disposal": (
            "Pre-owned inventory disposal cycle. No new BID was placed — this was a "
            "REDUCE_ONLY/EXIT_ONLY sell-only path. pnl_cents is measured relative to "
            "midpoint at cycle start (not real cost basis). Do not compare to new-entry "
            "fill_but_bad_exit cycles. Economics signal: pnl_cents > 0 = sold above ref."
        ),
    }
    print(f"  {diag.get(code, 'unknown outcome code — check record manually')}")


# ---------------------------------------------------------------------------
# Multi-cycle study summary
# ---------------------------------------------------------------------------

def print_study_summary(records: list[dict]) -> None:
    """
    Aggregate self-study summary across multiple cycle records.
    Identifies dominant failure mode and suggests one parameter change.
    """
    if not records:
        print("  [study_summary] no records to analyze")
        return

    codes = [classify_outcome(r) for r in records]
    n = len(records)

    # Count by code
    from collections import Counter
    counts = Counter(codes)

    # Aggregate economics
    pnl_list   = [r.get("pnl_cents", 0.0) or 0.0 for r in records]
    hold_list  = [r.get("hold_minutes", 0.0) or 0.0 for r in records]
    rdelta_list = [r["reward_delta_cents"] for r in records if r.get("reward_delta_cents") is not None]
    net_list   = [_net_cents(r) for r in records if _net_cents(r) is not None]

    bid_fills  = sum(1 for r in records if r.get("bid_filled"))
    ask_fills  = sum(1 for r in records if r.get("ask_filled"))

    print("\n" + "=" * 60)
    print(f"  STUDY SUMMARY  ({n} cycles)")
    print("=" * 60)

    print("\n  — Outcome distribution —")
    for code, cnt in counts.most_common():
        pct = round(cnt / n * 100)
        print(f"    {code:<40} {cnt}/{n}  ({pct}%)")

    print("\n  — Fill rates —")
    print(f"    BID fill rate        : {bid_fills}/{n}  ({round(bid_fills/n*100)}%)")
    print(f"    ASK fill rate        : {ask_fills}/{n}  ({round(ask_fills/n*100)}%)")

    print("\n  — Economics —")
    if pnl_list:
        print(f"    avg pnl_cents        : {sum(pnl_list)/len(pnl_list):.3f}¢")
        print(f"    min pnl_cents        : {min(pnl_list):.3f}¢")
        print(f"    max pnl_cents        : {max(pnl_list):.3f}¢")
    if rdelta_list:
        print(f"    avg reward_delta     : {sum(rdelta_list)/len(rdelta_list):.3f}¢  ({len(rdelta_list)}/{n} cycles measured)")
    if net_list:
        avg_net = sum(net_list) / len(net_list)
        print(f"    avg net_edge_cents   : {avg_net:.3f}¢")
    if hold_list:
        print(f"    avg hold_minutes     : {sum(hold_list)/len(hold_list):.1f} min")

    # Reward-aware pattern summary
    not_scoring_count  = sum(1 for r in records if r.get("outcome") == "NOT_SCORING"
                             or r.get("skip_reason") == "orders_not_scoring")
    out_of_zone_count  = sum(1 for r in records
                             if r.get("quotes_in_zone") is False)
    fallback_cfg_count = sum(1 for r in records
                             if r.get("reward_config_ok") is False)
    scoring_fail_count = sum(1 for r in records
                             if r.get("scoring_verified") is False)
    high_comp_count    = sum(1 for r in records
                             if (r.get("competitiveness") or 0) > 200)

    if any([not_scoring_count, out_of_zone_count, fallback_cfg_count,
            scoring_fail_count, high_comp_count]):
        print("\n  — Reward-Aware Patterns —")
        if not_scoring_count:
            print(f"    NOT_SCORING cycles       : {not_scoring_count}/{n}")
        if scoring_fail_count:
            print(f"    scoring_verified=False   : {scoring_fail_count}/{n}")
        if out_of_zone_count:
            print(f"    quotes outside zone      : {out_of_zone_count}/{n}")
        if fallback_cfg_count:
            print(f"    fallback reward config   : {fallback_cfg_count}/{n}  (live fetch failed)")
        if high_comp_count:
            print(f"    high competitiveness>200 : {high_comp_count}/{n}")

    print("\n  — Diagnosis —")
    dominant = counts.most_common(1)[0][0] if counts else "unknown"
    _print_multi_diagnosis(dominant, counts, n, pnl_list, rdelta_list, hold_list, bid_fills, ask_fills)

    print("=" * 60)


def _print_multi_diagnosis(
    dominant: str,
    counts,
    n: int,
    pnl_list, rdelta_list, hold_list,
    bid_fills: int, ask_fills: int,
) -> None:
    print(f"  Dominant failure mode : {dominant}")

    if dominant == "stranded_inventory":
        print("  CRITICAL: taker-sell path is broken. Fix before any further live runs.")
        print("  Suggested action: read notes[] from last stranded cycle; fix _taker_sell.")
        return

    if dominant == "weak_market_selection":
        print("  Inventory is the binding constraint, not strategy economics.")
        print("  Suggested action: restore inventory; add splitPosition automation.")
        return

    if dominant == "no_fill":
        print("  BID price is not competitive at current midpoint.")
        print("  Suggested action: check if midpoint has drifted; verify quote centering.")
        return

    if dominant == "fill_but_bad_exit":
        if hold_list:
            avg_hold = sum(hold_list) / len(hold_list)
            print(f"  BID fills but exit is below entry. Avg hold = {avg_hold:.1f} min.")
            if avg_hold > 50:
                print("  Suggested action: reduce max_hold_minutes to limit directional exposure.")
            else:
                print("  Suggested action: check if midpoint drops after BID fill (adverse selection).")
        return

    if dominant in ("reward_positive_pnl_negative", "reward_too_small_to_matter"):
        if rdelta_list:
            avg_r = sum(rdelta_list) / len(rdelta_list)
            print(f"  Reward is real (avg {avg_r:.3f}¢) but trading P&L is negative.")
            print("  Net edge is negative. Requires further cycles to confirm trend.")
            print("  Suggested action: measure net_edge_cents across 10 cycles before judgment.")
        return

    if dominant == "reward_positive_pnl_positive":
        print("  Both reward and spread captured consistently.")
        print("  Suggested action: extend cycle count to 10 before scaling judgment.")
        return

    print("  Insufficient data to isolate dominant cause. Continue collecting cycles.")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_runs(path) -> list[dict]:
    """Load all records from runs.jsonl. Returns empty list if file missing."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
