"""
Fill-realism stress test for maker-MM quote plans.

Applies 5 stress views to each plan from the latest maker_rewarded_event_mm_v1 report:
  1. spread_haircut_25   – spread reduced 25%, fill_prob held constant
  2. spread_haircut_50   – spread reduced 50%
  3. spread_haircut_75   – spread reduced 75%
  4. one_sided_fill      – only one leg fills (effective fill_prob halved)
  5. adverse_combined    – 50 % spread, one-sided fill, +50 % adverse selection cost,
                          reward_eligibility capped at 0.5

A plan is ROBUST_PAPER_MM if total_ev > 0 under **all** five views.
Verdict: THESIS_GENERALIZED if 3+ ROBUST plans across 2+ distinct events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── helpers ────────────────────────────────────────────────────────────────────

def _ev(
    *,
    quote_size: float,
    quote_spread: float,
    fill_prob: float,
    reward_daily_rate: float,
    reward_eligibility: float,
    adverse_sel_mult: float = 1.0,
    stability_ratio: float = 1.0,
    liquidity_support: float = 1.0,
    spread_churn: float = 0.0,
) -> float:
    spread_capture = quote_size * quote_spread * fill_prob * 0.5
    liquidity_reward = reward_daily_rate * reward_eligibility
    adverse_sel = quote_size * quote_spread * max(0.0, 1.0 - stability_ratio) * 0.35 * adverse_sel_mult
    inventory = quote_size * quote_spread * max(0.0, 1.0 - liquidity_support) * 0.20
    cancel_replace = quote_size * quote_spread * spread_churn * 0.15
    return round(spread_capture + liquidity_reward - adverse_sel - inventory - cancel_replace, 6)


def _derive_fill_prob(plan: dict) -> float:
    """Back-calculate fill_probability from reported spread_capture_ev."""
    quote_size = float(plan["quote_size"])
    quote_spread = float(plan["quote_ask"]) - float(plan["quote_bid"])
    spread_ev = float(plan["spread_capture_ev"])
    if quote_size <= 0 or quote_spread <= 0:
        return 0.5
    return min(0.95, max(0.05, spread_ev / (quote_size * quote_spread * 0.5)))


def _derive_reward_eligibility(plan: dict) -> float:
    """Back-calculate reward_eligibility from reported liquidity_reward_ev."""
    rate = float(plan.get("liquidity_reward_ev") or 0.0)
    # liquidity_reward_ev = reward_daily_rate * reward_eligibility
    # We don't have reward_daily_rate directly, but we do have liquidity_reward_ev
    # and can use it as-is (it IS reward_daily_rate * eligibility).
    # We return 1.0 and keep liquidity_reward_ev as the raw reward for stress scaling.
    return 1.0


def stress_plan(plan: dict) -> dict:
    candidate_id = plan["candidate_id"]
    event_slug = plan["event_slug"]
    market_slug = plan["market_slug"]
    quote_size = float(plan["quote_size"])
    quote_spread = float(plan["quote_ask"]) - float(plan["quote_bid"])
    reward_ev = float(plan.get("liquidity_reward_ev") or 0.0)  # already rate * eligibility

    fill_prob = _derive_fill_prob(plan)

    # Adverse selection proxy: we infer from reported adverse_selection_cost_proxy.
    # If it's 0 (stability_ratio=1), stress by imposing a nonzero value.
    reported_adverse = float(plan.get("adverse_selection_cost_proxy") or 0.0)
    reported_cancel = float(plan.get("cancel_replace_cost_proxy") or 0.0)
    reported_inventory = float(plan.get("inventory_cost_proxy") or 0.0)

    # Base EV (recalculated from components)
    base_ev = (
        float(plan["spread_capture_ev"])
        + reward_ev
        - reported_adverse
        - reported_inventory
        - reported_cancel
    )

    def _stressed_ev(
        spread_mult: float = 1.0,
        fill_mult: float = 1.0,
        adverse_cost_add: float = 0.0,
        reward_mult: float = 1.0,
    ) -> float:
        s = quote_spread * spread_mult
        sc = quote_size * s * (fill_prob * fill_mult) * 0.5
        lr = reward_ev * reward_mult
        # Adverse selection: base reported + added stress fraction of position cost
        adv = reported_adverse + adverse_cost_add * (quote_size * s)
        inv = reported_inventory * spread_mult  # inventory scales with spread
        crc = reported_cancel * spread_mult
        return round(sc + lr - adv - inv - crc, 6)

    views = {
        "base": round(base_ev, 6),
        "spread_haircut_25": _stressed_ev(spread_mult=0.75),
        "spread_haircut_50": _stressed_ev(spread_mult=0.50),
        "spread_haircut_75": _stressed_ev(spread_mult=0.25),
        "one_sided_fill": _stressed_ev(fill_mult=0.50),
        "adverse_combined": _stressed_ev(
            spread_mult=0.50,
            fill_mult=0.50,
            adverse_cost_add=0.02,   # 2% of notional as adverse selection
            reward_mult=0.50,
        ),
    }

    stress_views = [v for v in views if v != "base"]
    robust = all(views[v] > 0.0 for v in stress_views)
    label = "ROBUST_PAPER_MM" if robust else "FRAGILE"

    return {
        "candidate_id": candidate_id,
        "event_slug": event_slug,
        "market_slug": market_slug,
        "quote_spread": round(quote_spread, 6),
        "fill_prob_inferred": round(fill_prob, 4),
        "label": label,
        "ev_views": views,
        "failing_views": [v for v in stress_views if views[v] <= 0.0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="data/reports/maker_rewarded_event_mm_v1_latest.json")
    parser.add_argument("--out-dir", default="data/reports")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plans = report.get("quote_plans") or []

    results = [stress_plan(p) for p in plans]

    robust = [r for r in results if r["label"] == "ROBUST_PAPER_MM"]
    fragile = [r for r in results if r["label"] == "FRAGILE"]

    robust_events = {r["event_slug"] for r in robust}
    if len(robust) >= 3 and len(robust_events) >= 2:
        verdict = "THESIS_GENERALIZED"
    else:
        verdict = "STILL_SPECIAL_CASE"

    output = {
        "verdict": verdict,
        "robust_count": len(robust),
        "fragile_count": len(fragile),
        "robust_events": sorted(robust_events),
        "robust_plans": robust,
        "fragile_plans": fragile,
        "all_results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fill_realism_stress_latest.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    print(f"\n{'='*60}")
    print(f"FILL-REALISM STRESS VERDICT: {verdict}")
    print(f"  Robust plans  : {len(robust)}/{len(results)}")
    print(f"  Robust events : {sorted(robust_events)}")
    print(f"{'='*60}\n")

    for r in results:
        print(f"[{r['label']:>16s}] {r['event_slug']:40s} | {r['market_slug']}")
        for view_name, ev in r["ev_views"].items():
            flag = " <-- FAIL" if view_name != "base" and ev <= 0.0 else ""
            print(f"    {view_name:25s}: {ev:+.6f}{flag}")
        print()

    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
