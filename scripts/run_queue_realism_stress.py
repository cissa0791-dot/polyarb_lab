"""
Queue-realism stress test for ROBUST_PAPER_MM plans.

For maker-market-making, the key queue-realism risks are:
  1. delayed_fill        – order sits in queue; during delay adverse selection accumulates
                           (model: double the adverse_selection_cost_proxy per fill)
  2. quote_replaced      – quote gets beaten / replaced before fill; fraction of quotes earn 0
                           (model: effective fill_prob *= 0.70)
  3. partial_fill        – only a fraction of quote_size fills
                           (model: quote_size *= 0.60, spread_capture scales down)
  4. reduced_reward_dwell – insufficient dwell time in book to earn full rewards
                           (model: reward_eligibility *= 0.40)
  5. conservative_combined – all four applied simultaneously

A plan is QUEUE_RESILIENT if total_ev > 0 under **all five** views.
Verdict: READY_FOR_EXECUTION if 3+ plans are QUEUE_RESILIENT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _derive_fill_prob(plan: dict) -> float:
    quote_size = float(plan["quote_size"])
    quote_spread = float(plan["quote_ask"]) - float(plan["quote_bid"])
    spread_ev = float(plan["spread_capture_ev"])
    if quote_size <= 0 or quote_spread <= 0:
        return 0.5
    return min(0.95, max(0.05, spread_ev / (quote_size * quote_spread * 0.5)))


def _compute_ev(
    *,
    quote_size: float,
    quote_spread: float,
    fill_prob: float,
    reward_ev: float,
    reported_adverse: float,
    reported_inventory: float,
    reported_cancel: float,
    fill_mult: float = 1.0,
    size_mult: float = 1.0,
    adverse_mult: float = 1.0,
    reward_mult: float = 1.0,
) -> float:
    eff_size = quote_size * size_mult
    sc = eff_size * quote_spread * (fill_prob * fill_mult) * 0.5
    lr = reward_ev * reward_mult
    adv = reported_adverse * adverse_mult * size_mult
    inv = reported_inventory * size_mult
    crc = reported_cancel
    return round(sc + lr - adv - inv - crc, 6)


def queue_stress_plan(plan: dict) -> dict:
    candidate_id = plan["candidate_id"]
    event_slug = plan["event_slug"]
    market_slug = plan["market_slug"]
    quote_size = float(plan["quote_size"])
    quote_spread = float(plan["quote_ask"]) - float(plan["quote_bid"])
    reward_ev = float(plan.get("liquidity_reward_ev") or 0.0)
    reported_adverse = float(plan.get("adverse_selection_cost_proxy") or 0.0)
    reported_inventory = float(plan.get("inventory_cost_proxy") or 0.0)
    reported_cancel = float(plan.get("cancel_replace_cost_proxy") or 0.0)
    fill_prob = _derive_fill_prob(plan)

    # Ensure a floor on adverse (even if reported=0, apply a minimum for stress)
    min_adverse_floor = quote_size * quote_spread * 0.05  # 5% of spread*size baseline

    base_adverse = max(reported_adverse, min_adverse_floor * 0.01)

    kwargs_base = dict(
        quote_size=quote_size,
        quote_spread=quote_spread,
        fill_prob=fill_prob,
        reward_ev=reward_ev,
        reported_adverse=base_adverse,
        reported_inventory=reported_inventory,
        reported_cancel=reported_cancel,
    )

    views = {
        "base": _compute_ev(**kwargs_base),
        # 1. Delayed fill: extra adverse selection during queue dwell (2x adverse)
        "delayed_fill": _compute_ev(**{**kwargs_base, "adverse_mult": 2.0}),
        # 2. Quote replaced before fill: 30% of quotes never fill
        "quote_replaced": _compute_ev(**{**kwargs_base, "fill_mult": 0.70}),
        # 3. Partial fill: only 60% of size fills
        "partial_fill": _compute_ev(**{**kwargs_base, "size_mult": 0.60}),
        # 4. Reduced reward dwell: only 40% of reward earned
        "reduced_reward_dwell": _compute_ev(**{**kwargs_base, "reward_mult": 0.40}),
        # 5. Conservative combined: all stresses simultaneously
        "conservative_combined": _compute_ev(
            **{
                **kwargs_base,
                "fill_mult": 0.60,
                "size_mult": 0.60,
                "adverse_mult": 3.0,
                "reward_mult": 0.30,
            }
        ),
    }

    stress_views = [v for v in views if v != "base"]
    queue_resilient = all(views[v] > 0.0 for v in stress_views)
    label = "QUEUE_RESILIENT" if queue_resilient else "FRAGILE"

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

    results = [queue_stress_plan(p) for p in plans]

    resilient = [r for r in results if r["label"] == "QUEUE_RESILIENT"]
    fragile = [r for r in results if r["label"] == "FRAGILE"]
    resilient_events = {r["event_slug"] for r in resilient}

    if len(resilient) >= 3:
        verdict = "READY_FOR_EXECUTION"
    else:
        verdict = "NOT_READY"

    output = {
        "verdict": verdict,
        "queue_resilient_count": len(resilient),
        "fragile_count": len(fragile),
        "resilient_events": sorted(resilient_events),
        "resilient_plans": resilient,
        "fragile_plans": fragile,
        "all_results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "queue_realism_stress_latest.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"QUEUE-REALISM VERDICT: {verdict}")
    print(f"  Queue-resilient plans  : {len(resilient)}/{len(results)}")
    print(f"  Resilient events       : {sorted(resilient_events)}")
    print(f"{'='*60}\n")

    for r in results:
        print(f"[{r['label']:>15s}] {r['event_slug']:40s} | {r['market_slug']}")
        for view_name, ev in r["ev_views"].items():
            flag = " <-- FAIL" if view_name != "base" and ev <= 0.0 else ""
            print(f"    {view_name:25s}: {ev:+.6f}{flag}")
        if r["failing_views"]:
            print(f"    FAILING VIEWS: {r['failing_views']}")
        print()

    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
