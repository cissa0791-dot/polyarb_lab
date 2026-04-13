"""
scripts/run_math_selector_comparison.py

Research-only comparison: A (current selector) vs B (math-selector prototype).

Protocol
--------
1. Runs a fresh targeted research batch (cross_market_constraint + neg_risk family,
   market_limit=1000) that is known to generate raw candidates reaching the
   qualification layer.

2. After the batch, reads ALL qualification-stage rejection events from paper.db
   (both the fresh run and any historical ones that have full qualification metadata).

3. Applies B (MathCandidateSelector) to the stored qualification metadata for each
   candidate.  No orderbook reconstruction needed — B operates on the stored VWAP
   prices and depth metrics.

4. Reports the strict comparison:
   - raw candidate count (from the fresh batch)
   - A: evaluated / passed / rejected (from stored qual_funnel_reports)
   - B: evaluated / passed / rejected (from B applied to stored metadata)
   - rejection leaderboard by gate for both A and B
   - qualified shortlist count for both
   - family distribution of survivors
   - whether downstream gates activate under B
   - conclusion

Usage
-----
cd D:/Issac/polyarb_lab
python scripts/run_math_selector_comparison.py --settings config/settings.yaml

Optional flags:
  --market-limit INT    number of markets to scan (default: 1000)
  --cycles INT          number of batch cycles (default: 3)
  --skip-scan           skip the live scan, use only historical DB data
  --db-path PATH        explicit DB path (default: from settings)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.opportunity.math_selector import LegAdjustedThreshold, MathCandidateSelector
from src.reporting.analytics import resolve_sqlite_path
from src.runtime.batch import BatchResearchRunner
from src.runtime.runner import ResearchRunner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A vs B math-selector comparison.")
    p.add_argument("--settings", default="config/settings.yaml")
    p.add_argument("--constraints", default="config/constraints.yaml")
    p.add_argument("--market-limit", type=int, default=1000)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--skip-scan", action="store_true",
                   help="Skip the live scan; replay only historical DB data.")
    p.add_argument("--db-path", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Live scan
# ---------------------------------------------------------------------------

def _run_fresh_batch(args: argparse.Namespace) -> dict:
    """Run a research batch and return the aggregated RunSummary payload."""
    batch = BatchResearchRunner(
        runner_factory=lambda: ResearchRunner(
            settings_path=args.settings,
            constraints_path=args.constraints,
        )
    )
    summary = batch.run_batch(
        cycles=args.cycles,
        sleep_sec=0.0,
        experiment_label="math_selector_comparison",
        parameter_set_label="runtime_default",
        market_limit=args.market_limit,
    )
    return summary.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Load qualification-stage rejections from DB
# ---------------------------------------------------------------------------

def _load_qual_rejections(db_path: Path) -> list[dict]:
    """
    Return all qualification-stage rejection events that have full metadata
    (raw_candidate + qualification sub-dict in payload).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT payload_json FROM rejection_events WHERE stage = 'qualification'"
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except Exception:
            continue
        meta = payload.get("metadata", {})
        qual = meta.get("qualification", {})
        raw = meta.get("raw_candidate", {})
        # Only keep rows that have the qualification sub-dict with leg data
        if qual and qual.get("legs"):
            results.append({
                "candidate_id": payload.get("candidate_id", "?"),
                "run_id": payload.get("run_id", "?"),
                "reason_codes_a": [payload.get("reason_code", "?")],
                "family": meta.get("strategy_family") or raw.get("strategy_family", "?"),
                "qual": qual,
                "raw": raw,
            })
    return results


def _load_all_qual_rejections_deduplicated(db_path: Path) -> list[dict]:
    """
    Aggregate all rejection_events per candidate_id at stage=qualification.
    One entry per unique candidate.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT payload_json FROM rejection_events WHERE stage = 'qualification'"
    ).fetchall()
    conn.close()

    by_candidate: dict[str, dict] = {}
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except Exception:
            continue
        cid = payload.get("candidate_id", "?")
        meta = payload.get("metadata", {})
        qual = meta.get("qualification", {})
        raw = meta.get("raw_candidate", {})
        if not qual or not qual.get("legs"):
            continue
        if cid not in by_candidate:
            by_candidate[cid] = {
                "candidate_id": cid,
                "run_id": payload.get("run_id", "?"),
                "reason_codes_a": [],
                "family": meta.get("strategy_family") or raw.get("strategy_family", "?"),
                "qual": qual,
                "raw": raw,
            }
        rc = payload.get("reason_code")
        if rc and rc not in by_candidate[cid]["reason_codes_a"]:
            by_candidate[cid]["reason_codes_a"].append(rc)

    return list(by_candidate.values())


# ---------------------------------------------------------------------------
# Load qualification_funnel_reports for A metrics
# ---------------------------------------------------------------------------

def _load_funnel_reports(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT run_id, evaluated, passed, rejected, payload_json FROM qualification_funnel_reports"
        ).fetchall()
    except Exception:
        rows = []
    conn.close()
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
        except Exception:
            payload = {}
        out.append({
            "run_id": r["run_id"],
            "evaluated": r["evaluated"],
            "passed": r["passed"],
            "rejected": r["rejected"],
            "rejection_counts": payload.get("rejection_counts", {}),
        })
    return out


# ---------------------------------------------------------------------------
# Run B on collected candidates
# ---------------------------------------------------------------------------

def _apply_b(candidates: list[dict], selector: MathCandidateSelector) -> list[dict]:
    results = []
    for c in candidates:
        result = MathCandidateSelector.from_stored_qualification_metadata(
            selector=selector,
            candidate_id=c["candidate_id"],
            family=c["family"],
            raw_candidate=c["raw"],
            qual_meta=c["qual"],
        )
        results.append({
            "candidate_id": c["candidate_id"],
            "family": c["family"],
            "reason_codes_a": c["reason_codes_a"],
            "passed_a": False,  # all from rejection_events → A rejected
            "passed_b": result.passed_b,
            "reason_codes_b": result.reason_codes_b,
            "score_b": result.score,
            "divergence_b": result.divergence,
            "min_viable_edge_b": result.min_viable_edge,
            "fw_gap": result.fw_gap,
            "fw_converged": result.fw_converged,
            "fw_iters": result.fw_iters,
            "n_legs": result.n_legs,
            "kelly_ref": result.kelly_ref,
            "liquidity_cap_profit_usd": result.liquidity_cap_profit_usd,
        })
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report(
    *,
    scan_summary: dict | None,
    funnel_reports: list[dict],
    comparison: list[dict],
    selector: MathCandidateSelector,
) -> dict:
    # ---- A metrics from funnel_reports ----
    total_evaluated_a = sum(r["evaluated"] for r in funnel_reports)
    total_passed_a = sum(r["passed"] for r in funnel_reports)
    total_rejected_a = sum(r["rejected"] for r in funnel_reports)
    gate_counts_a: Counter[str] = Counter()
    for r in funnel_reports:
        gate_counts_a.update(r["rejection_counts"])

    # ---- B metrics from comparison ----
    evaluated_b = len(comparison)
    passed_b = sum(1 for c in comparison if c["passed_b"])
    rejected_b = evaluated_b - passed_b
    gate_counts_b: Counter[str] = Counter()
    for c in comparison:
        for rc in c["reason_codes_b"]:
            gate_counts_b[rc] += 1
    family_survivors_b: Counter[str] = Counter()
    for c in comparison:
        if c["passed_b"]:
            family_survivors_b[c["family"]] += 1

    # Score distribution for B
    scores = sorted([c["score_b"] for c in comparison])
    score_buckets = {"<0.25": 0, "0.25-0.50": 0, "0.50-0.75": 0, "0.75-1.00": 0, "≥1.00": 0}
    for s in scores:
        if s < 0.25:
            score_buckets["<0.25"] += 1
        elif s < 0.50:
            score_buckets["0.25-0.50"] += 1
        elif s < 0.75:
            score_buckets["0.50-0.75"] += 1
        elif s < 1.00:
            score_buckets["0.75-1.00"] += 1
        else:
            score_buckets["≥1.00"] += 1

    # Downstream gate activity under B
    downstream_gates_active = {
        "INSUFFICIENT_DEPTH": gate_counts_b.get("INSUFFICIENT_DEPTH", 0),
        "PARTIAL_FILL_RISK_TOO_HIGH": gate_counts_b.get("PARTIAL_FILL_RISK_TOO_HIGH", 0),
        "NON_ATOMIC_RISK_TOO_HIGH": gate_counts_b.get("NON_ATOMIC_RISK_TOO_HIGH", 0),
        "NET_PROFIT_BELOW_THRESHOLD": gate_counts_b.get("NET_PROFIT_BELOW_THRESHOLD", 0),
    }
    any_downstream_active = any(v > 0 for v in downstream_gates_active.values())

    # Candidates B passes that A rejects (the interesting set)
    b_only_pass = [c for c in comparison if c["passed_b"] and not c["passed_a"]]

    # Per-leg threshold info for B
    threshold_info = {
        "fee_per_leg_cents": selector.threshold.fee_per_leg_cents,
        "slip_per_leg_cents": selector.threshold.slip_per_leg_cents,
        "target_margin_cents": selector.threshold.target_margin_cents,
        "min_viable_2_leg": selector.threshold.min_viable(2),
        "min_viable_3_leg": selector.threshold.min_viable(3),
        "min_net_profit_usd_b": selector.min_net_profit_usd,
        "note": (
            "A uses flat min_edge_cents=0.030; B uses per-leg: "
            "2-leg min=%.3f, 3-leg min=%.3f"
            % (selector.threshold.min_viable(2), selector.threshold.min_viable(3))
        ),
    }

    # Raw candidate count from fresh scan (if available)
    raw_count = 0
    if scan_summary:
        meta = scan_summary.get("metadata", {})
        ob = meta.get("orderbook_funnel", {})
        raw_count = ob.get("raw_candidates_generated", 0)

    return {
        "fresh_scan": {
            "raw_candidates_generated": raw_count,
            "candidates_generated": scan_summary.get("candidates_generated", 0) if scan_summary else 0,
        } if scan_summary else None,
        "selector_A": {
            "description": "ExecutionFeasibilityEvaluator — flat threshold gates",
            "min_edge_cents": 0.03,
            "min_net_profit_usd": 0.50,
            "total_evaluated": total_evaluated_a,
            "total_passed": total_passed_a,
            "total_rejected": total_rejected_a,
            "qualified_shortlist_count": total_passed_a,
            "gate_rejection_leaderboard": [
                {"gate": gate, "count": cnt}
                for gate, cnt in gate_counts_a.most_common()
            ],
        },
        "selector_B": {
            "description": "MathCandidateSelector — FW Bregman projection + per-leg threshold",
            "threshold_params": threshold_info,
            "total_evaluated": evaluated_b,
            "total_passed": passed_b,
            "total_rejected": rejected_b,
            "qualified_shortlist_count": passed_b,
            "gate_rejection_leaderboard": [
                {"gate": gate, "count": cnt}
                for gate, cnt in gate_counts_b.most_common()
            ],
            "family_distribution_survivors": dict(family_survivors_b),
            "score_distribution": score_buckets,
            "downstream_gates_active": downstream_gates_active,
            "any_downstream_gate_active": any_downstream_active,
        },
        "b_only_passes": [
            {
                "candidate_id": c["candidate_id"],
                "family": c["family"],
                "n_legs": c["n_legs"],
                "score_b": c["score_b"],
                "divergence_b": c["divergence_b"],
                "min_viable_edge_b": c["min_viable_edge_b"],
                "fw_gap": c["fw_gap"],
                "fw_converged": c["fw_converged"],
                "kelly_ref": c["kelly_ref"],
                "liquidity_cap_profit_usd": c["liquidity_cap_profit_usd"],
                "reason_codes_a": c["reason_codes_a"],
            }
            for c in b_only_pass
        ],
        "top_candidates_by_b_score": sorted(
            [
                {
                    "candidate_id": c["candidate_id"],
                    "family": c["family"],
                    "n_legs": c["n_legs"],
                    "score_b": c["score_b"],
                    "divergence_cents_b": round(c["divergence_b"] * 100, 4),
                    "min_viable_cents_b": c["min_viable_edge_b"],
                    "fw_gap": round(c["fw_gap"], 6) if c["fw_gap"] == c["fw_gap"] else None,
                    "passed_b": c["passed_b"],
                    "reason_codes_a": c["reason_codes_a"],
                }
                for c in comparison
            ],
            key=lambda x: x["score_b"],
            reverse=True,
        )[:10],
        "conclusion": {
            "b_improved_throughput": passed_b > total_passed_a,
            "b_revealed_new_family": bool(family_survivors_b),
            "b_activated_downstream_gates": any_downstream_active,
            "candidates_b_passes_a_rejects": len(b_only_pass),
            "note": (
                "B improved throughput."
                if passed_b > total_passed_a
                else "B did not improve throughput vs A on this data."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    cfg = load_runtime_config(args.settings)
    db_path = (
        Path(args.db_path)
        if args.db_path
        else resolve_sqlite_path(cfg.storage.sqlite_url)
    )

    scan_summary = None
    if not args.skip_scan:
        print("[A+B] Running fresh research batch …", flush=True)
        scan_summary = _run_fresh_batch(args)
        print(
            f"[A+B] Batch done. "
            f"raw_candidates_generated="
            f"{scan_summary.get('metadata', {}).get('orderbook_funnel', {}).get('raw_candidates_generated', 0)}",
            flush=True,
        )

    print("[A] Loading qualification-stage rejections from DB …", flush=True)
    candidates = _load_all_qual_rejections_deduplicated(db_path)
    print(f"[A] {len(candidates)} unique candidates with full qualification metadata.", flush=True)

    funnel_reports = _load_funnel_reports(db_path)
    print(f"[A] {len(funnel_reports)} qualification_funnel_report rows.", flush=True)

    selector = MathCandidateSelector(
        threshold=LegAdjustedThreshold(
            fee_per_leg_cents=0.005,
            slip_per_leg_cents=0.005,
            target_margin_cents=0.005,
        ),
        max_partial_fill_risk=0.65,
        max_non_atomic_risk=0.60,
        min_net_profit_usd=0.10,
    )

    print("[B] Applying math selector to stored candidates …", flush=True)
    comparison = _apply_b(candidates, selector)

    report = _report(
        scan_summary=scan_summary,
        funnel_reports=funnel_reports,
        comparison=comparison,
        selector=selector,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
