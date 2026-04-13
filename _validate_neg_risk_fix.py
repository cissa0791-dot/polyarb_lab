"""
Research-only validation: does the negRisk books-fetch fix produce viable RawCandidates?

Reproduces exactly what _run_neg_risk_scan now does, then passes each RawCandidate
through ExecutionFeasibilityEvaluator to report qualification gate outcomes.
No live trading.  No paper broker.  Read-only.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.gamma import fetch_events
from src.ingest.clob import ReadOnlyClob
from src.scanner.neg_risk import build_eligible_neg_risk_event_groups
from src.strategies.opportunity_strategies import NegRiskRebalancingStrategy
from src.opportunity.qualification import ExecutionFeasibilityEvaluator
from src.config_runtime.models import OpportunityConfig

SETTINGS = ROOT / "config" / "settings.yaml"


def main():
    cfg = load_runtime_config(str(SETTINGS))
    gamma_host = cfg.market_data.gamma_host
    clob_host = cfg.market_data.clob_host
    max_notional = cfg.paper.max_notional_per_arb

    print("Fetching events...", flush=True)
    events = fetch_events(gamma_host, limit=2000)
    print(f"Events fetched: {len(events)}", flush=True)

    event_groups = build_eligible_neg_risk_event_groups(events)
    print(f"Eligible negRisk event groups: {len(event_groups)}", flush=True)

    # Collect YES token IDs
    token_ids = list({
        m["yes_token_id"]
        for g in event_groups
        for m in g.get("markets", [])
        if m.get("yes_token_id")
    })
    print(f"Fetching {len(token_ids)} YES token books (batch=20)...", flush=True)

    clob = ReadOnlyClob(clob_host)
    books: dict[str, object] = {}
    batch = 20
    for i in range(0, len(token_ids), batch):
        ids = token_ids[i: i + batch]
        try:
            for book in clob.get_books(ids):
                books[str(book.token_id)] = book
        except Exception:
            for tid in ids:
                try:
                    books[tid] = clob.get_book(tid)
                except Exception:
                    pass
    print(f"Books fetched: {len(books)}\n", flush=True)

    strategy = NegRiskRebalancingStrategy()
    evaluator = ExecutionFeasibilityEvaluator(cfg.opportunity)

    raw_candidates = []
    detect_failures = []
    for g in event_groups:
        raw, audit = strategy.detect_with_audit(g, books, max_notional)
        if raw is not None:
            raw_candidates.append(raw)
        else:
            detect_failures.append((g.get("event_slug"), (audit or {}).get("failure_reason")))

    print(f"RawCandidates produced: {len(raw_candidates)}")
    print(f"Detection failures: {len(detect_failures)}")
    if detect_failures:
        from collections import Counter
        reason_counts = Counter(r for _, r in detect_failures)
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

    # Run through qualification gate
    print("\n--- Qualification Results ---")
    qual_pass = []
    qual_fail = []
    for rc in raw_candidates:
        # Build books map for qualification (keyed by token_id)
        qual_books = {}
        for leg in rc.legs:
            b = books.get(leg.token_id)
            if b:
                qual_books[leg.token_id] = b
        result = evaluator.qualify(rc, qual_books)
        if result.passed:
            qual_pass.append((rc, result))
        else:
            qual_fail.append((rc, result))

    print(f"Qualification PASS: {len(qual_pass)}")
    print(f"Qualification FAIL: {len(qual_fail)}")

    if qual_pass:
        print("\nTop qualified candidates by gross_edge_cents:")
        qual_pass_sorted = sorted(qual_pass, key=lambda x: x[0].gross_edge_cents, reverse=True)
        print(f"  {'Event Slug':<55} {'N':>3} {'Edge$/sh':>10} {'NetProfit$':>11}")
        for rc, res in qual_pass_sorted[:10]:
            n_legs = len(rc.legs)
            slug = rc.detection_name[:55]
            print(f"  {slug:<55} {n_legs:>3} {rc.gross_edge_cents:>10.4f} {rc.gross_profit_usd:>11.4f}")

    if qual_fail:
        from collections import Counter
        fail_reasons = Counter()
        for rc, res in qual_fail:
            for code in (res.reason_codes or []):
                fail_reasons[code] += 1
        print(f"\nQualification failure gate counts:")
        for gate, count in fail_reasons.most_common():
            print(f"  {gate}: {count}")


if __name__ == "__main__":
    main()
