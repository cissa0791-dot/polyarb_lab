"""
Broader regression sweep for the maker-MM campaign path.

Expands the current 4-event cohort to 10 events covering all min_size tiers,
both neg_risk values, and a wider rate range. Runs 3 cycles via the named
campaign preset to verify:
  - summary accounting integrity
  - rejection accounting accuracy
  - persistence scaling
  - per-cycle component stability
  - no contamination of other strategy paths

Read-only intent: uses an isolated sweep DB, does not touch paper.db.

Usage:
    py -3 scripts/run_maker_mm_broader_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import MetaData, create_engine, func as sqlfunc, select

from src.runtime.batch import BatchResearchRunner
from src.runtime.campaigns import build_campaign_manifest_from_preset
from src.runtime.runner import ResearchRunner
from src.storage.event_store import ResearchStore
from src.utils.db import OpportunityStore

SWEEP_DB_URL = f"sqlite:///{ROOT / 'data' / 'processed' / 'runner_mm_broader_sweep.db'}"
CYCLES = 3

# Extended cohort: 10 events across all min_size tiers, both neg_risk values,
# rate range 10–200.  Kept bounded (not all 44 events).
EXTENDED_COHORT = [
    # -- current 4-event baseline --
    "next-prime-minister-of-hungary",        # min_size=200, rate=200, neg_risk=True
    "netanyahu-out-before-2027",             # min_size=200, rate=100, neg_risk=False
    "balance-of-power-2026-midterms",        # min_size=20,  rate=30-50, neg_risk=True
    "next-james-bond-actor-635",             # min_size=100, rate=1-35, neg_risk=True
    # -- new additions --
    "presidential-election-winner-2028",     # min_size=[20,200], rate=[2,100], neg_risk=True, 8 markets
    "republican-presidential-nominee-2028",  # min_size=[20,200], rate=[1-100], neg_risk=True, 11 markets
    "which-party-will-win-the-senate-in-2026",  # min_size=200, rate=50, neg_risk=True
    "texas-republican-senate-primary-winner",   # min_size=200, rate=50-100, neg_risk=True
    "russia-x-ukraine-ceasefire-before-2027",   # min_size=100, rate=20, neg_risk=False
    "will-china-invade-taiwan-before-2027",     # min_size=200, rate=10, neg_risk=False
]


def count_rows(db_url: str, table_names: list[str]) -> dict[str, int]:
    engine = create_engine(db_url, future=True)
    meta = MetaData()
    meta.reflect(bind=engine)
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for name in table_names:
            if name in meta.tables:
                counts[name] = conn.execute(
                    select(sqlfunc.count()).select_from(meta.tables[name])
                ).scalar()
            else:
                counts[name] = 0
    return counts


TABLES = [
    "opportunity_candidates",
    "risk_decisions",
    "order_intents",
    "execution_reports",
    "account_snapshots",
    "rejection_events",
    "position_events",
]


def main() -> None:
    print("=" * 72)
    print("MAKER-MM BROADER REGRESSION SWEEP  [P11-A: inter_cycle_reset=True]")
    print(f"  cohort_size        : {len(EXTENDED_COHORT)} events")
    print(f"  cycles             : {CYCLES}")
    print(f"  inter_cycle_reset  : True")
    print(f"  sweep_db           : {SWEEP_DB_URL}")
    print("=" * 72)

    manifest = build_campaign_manifest_from_preset(
        "maker_rewarded_event_mm_v1",
        campaign_label="maker-mm-broader-sweep",
        cycles=CYCLES,
        metadata={"maker_mm_cohort": EXTENDED_COHORT},
    )

    def make_runner() -> ResearchRunner:
        r = ResearchRunner()
        r.store = ResearchStore(SWEEP_DB_URL)
        r.opportunity_store = OpportunityStore(SWEEP_DB_URL)
        # P11-A: set on _base_config so the flag survives apply_runtime_parameter_set,
        # which rebuilds self.config from _base_config via model_copy(deep=True).
        r._base_config.paper.inter_cycle_reset = True
        r.config.paper.inter_cycle_reset = True
        return r

    before = count_rows(SWEEP_DB_URL, TABLES)
    print(f"\n[pre-run row counts] {before}")

    batch = BatchResearchRunner(runner_factory=make_runner)
    result = batch.run_campaign(manifest)

    after = count_rows(SWEEP_DB_URL, TABLES)
    delta = {k: after[k] - before.get(k, 0) for k in after}

    print("\n" + "=" * 72)
    print("SWEEP RESULTS")
    print("=" * 72)

    agg = result.aggregate_summary
    meta = agg.metadata or {}
    raw_by_family = meta.get("raw_candidates_by_family", {})
    mm_raw = raw_by_family.get("maker_rewarded_event_mm_v1", 0)
    arb_raw = sum(v for k, v in raw_by_family.items() if k != "maker_rewarded_event_mm_v1")

    print(f"\n  cycles_completed     : {result.cycles_completed}")
    print(f"  mm_raw_candidates    : {mm_raw}")
    print(f"  arb_raw_candidates   : {arb_raw}  (must be 0 — contamination check)")
    print(f"  candidates_generated : {agg.candidates_generated}")
    print(f"  risk_accepted        : {agg.risk_accepted}")
    print(f"  risk_rejected        : {agg.risk_rejected}")
    print(f"  paper_orders_created : {agg.paper_orders_created}")
    print(f"  fills                : {agg.fills}")
    print(f"  partial_fills        : {agg.partial_fills}")
    print(f"  system_errors        : {agg.system_errors}")
    print(f"  open_positions       : {agg.open_positions}")
    print(f"  realized_pnl         : {round(agg.realized_pnl, 4)}")

    print(f"\n  rejection_reason_counts: {dict(agg.rejection_reason_counts)}")
    print(f"  mm_rejection_by_family : {meta.get('rejection_reason_counts_by_family', {}).get('maker_rewarded_event_mm_v1', {})}")

    print("\n  DB rows written this sweep:")
    for tbl, cnt in delta.items():
        if cnt:
            print(f"    {tbl}: +{cnt}")

    print("\n  DB totals after sweep:")
    for tbl, cnt in after.items():
        if cnt:
            print(f"    {tbl}: {cnt}")

    # Per-cycle breakdown
    print("\n  Per-cycle breakdown:")
    cycle_pos_limit_exceeded: list[int] = []
    for i, run in enumerate(result.batch_summaries[0].per_run_summaries):
        r_meta = run.metadata or {}
        r_raw = r_meta.get("raw_candidates_by_family", {}).get("maker_rewarded_event_mm_v1", 0)
        mm_rejections = r_meta.get("rejection_reason_counts_by_family", {}).get("maker_rewarded_event_mm_v1", {})
        pos_limit_hits = mm_rejections.get("POSITION_LIMIT_EXCEEDED", 0)
        cycle_pos_limit_exceeded.append(pos_limit_hits)
        print(
            f"    cycle {i+1}: raw={r_raw}"
            f"  qual={run.candidates_generated}"
            f"  risk_ok={run.risk_accepted}"
            f"  orders={run.paper_orders_created}"
            f"  fills={run.fills}"
            f"  open_pos={run.open_positions}"
            f"  pos_limit_exceeded={pos_limit_hits}"
            f"  errors={run.system_errors}"
        )

    # Accounting integrity checks
    print("\n  Accounting integrity:")
    candidates_eq_risk = (agg.candidates_generated == agg.risk_accepted + agg.risk_rejected)
    print(f"    candidates == risk_accepted + risk_rejected : {candidates_eq_risk}"
          f"  ({agg.candidates_generated} == {agg.risk_accepted} + {agg.risk_rejected})")
    orders_eq_2x_risk_accepted = (agg.paper_orders_created == 2 * agg.risk_accepted)
    print(f"    orders == 2 * risk_accepted                 : {orders_eq_2x_risk_accepted}"
          f"  ({agg.paper_orders_created} == 2 * {agg.risk_accepted})")
    persistence_ok = (
        delta.get("opportunity_candidates", 0) > 0
        and delta.get("risk_decisions", 0) > 0
        and delta.get("order_intents", 0) > 0
        and delta.get("execution_reports", 0) > 0
    )
    print(f"    persistence_ok (all 4 tables written)       : {persistence_ok}")
    no_contamination = arb_raw == 0
    print(f"    no_arb_contamination                        : {no_contamination}")
    zero_errors = agg.system_errors == 0
    print(f"    system_errors == 0                          : {zero_errors}")
    zero_partials = agg.partial_fills == 0
    print(f"    partial_fills == 0                          : {zero_partials}")

    all_ok = all([candidates_eq_risk, orders_eq_2x_risk_accepted,
                  persistence_ok, no_contamination, zero_errors, zero_partials,
                  result.cycles_completed == CYCLES])

    # P11-A verdict: did inter_cycle_reset restore independent observation windows?
    # The blocker was cross-cycle saturation: cycle 2+ had risk_accepted == 0 because the
    # ledger carried 8 positions forward from cycle 1.  Intra-cycle saturation (some
    # POSITION_LIMIT_EXCEEDED hits within a single cycle) is normal and expected.
    # Pass condition: every cycle 2+ has risk_accepted > 0 (i.e. accepted at least one
    # opportunity, proving the cycle did not start fully saturated).
    per_run = result.batch_summaries[0].per_run_summaries
    cycle_risk_accepted = [run.risk_accepted for run in per_run]
    print("\n  P11-A verdict:")
    print(f"    pos_limit_exceeded per cycle : {cycle_pos_limit_exceeded}")
    print(f"    risk_accepted per cycle      : {cycle_risk_accepted}")
    p11a_blocker_resolved = (
        all(n > 0 for n in cycle_risk_accepted[1:]) if len(cycle_risk_accepted) > 1 else True
    )
    p11a_complete = p11a_blocker_resolved and result.cycles_completed == CYCLES
    print(f"    cycles 2+ have risk_accepted > 0 : {p11a_blocker_resolved}")
    if p11a_complete:
        print("    GOOD NEWS  : inter_cycle_reset restores independent observation windows")
        print("                 — cycles 2+ are no longer cross-cycle saturated")
        print("    BAD NEWS   : intra-cycle POSITION_LIMIT_EXCEEDED is expected and normal")
        print("                 (ledger fills within a cycle; resets cleanly before the next)")
        print("    ABANDONED state (P11-D) : NOT required — defer indefinitely")
    else:
        print("    BAD NEWS   : cycles 2+ still show risk_accepted == 0 despite reset")
        print("    ABANDONED state (P11-D) : consider scheduling if depth starvation confirmed")
    print(f"    P11_A_VERDICT : {'RESOLVED' if p11a_complete else 'PARTIAL — secondary blocker remains'}")

    print()
    print(f"  SWEEP_VERDICT: {'BROADER_REGRESSION_PASS' if all_ok else 'BROADER_REGRESSION_FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
