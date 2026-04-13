from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from src.opportunity.models import (
    QualificationDecision,
    QualificationFunnelReport,
    QualificationShortlistEntry,
)


class QualificationAuditor:
    """Accumulates QualificationDecision records during a scan cycle and
    produces a QualificationFunnelReport at the end.

    Usage::

        auditor = QualificationAuditor(run_id=run_id)
        for decision in decisions:
            auditor.record(decision)
        report = auditor.report()

    The report answers three questions:
      1. How many raw candidates were evaluated?
      2. Which gates rejected how many candidates? (rejection_counts)
      3. Which candidates passed? (shortlist, with key metrics)

    No state is mutated after report() is called, but the auditor can be
    queried again — each call to report() rebuilds from the full decision list.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._decisions: list[QualificationDecision] = []

    def record(self, decision: QualificationDecision) -> None:
        self._decisions.append(decision)

    def report(self) -> QualificationFunnelReport:
        passed_decisions = [d for d in self._decisions if d.passed]
        rejected_decisions = [d for d in self._decisions if not d.passed]

        rejection_counts: Counter[str] = Counter()
        for d in rejected_decisions:
            for code in d.reason_codes:
                rejection_counts[code] += 1

        shortlist: list[QualificationShortlistEntry] = []
        for d in passed_decisions:
            ec = d.executable_candidate
            if ec is None:
                continue
            shortlist.append(
                QualificationShortlistEntry(
                    candidate_id=ec.candidate_id,
                    strategy_family=ec.strategy_family.value,
                    market_slugs=list(ec.market_slugs),
                    pass_reason_codes=list(d.pass_reason_codes),
                    gross_edge_cents=ec.gross_edge_cents,
                    net_edge_cents=round(ec.net_edge_cents, 6),
                    expected_net_profit_usd=ec.estimated_net_profit_usd,
                    required_depth_usd=ec.required_depth_usd,
                    available_depth_usd=ec.available_depth_usd,
                    partial_fill_risk_score=ec.partial_fill_risk_score,
                    non_atomic_execution_risk_score=ec.non_atomic_execution_risk_score,
                    ts=d.ts,
                )
            )

        return QualificationFunnelReport(
            run_id=self.run_id,
            evaluated=len(self._decisions),
            passed=len(passed_decisions),
            rejected=len(rejected_decisions),
            rejection_counts=dict(rejection_counts),
            shortlist=shortlist,
            ts=datetime.now(timezone.utc),
        )
