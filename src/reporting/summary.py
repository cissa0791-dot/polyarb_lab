from __future__ import annotations

from collections import Counter
from collections import defaultdict
from datetime import datetime
from typing import Any

from src.domain.models import RunSummary


def _merge_nested_counts(target: dict[str, Counter[str]], source: dict[str, dict[str, int]]) -> None:
    for outer_key, nested in source.items():
        target[outer_key].update(nested)


class RunSummaryBuilder:
    def __init__(self, run_id: str, started_ts: datetime):
        self.run_id = run_id
        self.started_ts = started_ts
        self.snapshots_stored = 0
        self.markets_scanned = 0
        self.candidates_generated = 0
        self.risk_accepted = 0
        self.risk_rejected = 0
        self.near_miss_candidates = 0
        self.paper_orders_created = 0
        self.fills = 0
        self.partial_fills = 0
        self.cancellations = 0
        self.open_positions = 0
        self.closed_positions = 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.system_errors = 0
        self.rejection_reason_counts: Counter[str] = Counter()
        self.market_counts: Counter[str] = Counter()
        self.opportunity_type_counts: Counter[str] = Counter()
        self.raw_candidate_family_counts: Counter[str] = Counter()
        self.qualified_candidate_family_counts: Counter[str] = Counter()
        self.research_only_family_counts: Counter[str] = Counter()
        self.near_miss_family_counts: Counter[str] = Counter()
        self.rejection_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
        self.candidate_filter_failure_stage_counts: Counter[str] = Counter()
        self.candidate_filter_failure_stage_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
        self.strategy_family_funnel_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.strategy_family_markets_considered: dict[str, set[str]] = defaultdict(set)
        self.strategy_family_markets_with_any_signal: dict[str, set[str]] = defaultdict(set)
        self.books_fetched = 0
        self.books_structurally_valid = 0
        self.books_execution_feasible = 0
        self.books_skipped_due_to_recent_empty_asks = 0
        self.qualified_shortlist_count = 0
        self.qualification_rejection_counts_by_gate: Counter[str] = Counter()
        self.metadata: dict = {}

    def record_strategy_family_market_considered(self, strategy_family: str, market_slug: str | None) -> None:
        if strategy_family and market_slug:
            self.strategy_family_markets_considered[strategy_family].add(market_slug)

    def record_strategy_family_book_fetch(self, strategy_family: str, count: int = 1) -> None:
        if strategy_family and count > 0:
            self.strategy_family_funnel_counts[strategy_family]["books_fetched"] += count

    def record_strategy_family_book_validation(
        self,
        strategy_family: str,
        *,
        structurally_valid: bool,
        execution_feasible: bool,
    ) -> None:
        if not strategy_family:
            return
        if structurally_valid:
            self.strategy_family_funnel_counts[strategy_family]["books_structurally_valid"] += 1
        if execution_feasible:
            self.strategy_family_funnel_counts[strategy_family]["books_execution_feasible"] += 1

    def record_qualification_funnel(self, report: Any) -> None:
        """Incorporate a QualificationFunnelReport into the run's summary metadata."""
        self.qualified_shortlist_count = report.passed
        self.qualification_rejection_counts_by_gate.update(report.rejection_counts)

    def record_strategy_family_signal(self, strategy_family: str, market_slugs: list[str]) -> None:
        if not strategy_family:
            return
        for market_slug in market_slugs:
            if market_slug:
                self.strategy_family_markets_with_any_signal[strategy_family].add(market_slug)

    def _build_strategy_family_funnel_payload(self) -> dict[str, dict]:
        families = (
            set(self.strategy_family_funnel_counts)
            | set(self.strategy_family_markets_considered)
            | set(self.strategy_family_markets_with_any_signal)
            | set(self.raw_candidate_family_counts)
            | set(self.rejection_counts_by_family)
        )
        payload: dict[str, dict] = {}
        for family in sorted(families):
            counts = self.strategy_family_funnel_counts.get(family, Counter())
            payload[family] = {
                "markets_considered": int(counts.get("markets_considered", 0) + len(self.strategy_family_markets_considered.get(family, set()))),
                "books_fetched": int(counts.get("books_fetched", 0)),
                "books_structurally_valid": int(counts.get("books_structurally_valid", 0)),
                "books_execution_feasible": int(counts.get("books_execution_feasible", 0)),
                "raw_candidates_generated": int(self.raw_candidate_family_counts.get(family, 0)),
                "markets_with_any_signal": int(counts.get("markets_with_any_signal", 0) + len(self.strategy_family_markets_with_any_signal.get(family, set()))),
                "rejection_reason_counts": dict(self.rejection_counts_by_family.get(family, {})),
            }
        return payload

    def build(self, ended_ts: datetime) -> RunSummary:
        metadata = self.metadata.copy()
        metadata.update(
            {
                "raw_candidates_by_family": dict(self.raw_candidate_family_counts),
                "qualified_candidates_by_family": dict(self.qualified_candidate_family_counts),
                "research_only_candidates_by_family": dict(self.research_only_family_counts),
                "near_miss_by_family": dict(self.near_miss_family_counts),
                "rejection_reason_counts_by_family": {
                    family: dict(reason_counts)
                    for family, reason_counts in self.rejection_counts_by_family.items()
                },
                "candidate_filter_failure_stage_counts": dict(self.candidate_filter_failure_stage_counts),
                "candidate_filter_failure_stage_counts_by_family": {
                    family: dict(stage_counts)
                    for family, stage_counts in self.candidate_filter_failure_stage_counts_by_family.items()
                },
                "strategy_family_funnel": self._build_strategy_family_funnel_payload(),
                "orderbook_funnel": {
                    "books_fetched": self.books_fetched,
                    "books_structurally_valid": self.books_structurally_valid,
                    "books_execution_feasible": self.books_execution_feasible,
                    "raw_candidates_generated": int(sum(self.raw_candidate_family_counts.values())),
                    "qualified_candidates": self.candidates_generated,
                    "books_skipped_due_to_recent_empty_asks": self.books_skipped_due_to_recent_empty_asks,
                },
                "qualification_funnel": {
                    "qualified_shortlist_count": self.qualified_shortlist_count,
                    "rejection_counts_by_gate": dict(self.qualification_rejection_counts_by_gate),
                },
            }
        )
        return RunSummary(
            run_id=self.run_id,
            started_ts=self.started_ts,
            ended_ts=ended_ts,
            markets_scanned=self.markets_scanned,
            snapshots_stored=self.snapshots_stored,
            candidates_generated=self.candidates_generated,
            risk_accepted=self.risk_accepted,
            risk_rejected=self.risk_rejected,
            near_miss_candidates=self.near_miss_candidates,
            paper_orders_created=self.paper_orders_created,
            fills=self.fills,
            partial_fills=self.partial_fills,
            cancellations=self.cancellations,
            open_positions=self.open_positions,
            closed_positions=self.closed_positions,
            realized_pnl=round(self.realized_pnl, 6),
            unrealized_pnl=round(self.unrealized_pnl, 6),
            system_errors=self.system_errors,
            rejection_reason_counts=dict(self.rejection_reason_counts),
            top_markets_by_candidates=dict(self.market_counts.most_common(10)),
            top_opportunity_types=dict(self.opportunity_type_counts.most_common(10)),
            metadata=metadata,
        )


def aggregate_run_summaries(run_id: str, started_ts: datetime, ended_ts: datetime, summaries: list[RunSummary]) -> RunSummary:
    builder = RunSummaryBuilder(run_id=run_id, started_ts=started_ts)
    for summary in summaries:
        builder.markets_scanned += summary.markets_scanned
        builder.snapshots_stored += summary.snapshots_stored
        builder.candidates_generated += summary.candidates_generated
        builder.risk_accepted += summary.risk_accepted
        builder.risk_rejected += summary.risk_rejected
        builder.near_miss_candidates += summary.near_miss_candidates
        builder.paper_orders_created += summary.paper_orders_created
        builder.fills += summary.fills
        builder.partial_fills += summary.partial_fills
        builder.cancellations += summary.cancellations
        builder.open_positions += summary.open_positions
        builder.closed_positions += summary.closed_positions
        builder.realized_pnl += summary.realized_pnl
        builder.unrealized_pnl += summary.unrealized_pnl
        builder.system_errors += summary.system_errors
        builder.rejection_reason_counts.update(summary.rejection_reason_counts)
        builder.market_counts.update(summary.top_markets_by_candidates)
        builder.opportunity_type_counts.update(summary.top_opportunity_types)
        builder.raw_candidate_family_counts.update(summary.metadata.get("raw_candidates_by_family", {}))
        builder.qualified_candidate_family_counts.update(summary.metadata.get("qualified_candidates_by_family", {}))
        builder.research_only_family_counts.update(summary.metadata.get("research_only_candidates_by_family", {}))
        builder.near_miss_family_counts.update(summary.metadata.get("near_miss_by_family", {}))
        _merge_nested_counts(builder.rejection_counts_by_family, summary.metadata.get("rejection_reason_counts_by_family", {}))
        builder.candidate_filter_failure_stage_counts.update(summary.metadata.get("candidate_filter_failure_stage_counts", {}))
        _merge_nested_counts(
            builder.candidate_filter_failure_stage_counts_by_family,
            summary.metadata.get("candidate_filter_failure_stage_counts_by_family", {}),
        )
        for family, counts in summary.metadata.get("strategy_family_funnel", {}).items():
            family_counter = builder.strategy_family_funnel_counts[family]
            family_counter["markets_considered"] += int(counts.get("markets_considered", 0))
            family_counter["books_fetched"] += int(counts.get("books_fetched", 0))
            family_counter["books_structurally_valid"] += int(counts.get("books_structurally_valid", 0))
            family_counter["books_execution_feasible"] += int(counts.get("books_execution_feasible", 0))
            family_counter["markets_with_any_signal"] += int(counts.get("markets_with_any_signal", 0))
        funnel = summary.metadata.get("orderbook_funnel", {})
        builder.books_fetched += int(funnel.get("books_fetched", 0))
        builder.books_structurally_valid += int(funnel.get("books_structurally_valid", 0))
        builder.books_execution_feasible += int(funnel.get("books_execution_feasible", 0))
        builder.books_skipped_due_to_recent_empty_asks += int(funnel.get("books_skipped_due_to_recent_empty_asks", 0))
        qf = summary.metadata.get("qualification_funnel", {})
        builder.qualified_shortlist_count += int(qf.get("qualified_shortlist_count", 0))
        builder.qualification_rejection_counts_by_gate.update(qf.get("rejection_counts_by_gate", {}))
    return builder.build(ended_ts=ended_ts)
