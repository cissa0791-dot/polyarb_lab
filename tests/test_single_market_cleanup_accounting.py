from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.domain.models import RejectionReason
from src.reporting.summary import RunSummaryBuilder
from src.runtime.runner import ResearchRunner


class _Validation:
    def __init__(self, reason_code: str, problem_stage: str = "validate") -> None:
        self.reason_code = reason_code
        self.problem_stage = problem_stage
        self.required_action = "BUY"

    def to_debug_payload(self) -> dict:
        return {
            "token_id": "token-a",
            "required_action": self.required_action,
            "validation_rule": "required_ask_side_empty",
            "problem_stage": self.problem_stage,
        }


class SingleMarketCleanupAccountingTests(unittest.TestCase):
    def test_empty_asks_invalid_orderbook_is_normalized_to_pre_candidate_precheck(self) -> None:
        runner = ResearchRunner()
        runner._current_summary = RunSummaryBuilder(run_id="run-1", started_ts=datetime.now(timezone.utc))

        runner._record_invalid_orderbook(
            stage="candidate_filter",
            validation=_Validation(RejectionReason.EMPTY_ASKS.value),
            metadata={
                "market_slug": "market-a",
                "token_id": "token-a",
                "side": "YES",
                "strategy_family": runner.single_market_strategy.strategy_family.value,
            },
        )

        counts = runner._current_summary.candidate_filter_failure_stage_counts
        self.assertEqual(counts["pre_candidate_precheck"], 1)

    def test_summary_metadata_does_not_need_unknown_when_failure_stage_is_explicit(self) -> None:
        builder = RunSummaryBuilder(run_id="run-1", started_ts=datetime.now(timezone.utc))
        builder.candidate_filter_failure_stage_counts["pre_candidate_precheck"] += 2
        summary = builder.build(ended_ts=datetime.now(timezone.utc))

        self.assertEqual(summary.metadata["candidate_filter_failure_stage_counts"]["pre_candidate_precheck"], 2)
        self.assertNotIn("unknown", summary.metadata["candidate_filter_failure_stage_counts"])

    def test_summary_metadata_includes_failure_stage_counts_by_family(self) -> None:
        builder = RunSummaryBuilder(run_id="run-1", started_ts=datetime.now(timezone.utc))
        builder.candidate_filter_failure_stage_counts["pre_candidate_precheck"] += 1
        builder.candidate_filter_failure_stage_counts_by_family["single_market_mispricing"]["pre_candidate_precheck"] += 1
        summary = builder.build(ended_ts=datetime.now(timezone.utc))

        self.assertEqual(summary.metadata["candidate_filter_failure_stage_counts"]["pre_candidate_precheck"], 1)
        self.assertEqual(
            summary.metadata["candidate_filter_failure_stage_counts_by_family"]["single_market_mispricing"]["pre_candidate_precheck"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
