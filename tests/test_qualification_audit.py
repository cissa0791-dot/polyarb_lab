"""
tests/test_qualification_audit.py

Tests for two new components:

  QualificationAuditor  (src/opportunity/audit.py)
      Accumulates QualificationDecision records and emits a
      QualificationFunnelReport with per-gate rejection counts and a
      shortlist of passed candidates with explicit metrics.

  Pass-reason codes  (src/opportunity/qualification.py + models.py)
      qualify() now populates QualificationDecision.pass_reason_codes and
      ExecutableCandidate.qualification_reason_codes for every passed
      candidate, naming each gate that was explicitly cleared.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.config_runtime.models import OpportunityConfig
from src.opportunity.audit import QualificationAuditor
from src.opportunity.models import (
    CandidateLeg,
    QualificationDecision,
    QualificationPassReason,
    RawCandidate,
    StrategyFamily,
)
from src.opportunity.qualification import ExecutionFeasibilityEvaluator


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _raw(required_shares: float = 5.0, notional: float = 4.0, candidate_id: str = "audit-raw") -> RawCandidate:
    return RawCandidate(
        strategy_id="s1",
        strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
        candidate_id=candidate_id,
        kind="single_market",
        detection_name="yes_no_under_1",
        market_slugs=["mkt-a"],
        gross_edge_cents=0.10,
        expected_payout=required_shares,
        target_notional_usd=notional,
        target_shares=required_shares,
        legs=[
            CandidateLeg(token_id="yes-t", market_slug="mkt-a", action="BUY", side="YES", required_shares=required_shares),
            CandidateLeg(token_id="no-t",  market_slug="mkt-a", action="BUY", side="NO",  required_shares=required_shares),
        ],
        ts=_ts(),
    )


def _permissive_config() -> OpportunityConfig:
    return OpportunityConfig(
        min_edge_cents=0.01,
        fee_buffer_cents=0.005,
        slippage_buffer_cents=0.005,
        min_depth_multiple=1.5,
        max_spread_cents=0.20,
        max_partial_fill_risk=0.95,
        max_non_atomic_risk=0.95,
        min_net_profit_usd=0.01,
    )


def _good_books() -> dict:
    book = Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)])
    return {"yes-t": book, "no-t": book}


def _thin_books() -> dict:
    thin = Book(asks=[Level(0.40, 0.5)], bids=[Level(0.39, 0.5)])
    return {"yes-t": thin, "no-t": thin}


def _reject_decision(reason_codes: list[str], candidate_id: str = "r1") -> QualificationDecision:
    """Build a rejection decision directly — no evaluator needed."""
    return QualificationDecision(
        raw_candidate=_raw(candidate_id=candidate_id),
        passed=False,
        reason_codes=reason_codes,
        metadata={},
        ts=_ts(),
    )


def _pass_decision(candidate_id: str = "p1") -> QualificationDecision:
    """Run qualify() on a well-formed candidate to get a real passed decision."""
    evaluator = ExecutionFeasibilityEvaluator(_permissive_config())
    return evaluator.qualify(_raw(candidate_id=candidate_id), _good_books())


# ---------------------------------------------------------------------------
# QualificationAuditor
# ---------------------------------------------------------------------------

class QualificationAuditorTests(unittest.TestCase):

    def test_empty_auditor_produces_zero_counts(self) -> None:
        auditor = QualificationAuditor(run_id="run-empty")
        report = auditor.report()

        self.assertEqual(report.evaluated, 0)
        self.assertEqual(report.passed, 0)
        self.assertEqual(report.rejected, 0)
        self.assertEqual(report.rejection_counts, {})
        self.assertEqual(report.shortlist, [])

    def test_rejected_decision_increments_rejection_counts(self) -> None:
        auditor = QualificationAuditor(run_id="run-r1")
        auditor.record(_reject_decision(["EDGE_BELOW_THRESHOLD", "INSUFFICIENT_DEPTH"]))
        report = auditor.report()

        self.assertEqual(report.evaluated, 1)
        self.assertEqual(report.rejected, 1)
        self.assertEqual(report.passed, 0)
        self.assertEqual(report.rejection_counts["EDGE_BELOW_THRESHOLD"], 1)
        self.assertEqual(report.rejection_counts["INSUFFICIENT_DEPTH"], 1)

    def test_passed_decision_appears_in_shortlist(self) -> None:
        auditor = QualificationAuditor(run_id="run-p1")
        auditor.record(_pass_decision(candidate_id="p1"))
        report = auditor.report()

        self.assertEqual(report.passed, 1)
        self.assertEqual(report.rejected, 0)
        self.assertEqual(len(report.shortlist), 1)
        self.assertEqual(report.shortlist[0].candidate_id, "p1")

    def test_mixed_decisions_produce_correct_funnel(self) -> None:
        auditor = QualificationAuditor(run_id="run-mixed")
        auditor.record(_pass_decision(candidate_id="p1"))
        auditor.record(_reject_decision(["EDGE_BELOW_THRESHOLD"], candidate_id="r1"))
        auditor.record(_reject_decision(["EDGE_BELOW_THRESHOLD", "NET_PROFIT_BELOW_THRESHOLD"], candidate_id="r2"))
        report = auditor.report()

        self.assertEqual(report.evaluated, 3)
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.rejected, 2)
        self.assertEqual(report.rejection_counts["EDGE_BELOW_THRESHOLD"], 2)
        self.assertEqual(report.rejection_counts["NET_PROFIT_BELOW_THRESHOLD"], 1)

    def test_shortlist_entry_exposes_key_metrics(self) -> None:
        auditor = QualificationAuditor(run_id="run-metrics")
        auditor.record(_pass_decision(candidate_id="m1"))
        report = auditor.report()
        entry = report.shortlist[0]

        self.assertIsInstance(entry.gross_edge_cents, float)
        self.assertIsInstance(entry.net_edge_cents, float)
        self.assertGreater(entry.gross_edge_cents, 0.0)
        self.assertGreater(entry.expected_net_profit_usd, 0.0)
        self.assertGreater(entry.available_depth_usd, 0.0)
        self.assertGreaterEqual(entry.partial_fill_risk_score, 0.0)
        self.assertGreaterEqual(entry.non_atomic_execution_risk_score, 0.0)
        self.assertIsInstance(entry.pass_reason_codes, list)
        self.assertGreater(len(entry.pass_reason_codes), 0)

    def test_rejection_counts_aggregate_same_code_across_decisions(self) -> None:
        auditor = QualificationAuditor(run_id="run-agg")
        for i in range(5):
            auditor.record(_reject_decision(["INSUFFICIENT_DEPTH"], candidate_id=f"r{i}"))
        report = auditor.report()

        self.assertEqual(report.rejection_counts["INSUFFICIENT_DEPTH"], 5)
        self.assertEqual(report.evaluated, 5)
        self.assertEqual(report.passed, 0)


# ---------------------------------------------------------------------------
# Pass-reason codes
# ---------------------------------------------------------------------------

class PassReasonCodeTests(unittest.TestCase):

    def test_passed_decision_has_non_empty_pass_reason_codes(self) -> None:
        evaluator = ExecutionFeasibilityEvaluator(_permissive_config())
        decision = evaluator.qualify(_raw(), _good_books())

        self.assertTrue(decision.passed)
        self.assertGreater(len(decision.pass_reason_codes), 0)

    def test_passed_executable_candidate_has_qualification_reason_codes(self) -> None:
        evaluator = ExecutionFeasibilityEvaluator(_permissive_config())
        decision = evaluator.qualify(_raw(), _good_books())

        self.assertTrue(decision.passed)
        assert decision.executable_candidate is not None
        self.assertGreater(len(decision.executable_candidate.qualification_reason_codes), 0)

    def test_standard_pass_codes_all_present_for_healthy_candidate(self) -> None:
        evaluator = ExecutionFeasibilityEvaluator(_permissive_config())
        decision = evaluator.qualify(_raw(), _good_books())

        self.assertIn(QualificationPassReason.EDGE_SUFFICIENT,        decision.pass_reason_codes)
        self.assertIn(QualificationPassReason.DEPTH_SUFFICIENT,       decision.pass_reason_codes)
        self.assertIn(QualificationPassReason.NET_PROFIT_SUFFICIENT,  decision.pass_reason_codes)
        self.assertIn(QualificationPassReason.PARTIAL_FILL_RISK_OK,   decision.pass_reason_codes)
        self.assertIn(QualificationPassReason.NON_ATOMIC_RISK_OK,     decision.pass_reason_codes)

    def test_absolute_depth_ok_present_when_gate_enabled_and_passes(self) -> None:
        # Gate enabled at $1 floor — good books have $8 available per leg → should pass
        config = OpportunityConfig(
            min_edge_cents=0.01,
            fee_buffer_cents=0.005,
            slippage_buffer_cents=0.005,
            min_depth_multiple=1.5,
            max_spread_cents=0.20,
            max_partial_fill_risk=0.95,
            max_non_atomic_risk=0.95,
            min_net_profit_usd=0.01,
            min_absolute_leg_depth_usd=1.0,
        )
        evaluator = ExecutionFeasibilityEvaluator(config)
        decision = evaluator.qualify(_raw(), _good_books())

        self.assertTrue(decision.passed)
        self.assertIn(QualificationPassReason.ABSOLUTE_DEPTH_OK, decision.pass_reason_codes)

    def test_rejected_decision_has_empty_pass_reason_codes(self) -> None:
        evaluator = ExecutionFeasibilityEvaluator(_permissive_config())
        # Thin book causes INSUFFICIENT_DEPTH and PARTIAL_FILL_RISK_TOO_HIGH
        decision = evaluator.qualify(_raw(), _thin_books())

        self.assertFalse(decision.passed)
        self.assertEqual(decision.pass_reason_codes, [])

    def test_pass_reason_codes_match_executable_qualification_reason_codes(self) -> None:
        """QualificationDecision.pass_reason_codes and
        ExecutableCandidate.qualification_reason_codes must be identical."""
        evaluator = ExecutionFeasibilityEvaluator(_permissive_config())
        decision = evaluator.qualify(_raw(), _good_books())

        self.assertTrue(decision.passed)
        assert decision.executable_candidate is not None
        self.assertEqual(
            decision.pass_reason_codes,
            decision.executable_candidate.qualification_reason_codes,
        )


if __name__ == "__main__":
    unittest.main()
