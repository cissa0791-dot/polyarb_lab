"""Tests for L3/L4/L5/L6 dry-run guard, execution-mode routing, human
review gate, and dispatch routing — ResearchRunner._effective_live_enabled,
_check_live_dry_run_conflict, _order_mode, _blocked_by_human_review_gate,
and _dispatch_order.

All tests are offline.  ResearchRunner() is constructed with the default
config files; no run_once() call is made so no network access occurs.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.domain.models import ExecutionReport, OrderIntent, OrderMode, OrderStatus, OrderType, RejectionReason, RiskDecision, RiskStatus
from src.live.broker import LiveBroker
from src.reporting.summary import RunSummaryBuilder
from src.runtime.runner import ResearchRunner


def _runner() -> ResearchRunner:
    return ResearchRunner()


def _dummy_intent() -> OrderIntent:
    return OrderIntent(
        intent_id="test-intent-1",
        candidate_id="cand-1",
        mode=OrderMode.PAPER,
        market_slug="test-market",
        token_id="tok-1",
        position_id="pos-1",
        side="BUY",
        order_type=OrderType.LIMIT,
        size=10.0,
        limit_price=0.55,
        max_notional_usd=5.5,
        ts=datetime.now(timezone.utc),
    )


def _dummy_report() -> ExecutionReport:
    return ExecutionReport(
        intent_id="test-intent-1",
        status=OrderStatus.SUBMITTED,
        ts=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# _effective_live_enabled
# ---------------------------------------------------------------------------

class TestEffectiveLiveEnabled(unittest.TestCase):

    def test_false_by_default(self) -> None:
        # defaults: live_enabled=False, dry_run=True
        runner = _runner()
        self.assertFalse(runner._effective_live_enabled)

    def test_false_when_live_enabled_but_dry_run_on(self) -> None:
        # contradictory config — dry_run wins
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True
        self.assertFalse(runner._effective_live_enabled)

    def test_true_only_when_live_enabled_and_dry_run_off(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        self.assertTrue(runner._effective_live_enabled)

    def test_false_when_both_off(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = False
        runner.config.execution.dry_run = False
        self.assertFalse(runner._effective_live_enabled)


# ---------------------------------------------------------------------------
# _check_live_dry_run_conflict
# ---------------------------------------------------------------------------

class TestCheckLiveDryRunConflict(unittest.TestCase):

    def test_warns_when_live_enabled_and_dry_run(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True
        with patch.object(runner.logger, "warning") as mock_warn:
            runner._check_live_dry_run_conflict()
        mock_warn.assert_called_once()
        msg = mock_warn.call_args[0][0]
        self.assertIn("live_enabled=True", msg)
        self.assertIn("dry_run=True", msg)

    def test_no_warning_when_default_config(self) -> None:
        runner = _runner()
        with patch.object(runner.logger, "warning") as mock_warn:
            runner._check_live_dry_run_conflict()
        mock_warn.assert_not_called()

    def test_no_warning_when_live_disabled(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = False
        runner.config.execution.dry_run = True
        with patch.object(runner.logger, "warning") as mock_warn:
            runner._check_live_dry_run_conflict()
        mock_warn.assert_not_called()

    def test_no_warning_when_dry_run_cleared(self) -> None:
        # live_enabled=True + dry_run=False is valid live config — not a conflict
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        with patch.object(runner.logger, "warning") as mock_warn:
            runner._check_live_dry_run_conflict()
        mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# _order_mode (L4 routing)
# ---------------------------------------------------------------------------

class TestOrderMode(unittest.TestCase):

    def test_paper_by_default(self) -> None:
        # defaults: live_enabled=False, dry_run=True → PAPER
        runner = _runner()
        self.assertEqual(runner._order_mode, OrderMode.PAPER)

    def test_paper_when_live_disabled_dry_run_off(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = False
        runner.config.execution.dry_run = False
        self.assertEqual(runner._order_mode, OrderMode.PAPER)

    def test_paper_when_live_enabled_but_dry_run_on(self) -> None:
        # contradictory config — dry_run wins, intent stamped PAPER
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True
        self.assertEqual(runner._order_mode, OrderMode.PAPER)

    def test_live_when_effective_live_enabled(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        self.assertEqual(runner._order_mode, OrderMode.LIVE)


# ---------------------------------------------------------------------------
# _blocked_by_human_review_gate (L5)
# ---------------------------------------------------------------------------

def _decision(*, human_review_required: bool = False) -> RiskDecision:
    return RiskDecision(
        candidate_id="cand-1",
        status=RiskStatus.APPROVED,
        approved_notional_usd=50.0,
        human_review_required=human_review_required,
        ts=datetime.now(timezone.utc),
    )


class TestHumanReviewGate(unittest.TestCase):

    # -- paper mode: gate is always open --

    def test_not_blocked_in_paper_mode_flag_false(self) -> None:
        runner = _runner()  # dry_run=True, live_enabled=False → paper
        result = runner._blocked_by_human_review_gate(
            _decision(human_review_required=False), "cand-1", {}
        )
        self.assertFalse(result)

    def test_not_blocked_in_paper_mode_flag_true(self) -> None:
        # human_review_required=True should have NO effect in paper mode
        runner = _runner()
        result = runner._blocked_by_human_review_gate(
            _decision(human_review_required=True), "cand-1", {}
        )
        self.assertFalse(result)

    def test_not_blocked_when_contradictory_config(self) -> None:
        # live_enabled=True but dry_run=True → _effective_live_enabled=False → not blocked
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True
        result = runner._blocked_by_human_review_gate(
            _decision(human_review_required=True), "cand-1", {}
        )
        self.assertFalse(result)

    # -- live mode: gate is conditional --

    def test_not_blocked_when_live_and_flag_false(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        result = runner._blocked_by_human_review_gate(
            _decision(human_review_required=False), "cand-1", {}
        )
        self.assertFalse(result)

    def test_blocked_when_live_and_flag_true(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        result = runner._blocked_by_human_review_gate(
            _decision(human_review_required=True), "cand-1", {}
        )
        self.assertTrue(result)

    def test_rejection_recorded_with_correct_reason_code(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        runner._current_summary = RunSummaryBuilder(
            run_id="test-run", started_ts=datetime.now(timezone.utc)
        )
        runner._blocked_by_human_review_gate(
            _decision(human_review_required=True), "cand-1", {}
        )
        self.assertEqual(
            runner._current_summary.rejection_reason_counts[
                RejectionReason.HUMAN_REVIEW_REQUIRED.value
            ],
            1,
        )

    def test_warning_logged_when_blocked(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        with patch.object(runner.logger, "warning") as mock_warn:
            runner._blocked_by_human_review_gate(
                _decision(human_review_required=True), "cand-1", {}
            )
        mock_warn.assert_called_once()

    def test_no_warning_when_not_blocked(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        with patch.object(runner.logger, "warning") as mock_warn:
            runner._blocked_by_human_review_gate(
                _decision(human_review_required=False), "cand-1", {}
            )
        mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# _dispatch_order (L6 routing)
# ---------------------------------------------------------------------------

class TestDispatchOrder(unittest.TestCase):

    def test_routes_to_paper_broker_by_default(self) -> None:
        runner = _runner()
        # defaults: paper mode
        intent = _dummy_intent()
        fake_book = object()
        expected = _dummy_report()
        with patch.object(runner.paper_broker, "submit_limit_order", return_value=expected) as mock_paper:
            result = runner._dispatch_order(intent, fake_book)
        mock_paper.assert_called_once_with(intent, fake_book)
        self.assertIs(result, expected)

    def test_raises_when_live_mode_but_no_live_broker(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        self.assertIsNone(runner.live_broker)
        with self.assertRaises(RuntimeError) as ctx:
            runner._dispatch_order(_dummy_intent(), object())
        self.assertIn("live_broker", str(ctx.exception))

    def test_routes_to_live_broker_when_live_mode(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        expected = _dummy_report()
        mock_live = MagicMock(spec=LiveBroker)
        mock_live.submit_limit_order.return_value = expected
        runner.live_broker = mock_live
        intent = _dummy_intent()
        result = runner._dispatch_order(intent, object())
        mock_live.submit_limit_order.assert_called_once_with(intent)
        self.assertIs(result, expected)

    def test_paper_broker_not_called_when_live_mode(self) -> None:
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        mock_live = MagicMock(spec=LiveBroker)
        mock_live.submit_limit_order.return_value = _dummy_report()
        runner.live_broker = mock_live
        with patch.object(runner.paper_broker, "submit_limit_order") as mock_paper:
            runner._dispatch_order(_dummy_intent(), object())
        mock_paper.assert_not_called()

    def test_live_broker_not_called_when_contradictory_config(self) -> None:
        # live_enabled=True + dry_run=True → _effective_live_enabled=False → paper path
        runner = _runner()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True
        mock_live = MagicMock(spec=LiveBroker)
        runner.live_broker = mock_live
        with patch.object(runner.paper_broker, "submit_limit_order", return_value=_dummy_report()):
            runner._dispatch_order(_dummy_intent(), object())
        mock_live.submit_limit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
