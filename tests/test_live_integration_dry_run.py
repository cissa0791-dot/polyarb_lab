"""L10 dry-run integration tests for the live execution path.

Validates the full chain from runner routing through to LiveWriteClient
without sending real orders.  Uses LiveWriteClient(dry_run=True) so the
underlying ClobClient is never called.

Coverage goals:
  1. Full chain: runner._dispatch_order → LiveBroker → LiveWriteClient(dry_run=True)
     → no ClobClient.create_and_post_order call
  2. Contradictory config (live_enabled=True, dry_run=True) stays on paper path
  3. human_review_required gate blocks in live mode
  4. halt_on_data_errors behaves identically in live mode
  5. Paper mode unchanged when live_enabled=False
  6. run_once() completes cleanly in live+dry_run mode with mocked market data
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.domain.models import (
    ExecutionReport,
    OrderIntent,
    OrderMode,
    OrderStatus,
    OrderType,
    RejectionReason,
    RiskDecision,
    RiskStatus,
    RunSummary,
)
from src.live.broker import LiveBroker
from src.live.client import LiveWriteClient
from src.reporting.summary import RunSummaryBuilder
from src.runtime.runner import ResearchRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeClobClient:
    """Minimal ClobClient stand-in that records calls and never raises."""

    def __init__(self):
        self.calls: list[tuple] = []

    def create_and_post_order(self, order_args, options):
        self.calls.append(("create_and_post_order", order_args, options))
        return {"orderID": "live-test-99", "status": "live", "size_matched": "0.0"}


def _live_write_client_dry(fake_clob: _FakeClobClient) -> LiveWriteClient:
    """LiveWriteClient with dry_run=True — returns sentinel, never calls fake_clob."""
    return LiveWriteClient(fake_clob, dry_run=True)


def _live_write_client_live(fake_clob: _FakeClobClient) -> LiveWriteClient:
    """LiveWriteClient with dry_run=False — calls fake_clob for submit."""
    return LiveWriteClient(fake_clob, dry_run=False)


def _make_intent(mode: OrderMode = OrderMode.LIVE) -> OrderIntent:
    return OrderIntent(
        intent_id=str(uuid4()),
        candidate_id="cand-int-1",
        mode=mode,
        market_slug="test-market",
        token_id="tok-integration",
        position_id=str(uuid4()),
        side="BUY",
        order_type=OrderType.LIMIT,
        size=50.0,
        limit_price=0.55,
        max_notional_usd=27.5,
        ts=datetime.now(timezone.utc),
    )


def _runner_with_mock_store() -> ResearchRunner:
    runner = ResearchRunner()
    runner.store = MagicMock()
    runner.opportunity_store = MagicMock()
    return runner


def _live_runner(fake_clob: _FakeClobClient, client_dry_run: bool = True) -> ResearchRunner:
    """Runner wired for live execution with a fake CLOB client."""
    runner = _runner_with_mock_store()
    runner.config.execution.live_enabled = True
    runner.config.execution.dry_run = False
    lc = LiveWriteClient(fake_clob, dry_run=client_dry_run)
    runner.live_broker = LiveBroker(lc)
    return runner


# ---------------------------------------------------------------------------
# 1. Full chain: runner → LiveBroker → LiveWriteClient(dry_run=True)
#    LiveWriteClient.dry_run=True must return sentinel WITHOUT calling ClobClient
# ---------------------------------------------------------------------------

class TestFullChainWithDryRunClient(unittest.TestCase):

    def setUp(self) -> None:
        self.fake_clob = _FakeClobClient()
        self.runner = _live_runner(self.fake_clob, client_dry_run=True)

    def test_dispatch_routes_to_live_broker(self) -> None:
        # Paper broker must not be called when _order_mode == LIVE
        intent = _make_intent(mode=OrderMode.LIVE)
        with patch.object(self.runner.paper_broker, "submit_limit_order") as mock_paper:
            self.runner._dispatch_order(intent, object())
        mock_paper.assert_not_called()

    def test_no_real_clob_call_when_client_dry_run(self) -> None:
        # LiveWriteClient.dry_run=True short-circuits before create_and_post_order
        self.runner._dispatch_order(_make_intent(), object())
        self.assertEqual(self.fake_clob.calls, [])

    def test_returns_execution_report(self) -> None:
        report = self.runner._dispatch_order(_make_intent(), object())
        self.assertIsInstance(report, ExecutionReport)

    def test_report_status_is_submitted(self) -> None:
        # dry_run sentinel: size_matched=0 → SUBMITTED
        report = self.runner._dispatch_order(_make_intent(), object())
        self.assertEqual(report.status, OrderStatus.SUBMITTED)

    def test_report_metadata_shows_dry_run_status(self) -> None:
        report = self.runner._dispatch_order(_make_intent(), object())
        self.assertEqual(report.metadata.get("live_status"), "dry_run")

    def test_report_live_order_id_is_none(self) -> None:
        # sentinel has order_id=None
        report = self.runner._dispatch_order(_make_intent(), object())
        self.assertIsNone(report.metadata.get("live_order_id"))

    def test_report_filled_size_is_zero(self) -> None:
        report = self.runner._dispatch_order(_make_intent(), object())
        self.assertEqual(report.filled_size, 0.0)


# ---------------------------------------------------------------------------
# 2. Full chain when client dry_run=False: real CLOB call is made to fake
# ---------------------------------------------------------------------------

class TestFullChainWithLiveClient(unittest.TestCase):
    """Verifies that when LiveWriteClient.dry_run=False the CLOB IS called."""

    def test_clob_called_when_client_not_dry_run(self) -> None:
        fake_clob = _FakeClobClient()
        runner = _live_runner(fake_clob, client_dry_run=False)
        runner._dispatch_order(_make_intent(), object())
        self.assertEqual(len(fake_clob.calls), 1)
        self.assertEqual(fake_clob.calls[0][0], "create_and_post_order")

    def test_report_live_order_id_populated(self) -> None:
        fake_clob = _FakeClobClient()
        runner = _live_runner(fake_clob, client_dry_run=False)
        report = runner._dispatch_order(_make_intent(), object())
        self.assertEqual(report.metadata.get("live_order_id"), "live-test-99")


# ---------------------------------------------------------------------------
# 3. Contradictory config: live_enabled=True + dry_run=True → paper path
# ---------------------------------------------------------------------------

class TestContradictoryConfigStaysPaper(unittest.TestCase):

    def test_paper_broker_used_when_contradictory(self) -> None:
        fake_clob = _FakeClobClient()
        runner = _runner_with_mock_store()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True           # contradictory: dry_run wins
        runner.live_broker = LiveBroker(LiveWriteClient(fake_clob, dry_run=True))
        intent = _make_intent(mode=OrderMode.PAPER)       # mode resolved at dispatch time
        expected = ExecutionReport(
            intent_id=intent.intent_id,
            status=OrderStatus.SUBMITTED,
            ts=datetime.now(timezone.utc),
        )
        with patch.object(runner.paper_broker, "submit_limit_order", return_value=expected) as mock_paper:
            runner._dispatch_order(intent, object())
        mock_paper.assert_called_once()

    def test_no_clob_call_when_contradictory(self) -> None:
        fake_clob = _FakeClobClient()
        runner = _runner_with_mock_store()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True
        runner.live_broker = LiveBroker(LiveWriteClient(fake_clob, dry_run=False))
        with patch.object(runner.paper_broker, "submit_limit_order",
                          return_value=ExecutionReport(
                              intent_id="x", status=OrderStatus.SUBMITTED,
                              ts=datetime.now(timezone.utc)
                          )):
            runner._dispatch_order(_make_intent(), object())
        self.assertEqual(fake_clob.calls, [])

    def test_warning_logged_for_contradictory_config(self) -> None:
        runner = _runner_with_mock_store()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = True
        with patch.object(runner.logger, "warning") as mock_warn:
            runner._check_live_dry_run_conflict()
        mock_warn.assert_called_once()


# ---------------------------------------------------------------------------
# 4. human_review_required gate in live mode
# ---------------------------------------------------------------------------

class TestHumanReviewGateInLiveMode(unittest.TestCase):

    def _approved_live_decision(self, human_review: bool) -> RiskDecision:
        return RiskDecision(
            candidate_id="cand-1",
            status=RiskStatus.APPROVED,
            approved_notional_usd=50.0,
            human_review_required=human_review,
            ts=datetime.now(timezone.utc),
        )

    def test_gate_blocks_in_live_mode_when_review_required(self) -> None:
        runner = _live_runner(_FakeClobClient())
        runner._current_summary = RunSummaryBuilder(
            run_id="test", started_ts=datetime.now(timezone.utc)
        )
        blocked = runner._blocked_by_human_review_gate(
            self._approved_live_decision(human_review=True), "cand-1", {}
        )
        self.assertTrue(blocked)

    def test_gate_passes_in_live_mode_when_review_not_required(self) -> None:
        runner = _live_runner(_FakeClobClient())
        blocked = runner._blocked_by_human_review_gate(
            self._approved_live_decision(human_review=False), "cand-1", {}
        )
        self.assertFalse(blocked)

    def test_rejection_reason_code_is_human_review_required(self) -> None:
        runner = _live_runner(_FakeClobClient())
        runner._current_summary = RunSummaryBuilder(
            run_id="test", started_ts=datetime.now(timezone.utc)
        )
        runner._blocked_by_human_review_gate(
            self._approved_live_decision(human_review=True), "cand-1", {}
        )
        self.assertEqual(
            runner._current_summary.rejection_reason_counts[
                RejectionReason.HUMAN_REVIEW_REQUIRED.value
            ],
            1,
        )


# ---------------------------------------------------------------------------
# 5. halt_on_data_errors in live mode (identical behavior to paper mode)
# ---------------------------------------------------------------------------

class TestHaltOnDataErrorsInLiveMode(unittest.TestCase):

    def test_raises_in_live_mode_when_flag_true(self) -> None:
        runner = _live_runner(_FakeClobClient())
        runner.config.risk.halt_on_data_errors = True
        with patch("src.runtime.runner.fetch_markets", side_effect=ConnectionError("down")):
            with self.assertRaises(ConnectionError):
                runner.run_once()

    def test_returns_summary_in_live_mode_when_flag_false(self) -> None:
        runner = _live_runner(_FakeClobClient())
        runner.config.risk.halt_on_data_errors = False
        with patch("src.runtime.runner.fetch_markets", side_effect=ConnectionError("down")):
            result = runner.run_once()
        self.assertIsInstance(result, RunSummary)
        self.assertGreaterEqual(result.system_errors, 1)


# ---------------------------------------------------------------------------
# 6. Paper mode unchanged when live_enabled=False
# ---------------------------------------------------------------------------

class TestPaperModeUnchanged(unittest.TestCase):

    def test_paper_broker_called_when_live_disabled(self) -> None:
        runner = _runner_with_mock_store()
        # defaults: live_enabled=False, dry_run=True
        intent = _make_intent(mode=OrderMode.PAPER)
        expected = ExecutionReport(
            intent_id=intent.intent_id,
            status=OrderStatus.SUBMITTED,
            ts=datetime.now(timezone.utc),
        )
        with patch.object(runner.paper_broker, "submit_limit_order", return_value=expected) as mock_paper:
            runner._dispatch_order(intent, object())
        mock_paper.assert_called_once()

    def test_run_once_completes_with_live_disabled_and_empty_markets(self) -> None:
        runner = _runner_with_mock_store()
        with patch("src.runtime.runner.fetch_markets", return_value=[]):
            result = runner.run_once()
        self.assertIsInstance(result, RunSummary)
        self.assertEqual(result.markets_scanned, 0)

    def test_run_once_completes_with_live_enabled_and_empty_markets(self) -> None:
        runner = _live_runner(_FakeClobClient())
        with patch("src.runtime.runner.fetch_markets", return_value=[]):
            result = runner.run_once()
        self.assertIsInstance(result, RunSummary)
        self.assertEqual(result.markets_scanned, 0)

    def test_no_clob_call_on_empty_market_list_live_mode(self) -> None:
        fake_clob = _FakeClobClient()
        runner = _live_runner(fake_clob, client_dry_run=False)
        with patch("src.runtime.runner.fetch_markets", return_value=[]):
            runner.run_once()
        self.assertEqual(fake_clob.calls, [])


if __name__ == "__main__":
    unittest.main()
