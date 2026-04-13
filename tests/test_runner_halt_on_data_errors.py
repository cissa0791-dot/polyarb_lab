"""Tests for L7 halt_on_data_errors circuit breaker in ResearchRunner.

All tests are offline.  discovery fetches are patched to raise so no network
call is made.  The runner store is replaced with a MagicMock so no SQLite
writes occur.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.domain.models import RunSummary
from src.runtime.runner import ResearchRunner


def _runner_with_mock_store() -> ResearchRunner:
    runner = ResearchRunner()
    runner.store = MagicMock()
    runner.opportunity_store = MagicMock()
    return runner


class TestHaltOnDataErrors(unittest.TestCase):

    # ------------------------------------------------------------------
    # halt_on_data_errors=True (default): exception must propagate
    # ------------------------------------------------------------------

    def test_raises_when_flag_true_and_market_fetch_fails(self) -> None:
        runner = _runner_with_mock_store()
        self.assertTrue(runner.config.risk.halt_on_data_errors)  # verify default
        with (
            patch("src.runtime.runner.fetch_events", side_effect=ConnectionError("gamma down")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            with self.assertRaises(ConnectionError):
                runner.run_once()

    def test_raises_when_flag_explicitly_true(self) -> None:
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = True
        with (
            patch("src.runtime.runner.fetch_events", side_effect=RuntimeError("service unavailable")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            with self.assertRaises(RuntimeError):
                runner.run_once()

    # ------------------------------------------------------------------
    # halt_on_data_errors=False: degrade gracefully, return RunSummary
    # ------------------------------------------------------------------

    def test_returns_run_summary_when_flag_false(self) -> None:
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = False
        with (
            patch("src.runtime.runner.fetch_events", side_effect=ConnectionError("gamma down")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            result = runner.run_once()
        self.assertIsInstance(result, RunSummary)

    def test_no_exception_when_flag_false(self) -> None:
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = False
        with (
            patch("src.runtime.runner.fetch_events", side_effect=OSError("timeout")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            try:
                runner.run_once()
            except Exception as exc:
                self.fail(f"run_once raised unexpectedly: {exc}")

    def test_summary_has_system_error_recorded(self) -> None:
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = False
        with (
            patch("src.runtime.runner.fetch_events", side_effect=ConnectionError("timeout")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            result = runner.run_once()
        self.assertGreaterEqual(result.system_errors, 1)

    def test_summary_has_zero_markets_scanned(self) -> None:
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = False
        with (
            patch("src.runtime.runner.fetch_events", side_effect=ConnectionError("fail")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            result = runner.run_once()
        self.assertEqual(result.markets_scanned, 0)

    def test_run_summary_saved_to_store(self) -> None:
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = False
        with (
            patch("src.runtime.runner.fetch_events", side_effect=ConnectionError("fail")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            runner.run_once()
        runner.store.save_run_summary.assert_called_once()

    def test_runner_state_reset_after_abort(self) -> None:
        # _current_summary and _current_run_id must be None after early abort
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = False
        with (
            patch("src.runtime.runner.fetch_events", side_effect=ConnectionError("fail")),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            runner.run_once()
        self.assertIsNone(runner._current_summary)
        self.assertIsNone(runner._current_run_id)

    # ------------------------------------------------------------------
    # Default paper behavior unchanged (fetch succeeds: no early abort)
    # ------------------------------------------------------------------

    def test_default_paper_behavior_unaffected_when_no_error(self) -> None:
        # When discovery fetch succeeds, halt_on_data_errors has no observable effect
        runner = _runner_with_mock_store()
        runner.config.risk.halt_on_data_errors = True
        with (
            patch("src.runtime.runner.fetch_events", return_value=[]),
            patch("src.runtime.runner.fetch_markets_from_events", return_value=[]),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        ):
            # Empty market list returns a valid summary without touching CLOB
            result = runner.run_once()
        self.assertIsInstance(result, RunSummary)
        self.assertEqual(result.markets_scanned, 0)


if __name__ == "__main__":
    unittest.main()
