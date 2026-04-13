"""Tests that _run_maker_mm_scan persists CLOB depth snapshots for every
market whose books are fetched successfully.

Before the fix, _run_maker_mm_scan fetched yes/no books and stored them in
book_cache but never called _save_raw_snapshot("clob", ...).  Every other
scan method (single_market, cross_market, manage_open_positions) saves the
books it fetches.  This was a code omission.

Coverage:
  1. Both tokens saved with source='clob' when EV passes (positive-ev market)
  2. Both tokens saved with source='clob' even when EV <= 0 (save happens
     before the EV continue, so no snapshot is silently dropped)
  3. Two markets in one scan → four snapshot rows total
  4. Snapshot not saved when book fetch raises (exception path → continue
     before the save call, consistent with all other scan methods)
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, call, patch


def _build_runner():
    from src.runtime.runner import ResearchRunner

    temp_dir = tempfile.mkdtemp()
    runner = ResearchRunner(
        settings_path="config/settings.yaml",
        constraints_path="config/constraints.yaml",
        debug_output_dir=temp_dir,
    )
    runner.store = Mock()
    runner.store.save_raw_snapshot = Mock()
    runner.opportunity_store = Mock()
    runner.clob = Mock()
    return runner


def _fake_book(bids=None, asks=None):
    book = Mock()
    book.bids = bids or []
    book.asks = asks or []
    book.model_dump.return_value = {"bids": bids or [], "asks": asks or []}
    return book


def _fake_market(yes_token="tok-yes", no_token="tok-no", slug="test-market"):
    return {
        "yes_token_id": yes_token,
        "no_token_id":  no_token,
        "market_slug":  slug,
        "rewards_min_size": 20.0,
        "rewards_max_spread": 3.5,
        "reward_daily_rate": 2.0,
        "best_bid": 0.45,
        "best_ask": 0.55,
    }


_SCAN_PATH = "src.scanner.wide_scan_maker_mm"
_NOW = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Both tokens saved with source='clob' — positive EV path
# ---------------------------------------------------------------------------

class TestMakerMmClobSnapshotPositiveEv(unittest.TestCase):

    def test_yes_and_no_token_snapshots_saved_on_positive_ev(self) -> None:
        runner = _build_runner()
        yes_book = _fake_book()
        no_book  = _fake_book()
        runner.clob.get_book.side_effect = [yes_book, no_book]

        market = _fake_market()

        # Patch qualify at class level: returns failed decision so the loop
        # short-circuits cleanly after the snapshot save without needing a
        # fully-formed RawCandidate mock.
        from src.opportunity.qualification import QualificationDecision
        failed_decision = MagicMock(spec=QualificationDecision)
        failed_decision.passed = False
        failed_decision.reason_codes = []

        with (
            patch(f"{_SCAN_PATH}.fetch_wide_scan_maker_mm_candidates", return_value=[market]),
            patch(f"{_SCAN_PATH}.compute_wide_scan_ev", return_value={"total_ev": 0.05}),
            patch(f"{_SCAN_PATH}.build_wide_scan_raw_candidate", return_value=MagicMock()),
            patch.object(runner, "_decorate_raw_candidate", side_effect=lambda r: r),
            patch.object(runner, "_record_raw_candidate"),
            patch.object(runner, "_record_rejection"),
            patch("src.opportunity.qualification.ExecutionFeasibilityEvaluator.qualify",
                  return_value=failed_decision),
        ):
            runner._run_maker_mm_scan(_NOW, {})

        calls = runner.store.save_raw_snapshot.call_args_list
        sources    = [c.args[0] for c in calls]
        entity_ids = [c.args[1] for c in calls]

        self.assertIn("clob", sources)
        self.assertEqual(sources.count("clob"), 2)
        self.assertIn("tok-yes", entity_ids)
        self.assertIn("tok-no",  entity_ids)


# ---------------------------------------------------------------------------
# 2. Both tokens saved even when EV <= 0 (save is before the EV filter)
# ---------------------------------------------------------------------------

class TestMakerMmClobSnapshotNegativeEv(unittest.TestCase):

    def test_snapshots_written_before_ev_continue(self) -> None:
        runner = _build_runner()
        yes_book = _fake_book()
        no_book  = _fake_book()
        runner.clob.get_book.side_effect = [yes_book, no_book]

        market = _fake_market()

        with (
            patch(f"{_SCAN_PATH}.fetch_wide_scan_maker_mm_candidates", return_value=[market]),
            patch(f"{_SCAN_PATH}.compute_wide_scan_ev", return_value={"total_ev": -1.0}),
        ):
            runner._run_maker_mm_scan(_NOW, {})

        calls = runner.store.save_raw_snapshot.call_args_list
        sources    = [c.args[0] for c in calls]
        entity_ids = [c.args[1] for c in calls]

        self.assertEqual(sources.count("clob"), 2)
        self.assertIn("tok-yes", entity_ids)
        self.assertIn("tok-no",  entity_ids)

    def test_snapshot_timestamp_is_fetch_time_not_cycle_started(self) -> None:
        # After the timestamp-alignment fix, maker_mm CLOB snapshots use
        # datetime.now(timezone.utc) at fetch time, NOT the cycle_started
        # sentinel passed to _run_maker_mm_scan.  Verify the saved ts is a
        # UTC-aware datetime that differs from cycle_started (_NOW).
        runner = _build_runner()
        runner.clob.get_book.return_value = _fake_book()

        market = _fake_market()

        with (
            patch(f"{_SCAN_PATH}.fetch_wide_scan_maker_mm_candidates", return_value=[market]),
            patch(f"{_SCAN_PATH}.compute_wide_scan_ev", return_value={"total_ev": -1.0}),
        ):
            runner._run_maker_mm_scan(_NOW, {})

        for c in runner.store.save_raw_snapshot.call_args_list:
            if c.args[0] == "clob":
                saved_ts = c.args[3]
                self.assertIsInstance(saved_ts, datetime)
                self.assertIsNotNone(saved_ts.tzinfo)
                self.assertNotEqual(saved_ts, _NOW)


# ---------------------------------------------------------------------------
# 3. Two markets → four snapshot rows total
# ---------------------------------------------------------------------------

class TestMakerMmClobSnapshotMultipleMarkets(unittest.TestCase):

    def test_four_snapshots_for_two_markets(self) -> None:
        runner = _build_runner()
        runner.clob.get_book.return_value = _fake_book()

        markets = [
            _fake_market(yes_token="yes-1", no_token="no-1", slug="market-one"),
            _fake_market(yes_token="yes-2", no_token="no-2", slug="market-two"),
        ]

        with (
            patch(f"{_SCAN_PATH}.fetch_wide_scan_maker_mm_candidates", return_value=markets),
            patch(f"{_SCAN_PATH}.compute_wide_scan_ev", return_value={"total_ev": -1.0}),
        ):
            runner._run_maker_mm_scan(_NOW, {})

        clob_calls = [
            c for c in runner.store.save_raw_snapshot.call_args_list
            if c.args[0] == "clob"
        ]
        self.assertEqual(len(clob_calls), 4)
        entity_ids = {c.args[1] for c in clob_calls}
        self.assertEqual(entity_ids, {"yes-1", "no-1", "yes-2", "no-2"})


# ---------------------------------------------------------------------------
# 4. Book fetch exception → no snapshot written for that market
# ---------------------------------------------------------------------------

class TestMakerMmClobSnapshotFetchError(unittest.TestCase):

    def test_no_snapshot_when_book_fetch_raises(self) -> None:
        runner = _build_runner()
        from src.live.client import LiveClientError
        runner.clob.get_book.side_effect = Exception("network timeout")

        market = _fake_market()

        with (
            patch(f"{_SCAN_PATH}.fetch_wide_scan_maker_mm_candidates", return_value=[market]),
            patch.object(runner, "_record_event"),
        ):
            runner._run_maker_mm_scan(_NOW, {})

        clob_calls = [
            c for c in runner.store.save_raw_snapshot.call_args_list
            if c.args[0] == "clob"
        ]
        self.assertEqual(len(clob_calls), 0)


if __name__ == "__main__":
    unittest.main()
