from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from src.constraints.political_rules import POLITICAL_BINARY_MODEL_ID
from src.reporting.summary import RunSummaryBuilder
from src.runtime.political_binary_runner import PoliticalBinaryPaperRunner
from src.strategies.opportunity_strategies import PoliticalBinaryConstraintPaperStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []

    def model_dump(self, mode: str = "json"):
        return {
            "asks": [{"price": level.price, "size": level.size} for level in self.asks],
            "bids": [{"price": level.price, "size": level.size} for level in self.bids],
        }


def _political_rule(relation_type: str, lhs_slug: str, rhs_slug: str, lhs_exec_side: str, rhs_exec_side: str) -> dict:
    return {
        "name": f"{relation_type}:{lhs_slug}:{rhs_slug}",
        "relation": "political_binary",
        "relation_type": relation_type,
        "strategy_family": "political_binary_constraint_paper",
        "lhs": {"market_slug": lhs_slug, "side": "YES"},
        "rhs": {"market_slug": rhs_slug, "side": "YES"},
        "lhs_execution": {"side": lhs_exec_side},
        "rhs_execution": {"side": rhs_exec_side},
        "tier": "A",
        "assertion": "approved-test-assertion",
        "trade_enabled": True,
        "notes": "focused test rule",
        "preconditions": {"scope": "focused-test"},
        "constraint_model_id": POLITICAL_BINARY_MODEL_ID,
    }


class PoliticalBinaryConstraintStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = PoliticalBinaryConstraintPaperStrategy()

    def test_nominee_family_mutex_detects_no_no_pair(self) -> None:
        rule = _political_rule(
            "nominee_family_mutex",
            "dem-a",
            "dem-b",
            "NO",
            "NO",
        )
        raw_candidate, audit = self.strategy.detect_with_audit(
            rule,
            Book(asks=[Level(0.64, 200.0)]),
            Book(asks=[Level(0.47, 200.0)]),
            "dem-a-no",
            "NO",
            Book(asks=[Level(0.31, 200.0)]),
            "dem-b-no",
            "NO",
            Book(asks=[Level(0.42, 200.0)]),
            100.0,
            0.02,
        )

        self.assertIsNone(audit)
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None
        self.assertEqual(raw_candidate.strategy_family.value, "political_binary_constraint_paper")
        self.assertEqual([leg.side for leg in raw_candidate.legs], ["NO", "NO"])
        self.assertFalse(raw_candidate.research_only)
        self.assertEqual(raw_candidate.execution_mode, "paper_eligible")

    def test_winner_family_mutex_detects_no_no_pair(self) -> None:
        rule = _political_rule(
            "winner_family_mutex",
            "winner-a",
            "winner-b",
            "NO",
            "NO",
        )
        raw_candidate, audit = self.strategy.detect_with_audit(
            rule,
            Book(asks=[Level(0.58, 200.0)]),
            Book(asks=[Level(0.46, 200.0)]),
            "winner-a-no",
            "NO",
            Book(asks=[Level(0.33, 200.0)]),
            "winner-b-no",
            "NO",
            Book(asks=[Level(0.44, 200.0)]),
            100.0,
            0.02,
        )

        self.assertIsNone(audit)
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None
        self.assertEqual(raw_candidate.metadata["relation_type"], "winner_family_mutex")
        self.assertEqual([leg.side for leg in raw_candidate.legs], ["NO", "NO"])

    def test_implication_rule_detects_no_yes_pair(self) -> None:
        rule = _political_rule(
            "time_monotone_implication",
            "event-earlier",
            "event-later",
            "NO",
            "YES",
        )
        raw_candidate, audit = self.strategy.detect_with_audit(
            rule,
            Book(asks=[Level(0.67, 200.0)]),
            Book(asks=[Level(0.54, 200.0)]),
            "event-earlier-no",
            "NO",
            Book(asks=[Level(0.22, 200.0)]),
            "event-later-yes",
            "YES",
            Book(asks=[Level(0.43, 200.0)]),
            100.0,
            0.02,
        )

        self.assertIsNone(audit)
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None
        self.assertEqual(raw_candidate.metadata["relation_type"], "time_monotone_implication")
        self.assertEqual([leg.side for leg in raw_candidate.legs], ["NO", "YES"])


class PoliticalBinaryPaperExecutionPathTests(unittest.TestCase):
    def test_political_runner_can_flow_into_risk_and_paper_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = PoliticalBinaryPaperRunner(
                settings_path="config/settings.yaml",
                constraints_path="config/political_constraint_rules.yaml",
                debug_output_dir=tmp_dir,
            )
            runner.set_experiment_context(
                campaign_target_strategy_families=["political_binary_constraint_paper"],
            )
            runner.store = Mock()
            runner.opportunity_store = Mock()
            runner._current_run_id = "test-political-run"
            runner._current_summary = RunSummaryBuilder(
                run_id=runner._current_run_id,
                started_ts=datetime.now(timezone.utc),
            )
            runner._current_summary.metadata.update(runner._build_experiment_metadata())
            runner._invalid_orderbook_export_path = Path(tmp_dir) / "invalid.jsonl"

            rule = {
                "cross_market": [
                    _political_rule(
                        "winner_family_mutex",
                        "winner-a",
                        "winner-b",
                        "NO",
                        "NO",
                    )
                ]
            }
            token_map = {
                ("winner-a", "YES"): "winner-a-yes",
                ("winner-b", "YES"): "winner-b-yes",
                ("winner-a", "NO"): "winner-a-no",
                ("winner-b", "NO"): "winner-b-no",
            }

            runner.clob = Mock()
            runner.clob.get_book.side_effect = [
                Book(asks=[Level(0.64, 1000.0)], bids=[Level(0.63, 1000.0)]),
                Book(asks=[Level(0.47, 1000.0)], bids=[Level(0.46, 1000.0)]),
                Book(asks=[Level(0.31, 1000.0)], bids=[Level(0.30, 1000.0)]),
                Book(asks=[Level(0.33, 1000.0)], bids=[Level(0.32, 1000.0)]),
            ]

            runner._run_political_constraint_scan(rule, token_map, datetime.now(timezone.utc), {})

            self.assertEqual(runner._current_summary.candidates_generated, 1)
            self.assertEqual(runner._current_summary.risk_accepted, 1)
            self.assertEqual(runner._current_summary.paper_orders_created, 2)
            self.assertEqual(runner.store.save_candidate.call_count, 1)
            self.assertEqual(runner.store.save_risk_decision.call_count, 1)
            self.assertEqual(runner.store.save_order_intent.call_count, 2)
            self.assertEqual(runner.store.save_execution_report.call_count, 2)
            self.assertEqual(runner.store.save_position_event.call_count, 2)
            self.assertGreaterEqual(runner.store.save_account_snapshot.call_count, 1)


if __name__ == "__main__":
    unittest.main()
