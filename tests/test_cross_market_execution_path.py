from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from src.reporting.summary import RunSummaryBuilder
from src.runtime.runner import ResearchRunner


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


class CrossMarketExecutionPathTests(unittest.TestCase):
    def test_cross_market_execution_gross_can_flow_into_risk_and_paper_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = ResearchRunner(
                settings_path="config/settings.yaml",
                constraints_path="config/constraints.yaml",
                debug_output_dir=tmp_dir,
            )
            runner.apply_runtime_parameter_set("execution_tinybuffer_probe")
            runner.set_experiment_context(
                campaign_target_strategy_families=["cross_market_execution_gross_constraint"],
                parameter_set_label="execution_tinybuffer_probe",
            )
            runner.store = Mock()
            runner.opportunity_store = Mock()
            runner._current_run_id = "test-cross-market-run"
            runner._current_summary = RunSummaryBuilder(
                run_id=runner._current_run_id,
                started_ts=datetime.now(timezone.utc),
            )
            runner._current_summary.metadata.update(runner._build_experiment_metadata())
            runner._invalid_orderbook_export_path = Path(tmp_dir) / "invalid.jsonl"

            rule = {
                "cross_market": [
                    {
                        "name": "lhs_leq_rhs",
                        "relation": "leq",
                        "lhs": {"market_slug": "market-a", "side": "YES"},
                        "rhs": {"market_slug": "market-b", "side": "YES"},
                    }
                ]
            }
            token_map = {
                ("market-a", "YES"): "lhs-rel",
                ("market-b", "YES"): "rhs-rel",
                ("market-a", "NO"): "lhs-exec",
                ("market-b", "YES"): "rhs-rel",
            }

            relation_lhs = Book(asks=[Level(0.025, 200.0)], bids=[Level(0.024, 200.0)])
            relation_rhs = Book(asks=[Level(0.019, 200.0)], bids=[Level(0.018, 200.0)])
            lhs_exec = Book(asks=[Level(0.977, 200.0)], bids=[Level(0.976, 200.0)])
            rhs_exec = Book(asks=[Level(0.019, 200.0)], bids=[Level(0.018, 200.0)])

            books = {
                "lhs-rel": relation_lhs,
                "rhs-rel": relation_rhs,
                "lhs-exec": lhs_exec,
            }

            def get_book(token_id: str):
                if token_id == "rhs-rel":
                    return rhs_exec if get_book.calls == 3 else relation_rhs
                return books[token_id]

            get_book.calls = 0

            def side_effect(token_id: str):
                result = get_book(token_id)
                get_book.calls += 1
                return result

            runner.clob = Mock()
            runner.clob.get_book.side_effect = side_effect

            runner._run_cross_market_scan(rule, token_map, datetime.now(timezone.utc), {})

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
