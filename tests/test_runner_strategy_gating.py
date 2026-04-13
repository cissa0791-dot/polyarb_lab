from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.runtime.runner import ResearchRunner


class RunnerStrategyGatingTests(unittest.TestCase):
    def _build_runner(self) -> ResearchRunner:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        runner = ResearchRunner(
            settings_path="config/settings.yaml",
            constraints_path="config/constraints.yaml",
            debug_output_dir=temp_dir.name,
        )
        runner.store = Mock()
        runner.store.save_raw_snapshot = Mock()
        runner.store.save_account_snapshot = Mock()
        runner.store.save_run_summary = Mock()
        runner.paper_ledger = Mock()
        runner.paper_ledger.snapshot.return_value = SimpleNamespace(
            open_positions=0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
        )
        return runner

    def test_run_once_only_runs_cross_market_when_cross_market_is_targeted(self) -> None:
        runner = self._build_runner()

        with (
            patch("src.runtime.runner.fetch_events", return_value=[]),
            patch("src.runtime.runner.fetch_markets_from_events", return_value=[]),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
            patch("src.runtime.runner._read_yaml", return_value={"cross_market": []}),
            patch.object(runner, "_run_single_market_scan") as single_scan,
            patch.object(runner, "_run_cross_market_scan") as cross_scan,
            patch.object(runner, "_manage_open_positions"),
        ):
            runner.run_once(
                experiment_context={
                    "campaign_target_strategy_families": ["cross_market_constraint"],
                }
            )

        single_scan.assert_not_called()
        cross_scan.assert_called_once()

    def test_run_once_only_runs_single_market_when_single_market_is_targeted(self) -> None:
        runner = self._build_runner()

        with (
            patch("src.runtime.runner.fetch_events", return_value=[]),
            patch("src.runtime.runner.fetch_markets_from_events", return_value=[]),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
            patch("src.runtime.runner._read_yaml", return_value={"cross_market": []}),
            patch.object(runner, "_run_single_market_scan") as single_scan,
            patch.object(runner, "_run_cross_market_scan") as cross_scan,
            patch.object(runner, "_manage_open_positions"),
        ):
            runner.run_once(
                experiment_context={
                    "campaign_target_strategy_families": ["single_market_mispricing"],
                }
            )

        single_scan.assert_called_once()
        cross_scan.assert_not_called()

    def test_run_once_only_runs_neg_risk_when_neg_risk_is_targeted(self) -> None:
        runner = self._build_runner()

        with (
            patch("src.runtime.runner.fetch_events", return_value=[]),
            patch("src.runtime.runner.fetch_markets_from_events", return_value=[]),
            patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
            patch("src.runtime.runner._read_yaml", return_value={"cross_market": []}),
            patch.object(runner, "_run_single_market_scan") as single_scan,
            patch.object(runner, "_run_cross_market_scan") as cross_scan,
            patch.object(runner, "_run_neg_risk_scan") as neg_risk_scan,
            patch.object(runner, "_manage_open_positions"),
        ):
            runner.run_once(
                experiment_context={
                    "campaign_target_strategy_families": ["neg_risk_rebalancing"],
                }
            )

        single_scan.assert_not_called()
        cross_scan.assert_not_called()
        neg_risk_scan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
