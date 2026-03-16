from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.config_runtime.loader import load_runtime_config
from src.domain.models import AccountSnapshot, OpportunityCandidate
from src.risk.manager import RiskManager


class RuntimeFoundationTests(unittest.TestCase):
    def test_load_runtime_config_supports_legacy_flat_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        'gamma_host: "https://gamma-api.polymarket.com"',
                        'clob_host: "https://clob.polymarket.com"',
                        "market_limit: 99",
                        "min_depth_multiple: 2.5",
                        "starting_cash: 1234.5",
                        'sqlite_url: "sqlite:///data/test.db"',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_runtime_config(settings_path)

        self.assertEqual(config.market_data.market_limit, 99)
        self.assertEqual(config.opportunity.min_depth_multiple, 2.5)
        self.assertEqual(config.paper.starting_cash, 1234.5)
        self.assertEqual(config.storage.sqlite_url, "sqlite:///data/test.db")

    def test_risk_manager_blocks_low_score_candidates(self) -> None:
        config = load_runtime_config("config/settings.yaml")
        risk_manager = RiskManager(config.risk, config.opportunity, config.execution)

        candidate = OpportunityCandidate(
            strategy_id="test",
            candidate_id="cand-1",
            kind="single_market",
            market_slugs=["example-market"],
            gross_edge_cents=0.02,
            fee_estimate_cents=0.01,
            slippage_estimate_cents=0.01,
            target_notional_usd=10.0,
            estimated_depth_usd=50.0,
            score=10.0,
            estimated_net_profit_usd=0.01,
            ts=datetime.now(timezone.utc),
        )
        account = AccountSnapshot(cash=1000.0, ts=datetime.now(timezone.utc))

        decision = risk_manager.evaluate(candidate, account)

        self.assertIn(decision.status.value, {"blocked", "halted"})
        self.assertIn("EDGE_BELOW_THRESHOLD", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
