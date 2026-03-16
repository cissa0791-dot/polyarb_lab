from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.models import RejectionEvent, RunSummary
from src.opportunity.models import CandidateLeg, RankedOpportunity, StrategyFamily
from src.reporting.calibration import ThresholdCalibrationService
from src.reporting.exporter import export_calibration_report
from src.reporting.models import CalibrationParameterSet
from src.runtime.batch import BatchResearchRunner
from src.runtime.campaigns import ResearchCampaignManifest, load_campaign_manifest
from src.runtime.runner import ResearchRunner
from src.storage.event_store import ResearchStore


class FakeRunner:
    def __init__(self):
        self.config = type("Config", (), {"market_data": type("MarketData", (), {"market_limit": 5})()})()
        self.calls = 0

    def run_once(self, experiment_context=None):
        self.calls += 1
        now = datetime(2026, 3, 15, 0, self.calls, tzinfo=timezone.utc)
        return RunSummary(
            run_id=f"run-{self.calls}",
            started_ts=now,
            ended_ts=now + timedelta(seconds=10),
            markets_scanned=5,
            snapshots_stored=10,
            candidates_generated=2,
            risk_accepted=1,
            risk_rejected=1,
            near_miss_candidates=1,
            paper_orders_created=0,
            fills=0,
            partial_fills=0,
            cancellations=0,
            open_positions=0,
            closed_positions=0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            system_errors=0,
            metadata={
                **(experiment_context or {}),
                "raw_candidates_by_family": {"single_market_mispricing": 2},
                "qualified_candidates_by_family": {"single_market_mispricing": 1},
                "near_miss_by_family": {"single_market_mispricing": 1},
                "rejection_reason_counts_by_family": {"single_market_mispricing": {"EDGE_BELOW_THRESHOLD": 1}},
            },
        )


class BatchAndCalibrationTests(unittest.TestCase):
    def test_runner_experiment_metadata_snapshot(self) -> None:
        runner = ResearchRunner()
        runner.set_experiment_context(experiment_id="exp-1", experiment_label="batch-a", parameter_set_label="base")

        metadata = runner._build_experiment_metadata()

        self.assertEqual(metadata["experiment_id"], "exp-1")
        self.assertEqual(metadata["experiment_label"], "batch-a")
        self.assertEqual(metadata["parameter_set_label"], "base")
        self.assertIn("parameter_set", metadata)
        self.assertIn("scan_scope", metadata)

    def test_batch_runner_aggregates_runs_and_propagates_experiment(self) -> None:
        batch = BatchResearchRunner(runner_factory=FakeRunner)

        result = batch.run_batch(cycles=2, sleep_sec=0.0, experiment_label="exp-batch", parameter_set_label="base", market_limit=7)

        self.assertEqual(result.cycles_completed, 2)
        self.assertEqual(result.aggregate_summary.candidates_generated, 4)
        self.assertEqual(result.aggregate_summary.metadata["experiment_label"], "exp-batch")
        self.assertEqual(result.aggregate_summary.metadata["batch_cycles_completed"], 2)
        self.assertEqual(result.metadata["market_limit"], 7)

    def test_campaign_manifest_and_batch_runner_propagate_campaign_metadata(self) -> None:
        manifest = ResearchCampaignManifest(
            campaign_id="campaign-1",
            campaign_label="coverage-campaign",
            purpose="Collect more single-market evidence",
            notes="Focus on base and strict",
            target_strategy_families=["single_market_mispricing"],
            target_parameter_sets=["base", "strict"],
            cycles=1,
            market_limit=9,
            run_cadence_note="manual-daily",
            metadata={"campaign_owner": "local-test"},
        )
        batch = BatchResearchRunner(runner_factory=FakeRunner)

        result = batch.run_campaign(manifest)

        self.assertEqual(result.campaign_id, "campaign-1")
        self.assertEqual(result.campaign_label, "coverage-campaign")
        self.assertEqual(result.cycles_completed, 2)
        self.assertEqual(result.target_parameter_sets, ["base", "strict"])
        self.assertEqual(len(result.batch_summaries), 2)
        self.assertEqual(result.aggregate_summary.metadata["campaign_label"], "coverage-campaign")
        self.assertEqual(result.aggregate_summary.metadata["campaign_owner"], "local-test")
        self.assertEqual(result.batch_summaries[0].campaign_label, "coverage-campaign")
        self.assertEqual(result.batch_summaries[0].metadata["campaign_run_cadence_note"], "manual-daily")

    def test_campaign_manifest_loader_supports_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "campaign.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "campaign_label: smoke-campaign",
                        "target_strategy_families:",
                        "  - single_market_mispricing",
                        "target_parameter_sets:",
                        "  - base",
                        "cycles: 2",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = load_campaign_manifest(manifest_path)

            self.assertEqual(manifest.campaign_label, "smoke-campaign")
            self.assertEqual(manifest.resolved_parameter_sets(), ["base"])
            self.assertEqual(manifest.cycles, 2)

    def test_threshold_calibration_reports_family_yield_and_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "calibration.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

            store.save_run_summary(
                RunSummary(
                    run_id="run-1",
                    started_ts=base,
                    ended_ts=base + timedelta(minutes=1),
                    candidates_generated=2,
                    metadata={
                        "experiment_id": "exp-1",
                        "experiment_label": "family-study",
                        "parameter_set_label": "base",
                        "raw_candidates_by_family": {"single_market_mispricing": 2, "cross_market_constraint": 2},
                    },
                )
            )

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-single",
                    kind="single_market",
                    market_slugs=["market-a"],
                    gross_edge_cents=0.08,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    expected_payout=10.0,
                    target_notional_usd=20.0,
                    estimated_depth_usd=60.0,
                    score=81.0,
                    estimated_net_profit_usd=1.0,
                    available_depth_usd=60.0,
                    required_depth_usd=20.0,
                    partial_fill_risk_score=0.10,
                    non_atomic_execution_risk_score=0.10,
                    execution_mode="paper_eligible",
                    research_only=False,
                    strategy_tag="single_market_mispricing:single_market_sum_under_1",
                    ranking_score=81.0,
                    sizing_hint_usd=20.0,
                    sizing_hint_shares=10.0,
                    legs=[
                        CandidateLeg(token_id="tok-single-yes", market_slug="market-a", action="BUY", side="YES", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.03),
                        CandidateLeg(token_id="tok-single-no", market_slug="market-a", action="BUY", side="NO", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.04),
                    ],
                    metadata={
                        "run_id": "run-1",
                        "experiment_id": "exp-1",
                        "experiment_label": "family-study",
                        "parameter_set_label": "base",
                        "strategy_family": "single_market_mispricing",
                        "qualification": {
                            "expected_net_edge_cents": 0.06,
                            "expected_net_profit_usd": 1.0,
                            "required_depth_usd": 20.0,
                            "available_depth_usd": 60.0,
                            "legs": [
                                {"token_id": "tok-single-yes", "market_slug": "market-a", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.45, "vwap_price": 0.45, "spread_cents": 0.03},
                                {"token_id": "tok-single-no", "market_slug": "market-a", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.45, "vwap_price": 0.45, "spread_cents": 0.04},
                            ],
                        },
                    },
                    ts=base + timedelta(seconds=5),
                )
            )
            store.save_candidate(
                RankedOpportunity(
                    strategy_id="cross_market_leq",
                    strategy_family=StrategyFamily.CROSS_MARKET_CONSTRAINT,
                    candidate_id="cand-cross",
                    kind="cross_market",
                    market_slugs=["market-b", "market-c"],
                    gross_edge_cents=0.20,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    expected_payout=10.0,
                    target_notional_usd=25.0,
                    estimated_depth_usd=80.0,
                    score=88.0,
                    estimated_net_profit_usd=2.0,
                    available_depth_usd=80.0,
                    required_depth_usd=25.0,
                    partial_fill_risk_score=0.20,
                    non_atomic_execution_risk_score=0.25,
                    execution_mode="research_only",
                    research_only=True,
                    strategy_tag="cross_market_constraint:cross_market_leq",
                    ranking_score=88.0,
                    sizing_hint_usd=25.0,
                    sizing_hint_shares=10.0,
                    legs=[
                        CandidateLeg(token_id="tok-cross-left", market_slug="market-b", action="BUY", side="NO", required_shares=10.0, best_price=0.40, vwap_price=0.40, spread_cents=0.05),
                        CandidateLeg(token_id="tok-cross-right", market_slug="market-c", action="BUY", side="YES", required_shares=10.0, best_price=0.40, vwap_price=0.40, spread_cents=0.06),
                    ],
                    metadata={
                        "run_id": "run-1",
                        "experiment_id": "exp-1",
                        "experiment_label": "family-study",
                        "parameter_set_label": "base",
                        "strategy_family": "cross_market_constraint",
                        "qualification": {
                            "expected_net_edge_cents": 0.18,
                            "expected_net_profit_usd": 2.0,
                            "required_depth_usd": 25.0,
                            "available_depth_usd": 80.0,
                            "legs": [
                                {"token_id": "tok-cross-left", "market_slug": "market-b", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.40, "vwap_price": 0.40, "spread_cents": 0.05},
                                {"token_id": "tok-cross-right", "market_slug": "market-c", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.40, "vwap_price": 0.40, "spread_cents": 0.06},
                            ],
                        },
                    },
                    ts=base + timedelta(seconds=10),
                )
            )

            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-1",
                    candidate_id="rej-cross",
                    stage="qualification",
                    reason_code="PARTIAL_FILL_RISK_TOO_HIGH",
                    metadata={
                        "run_id": "run-1",
                        "experiment_id": "exp-1",
                        "experiment_label": "family-study",
                        "parameter_set_label": "base",
                        "strategy_family": "cross_market_constraint",
                        "execution_mode": "research_only",
                        "research_only": True,
                        "market_slugs": ["market-d", "market-e"],
                        "raw_candidate": {
                            "strategy_id": "cross_market_leq",
                            "kind": "cross_market",
                            "market_slugs": ["market-d", "market-e"],
                            "gross_edge_cents": 0.16,
                            "expected_payout": 10.0,
                            "target_notional_usd": 15.0,
                            "legs": [
                                {"token_id": "tok-rej-cross-left", "market_slug": "market-d", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.42},
                                {"token_id": "tok-rej-cross-right", "market_slug": "market-e", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.42},
                            ],
                        },
                        "qualification": {
                            "expected_net_edge_cents": 0.14,
                            "expected_net_profit_usd": 0.80,
                            "required_depth_usd": 15.0,
                            "available_depth_usd": 18.0,
                            "partial_fill_risk_score": 0.62,
                            "non_atomic_execution_risk_score": 0.35,
                            "legs": [
                                {"token_id": "tok-rej-cross-left", "market_slug": "market-d", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.42, "vwap_price": 0.42, "spread_cents": 0.05},
                                {"token_id": "tok-rej-cross-right", "market_slug": "market-e", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.42, "vwap_price": 0.42, "spread_cents": 0.05},
                            ],
                        },
                    },
                    ts=base + timedelta(seconds=20),
                )
            )
            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-1",
                    candidate_id="rej-single",
                    stage="qualification",
                    reason_code="EDGE_BELOW_THRESHOLD",
                    metadata={
                        "run_id": "run-1",
                        "experiment_id": "exp-1",
                        "experiment_label": "family-study",
                        "parameter_set_label": "base",
                        "strategy_family": "single_market_mispricing",
                        "execution_mode": "paper_eligible",
                        "research_only": False,
                        "market_slugs": ["market-f"],
                        "raw_candidate": {
                            "strategy_id": "single_market_sum_under_1",
                            "kind": "single_market",
                            "market_slugs": ["market-f"],
                            "gross_edge_cents": 0.04,
                            "expected_payout": 10.0,
                            "target_notional_usd": 12.0,
                            "legs": [
                                {"token_id": "tok-rej-single-yes", "market_slug": "market-f", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.46},
                                {"token_id": "tok-rej-single-no", "market_slug": "market-f", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.46},
                            ],
                        },
                        "qualification": {
                            "expected_net_edge_cents": 0.045,
                            "expected_net_profit_usd": 0.60,
                            "required_depth_usd": 12.0,
                            "available_depth_usd": 18.0,
                            "partial_fill_risk_score": 0.20,
                            "non_atomic_execution_risk_score": 0.10,
                            "legs": [
                                {"token_id": "tok-rej-single-yes", "market_slug": "market-f", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.46, "vwap_price": 0.46, "spread_cents": 0.02},
                                {"token_id": "tok-rej-single-no", "market_slug": "market-f", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.46, "vwap_price": 0.46, "spread_cents": 0.02},
                            ],
                        },
                    },
                    ts=base + timedelta(seconds=25),
                )
            )

            def save_book_snapshot(token_id: str, bid: float, ask: float, offset_sec: int) -> None:
                store.save_raw_snapshot(
                    "clob",
                    token_id,
                    {
                        "token_id": token_id,
                        "bids": [{"price": bid, "size": 100.0}],
                        "asks": [{"price": ask, "size": 100.0}],
                    },
                    base + timedelta(seconds=offset_sec),
                )

            for offset_sec, prices in (
                (
                    5,
                    {
                        "tok-single-yes": 0.44,
                        "tok-single-no": 0.44,
                    },
                ),
                (
                    10,
                    {
                        "tok-cross-left": 0.39,
                        "tok-cross-right": 0.39,
                    },
                ),
                (
                    20,
                    {
                        "tok-rej-cross-left": 0.41,
                        "tok-rej-cross-right": 0.41,
                    },
                ),
                (
                    25,
                    {
                        "tok-rej-single-yes": 0.45,
                        "tok-rej-single-no": 0.45,
                    },
                ),
            ):
                for token_id, bid in prices.items():
                    save_book_snapshot(token_id, bid=bid, ask=bid + 0.01, offset_sec=offset_sec)

            for offset_sec, prices in (
                (
                    40,
                    {
                        "tok-single-yes": 0.47,
                        "tok-single-no": 0.46,
                        "tok-cross-left": 0.42,
                        "tok-cross-right": 0.41,
                        "tok-rej-cross-left": 0.41,
                        "tok-rej-cross-right": 0.41,
                        "tok-rej-single-yes": 0.47,
                        "tok-rej-single-no": 0.47,
                    },
                ),
                (
                    330,
                    {
                        "tok-single-yes": 0.44,
                        "tok-single-no": 0.44,
                        "tok-cross-left": 0.43,
                        "tok-cross-right": 0.42,
                        "tok-rej-cross-left": 0.43,
                        "tok-rej-cross-right": 0.42,
                        "tok-rej-single-yes": 0.45,
                        "tok-rej-single-no": 0.45,
                    },
                ),
            ):
                for token_id, bid in prices.items():
                    save_book_snapshot(token_id, bid=bid, ask=bid + 0.01, offset_sec=offset_sec)
            store.close()

            service = ThresholdCalibrationService(db_path=db_path)
            report = service.build_report(
                parameter_sets=[
                    CalibrationParameterSet(
                        label="base",
                        min_net_edge_cents=0.05,
                        min_net_profit_usd=0.50,
                        max_spread_cents=0.08,
                        min_depth_ratio=1.0,
                        min_target_notional_usd=5.0,
                        max_partial_fill_risk=0.60,
                        max_non_atomic_risk=0.60,
                    ),
                    CalibrationParameterSet(
                        label="loose",
                        min_net_edge_cents=0.04,
                        min_net_profit_usd=0.30,
                        max_spread_cents=0.10,
                        min_depth_ratio=1.0,
                        min_target_notional_usd=5.0,
                        max_partial_fill_risk=0.70,
                        max_non_atomic_risk=0.70,
                    ),
                ],
                experiment_label="family-study",
            )
            service.close()

            self.assertEqual(report.record_count, 4)
            results = {result.parameter_set_label: result for result in report.parameter_results}
            self.assertEqual(results["base"].qualified_count, 2)
            self.assertEqual(results["base"].rejected_count, 2)
            self.assertEqual(results["loose"].qualified_count, 4)
            self.assertEqual(results["base"].qualified_by_family["cross_market_constraint"], 1)
            self.assertEqual(results["base"].near_miss_by_family["single_market_mispricing"], 1)
            base_family = {item.strategy_family: item for item in results["base"].family_summaries}
            self.assertAlmostEqual(base_family["cross_market_constraint"].conversion_rate or 0.0, 0.5, places=6)
            self.assertEqual(results["base"].rejection_reason_counts["PARTIAL_FILL_RISK_TOO_HIGH"], 1)
            top_ids = [item.candidate_id for item in results["base"].top_ranked_opportunities]
            self.assertIn("cand-cross", top_ids)
            self.assertEqual(results["base"].outcome_horizon_stats[0].labeled_candidates, 2)
            self.assertEqual(results["loose"].outcome_horizon_stats[0].labeled_candidates, 4)
            self.assertGreater(results["base"].outcome_horizon_stats[0].mean_forward_markout_usd or 0.0, 0.0)
            self.assertGreater(results["loose"].outcome_horizon_stats[0].mean_forward_markout_usd or 0.0, 0.0)
            base_outcome_family = {
                (item.group_key, item.horizon_sec): item
                for item in results["base"].outcome_family_scorecards
            }
            self.assertEqual(base_outcome_family[("cross_market_constraint", 30)].labeled_candidates, 1)
            base_rank_buckets = {
                (item.group_key, item.horizon_sec): item
                for item in results["base"].outcome_rank_bucket_scorecards
            }
            self.assertEqual(base_rank_buckets[("score_80_89", 30)].total_candidates, 2)
            self.assertEqual(base_rank_buckets[("score_80_89", 30)].labeled_candidates, 2)
            self.assertIsNotNone(results["base"].shadow_execution_summary)
            self.assertEqual(results["base"].shadow_execution_summary.total_candidates, 2)
            self.assertEqual(results["base"].shadow_execution_summary.data_sufficient_count, 2)
            self.assertEqual(results["base"].shadow_execution_summary.viable_count, 2)
            self.assertEqual(results["loose"].shadow_execution_summary.total_candidates, 4)
            self.assertEqual(results["loose"].shadow_execution_summary.data_sufficient_count, 4)
            base_shadow_family = {
                item.group_key: item
                for item in results["base"].shadow_execution_family_scorecards
            }
            self.assertEqual(base_shadow_family["cross_market_constraint"].viable_count, 1)
            loose_shadow_rank = {
                item.group_key: item
                for item in results["loose"].shadow_execution_rank_bucket_scorecards
            }
            self.assertEqual(loose_shadow_rank["score_80_89"].total_candidates, 2)

            out_dir = Path(tmp_dir) / "reports"
            files = export_calibration_report(report, out_dir=out_dir)
            self.assertTrue(files["full_json"].exists())
            self.assertTrue(files["parameter_results_csv"].exists())
            self.assertTrue(files["family_summaries_csv"].exists())
            self.assertTrue(files["outcome_horizons_csv"].exists())
            self.assertTrue(files["outcome_families_csv"].exists())
            self.assertTrue(files["shadow_summary_csv"].exists())
            self.assertTrue(files["shadow_family_csv"].exists())

    def test_sparse_calibration_is_backward_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "empty.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            store.close()

            service = ThresholdCalibrationService(db_path=db_path)
            report = service.build_report(parameter_sets=[CalibrationParameterSet(label="base", min_net_edge_cents=0.05)])
            service.close()

            self.assertEqual(report.record_count, 0)
            self.assertEqual(report.parameter_results[0].qualified_count, 0)
            self.assertEqual(report.parameter_results[0].family_summaries, [])


if __name__ == "__main__":
    unittest.main()
