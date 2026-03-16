from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.models import RunSummary
from src.opportunity.models import CandidateLeg, RankedOpportunity, StrategyFamily
from src.reporting.analytics import OfflineAnalyticsService
from src.storage.event_store import ResearchStore


def _book_payload(bids: list[tuple[float, float]], asks: list[tuple[float, float]], token_id: str) -> dict:
    return {
        "token_id": token_id,
        "bids": [{"price": price, "size": size} for price, size in bids],
        "asks": [{"price": price, "size": size} for price, size in asks],
    }


class ShadowExecutionAnalyticsTests(unittest.TestCase):
    def test_shadow_execution_scorecards_and_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "shadow.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

            store.save_run_summary(
                RunSummary(
                    run_id="run-shadow",
                    started_ts=base,
                    ended_ts=base + timedelta(minutes=20),
                    candidates_generated=3,
                    metadata={
                        "experiment_label": "shadow-exp",
                        "raw_candidates_by_family": {
                            "single_market_mispricing": 2,
                            "cross_market_constraint": 1,
                        },
                        "qualified_candidates_by_family": {
                            "single_market_mispricing": 2,
                            "cross_market_constraint": 1,
                        },
                        "research_only_candidates_by_family": {"cross_market_constraint": 1},
                    },
                )
            )

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-fill",
                    kind="single_market",
                    market_slugs=["market-a"],
                    gross_edge_cents=0.10,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    expected_payout=10.0,
                    target_notional_usd=9.0,
                    estimated_depth_usd=40.0,
                    score=92.0,
                    estimated_net_profit_usd=0.8,
                    execution_mode="paper_eligible",
                    research_only=False,
                    strategy_tag="single_market_mispricing:single_market_sum_under_1",
                    ranking_score=92.0,
                    sizing_hint_usd=9.0,
                    sizing_hint_shares=10.0,
                    required_shares=10.0,
                    partial_fill_risk_score=0.10,
                    non_atomic_execution_risk_score=0.10,
                    legs=[
                        CandidateLeg(token_id="tok-fill-yes", market_slug="market-a", action="BUY", side="YES", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.02),
                        CandidateLeg(token_id="tok-fill-no", market_slug="market-a", action="BUY", side="NO", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.02),
                    ],
                    metadata={
                        "experiment_label": "shadow-exp",
                        "parameter_set_label": "base",
                        "strategy_family": "single_market_mispricing",
                        "qualification": {"expected_net_edge_cents": 0.08},
                    },
                    ts=base,
                )
            )
            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-partial",
                    kind="single_market",
                    market_slugs=["market-b"],
                    gross_edge_cents=0.11,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    expected_payout=10.0,
                    target_notional_usd=9.0,
                    estimated_depth_usd=20.0,
                    score=82.0,
                    estimated_net_profit_usd=0.6,
                    execution_mode="paper_eligible",
                    research_only=False,
                    strategy_tag="single_market_mispricing:single_market_sum_under_1",
                    ranking_score=82.0,
                    sizing_hint_usd=9.0,
                    sizing_hint_shares=10.0,
                    required_shares=10.0,
                    partial_fill_risk_score=0.10,
                    non_atomic_execution_risk_score=0.10,
                    legs=[
                        CandidateLeg(token_id="tok-partial-yes", market_slug="market-b", action="BUY", side="YES", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.02),
                        CandidateLeg(token_id="tok-partial-no", market_slug="market-b", action="BUY", side="NO", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.02),
                    ],
                    metadata={
                        "experiment_label": "shadow-exp",
                        "parameter_set_label": "strict",
                        "strategy_family": "single_market_mispricing",
                        "qualification": {"expected_net_edge_cents": 0.09},
                    },
                    ts=base + timedelta(seconds=1),
                )
            )
            store.save_candidate(
                RankedOpportunity(
                    strategy_id="cross_market_leq",
                    strategy_family=StrategyFamily.CROSS_MARKET_CONSTRAINT,
                    candidate_id="cand-cross",
                    kind="cross_market",
                    market_slugs=["market-c", "market-d"],
                    gross_edge_cents=0.15,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    expected_payout=10.0,
                    target_notional_usd=8.8,
                    estimated_depth_usd=30.0,
                    score=88.0,
                    estimated_net_profit_usd=0.5,
                    execution_mode="research_only",
                    research_only=True,
                    strategy_tag="cross_market_constraint:cross_market_leq",
                    ranking_score=88.0,
                    sizing_hint_usd=8.8,
                    sizing_hint_shares=10.0,
                    required_shares=10.0,
                    partial_fill_risk_score=0.10,
                    non_atomic_execution_risk_score=0.10,
                    legs=[
                        CandidateLeg(token_id="tok-cross-left", market_slug="market-c", action="BUY", side="NO", required_shares=10.0, best_price=0.44, vwap_price=0.44, spread_cents=0.02),
                        CandidateLeg(token_id="tok-cross-right", market_slug="market-d", action="BUY", side="YES", required_shares=10.0, best_price=0.44, vwap_price=0.44, spread_cents=0.02),
                    ],
                    metadata={
                        "experiment_label": "shadow-exp",
                        "parameter_set_label": "base",
                        "strategy_family": "cross_market_constraint",
                        "qualification": {"expected_net_edge_cents": 0.13},
                    },
                    ts=base + timedelta(seconds=2),
                )
            )

            # Entry snapshots around candidate detection.
            store.save_raw_snapshot(
                "clob",
                "tok-fill-yes",
                _book_payload(bids=[(0.44, 20.0)], asks=[(0.45, 5.0), (0.46, 10.0)], token_id="tok-fill-yes"),
                base,
            )
            store.save_raw_snapshot(
                "clob",
                "tok-fill-no",
                _book_payload(bids=[(0.44, 20.0)], asks=[(0.45, 10.0)], token_id="tok-fill-no"),
                base,
            )
            store.save_raw_snapshot(
                "clob",
                "tok-partial-yes",
                _book_payload(bids=[(0.44, 20.0)], asks=[(0.45, 6.0)], token_id="tok-partial-yes"),
                base + timedelta(seconds=1),
            )
            store.save_raw_snapshot(
                "clob",
                "tok-partial-no",
                _book_payload(bids=[(0.44, 20.0)], asks=[(0.45, 10.0)], token_id="tok-partial-no"),
                base + timedelta(seconds=1),
            )
            store.save_raw_snapshot(
                "clob",
                "tok-cross-left",
                _book_payload(bids=[(0.43, 20.0)], asks=[(0.44, 10.0)], token_id="tok-cross-left"),
                base + timedelta(seconds=2),
            )
            store.save_raw_snapshot(
                "clob",
                "tok-cross-right",
                _book_payload(bids=[(0.43, 20.0)], asks=[(0.44, 10.0)], token_id="tok-cross-right"),
                base + timedelta(seconds=6),
            )

            # Future snapshots for forward outcome labeling.
            for offset_sec, bids in (
                (70, {"tok-fill-yes": 0.48, "tok-fill-no": 0.47, "tok-partial-yes": 0.47, "tok-partial-no": 0.46, "tok-cross-left": 0.46, "tok-cross-right": 0.46}),
                (330, {"tok-fill-yes": 0.49, "tok-fill-no": 0.48, "tok-partial-yes": 0.48, "tok-partial-no": 0.47, "tok-cross-left": 0.47, "tok-cross-right": 0.47}),
            ):
                ts = base + timedelta(seconds=offset_sec)
                for token_id, bid in bids.items():
                    store.save_raw_snapshot(
                        "clob",
                        token_id,
                        _book_payload(bids=[(bid, 20.0)], asks=[(bid + 0.01, 20.0)], token_id=token_id),
                        ts,
                    )

            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            shadow_by_candidate = {item.candidate_id: item for item in report.shadow_execution_observations}
            self.assertEqual(shadow_by_candidate["cand-fill"].shadow_status, "FILLABLE")
            self.assertTrue(shadow_by_candidate["cand-fill"].execution_viable)
            self.assertTrue(shadow_by_candidate["cand-fill"].full_size_fillable)
            self.assertAlmostEqual(shadow_by_candidate["cand-fill"].fillability_ratio or 0.0, 1.0, places=6)
            self.assertAlmostEqual(shadow_by_candidate["cand-fill"].estimated_entry_bundle_vwap or 0.0, 0.905, places=6)
            self.assertAlmostEqual(shadow_by_candidate["cand-fill"].execution_adjusted_net_edge_cents or 0.0, 0.085, places=6)
            self.assertAlmostEqual(shadow_by_candidate["cand-fill"].execution_gap_cents or 0.0, -0.005, places=6)

            self.assertEqual(shadow_by_candidate["cand-partial"].shadow_status, "PARTIALLY_FILLABLE")
            self.assertTrue(shadow_by_candidate["cand-partial"].execution_viable)
            self.assertAlmostEqual(shadow_by_candidate["cand-partial"].fillability_ratio or 0.0, 0.6, places=6)
            self.assertAlmostEqual(shadow_by_candidate["cand-partial"].executable_bundle_shares or 0.0, 6.0, places=6)

            self.assertEqual(shadow_by_candidate["cand-cross"].shadow_status, "INSUFFICIENT_SYNCHRONIZED_DEPTH")
            self.assertFalse(shadow_by_candidate["cand-cross"].data_sufficient)
            self.assertFalse(shadow_by_candidate["cand-cross"].execution_viable)

            family_shadow = {
                card.group_key: card
                for card in report.family_shadow_execution_scorecards
            }
            self.assertEqual(family_shadow["single_market_mispricing"].total_candidates, 2)
            self.assertEqual(family_shadow["single_market_mispricing"].data_sufficient_count, 2)
            self.assertEqual(family_shadow["single_market_mispricing"].viable_count, 2)
            self.assertAlmostEqual(family_shadow["single_market_mispricing"].fillability_rate or 0.0, 0.5, places=6)
            self.assertAlmostEqual(family_shadow["single_market_mispricing"].viability_rate or 0.0, 1.0, places=6)
            self.assertEqual(family_shadow["cross_market_constraint"].unresolved_count, 1)

            rank_shadow = {
                card.group_key: card
                for card in report.rank_bucket_shadow_execution_scorecards
            }
            self.assertEqual(rank_shadow["score_90_plus"].viable_count, 1)
            self.assertEqual(rank_shadow["score_80_89"].total_candidates, 2)

            parameter_shadow = {
                card.group_key: card
                for card in report.parameter_set_shadow_execution_scorecards
            }
            self.assertEqual(parameter_shadow["base"].total_candidates, 2)
            self.assertEqual(parameter_shadow["strict"].total_candidates, 1)

            readiness = {
                card.strategy_family: card
                for card in report.strategy_family_live_readiness
            }
            self.assertEqual(
                readiness["single_market_mispricing"].recommendation_bucket,
                "candidate_for_future_tiny_live_preparation",
            )
            self.assertEqual(readiness["cross_market_constraint"].recommendation_bucket, "not_ready")

    def test_shadow_execution_handles_missing_entry_data_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "shadow_sparse.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-missing",
                    kind="single_market",
                    market_slugs=["market-x"],
                    gross_edge_cents=0.08,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    expected_payout=10.0,
                    target_notional_usd=9.0,
                    estimated_depth_usd=20.0,
                    score=75.0,
                    estimated_net_profit_usd=0.5,
                    execution_mode="paper_eligible",
                    research_only=False,
                    strategy_tag="single_market_mispricing:single_market_sum_under_1",
                    ranking_score=75.0,
                    sizing_hint_usd=9.0,
                    sizing_hint_shares=10.0,
                    required_shares=10.0,
                    legs=[
                        CandidateLeg(token_id="tok-missing-yes", market_slug="market-x", action="BUY", side="YES", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.02),
                        CandidateLeg(token_id="tok-missing-no", market_slug="market-x", action="BUY", side="NO", required_shares=10.0, best_price=0.45, vwap_price=0.45, spread_cents=0.02),
                    ],
                    ts=base,
                )
            )
            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            self.assertEqual(len(report.shadow_execution_observations), 1)
            observation = report.shadow_execution_observations[0]
            self.assertEqual(observation.shadow_status, "UNRESOLVED_MISSING_DATA")
            self.assertFalse(observation.data_sufficient)
            family_shadow = report.family_shadow_execution_scorecards[0]
            self.assertEqual(family_shadow.data_sufficient_count, 0)
            self.assertEqual(family_shadow.unresolved_count, 1)


if __name__ == "__main__":
    unittest.main()
