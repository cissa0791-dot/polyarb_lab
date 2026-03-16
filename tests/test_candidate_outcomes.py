from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.models import RejectionEvent, RunSummary
from src.opportunity.models import CandidateLeg, RankedOpportunity, StrategyFamily
from src.reporting.analytics import OfflineAnalyticsService
from src.storage.event_store import ResearchStore


def _book(token_id: str, bid: float, ask: float, size: float = 100.0) -> dict:
    return {
        "token_id": token_id,
        "bids": [{"price": bid, "size": size}],
        "asks": [{"price": ask, "size": size}],
    }


class CandidateOutcomeAnalyticsTests(unittest.TestCase):
    def test_candidate_outcome_labeling_and_scorecards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "outcomes.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

            store.save_run_summary(
                RunSummary(
                    run_id="run-1",
                    started_ts=base,
                    ended_ts=base + timedelta(minutes=20),
                    candidates_generated=3,
                    metadata={
                        "experiment_label": "exp-a",
                        "raw_candidates_by_family": {
                            "single_market_mispricing": 2,
                            "cross_market_constraint": 1,
                        },
                        "qualified_candidates_by_family": {
                            "single_market_mispricing": 1,
                            "cross_market_constraint": 1,
                        },
                        "near_miss_by_family": {"single_market_mispricing": 1},
                    },
                )
            )

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-top",
                    kind="single_market",
                    market_slugs=["market-a"],
                    gross_edge_cents=0.10,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
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
                    legs=[
                        CandidateLeg(token_id="tok-a-yes", market_slug="market-a", action="BUY", side="YES", required_shares=10.0, best_price=0.45, vwap_price=0.45),
                        CandidateLeg(token_id="tok-a-no", market_slug="market-a", action="BUY", side="NO", required_shares=10.0, best_price=0.45, vwap_price=0.45),
                    ],
                    metadata={
                        "experiment_label": "exp-a",
                        "parameter_set_label": "base",
                        "strategy_family": "single_market_mispricing",
                        "qualification": {"expected_net_edge_cents": 0.08},
                    },
                    ts=base,
                )
            )

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="cross_market_leq",
                    strategy_family=StrategyFamily.CROSS_MARKET_CONSTRAINT,
                    candidate_id="cand-cross",
                    kind="cross_market",
                    market_slugs=["market-b", "market-c"],
                    gross_edge_cents=0.16,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    target_notional_usd=4.0,
                    estimated_depth_usd=30.0,
                    score=78.0,
                    estimated_net_profit_usd=0.4,
                    execution_mode="research_only",
                    research_only=True,
                    strategy_tag="cross_market_constraint:cross_market_leq",
                    ranking_score=78.0,
                    sizing_hint_usd=4.0,
                    sizing_hint_shares=5.0,
                    legs=[
                        CandidateLeg(token_id="tok-b", market_slug="market-b", action="BUY", side="NO", required_shares=5.0, best_price=0.40, vwap_price=0.40),
                        CandidateLeg(token_id="tok-c", market_slug="market-c", action="BUY", side="YES", required_shares=5.0, best_price=0.40, vwap_price=0.40),
                    ],
                    metadata={
                        "experiment_label": "exp-a",
                        "parameter_set_label": "strict",
                        "strategy_family": "cross_market_constraint",
                        "qualification": {"expected_net_edge_cents": 0.14},
                    },
                    ts=base + timedelta(seconds=5),
                )
            )

            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-1",
                    candidate_id="cand-near",
                    stage="qualification",
                    reason_code="EDGE_BELOW_THRESHOLD",
                    metadata={
                        "experiment_label": "exp-a",
                        "parameter_set_label": "base",
                        "strategy_family": "single_market_mispricing",
                        "market_slugs": ["market-d"],
                        "raw_candidate": {
                            "strategy_id": "single_market_sum_under_1",
                            "kind": "single_market",
                            "market_slugs": ["market-d"],
                            "gross_edge_cents": 0.04,
                            "legs": [
                                {"token_id": "tok-d-yes", "market_slug": "market-d", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.46},
                                {"token_id": "tok-d-no", "market_slug": "market-d", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.46},
                            ],
                        },
                        "qualification": {
                            "expected_net_edge_cents": 0.045,
                            "legs": [
                                {"token_id": "tok-d-yes", "market_slug": "market-d", "action": "BUY", "side": "YES", "required_shares": 10.0, "best_price": 0.46, "vwap_price": 0.46},
                                {"token_id": "tok-d-no", "market_slug": "market-d", "action": "BUY", "side": "NO", "required_shares": 10.0, "best_price": 0.46, "vwap_price": 0.46},
                            ],
                        },
                    },
                    ts=base + timedelta(seconds=10),
                )
            )

            for ts_offset, bids in (
                (40, {"tok-a-yes": 0.47, "tok-a-no": 0.46, "tok-b": 0.39, "tok-c": 0.38, "tok-d-yes": 0.47, "tok-d-no": 0.47}),
                (70, {"tok-a-yes": 0.44, "tok-a-no": 0.44, "tok-b": 0.41, "tok-c": 0.40, "tok-d-yes": 0.45, "tok-d-no": 0.45}),
                (330, {"tok-a-yes": 0.49, "tok-a-no": 0.48, "tok-b": 0.42, "tok-c": 0.41, "tok-d-yes": 0.48, "tok-d-no": 0.47}),
            ):
                ts = base + timedelta(seconds=ts_offset)
                for token_id, bid in bids.items():
                    store.save_raw_snapshot("clob", token_id, _book(token_id, bid=bid, ask=bid + 0.01), ts)

            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            observations = [obs for obs in report.candidate_outcome_observations if obs.horizon_sec == 30]
            self.assertEqual(len(observations), 3)

            by_candidate = {obs.candidate_id: obs for obs in observations}
            self.assertAlmostEqual(by_candidate["cand-top"].selected_mark_age_sec, 40.0, places=6)
            self.assertAlmostEqual(by_candidate["cand-top"].forward_markout_usd, 0.3, places=6)
            self.assertEqual(by_candidate["cand-top"].rank_bucket, "score_90_plus")
            self.assertTrue(by_candidate["cand-top"].positive_outcome)
            self.assertEqual(by_candidate["cand-near"].source, "near_miss")
            self.assertTrue(by_candidate["cand-near"].positive_outcome)
            self.assertFalse(by_candidate["cand-cross"].positive_outcome)

            sixty_second = next(
                obs for obs in report.candidate_outcome_observations
                if obs.candidate_id == "cand-top" and obs.horizon_sec == 60
            )
            self.assertAlmostEqual(sixty_second.selected_mark_age_sec, 70.0, places=6)
            self.assertAlmostEqual(sixty_second.forward_markout_usd, -0.2, places=6)

            overall_30 = next(stat for stat in report.candidate_outcome_horizon_stats if stat.horizon_sec == 30)
            self.assertEqual(overall_30.total_candidates, 3)
            self.assertEqual(overall_30.labeled_candidates, 3)
            self.assertAlmostEqual(overall_30.positive_ratio or 0.0, 2 / 3, places=6)

            family_cards = {
                (card.group_key, card.horizon_sec): card
                for card in report.family_outcome_scorecards
            }
            self.assertEqual(family_cards[("single_market_mispricing", 30)].total_candidates, 2)
            self.assertEqual(family_cards[("single_market_mispricing", 30)].labeled_candidates, 2)
            self.assertAlmostEqual(family_cards[("single_market_mispricing", 30)].positive_ratio or 0.0, 1.0, places=6)

            rank_cards = {
                (card.group_key, card.horizon_sec): card
                for card in report.rank_bucket_outcome_scorecards
            }
            self.assertEqual(rank_cards[("score_90_plus", 30)].labeled_candidates, 1)
            self.assertAlmostEqual(rank_cards[("score_90_plus", 30)].mean_forward_markout_usd or 0.0, 0.3, places=6)

            parameter_cards = {
                (card.group_key, card.horizon_sec): card
                for card in report.parameter_set_outcome_scorecards
            }
            self.assertEqual(parameter_cards[("base", 30)].total_candidates, 2)
            self.assertEqual(parameter_cards[("strict", 30)].total_candidates, 1)

            source_cards = {
                (card.group_key, card.horizon_sec): card
                for card in report.candidate_source_outcome_scorecards
            }
            self.assertEqual(source_cards[("near_miss", 30)].total_candidates, 1)
            self.assertEqual(source_cards[("qualified", 30)].total_candidates, 2)

    def test_candidate_outcomes_are_backward_safe_with_sparse_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "sparse.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-old",
                    kind="single_market",
                    market_slugs=["market-old"],
                    gross_edge_cents=0.05,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    target_notional_usd=10.0,
                    estimated_depth_usd=10.0,
                    score=75.0,
                    estimated_net_profit_usd=0.4,
                    execution_mode="paper_eligible",
                    research_only=False,
                    strategy_tag="single_market_mispricing:single_market_sum_under_1",
                    ranking_score=75.0,
                    sizing_hint_usd=10.0,
                    sizing_hint_shares=10.0,
                    ts=base,
                )
            )
            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            self.assertEqual(report.candidate_outcome_observations, [])
            horizon_30 = next(stat for stat in report.candidate_outcome_horizon_stats if stat.horizon_sec == 30)
            self.assertEqual(horizon_30.total_candidates, 1)
            self.assertEqual(horizon_30.labeled_candidates, 0)


if __name__ == "__main__":
    unittest.main()
