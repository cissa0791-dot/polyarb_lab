from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.reporting.models import (
    CandidateOutcomeObservation,
    LiveReadinessScorecard,
    ShadowExecutionObservation,
)
from src.reporting.outcomes import OutcomeEvaluationCandidate
from src.reporting.promotion import (
    build_promotion_gate_reports,
    build_sample_sufficiency_scorecards,
    build_stability_scorecards,
)


def _rank_bucket(score: float | None) -> str:
    if score is None:
        return "unranked"
    if score >= 90.0:
        return "score_90_plus"
    if score >= 80.0:
        return "score_80_89"
    if score >= 70.0:
        return "score_70_79"
    return "score_below_70"


def _record(
    *,
    candidate_id: str,
    family: str,
    score: float,
    run_id: str,
    experiment: str | None,
    parameter_set: str | None,
    ts: datetime,
) -> OutcomeEvaluationCandidate:
    return OutcomeEvaluationCandidate(
        candidate_id=candidate_id,
        source="qualified",
        strategy_family=family,
        strategy_id=f"{family}_strategy",
        kind="single_market" if family == "single_market_mispricing" else "cross_market",
        market_slugs=[f"market-{candidate_id}"],
        run_id=run_id,
        legs=[
            {
                "token_id": f"tok-{candidate_id}",
                "market_slug": f"market-{candidate_id}",
                "action": "BUY",
                "side": "YES",
                "required_shares": 10.0,
                "best_price": 0.45,
                "vwap_price": 0.45,
            }
        ],
        ts=ts,
        ranking_score=score,
        gross_edge_cents=0.10,
        net_edge_cents=0.08,
        expected_payout=10.0,
        fee_estimate_cents=0.01,
        latency_penalty_cents=0.005,
        partial_fill_risk_score=0.10,
        non_atomic_execution_risk_score=0.10,
        execution_mode="paper_eligible",
        research_only=False,
        experiment_id=None,
        experiment_label=experiment,
        parameter_set_label=parameter_set,
        metadata={},
    )


def _outcome(
    *,
    record: OutcomeEvaluationCandidate,
    markout: float,
    horizon_sec: int = 60,
) -> CandidateOutcomeObservation:
    return CandidateOutcomeObservation(
        candidate_id=record.candidate_id,
        source=record.source,
        strategy_family=record.strategy_family,
        strategy_id=record.strategy_id,
        kind=record.kind,
        market_slugs=list(record.market_slugs),
        rank_bucket=_rank_bucket(record.ranking_score),
        execution_mode=record.execution_mode,
        research_only=record.research_only,
        experiment_id=record.experiment_id,
        experiment_label=record.experiment_label,
        parameter_set_label=record.parameter_set_label,
        horizon_label="1m",
        horizon_sec=horizon_sec,
        selected_mark_age_sec=float(horizon_sec),
        selected_mark_ts=record.ts + timedelta(seconds=horizon_sec),
        forward_markout_usd=markout,
        return_pct=(markout / 4.5) * 100.0,
        positive_outcome=markout > 0.0,
        ranking_score=record.ranking_score,
        gross_edge_cents=record.gross_edge_cents,
        net_edge_cents=record.net_edge_cents,
        metadata={},
    )


def _shadow(
    *,
    record: OutcomeEvaluationCandidate,
    viable: bool,
    full_size_fillable: bool,
    fillability_ratio: float,
    execution_gap_cents: float,
    data_sufficient: bool = True,
) -> ShadowExecutionObservation:
    return ShadowExecutionObservation(
        candidate_id=record.candidate_id,
        source=record.source,
        strategy_family=record.strategy_family,
        strategy_id=record.strategy_id,
        kind=record.kind,
        market_slugs=list(record.market_slugs),
        rank_bucket=_rank_bucket(record.ranking_score),
        execution_mode=record.execution_mode,
        research_only=record.research_only,
        experiment_id=record.experiment_id,
        experiment_label=record.experiment_label,
        parameter_set_label=record.parameter_set_label,
        shadow_status="FILLABLE" if full_size_fillable else "PARTIALLY_FILLABLE",
        execution_viable=viable,
        data_sufficient=data_sufficient,
        full_size_fillable=full_size_fillable,
        theoretical_gross_edge_cents=record.gross_edge_cents,
        theoretical_net_edge_cents=record.net_edge_cents,
        execution_adjusted_gross_edge_cents=0.09,
        execution_adjusted_net_edge_cents=0.07,
        execution_gap_cents=execution_gap_cents,
        required_bundle_shares=10.0,
        executable_bundle_shares=10.0 * fillability_ratio,
        fillability_ratio=fillability_ratio,
        required_entry_notional_usd=9.0,
        executable_entry_notional_usd=9.0 * fillability_ratio,
        estimated_entry_bundle_vwap=0.90,
        expected_slippage_cost_usd=0.02,
        partial_fill_risk_score=0.10,
        non_atomic_fragility_score=0.10,
        ranking_score=record.ranking_score,
        metadata={},
    )


class PromotionGateTests(unittest.TestCase):
    def test_sample_sufficiency_stability_and_promotion_buckets(self) -> None:
        base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)
        records: list[OutcomeEvaluationCandidate] = []
        outcomes: list[CandidateOutcomeObservation] = []
        shadows: list[ShadowExecutionObservation] = []

        # Stable family: enough samples, consistent across parameter sets / experiments / rank buckets.
        stable_marks = [0.30, 0.20, -0.05, 0.25, 0.22, -0.03, 0.28, 0.18, -0.02, 0.24, 0.21, -0.04]
        stable_viable = [True, True, False, True, True, True, True, False, True, True, True, True]
        stable_fillable = [True, True, False, True, True, False, True, True, True, False, True, True]
        stable_params = ["base", "strict"] * 6
        stable_experiments = ["exp-a"] * 6 + ["exp-b"] * 6
        stable_runs = ["run-a"] * 4 + ["run-b"] * 4 + ["run-c"] * 4
        stable_scores = [92.0] * 6 + [84.0] * 6
        stable_days = [0] * 6 + [1] * 6
        for index in range(12):
            record = _record(
                candidate_id=f"stable-{index}",
                family="single_market_mispricing",
                score=stable_scores[index],
                run_id=stable_runs[index],
                experiment=stable_experiments[index],
                parameter_set=stable_params[index],
                ts=base + timedelta(days=stable_days[index], minutes=index),
            )
            records.append(record)
            outcomes.append(_outcome(record=record, markout=stable_marks[index]))
            shadows.append(
                _shadow(
                    record=record,
                    viable=stable_viable[index],
                    full_size_fillable=stable_fillable[index],
                    fillability_ratio=1.0 if stable_fillable[index] else 0.7,
                    execution_gap_cents=0.03 if index % 2 == 0 else 0.035,
                )
            )

        # Sparse family: good-looking but not enough evidence.
        for index in range(2):
            record = _record(
                candidate_id=f"sparse-{index}",
                family="cross_market_constraint",
                score=88.0,
                run_id="run-sparse",
                experiment=None,
                parameter_set="base",
                ts=base + timedelta(minutes=100 + index),
            )
            records.append(record)
            outcomes.append(_outcome(record=record, markout=0.25))
            shadows.append(
                _shadow(
                    record=record,
                    viable=True,
                    full_size_fillable=True,
                    fillability_ratio=1.0,
                    execution_gap_cents=0.02,
                )
            )

        # Unstable family: weak forward quality and large execution gap.
        unstable_marks = [0.28, 0.22, 0.24, -0.20, -0.35, -0.30, -0.32, -0.27]
        unstable_params = ["base"] * 4 + ["strict"] * 4
        unstable_runs = ["run-u1"] * 3 + ["run-u2"] * 3 + ["run-u3"] * 2
        for index in range(8):
            record = _record(
                candidate_id=f"unstable-{index}",
                family="rebalancing",
                score=78.0,
                run_id=unstable_runs[index],
                experiment="exp-unstable",
                parameter_set=unstable_params[index],
                ts=base + timedelta(days=index // 4, minutes=200 + index),
            )
            records.append(record)
            outcomes.append(_outcome(record=record, markout=unstable_marks[index]))
            shadows.append(
                _shadow(
                    record=record,
                    viable=unstable_params[index] == "base",
                    full_size_fillable=unstable_params[index] == "base",
                    fillability_ratio=0.8 if unstable_params[index] == "base" else 0.5,
                    execution_gap_cents=0.10 if unstable_params[index] == "strict" else 0.09,
                )
            )

        sufficiency_cards = build_sample_sufficiency_scorecards(
            records,
            outcomes,
            shadows,
            group_type="strategy_family",
            record_group_getter=lambda record: record.strategy_family,
        )
        stability_cards = build_stability_scorecards(
            records,
            outcomes,
            shadows,
            group_type="strategy_family",
            record_group_getter=lambda record: record.strategy_family,
        )
        readiness_cards = [
            LiveReadinessScorecard(
                strategy_family="single_market_mispricing",
                reference_horizon_label="1m",
                reference_horizon_sec=60,
                total_candidates=12,
                outcome_labeled_count=12,
                positive_outcome_ratio=8 / 12,
                mean_forward_markout_usd=0.145,
                shadow_data_sufficient_count=12,
                shadow_data_sufficiency_rate=1.0,
                shadow_viable_count=10,
                shadow_viability_rate=10 / 12,
                shadow_fillability_rate=8 / 12,
                mean_execution_adjusted_net_edge_cents=0.07,
                mean_execution_gap_cents=0.0325,
                forward_quality_score=0.9,
                fillability_score=0.83,
                execution_gap_score=0.85,
                data_sufficiency_score=1.0,
                overall_score=0.89,
                recommendation_bucket="candidate_for_future_tiny_live_preparation",
            ),
            LiveReadinessScorecard(
                strategy_family="cross_market_constraint",
                reference_horizon_label="1m",
                reference_horizon_sec=60,
                total_candidates=2,
                outcome_labeled_count=2,
                positive_outcome_ratio=1.0,
                shadow_data_sufficient_count=2,
                shadow_data_sufficiency_rate=1.0,
                shadow_viable_count=2,
                shadow_viability_rate=1.0,
                shadow_fillability_rate=1.0,
                mean_execution_adjusted_net_edge_cents=0.08,
                mean_execution_gap_cents=0.02,
                overall_score=0.92,
                recommendation_bucket="candidate_for_future_tiny_live_preparation",
            ),
            LiveReadinessScorecard(
                strategy_family="rebalancing",
                reference_horizon_label="1m",
                reference_horizon_sec=60,
                total_candidates=8,
                outcome_labeled_count=8,
                positive_outcome_ratio=0.5,
                shadow_data_sufficient_count=8,
                shadow_data_sufficiency_rate=1.0,
                shadow_viable_count=4,
                shadow_viability_rate=0.5,
                shadow_fillability_rate=0.625,
                mean_execution_adjusted_net_edge_cents=0.03,
                mean_execution_gap_cents=0.095,
                overall_score=0.35,
                recommendation_bucket="not_ready",
            ),
        ]

        sufficiency_by_family = {card.group_key: card for card in sufficiency_cards}
        self.assertEqual(sufficiency_by_family["single_market_mispricing"].sufficiency_bucket, "moderate_data")
        self.assertEqual(sufficiency_by_family["cross_market_constraint"].sufficiency_bucket, "insufficient_data")

        stability_by_family_and_slice = {
            (card.group_key, card.slice_dimension): card
            for card in stability_cards
        }
        self.assertEqual(
            stability_by_family_and_slice[("single_market_mispricing", "parameter_set")].stability_bucket,
            "stable",
        )
        self.assertEqual(
            stability_by_family_and_slice[("cross_market_constraint", "parameter_set")].stability_bucket,
            "insufficient_slices",
        )
        self.assertEqual(
            stability_by_family_and_slice[("rebalancing", "parameter_set")].stability_bucket,
            "unstable",
        )

        promotion_reports, watchlist, blockers = build_promotion_gate_reports(
            records,
            outcomes,
            shadows,
            readiness_cards,
            sufficiency_cards,
            stability_cards,
        )
        promotion_by_family = {report.strategy_family: report for report in promotion_reports}
        self.assertEqual(
            promotion_by_family["single_market_mispricing"].promotion_bucket,
            "candidate_for_future_tiny_live_preparation",
        )
        self.assertEqual(
            promotion_by_family["cross_market_constraint"].promotion_bucket,
            "insufficient_evidence",
        )
        self.assertEqual(
            promotion_by_family["rebalancing"].promotion_bucket,
            "deprioritize_for_now",
        )
        self.assertIn("LOW_OUTCOME_SAMPLE", promotion_by_family["cross_market_constraint"].blocker_codes)
        self.assertIn("FORWARD_QUALITY_WEAK", promotion_by_family["rebalancing"].blocker_codes)
        self.assertIn("EXECUTION_GAP_TOO_LARGE", promotion_by_family["rebalancing"].blocker_codes)

        watchlist_families = [entry.strategy_family for entry in watchlist]
        self.assertEqual(watchlist_families[0], "single_market_mispricing")
        self.assertIn("cross_market_constraint", watchlist_families)
        self.assertNotIn("rebalancing", watchlist_families)

        blocker_pairs = {(entry.strategy_family, entry.blocker_code) for entry in blockers}
        self.assertIn(("cross_market_constraint", "LOW_OUTCOME_SAMPLE"), blocker_pairs)
        self.assertIn(("rebalancing", "EXECUTION_GAP_TOO_LARGE"), blocker_pairs)


if __name__ == "__main__":
    unittest.main()
