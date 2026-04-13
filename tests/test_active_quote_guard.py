from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.config_runtime.models import ExecutionConfig, OpportunityConfig, PaperConfig
from src.opportunity.models import CandidateLeg, RankedOpportunity, StrategyFamily
from src.paper.ledger import PaperPositionRecord
from src.runtime.runner import (
    ResearchRunner,
    _apply_inventory_skew,
    _basket_idle_release_eligible,
    _evaluate_quote_health,
)


def _book(best_bid: float, best_ask: float, bid_size: float = 100.0, ask_size: float = 100.0):
    return SimpleNamespace(
        bids=[SimpleNamespace(price=best_bid, size=bid_size)],
        asks=[SimpleNamespace(price=best_ask, size=ask_size)],
    )


def _exec_cfg(**overrides) -> ExecutionConfig:
    cfg = ExecutionConfig(
        maker_quote_min_expected_net_edge_cents=0.01,
        maker_quote_max_fair_value_drift_cents=0.03,
        maker_quote_max_age_sec=45.0,
        maker_quote_inventory_soft_limit_shares=100.0,
        maker_quote_inventory_hard_limit_shares=150.0,
        maker_quote_inventory_skew_cents=0.01,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _opp_cfg() -> OpportunityConfig:
    return OpportunityConfig(fee_buffer_cents=0.01, slippage_buffer_cents=0.01)


def _qualified_health(
    *,
    gross_edge: float,
    fee_impact: float,
    slippage: float,
    net_edge: float,
    threshold: float,
):
    now = datetime.now(timezone.utc)
    return _evaluate_quote_health(
        side="BUY",
        quote_price=0.46,
        size=10.0,
        book=_book(0.45, 0.47),
        inventory_shares=0.0,
        posted_ts=now,
        now_ts=now,
        opportunity_config=_opp_cfg(),
        execution_config=_exec_cfg(maker_quote_min_expected_net_edge_cents=threshold),
        qualified_gross_edge_cents=gross_edge,
        qualified_fee_impact_cents=fee_impact,
        qualified_slippage_cents=slippage,
        qualified_net_edge_cents=net_edge,
    )


def _maker_candidate(*, gross_edge: float, fee_impact: float, slippage: float) -> RankedOpportunity:
    now = datetime.now(timezone.utc)
    return RankedOpportunity(
        strategy_id="neg_risk_rebalancing",
        strategy_family=StrategyFamily.NEG_RISK_REBALANCING,
        candidate_id="cand-maker",
        kind="neg_risk_rebalancing",
        market_slugs=["maker-market"],
        gross_edge_cents=gross_edge,
        fee_estimate_cents=fee_impact,
        slippage_estimate_cents=slippage,
        expected_payout=10.0,
        target_notional_usd=4.5,
        estimated_depth_usd=20.0,
        score=50.0,
        estimated_net_profit_usd=(gross_edge - fee_impact - slippage) * 10.0,
        available_depth_usd=20.0,
        required_depth_usd=4.5,
        partial_fill_risk_score=0.0,
        non_atomic_execution_risk_score=0.0,
        execution_mode="paper_eligible",
        research_only=False,
        strategy_tag="neg_risk_rebalancing:neg_risk_rebalancing",
        ranking_score=50.0,
        sizing_hint_usd=4.5,
        sizing_hint_shares=10.0,
        legs=[
            CandidateLeg(
                token_id="tok-maker",
                market_slug="maker-market",
                action="BUY",
                side="YES",
                required_shares=10.0,
                best_price=0.45,
                vwap_price=0.45,
                metadata={"maker_first": True},
            )
        ],
        ts=now,
    )


def _runner_harness(exec_cfg: ExecutionConfig) -> ResearchRunner:
    runner = object.__new__(ResearchRunner)
    runner.config = SimpleNamespace(opportunity=_opp_cfg(), execution=exec_cfg)
    runner.paper_ledger = SimpleNamespace(position_records={})
    runner._current_summary = None
    runner._record_event = lambda *args, **kwargs: None
    runner._dispatch_order = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dispatch should not be reached"))
    ResearchRunner._reset_quote_guard_metrics(runner)
    return runner


def test_positive_qualified_edge_remains_postable_when_above_threshold() -> None:
    health = _qualified_health(
        gross_edge=0.04,
        fee_impact=0.01,
        slippage=0.01,
        net_edge=0.02,
        threshold=0.01,
    )
    assert health["should_cancel"] is False
    assert health["expected_net_edge"] == 0.02
    assert health["baseline_source"] == "qualification"


def test_zero_edge_behavior_is_explicit_and_tested() -> None:
    health = _qualified_health(
        gross_edge=0.02,
        fee_impact=0.01,
        slippage=0.01,
        net_edge=0.0,
        threshold=0.0,
    )
    assert health["should_cancel"] is False
    assert health["expected_net_edge"] == 0.0
    assert health["comparison_operator"] == "<"


def test_negative_edge_is_blocked() -> None:
    health = _qualified_health(
        gross_edge=0.015,
        fee_impact=0.01,
        slippage=0.01,
        net_edge=-0.005,
        threshold=0.0,
    )
    assert health["should_cancel"] is True
    assert health["cancel_reason"] == "profitability"


def test_quote_canceled_when_fair_value_drift_exceeds_cap() -> None:
    now = datetime.now(timezone.utc)
    health = _evaluate_quote_health(
        side="BUY",
        quote_price=0.40,
        size=10.0,
        book=_book(0.45, 0.47),
        inventory_shares=0.0,
        posted_ts=now,
        now_ts=now,
        opportunity_config=_opp_cfg(),
        execution_config=_exec_cfg(maker_quote_max_fair_value_drift_cents=0.02),
    )
    assert health["should_cancel"] is True
    assert health["cancel_reason"] == "price_guard"


def test_quote_canceled_when_age_exceeds_cap() -> None:
    now = datetime.now(timezone.utc)
    health = _evaluate_quote_health(
        side="BUY",
        quote_price=0.44,
        size=10.0,
        book=_book(0.45, 0.47),
        inventory_shares=0.0,
        posted_ts=now - timedelta(seconds=60),
        now_ts=now,
        opportunity_config=_opp_cfg(),
        execution_config=_exec_cfg(maker_quote_max_age_sec=30.0),
    )
    assert health["should_cancel"] is True
    assert health["cancel_reason"] == "stale_quote"


def test_skew_suppresses_or_worsens_one_side_under_inventory_imbalance() -> None:
    worsened = _apply_inventory_skew(
        "BUY",
        0.45,
        inventory_shares=125.0,
        config=_exec_cfg(),
    )
    suppressed = _apply_inventory_skew(
        "BUY",
        0.45,
        inventory_shares=150.0,
        config=_exec_cfg(),
    )
    assert float(worsened["adjusted_quote_price"]) < 0.45
    assert worsened["suppressed"] is False
    assert suppressed["suppressed"] is True


def test_blocked_before_post_events_are_reflected_in_summary_counters() -> None:
    runner = _runner_harness(_exec_cfg(maker_quote_min_expected_net_edge_cents=0.01))
    candidate = _maker_candidate(gross_edge=0.026, fee_impact=0.01, slippage=0.01)
    try:
        ResearchRunner._submit_candidate_orders(
            runner,
            candidate,
            {"tok-maker": _book(0.45, 0.47)},
            10.0,
        )
    except ValueError as exc:
        assert "profitability" in str(exc)
    else:
        raise AssertionError("expected profitability block before post")

    summary = ResearchRunner._quote_guard_summary(runner)
    assert summary["profitability_cancel_count"] == 1
    assert summary["avg_expected_net_edge_at_post"] == 0.006
    assert summary["quote_guard_reason_counts"] == {"profitability": 1}


def test_idle_hold_release_behavior_unchanged() -> None:
    cfg = PaperConfig(
        max_holding_sec=300.0,
        idle_hold_release_check_sec=180.0,
        idle_hold_release_max_repricing_events=1,
        idle_hold_release_max_abs_unrealized_pnl=0.01,
        idle_hold_release_max_drawdown=0.01,
    )
    now = datetime(2026, 3, 23, 8, 0, 0, tzinfo=timezone.utc)
    r1 = PaperPositionRecord(
        position_id="p1",
        candidate_id="idle-bsk",
        symbol="tok-1",
        market_slug="mkt-1",
        opened_ts=(now - timedelta(seconds=240)).isoformat(),
    )
    r2 = PaperPositionRecord(
        position_id="p2",
        candidate_id="idle-bsk",
        symbol="tok-2",
        market_slug="mkt-2",
        opened_ts=(now - timedelta(seconds=240)).isoformat(),
    )
    r1.last_unrealized_pnl_usd = 0.004
    r1.peak_unrealized_pnl_usd = 0.004
    r1.repricing_event_count = 1
    r1.total_entry_shares = 1.0
    r2.last_unrealized_pnl_usd = -0.003
    r2.total_entry_shares = 1.0
    records = {r1.position_id: r1, r2.position_id: r2}
    assert _basket_idle_release_eligible(records, "idle-bsk", cfg, now) is True
