from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.config_runtime.models import MarketDataConfig
from src.core.models import MarketPair
from src.ingest.clob import ReadOnlyClob
from src.opportunity.models import CandidateLeg, RankedOpportunity, StrategyFamily
from src.runtime.market_universe import MarketTier, MarketUniverseManager, build_pair_snapshot
from src.runtime.runner import ResearchRunner


def _book(best_bid: float, best_ask: float, bid_size: float = 100.0, ask_size: float = 100.0):
    return SimpleNamespace(
        bids=[SimpleNamespace(price=best_bid, size=bid_size)],
        asks=[SimpleNamespace(price=best_ask, size=ask_size)],
        model_dump=lambda mode="json": {
            "bids": [{"price": best_bid, "size": bid_size}],
            "asks": [{"price": best_ask, "size": ask_size}],
        },
    )


def _pair(slug: str) -> MarketPair:
    return MarketPair(
        market_slug=slug,
        yes_token_id=f"{slug}-yes",
        no_token_id=f"{slug}-no",
        question=f"Question {slug}",
    )


def _market(slug: str) -> dict:
    return {
        "slug": slug,
        "question": f"Question {slug}",
        "outcomes": ["YES", "NO"],
        "clobTokenIds": [f"{slug}-yes", f"{slug}-no"],
        "eventSlug": f"event-{slug}",
        "eventTitle": f"Event {slug}",
    }


def _event(slug: str) -> dict:
    return {
        "id": f"evt-{slug}",
        "slug": f"event-{slug}",
        "title": f"Event {slug}",
        "markets": [_market(slug)],
    }


def _neg_risk_market(slug: str, *, event_slug: str) -> dict:
    return {
        "slug": slug,
        "question": f"Question {slug}",
        "groupItemTitle": f"Bucket {slug}",
        "outcomes": '["YES","NO"]',
        "clobTokenIds": f'["{slug}-yes","{slug}-no"]',
        "eventSlug": event_slug,
        "eventTitle": f"Event {event_slug}",
        "negRisk": True,
        "negRiskOther": False,
        "feesEnabled": False,
        "enableOrderBook": True,
    }


def _neg_risk_event(event_slug: str, market_slugs: list[str]) -> dict:
    return {
        "id": f"evt-{event_slug}",
        "slug": event_slug,
        "title": f"Event {event_slug}",
        "negRisk": True,
        "enableNegRisk": True,
        "negRiskAugmented": False,
        "markets": [_neg_risk_market(slug, event_slug=event_slug) for slug in market_slugs],
    }


def _neg_risk_event_group(event_slug: str, market_slugs: list[str]) -> dict:
    return {
        "event_id": f"evt-{event_slug}",
        "event_slug": event_slug,
        "event_title": f"Event {event_slug}",
        "neg_risk_market_id": f"neg-{event_slug}",
        "markets": [
            {
                "slug": slug,
                "question": f"Question {slug}",
                "group_item_title": f"Bucket {slug}",
                "yes_token_id": f"{slug}-yes",
                "best_bid": 0.45,
                "best_ask": 0.47,
            }
            for slug in market_slugs
        ],
    }


def _build_runner() -> ResearchRunner:
    temp_dir = tempfile.TemporaryDirectory()
    runner = ResearchRunner(
        settings_path="config/settings.yaml",
        constraints_path="config/constraints.yaml",
        debug_output_dir=temp_dir.name,
    )
    runner.store = Mock()
    runner.store.save_raw_snapshot = Mock()
    runner.store.save_account_snapshot = Mock()
    runner.store.save_run_summary = Mock()
    runner.store.save_qualification_funnel_report = Mock()
    runner.store.load_recent_productive_market_slugs = Mock(return_value=[])
    runner.store.load_recent_candidate_market_slugs = Mock(return_value=[])
    runner.store.load_recent_rejection_market_slugs = Mock(return_value=[])
    runner.opportunity_store = Mock()
    runner.paper_ledger = Mock()
    runner.paper_ledger.position_records = {}
    runner.paper_ledger.snapshot.return_value = SimpleNamespace(
        open_positions=0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
    )
    runner.market_universe = MarketUniverseManager(runner.config.market_data)
    runner._hot_tier_zero_cycles = 0
    return runner


def _ranked(candidate_id: str, ranking_score: float, expected_profit_usd: float) -> RankedOpportunity:
    now = datetime.now(timezone.utc)
    return RankedOpportunity(
        strategy_id="single_market_mispricing",
        strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
        candidate_id=candidate_id,
        kind="single_market",
        market_slugs=[candidate_id],
        gross_edge_cents=0.05,
        fee_estimate_cents=0.01,
        slippage_estimate_cents=0.01,
        expected_payout=1.0,
        target_notional_usd=10.0,
        estimated_depth_usd=50.0,
        score=ranking_score,
        estimated_net_profit_usd=expected_profit_usd,
        available_depth_usd=50.0,
        required_depth_usd=10.0,
        partial_fill_risk_score=0.0,
        non_atomic_execution_risk_score=0.0,
        execution_mode="paper_eligible",
        research_only=False,
        strategy_tag="single_market:single_market_mispricing",
        ranking_score=ranking_score,
        sizing_hint_usd=10.0,
        sizing_hint_shares=10.0,
        expected_profit_usd=expected_profit_usd,
        legs=[
            CandidateLeg(
                token_id=f"{candidate_id}-yes",
                market_slug=candidate_id,
                action="BUY",
                side="YES",
                required_shares=10.0,
            )
        ],
        ts=now,
    )


def test_market_universe_tier_assignment() -> None:
    cfg = MarketDataConfig(
        discovery_refresh_interval_sec=60.0,
        hot_refresh_interval_sec=2.0,
        warm_refresh_interval_sec=8.0,
        cold_refresh_interval_sec=30.0,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    pairs = [_pair("hot"), _pair("warm"), _pair("cold")]
    manager.update_discovery(events=[], markets=[], pairs=pairs, discovered_at=now)
    manager.record_qualified_candidate(["hot"], now)
    manager.record_raw_signal(["warm"], now)

    counts = manager.tier_counts()
    assert counts == {"hot": 0, "warm": 0, "cold": 3}

    manager.select_refresh_plan(now + timedelta(seconds=3))
    counts = manager.tier_counts()
    assert counts["hot"] == 1
    assert counts["warm"] == 1
    assert counts["cold"] == 1


def test_hot_tier_cannot_collapse_to_zero_with_seeded_productive_markets() -> None:
    cfg = MarketDataConfig(
        discovery_refresh_interval_sec=60.0,
        hot_refresh_interval_sec=2.0,
        warm_refresh_interval_sec=8.0,
        cold_refresh_interval_sec=30.0,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    manager.seed_productive_markets(["pinned-market"], observed_at=now)
    manager.update_discovery(events=[], markets=[_market("pinned-market")], pairs=[_pair("pinned-market")], discovered_at=now)

    counts = manager.tier_counts(now)
    assert counts["hot"] == 1
    plan = manager.select_refresh_plan(now, cycle_index=1)
    assert [entry.market_slug for entry in plan.hot] == ["pinned-market"]


def test_tier_based_refresh_selection_respects_due_intervals() -> None:
    cfg = MarketDataConfig(
        hot_refresh_interval_sec=2.0,
        warm_refresh_interval_sec=8.0,
        cold_refresh_interval_sec=30.0,
        hot_market_count=10,
        warm_market_count=10,
        cold_market_count=10,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    pairs = [_pair("hot"), _pair("warm"), _pair("cold")]
    manager.update_discovery(events=[], markets=[], pairs=pairs, discovered_at=now)
    manager.record_qualified_candidate(["hot"], now)
    manager.record_raw_signal(["warm"], now)
    manager.select_refresh_plan(now)
    manager.entries["hot"].last_refreshed_at = now
    manager.entries["warm"].last_refreshed_at = now
    manager.entries["cold"].last_refreshed_at = now

    plan = manager.select_refresh_plan(now + timedelta(seconds=5))
    assert [entry.market_slug for entry in plan.hot] == ["hot"]
    assert plan.warm == []
    assert plan.cold == []


def test_backstop_rescan_reintroduces_markets_without_recent_change_events() -> None:
    cfg = MarketDataConfig(
        hot_refresh_interval_sec=20.0,
        warm_refresh_interval_sec=20.0,
        cold_refresh_interval_sec=300.0,
        hot_market_count=0,
        warm_market_count=0,
        cold_market_count=0,
        backstop_full_rescan_interval_sec=1.0,
        backstop_refresh_market_budget=2,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    pairs = [_pair("cold-a"), _pair("cold-b"), _pair("cold-c")]
    manager.update_discovery(events=[], markets=[_market("cold-a"), _market("cold-b"), _market("cold-c")], pairs=pairs, discovered_at=now)
    for entry in manager.active_entries():
        entry.last_refreshed_at = now
        entry.last_evaluated_at = now

    plan = manager.select_refresh_plan(now + timedelta(seconds=2), cycle_index=1)
    assert len(plan.backstop) == 2
    assert len(plan.force_scan_market_slugs) == 2


def test_changed_leg_triggers_full_neg_risk_family_recompute() -> None:
    cfg = MarketDataConfig(
        neg_risk_family_due_refresh_interval_sec=120.0,
        neg_risk_family_backstop_every_n_cycles=99,
        neg_risk_family_backstop_budget=0,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    family_slug = "family-alpha"
    manager.seed_productive_markets(["leg-a"], observed_at=now)
    manager.update_discovery(
        events=[],
        markets=[_neg_risk_market("leg-a", event_slug=family_slug), _neg_risk_market("leg-b", event_slug=family_slug)],
        pairs=[_pair("leg-a"), _pair("leg-b")],
        discovered_at=now,
    )

    selected, metrics = manager.select_neg_risk_groups_for_recompute(
        event_groups=[_neg_risk_event_group(family_slug, ["leg-a", "leg-b"])],
        changed_market_slugs={"leg-b"},
        now=now + timedelta(seconds=1),
        cycle_index=1,
    )

    assert [group["event_slug"] for group in selected] == [family_slug]
    assert metrics["families_recomputed_due_to_change"] == 1
    assert metrics["families_considered"] == 1


def test_pinned_productive_family_is_reevaluated_when_due_without_market_change() -> None:
    cfg = MarketDataConfig(
        neg_risk_family_due_refresh_interval_sec=20.0,
        neg_risk_family_backstop_every_n_cycles=99,
        neg_risk_family_backstop_budget=0,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    family_slug = "family-beta"
    manager.seed_productive_markets(["leg-c"], observed_at=now)
    manager.update_discovery(
        events=[],
        markets=[_neg_risk_market("leg-c", event_slug=family_slug), _neg_risk_market("leg-d", event_slug=family_slug)],
        pairs=[_pair("leg-c"), _pair("leg-d")],
        discovered_at=now,
    )
    manager.record_family_recompute(family_slug=family_slug, observed_at=now, market_slugs={"leg-c", "leg-d"})

    selected, metrics = manager.select_neg_risk_groups_for_recompute(
        event_groups=[_neg_risk_event_group(family_slug, ["leg-c", "leg-d"])],
        changed_market_slugs=set(),
        now=now + timedelta(seconds=25),
        cycle_index=1,
    )

    assert [group["event_slug"] for group in selected] == [family_slug]
    assert metrics["families_recomputed_due_to_due_interval"] == 1
    assert metrics["pinned_productive_families_evaluated"] == 1


def test_family_backstop_rescan_restores_family_consideration_without_changes() -> None:
    cfg = MarketDataConfig(
        neg_risk_family_due_refresh_interval_sec=999.0,
        neg_risk_family_backstop_every_n_cycles=2,
        neg_risk_family_backstop_budget=1,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    family_slug = "family-gamma"
    manager.seed_recent_qualified_markets(["leg-e"], observed_at=now)
    manager.update_discovery(
        events=[],
        markets=[_neg_risk_market("leg-e", event_slug=family_slug), _neg_risk_market("leg-f", event_slug=family_slug)],
        pairs=[_pair("leg-e"), _pair("leg-f")],
        discovered_at=now,
    )
    manager.record_family_recompute(family_slug=family_slug, observed_at=now, market_slugs={"leg-e", "leg-f"})

    selected, metrics = manager.select_neg_risk_groups_for_recompute(
        event_groups=[_neg_risk_event_group(family_slug, ["leg-e", "leg-f"])],
        changed_market_slugs=set(),
        now=now + timedelta(seconds=5),
        cycle_index=2,
    )

    assert [group["event_slug"] for group in selected] == [family_slug]
    assert metrics["family_backstop_recompute_count"] == 1


def test_recent_near_miss_family_gets_due_refresh() -> None:
    cfg = MarketDataConfig(
        neg_risk_family_due_refresh_interval_sec=15.0,
        neg_risk_family_backstop_every_n_cycles=99,
        neg_risk_family_backstop_budget=0,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    family_slug = "family-delta"
    manager.seed_recent_near_miss_markets(["leg-g"], observed_at=now)
    manager.update_discovery(
        events=[],
        markets=[_neg_risk_market("leg-g", event_slug=family_slug), _neg_risk_market("leg-h", event_slug=family_slug)],
        pairs=[_pair("leg-g"), _pair("leg-h")],
        discovered_at=now,
    )
    manager.record_family_recompute(family_slug=family_slug, observed_at=now, market_slugs={"leg-g", "leg-h"})

    selected, metrics = manager.select_neg_risk_groups_for_recompute(
        event_groups=[_neg_risk_event_group(family_slug, ["leg-g", "leg-h"])],
        changed_market_slugs=set(),
        now=now + timedelta(seconds=20),
        cycle_index=1,
    )

    assert [group["event_slug"] for group in selected] == [family_slug]
    assert metrics["recent_near_miss_families_evaluated"] == 1
    assert metrics["families_recomputed_due_to_due_interval"] == 1


def test_batch_fetch_wrappers_use_batch_results_and_fallbacks() -> None:
    class FakeClient:
        def get_order_books(self, params):
            return [
                SimpleNamespace(
                    asset_id="tok-a",
                    bids=[SimpleNamespace(price="0.45", size="100")],
                    asks=[SimpleNamespace(price="0.47", size="100")],
                )
            ]

        def get_order_book(self, token_id):
            return SimpleNamespace(
                asset_id=token_id,
                bids=[SimpleNamespace(price="0.40", size="50")],
                asks=[SimpleNamespace(price="0.42", size="60")],
            )

        def get_midpoints(self, params):
            return [{"token_id": "tok-a", "midpoint": "0.46"}, {"token_id": "tok-b", "midpoint": "0.41"}]

        def get_spreads(self, params):
            return {"tok-a": "0.02", "tok-b": "0.02"}

        def get_prices(self, params):
            return [{"token_id": "tok-a", "price": "0.45"}, {"token_id": "tok-b", "price": "0.41"}]

    clob = object.__new__(ReadOnlyClob)
    clob.client = FakeClient()
    clob.host = "https://clob.polymarket.com"

    books = ReadOnlyClob.fetch_books_batch(clob, ["tok-a", "tok-b"])
    assert sorted(books) == ["tok-a", "tok-b"]
    assert ReadOnlyClob.fetch_midpoints_batch(clob, ["tok-a", "tok-b"]) == {"tok-a": 0.46, "tok-b": 0.41}
    assert ReadOnlyClob.fetch_spreads_batch(clob, ["tok-a", "tok-b"]) == {"tok-a": 0.02, "tok-b": 0.02}
    assert ReadOnlyClob.fetch_prices_batch(clob, ["tok-a", "tok-b"]) == {"tok-a": 0.45, "tok-b": 0.41}


def test_recompute_gate_behaves_deterministically() -> None:
    cfg = MarketDataConfig(
        recompute_midpoint_delta_cents=0.01,
        recompute_spread_delta_cents=0.01,
        recompute_top_depth_delta_ratio=0.25,
        recompute_inventory_delta_shares=5.0,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    pair = _pair("sample")
    manager.update_discovery(events=[], markets=[], pairs=[pair], discovered_at=now)

    first = build_pair_snapshot(
        yes_book=_book(0.45, 0.47),
        no_book=_book(0.52, 0.54),
        inventory_abs=0.0,
        source="rest",
        ts=now,
    )
    second = build_pair_snapshot(
        yes_book=_book(0.45, 0.47),
        no_book=_book(0.52, 0.54),
        inventory_abs=0.0,
        source="rest",
        ts=now + timedelta(seconds=2),
    )
    third = build_pair_snapshot(
        yes_book=_book(0.48, 0.50),
        no_book=_book(0.52, 0.54),
        inventory_abs=0.0,
        source="rest",
        ts=now + timedelta(seconds=4),
    )

    assert manager.evaluate_recompute(market_slug="sample", snapshot=first, stream_event=False).triggered is True
    unchanged = manager.evaluate_recompute(market_slug="sample", snapshot=second, stream_event=False)
    assert unchanged.triggered is False
    assert unchanged.reasons == ["unchanged"]
    moved = manager.evaluate_recompute(market_slug="sample", snapshot=third, stream_event=False)
    assert moved.triggered is True
    assert "midpoint_delta" in moved.reasons


def test_changed_or_due_scan_selection_keeps_due_market_even_when_snapshot_is_unchanged() -> None:
    runner = _build_runner()
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    pair = _pair("due-market")
    runner.market_universe.update_discovery(events=[], markets=[_market("due-market")], pairs=[pair], discovered_at=now)
    runner.market_universe.record_qualified_candidate([pair.market_slug], now)
    first_snapshot = build_pair_snapshot(
        yes_book=_book(0.45, 0.47),
        no_book=_book(0.53, 0.55),
        inventory_abs=0.0,
        source="rest",
        ts=now,
    )
    runner.market_universe.entries[pair.market_slug].last_snapshot = first_snapshot
    runner.market_universe.entries[pair.market_slug].last_refreshed_at = now
    refresh_plan = runner.market_universe.select_refresh_plan(now + timedelta(seconds=3), cycle_index=1)

    with (
        patch.object(runner.clob, "fetch_books_batch", return_value={pair.yes_token_id: _book(0.45, 0.47), pair.no_token_id: _book(0.53, 0.55)}),
        patch.object(runner.clob, "fetch_midpoints_batch", return_value={}),
        patch.object(runner.clob, "fetch_spreads_batch", return_value={}),
        patch.object(runner.clob, "fetch_prices_batch", return_value={}),
    ):
        changed, due = runner._refresh_market_books(
            cycle_started=now + timedelta(seconds=3),
            refresh_plan=refresh_plan,
            book_cache={},
            forced_hot_market_slugs=set(),
            force_rest_hot=False,
        )

    assert changed == set()
    assert due == {"due-market"}


def test_hot_tier_websocket_fallback_forces_rest_refresh() -> None:
    runner = _build_runner()
    runner._reset_cycle_metrics()
    runner.config.market_data.enable_hot_tier_websocket = True
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    pair = _pair("hot-market")
    runner.market_universe.update_discovery(events=[], markets=[], pairs=[pair], discovered_at=now)
    runner.market_universe.record_qualified_candidate([pair.market_slug], now)
    runner.market_universe.select_refresh_plan(now)
    runner.market_universe.entries[pair.market_slug].last_refreshed_at = now
    refresh_plan = runner.market_universe.select_refresh_plan(now + timedelta(seconds=1))
    runner.hot_market_stream = Mock()
    runner.hot_market_stream.collect_updates.return_value = {
        "updated_tokens": [],
        "update_count": 0,
        "fallback_to_rest": True,
    }

    changed_slugs, force_rest_hot = runner._collect_hot_tier_updates(refresh_plan, now + timedelta(seconds=1))
    with (
        patch.object(runner.clob, "fetch_books_batch", return_value={pair.yes_token_id: _book(0.45, 0.47), pair.no_token_id: _book(0.53, 0.55)}),
        patch.object(runner.clob, "fetch_midpoints_batch", return_value={}),
        patch.object(runner.clob, "fetch_spreads_batch", return_value={}),
        patch.object(runner.clob, "fetch_prices_batch", return_value={}),
    ):
        runner._refresh_market_books(
            cycle_started=now + timedelta(seconds=1),
            refresh_plan=refresh_plan,
            book_cache={},
            forced_hot_market_slugs=changed_slugs,
            force_rest_hot=force_rest_hot,
        )

    assert force_rest_hot is True
    assert runner._cycle_metrics["books_refreshed_by_tier"]["hot"] == 2


def test_negative_cache_suppresses_retries_and_expires_correctly() -> None:
    class FakeClient:
        def __init__(self):
            self.single_fetches = 0

        def get_order_books(self, params):
            return []

        def get_order_book(self, token_id):
            self.single_fetches += 1
            raise RuntimeError(f"404 Not Found for /book?token_id={token_id}")

    clob = object.__new__(ReadOnlyClob)
    clob.client = FakeClient()
    clob.host = "https://clob.polymarket.com"
    clob.configure_negative_cache(no_orderbook_ttl_sec=60.0, invalid_token_retry_interval_sec=120.0)
    clob.reset_request_stats()

    assert ReadOnlyClob.fetch_books_batch(clob, ["missing-token"]) == {}
    assert clob.client.single_fetches == 1
    assert ReadOnlyClob.fetch_books_batch(clob, ["missing-token"]) == {}
    assert clob.client.single_fetches == 1

    clob._negative_cache["missing-token"]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert ReadOnlyClob.fetch_books_batch(clob, ["missing-token"]) == {}
    assert clob.client.single_fetches == 2
    stats = clob.request_stats_snapshot()
    assert stats["negative_cache_hits"] >= 1
    assert stats["negative_cache_expired_rechecks"] >= 1


def test_negative_cached_dead_token_does_not_block_family_recompute_for_valid_sibling() -> None:
    cfg = MarketDataConfig(
        neg_risk_family_due_refresh_interval_sec=120.0,
        neg_risk_family_backstop_every_n_cycles=99,
        neg_risk_family_backstop_budget=0,
    )
    manager = MarketUniverseManager(cfg)
    now = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
    family_slug = "family-epsilon"
    manager.seed_recent_qualified_markets(["live-leg"], observed_at=now)
    manager.update_discovery(
        events=[],
        markets=[_neg_risk_market("dead-leg", event_slug=family_slug), _neg_risk_market("live-leg", event_slug=family_slug)],
        pairs=[_pair("dead-leg"), _pair("live-leg")],
        discovered_at=now,
    )

    selected, metrics = manager.select_neg_risk_groups_for_recompute(
        event_groups=[_neg_risk_event_group(family_slug, ["dead-leg", "live-leg"])],
        changed_market_slugs={"live-leg"},
        now=now + timedelta(seconds=1),
        cycle_index=1,
    )

    assert [group["event_slug"] for group in selected] == [family_slug]
    assert metrics["families_considered"] == 1


def test_cycle_metrics_are_emitted_on_run_summary() -> None:
    runner = _build_runner()
    event = _event("alpha")
    market = _market("alpha")
    with (
        patch("src.runtime.runner.fetch_events", return_value=[event]),
        patch("src.runtime.runner.fetch_markets_from_events", return_value=[market]),
        patch.object(runner.clob, "fetch_simplified_markets", return_value=[]),
        patch.object(runner.clob, "fetch_books_batch", return_value={"alpha-yes": _book(0.45, 0.47), "alpha-no": _book(0.53, 0.55)}),
        patch.object(runner.clob, "fetch_midpoints_batch", return_value={"alpha-yes": 0.46, "alpha-no": 0.54}),
        patch.object(runner.clob, "fetch_spreads_batch", return_value={"alpha-yes": 0.02, "alpha-no": 0.02}),
        patch.object(runner.clob, "fetch_prices_batch", return_value={"alpha-yes": 0.45, "alpha-no": 0.55}),
        patch.object(runner, "_run_single_market_scan"),
        patch.object(runner, "_run_cross_market_scan"),
        patch.object(runner, "_run_neg_risk_scan"),
        patch.object(runner, "_manage_open_positions"),
    ):
        summary = runner.run_once(experiment_context={"campaign_target_strategy_families": ["single_market_mispricing"]})

    metrics = summary.metadata["scan_metrics"]
    assert "discovery_duration_ms" in metrics
    assert "refresh_duration_ms" in metrics
    assert "candidate_eval_duration_ms" in metrics
    assert "total_cycle_duration_ms" in metrics
    assert "batch_request_counts_by_endpoint" in metrics
    assert "qualified_candidates_per_minute" in metrics
    assert "hot_tier_seed_count" in metrics
    assert "backstop_markets_refreshed" in metrics
    assert "negative_cache_active_count" in metrics
    assert "recompute_due_count" in metrics
    assert "markets_with_any_signal" in metrics
    assert "families_considered" in metrics
    assert "neg_risk_event_groups_available" in metrics
    assert "neg_risk_audit_watchlist_matches" in metrics
    assert "neg_risk_audit_forced_families" in metrics
    assert "neg_risk_family_qualification_audit_count" in metrics
    assert "families_recomputed_due_to_change" in metrics
    assert "families_recomputed_due_to_due_interval" in metrics
    assert "pinned_productive_families_evaluated" in metrics
    assert "recent_near_miss_families_evaluated" in metrics
    assert "family_backstop_recompute_count" in metrics
    assert "avg_markets_per_family_recompute" in metrics
    assert "neg_risk_family_coverage_rate" in metrics
    assert "raw_candidates_by_family_per_cycle" in metrics
    assert "watched_families_audited" in metrics
    assert "watched_families_fast_vs_broad_mismatch_count" in metrics
    assert "watched_families_incomplete_input_count" in metrics
    assert "watched_families_time_skew_count" in metrics
    assert "watched_families_negative_cache_side_effect_count" in metrics
    assert "watched_families_depth_failure_count" in metrics
    assert "watched_families_net_profit_failure_count" in metrics
    assert "productive_families_audited" in metrics
    assert "productive_families_currently_viable_count" in metrics
    assert "productive_families_marginal_count" in metrics
    assert "productive_families_edge_vanished_count" in metrics
    assert "productive_families_depth_failure_count" in metrics
    assert "productive_families_spread_failure_count" in metrics
    assert "selector_refresh_families_audited" in metrics
    assert "selector_refresh_currently_viable_count" in metrics
    assert "selector_refresh_near_miss_count" in metrics
    assert "selector_refresh_marginal_count" in metrics
    assert "selector_refresh_downgrade_count" in metrics
    assert "selector_refresh_park_count" in metrics
    assert "raw_candidates_per_minute" in metrics
    assert "near_miss_candidates_per_minute" in metrics


def test_ranked_opportunity_hook_is_pluggable() -> None:
    runner = object.__new__(ResearchRunner)
    runner._experiment_context = {"ranked_opportunity_sort": "expected_profit_usd"}
    candidates = [
        _ranked("low-rank-high-profit", ranking_score=10.0, expected_profit_usd=5.0),
        _ranked("high-rank-low-profit", ranking_score=99.0, expected_profit_usd=1.0),
    ]
    ordered = ResearchRunner._apply_ranked_opportunity_hook(runner, candidates)
    assert [candidate.candidate_id for candidate in ordered] == [
        "low-rank-high-profit",
        "high-rank-low-profit",
    ]
