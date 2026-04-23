from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.domain.models import AccountSnapshot
from src.runtime.market_universe import FamilyEntry
from src.runtime.runner import ResearchRunner


def _book(
    best_bid: float,
    best_ask: float,
    *,
    bid_size: float = 1000.0,
    ask_size: float = 1000.0,
    ts: datetime | None = None,
):
    return SimpleNamespace(
        bids=[SimpleNamespace(price=best_bid, size=bid_size)],
        asks=[SimpleNamespace(price=best_ask, size=ask_size)],
        ts=ts or datetime.now(timezone.utc),
        model_dump=lambda mode="json": {
            "bids": [{"price": best_bid, "size": bid_size}],
            "asks": [{"price": best_ask, "size": ask_size}],
        },
    )


def _event_group(event_slug: str, market_slugs: list[str]) -> dict:
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
                "best_bid": 0.2,
                "best_ask": 0.21,
            }
            for slug in market_slugs
        ],
    }


def _raw_neg_risk_event(event_slug: str, market_slugs: list[str]) -> dict:
    return {
        "id": f"evt-{event_slug}",
        "slug": event_slug,
        "title": f"Event {event_slug}",
        "negRisk": True,
        "enableNegRisk": True,
        "negRiskAugmented": False,
        "negRiskMarketID": f"neg-{event_slug}",
        "markets": [
            {
                "slug": slug,
                "question": f"Question {slug}",
                "groupItemTitle": f"Bucket {slug}",
                "negRisk": True,
                "negRiskOther": False,
                "feesEnabled": False,
                "enableOrderBook": True,
                "outcomes": '["YES", "NO"]',
                "clobTokenIds": f'["{slug}-yes", "{slug}-no"]',
                "bestBid": 0.2,
                "bestAsk": 0.21,
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
    runner.store.save_candidate = Mock()
    runner.store.save_risk_decision = Mock()
    runner.store.save_order_intent = Mock()
    runner.store.save_execution_report = Mock()
    runner.store.save_system_event = Mock()
    runner.store.save_rejection_event = Mock()
    runner.store.load_recent_productive_market_slugs = Mock(return_value=[])
    runner.store.load_recent_candidate_market_slugs = Mock(return_value=[])
    runner.store.load_recent_rejection_market_slugs = Mock(return_value=[])
    runner.opportunity_store = Mock()
    runner._ab_sidecar = Mock()
    runner.paper_ledger = Mock()
    runner.paper_ledger.position_records = {}
    runner.paper_ledger.snapshot.return_value = AccountSnapshot(
        cash=10000.0,
        frozen_cash=0.0,
        ts=datetime.now(timezone.utc),
    )
    runner.market_universe._pinned_productive_market_slugs.clear()
    runner.market_universe._seeded_recent_qualified_market_slugs.clear()
    runner.market_universe._seeded_recent_near_miss_market_slugs.clear()
    runner.market_universe.family_entries.clear()
    runner.config.market_data.neg_risk_condition_monitor_mode_enabled = False
    runner.config.market_data.neg_risk_condition_monitor_watchlist = []
    runner._reset_cycle_metrics()
    runner._run_sequence = 1
    return runner


def test_family_audit_reports_incomplete_input_when_sibling_leg_missing() -> None:
    runner = _build_runner()
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_group = _event_group("family-missing", ["leg-a", "leg-b"])
    books_by_token = {
        "leg-a-yes": _book(0.20, 0.21, ts=now),
    }

    audit = runner._build_neg_risk_family_input_audit(
        event_group=event_group,
        books_by_token=books_by_token,
        cycle_started=now,
        naturally_selected=True,
        audit_forced=False,
    )

    assert audit["expected_leg_count"] == 2
    assert audit["valid_book_leg_count"] == 1
    assert audit["missing_leg_count"] == 1
    assert audit["all_expected_legs_present"] is False
    assert audit["inputs_coherent"] is False


def test_family_audit_reports_snapshot_skew() -> None:
    runner = _build_runner()
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_group = _event_group("family-skew", ["leg-a", "leg-b"])
    books_by_token = {
        "leg-a-yes": _book(0.20, 0.21, ts=now - timedelta(milliseconds=250)),
        "leg-b-yes": _book(0.30, 0.31, ts=now - timedelta(milliseconds=2550)),
    }

    audit = runner._build_neg_risk_family_input_audit(
        event_group=event_group,
        books_by_token=books_by_token,
        cycle_started=now,
        naturally_selected=True,
        audit_forced=False,
    )

    assert audit["valid_book_leg_count"] == 2
    assert float(audit["family_snapshot_time_skew_ms"]) >= 2000.0
    assert audit["all_legs_refreshed_same_window"] is False
    assert audit["inputs_coherent"] is False


def test_negative_cached_dead_token_still_surfaces_in_family_audit() -> None:
    runner = _build_runner()
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_group = _event_group("family-negative-cache", ["dead-leg", "live-leg"])
    runner.clob._negative_cache["dead-leg-yes"] = {
        "reason": "no_orderbook",
        "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
    }
    books_by_token = {
        "live-leg-yes": _book(0.25, 0.26, ts=now),
    }

    audit = runner._build_neg_risk_family_input_audit(
        event_group=event_group,
        books_by_token=books_by_token,
        cycle_started=now,
        naturally_selected=False,
        audit_forced=True,
    )

    assert audit["negative_cached_leg_count"] == 1
    assert audit["missing_leg_count"] == 1
    assert audit["negative_cached_legs"] == ["dead-leg"]
    assert audit["family_considered"] is True


def test_complete_family_input_can_pass_raw_to_qualification() -> None:
    runner = _build_runner()
    runner.config.opportunity.min_edge_cents = 0.01
    runner.config.opportunity.min_net_profit_usd = 0.01
    runner.config.opportunity.max_spread_cents = 0.10
    runner.config.opportunity.min_absolute_leg_depth_usd = 0.0
    runner.config.opportunity.max_single_leg_bid = 1.0
    runner.config.opportunity.max_partial_fill_risk = 1.0
    runner.config.opportunity.max_non_atomic_risk = 1.0
    runner.config.opportunity.min_depth_multiple = 1.0

    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_group = _event_group("family-pass", ["leg-a", "leg-b", "leg-c"])
    books_by_token = {
        "leg-a-yes": _book(0.20, 0.21, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-b-yes": _book(0.25, 0.26, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-c-yes": _book(0.30, 0.31, bid_size=5000.0, ask_size=5000.0, ts=now),
    }
    raw_candidate, audit = runner.neg_risk_strategy.detect_with_audit(
        event_group,
        books_by_token,
        max_notional=100.0,
    )

    assert raw_candidate is not None
    assert audit is None

    audit_sink = runner._build_neg_risk_family_input_audit(
        event_group=event_group,
        books_by_token=books_by_token,
        cycle_started=now,
        naturally_selected=True,
        audit_forced=False,
    )

    candidate = runner._qualify_and_rank_candidate(
        raw_candidate=raw_candidate,
        books_by_token=books_by_token,
        account_snapshot=AccountSnapshot(cash=10000.0, frozen_cash=0.0, ts=now),
        audit_sink=audit_sink,
    )

    assert candidate is not None
    assert audit_sink["qualified"] is True
    assert audit_sink["qualification_rejection_reasons"] == []
    assert float(audit_sink["gross_edge"]) > 0.0
    assert float(audit_sink["expected_net_profit_usd"]) > 0.0


def test_pinned_family_audit_runs_even_without_natural_trigger() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_family_audit_mode_enabled = True
    runner.config.market_data.neg_risk_family_audit_watchlist = ["gc-settle-3800-4200-jun-2026"]
    runner.config.market_data.neg_risk_family_audit_budget = 2
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    watched_group = _event_group(
        "family-watch",
        ["gc-settle-3800-4200-jun-2026", "gc-settle-4200-4600-jun-2026"],
    )

    selected, _metrics, natural_keys, audit_forced_keys = runner._select_neg_risk_event_groups_for_scan(
        event_groups=[watched_group],
        changed_market_slugs=set(),
        cycle_started=now,
    )

    assert [group["event_slug"] for group in selected] == ["family-watch"]
    assert natural_keys == set()
    assert audit_forced_keys == {"family-watch"}


def test_condition_monitor_mode_filters_natural_selection_to_monitored_families() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_condition_monitor_mode_enabled = True
    runner.config.market_data.neg_risk_condition_monitor_watchlist = [
        "how-many-gold-cards-will-trump-sell-in-2026",
    ]
    runner.config.market_data.neg_risk_family_audit_mode_enabled = False
    watched_group = _event_group(
        "how-many-gold-cards-will-trump-sell-in-2026",
        [
            "will-trump-sell-0-gold-cards-in-2026",
            "will-trump-sell-1-100-gold-cards-in-2026",
        ],
    )
    other_group = _event_group(
        "balance-of-power-2026-midterms",
        [
            "which-party-will-win-the-house-in-2026",
            "which-party-will-win-the-senate-in-2026",
        ],
    )
    runner.market_universe.select_neg_risk_groups_for_recompute = Mock(
        return_value=([watched_group, other_group], {"families_considered": 2})
    )

    selected, metrics, natural_keys, audit_forced_keys = runner._select_neg_risk_event_groups_for_scan(
        event_groups=[watched_group, other_group],
        changed_market_slugs=set(),
        cycle_started=datetime(2026, 3, 24, 5, 0, 0, tzinfo=timezone.utc),
    )

    assert [group["event_slug"] for group in selected] == ["how-many-gold-cards-will-trump-sell-in-2026"]
    assert metrics["families_considered"] == 1
    assert natural_keys == {"how-many-gold-cards-will-trump-sell-in-2026"}
    assert audit_forced_keys == set()


def test_condition_monitor_mode_forces_only_watched_family_and_skips_broader_refresh() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_condition_monitor_mode_enabled = True
    runner.config.market_data.neg_risk_condition_monitor_watchlist = [
        "harvey-weinstein-prison-time",
    ]
    runner.config.market_data.neg_risk_family_audit_mode_enabled = True
    runner.config.market_data.neg_risk_family_audit_budget = 2
    runner.config.market_data.neg_risk_selector_refresh_budget = 10
    watched_group = _event_group(
        "harvey-weinstein-prison-time",
        [
            "how-much-prison-time-will-harvey-weinstein-be-sentenced-to",
            "harvey-weinstein-no-prison-time",
        ],
    )
    other_group = _event_group(
        "another-family",
        [
            "another-leg-a",
            "another-leg-b",
        ],
    )
    runner.market_universe.select_neg_risk_groups_for_recompute = Mock(
        return_value=([], {"families_considered": 0})
    )

    selected, metrics, natural_keys, audit_forced_keys = runner._select_neg_risk_event_groups_for_scan(
        event_groups=[watched_group, other_group],
        changed_market_slugs=set(),
        cycle_started=datetime(2026, 3, 24, 5, 0, 0, tzinfo=timezone.utc),
    )

    assert [group["event_slug"] for group in selected] == ["harvey-weinstein-prison-time"]
    assert metrics["families_considered"] == 1
    assert natural_keys == set()
    assert audit_forced_keys == {"harvey-weinstein-prison-time"}


def test_condition_monitor_report_filters_to_monitored_families_only() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_condition_monitor_mode_enabled = True
    runner.config.market_data.neg_risk_condition_monitor_watchlist = [
        "harvey-weinstein-prison-time",
    ]
    runner._build_neg_risk_productive_family_economics_report = Mock(
        return_value=[
            {
                "family_key": "harvey-weinstein-prison-time",
                "source_tags": ["config", "history_near_miss"],
                "historically_productive_family": False,
                "current_audit_result": "depth_insufficient_now",
                "remains_viable_now": False,
                "economics_class": "depth_insufficient_now",
                "gross_edge_cents": 0.049,
                "net_edge_cents": 0.029,
                "expected_net_profit_usd": 0.16443,
                "required_depth_usd": 5.7267,
                "available_depth_usd": 2017.73232,
                "max_spread_observed": 0.019,
                "rejection_reason_codes": [
                    "ABSOLUTE_DEPTH_BELOW_FLOOR",
                    "EDGE_BELOW_THRESHOLD",
                    "NET_PROFIT_BELOW_THRESHOLD",
                ],
                "watchlist_recommendation": "near_miss_now",
            },
            {
                "family_key": "how-many-gold-cards-will-trump-sell-in-2026",
                "source_tags": ["config", "history"],
                "historically_productive_family": False,
                "current_audit_result": "spread_too_wide_now",
                "remains_viable_now": False,
                "economics_class": "spread_too_wide_now",
                "gross_edge_cents": 0.042,
                "net_edge_cents": 0.022,
                "expected_net_profit_usd": 0.3498,
                "required_depth_usd": 17.81844,
                "available_depth_usd": 2737388.46084,
                "max_spread_observed": 0.083,
                "rejection_reason_codes": [
                    "EDGE_BELOW_THRESHOLD",
                    "NET_PROFIT_BELOW_THRESHOLD",
                    "SPREAD_TOO_WIDE",
                ],
                "watchlist_recommendation": "near_miss_now",
            },
        ]
    )

    report = runner._build_neg_risk_selector_refresh_report()

    assert [row["family_key"] for row in report] == ["harvey-weinstein-prison-time"]


def test_same_family_passes_under_broad_input_but_fails_under_incomplete_fast_input() -> None:
    runner = _build_runner()
    runner.config.opportunity.min_edge_cents = 0.01
    runner.config.opportunity.min_net_profit_usd = 0.01
    runner.config.opportunity.max_spread_cents = 0.10
    runner.config.opportunity.min_absolute_leg_depth_usd = 0.0
    runner.config.opportunity.max_single_leg_bid = 1.0
    runner.config.opportunity.max_partial_fill_risk = 1.0
    runner.config.opportunity.max_non_atomic_risk = 1.0
    runner.config.opportunity.min_depth_multiple = 1.0
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_group = _event_group("family-parity-incomplete", ["leg-a", "leg-b", "leg-c"])
    fast_books = {
        "leg-a-yes": _book(0.20, 0.21, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-b-yes": _book(0.25, 0.26, bid_size=5000.0, ask_size=5000.0, ts=now),
    }
    broad_books = {
        **fast_books,
        "leg-c-yes": _book(0.30, 0.31, bid_size=5000.0, ask_size=5000.0, ts=now),
    }

    fast_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=fast_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=False,
    )
    broad_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=broad_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=False,
    )
    parity = runner._build_neg_risk_family_parity_payload(
        fast_path_result=fast_payload,
        broad_path_result=broad_payload,
    )

    assert fast_payload["qualified"] is False
    assert broad_payload["qualified"] is True
    assert parity["qualification_outcome_differs"] is True
    assert parity["failure_class"] == "incomplete_family_input"


def test_time_skewed_family_input_is_surfaced_in_parity_audit() -> None:
    runner = _build_runner()
    runner.config.opportunity.min_edge_cents = 0.01
    runner.config.opportunity.min_net_profit_usd = 0.01
    runner.config.opportunity.max_spread_cents = 0.10
    runner.config.opportunity.min_absolute_leg_depth_usd = 0.0
    runner.config.opportunity.max_single_leg_bid = 1.0
    runner.config.opportunity.max_partial_fill_risk = 1.0
    runner.config.opportunity.max_non_atomic_risk = 1.0
    runner.config.opportunity.min_depth_multiple = 1.0
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_group = _event_group("family-parity-skew", ["leg-a", "leg-b", "leg-c"])
    fast_books = {
        "leg-a-yes": _book(0.20, 0.21, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-b-yes": _book(0.25, 0.26, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-c-yes": _book(0.34, 0.35, bid_size=5000.0, ask_size=5000.0, ts=now - timedelta(seconds=5)),
    }
    broad_books = {
        "leg-a-yes": _book(0.20, 0.21, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-b-yes": _book(0.25, 0.26, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-c-yes": _book(0.30, 0.31, bid_size=5000.0, ask_size=5000.0, ts=now),
    }

    fast_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=fast_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=False,
    )
    broad_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=broad_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=False,
    )
    parity = runner._build_neg_risk_family_parity_payload(
        fast_path_result=fast_payload,
        broad_path_result=broad_payload,
    )

    assert float(fast_payload["family_snapshot_time_skew_ms"]) > float(broad_payload["family_snapshot_time_skew_ms"])
    assert float(broad_payload["expected_net_profit_usd"]) > float(fast_payload["expected_net_profit_usd"])
    assert parity["failure_class"] == "time_skewed_family_input"


def test_negative_cached_dead_leg_is_classified_as_side_effect() -> None:
    runner = _build_runner()
    runner.config.opportunity.min_edge_cents = 0.01
    runner.config.opportunity.min_net_profit_usd = 0.01
    runner.config.opportunity.max_spread_cents = 0.10
    runner.config.opportunity.min_absolute_leg_depth_usd = 0.0
    runner.config.opportunity.max_single_leg_bid = 1.0
    runner.config.opportunity.max_partial_fill_risk = 1.0
    runner.config.opportunity.max_non_atomic_risk = 1.0
    runner.config.opportunity.min_depth_multiple = 1.0
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    runner.clob._negative_cache["dead-leg-yes"] = {
        "reason": "no_orderbook",
        "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
    }
    event_group = _event_group("family-parity-negative-cache", ["dead-leg", "live-leg", "hedge-leg"])
    fast_books = {
        "live-leg-yes": _book(0.20, 0.21, bid_size=5000.0, ask_size=5000.0, ts=now),
        "hedge-leg-yes": _book(0.25, 0.26, bid_size=5000.0, ask_size=5000.0, ts=now),
    }
    broad_books = {
        "dead-leg-yes": _book(0.30, 0.31, bid_size=5000.0, ask_size=5000.0, ts=now),
        **fast_books,
    }

    fast_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=fast_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=True,
    )
    broad_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=broad_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=True,
    )
    parity = runner._build_neg_risk_family_parity_payload(
        fast_path_result=fast_payload,
        broad_path_result=broad_payload,
    )

    assert fast_payload["negative_cached_leg_count"] == 1
    assert broad_payload["qualified"] is True
    assert parity["failure_class"] == "negative_cache_side_effect"


def test_true_market_deterioration_classification_when_both_paths_fail_similarly() -> None:
    runner = _build_runner()
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_group = _event_group("family-market-deterioration", ["leg-a", "leg-b"])
    fast_books = {
        "leg-a-yes": _book(0.60, 0.61, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-b-yes": _book(0.41, 0.42, bid_size=5000.0, ask_size=5000.0, ts=now),
    }
    broad_books = {
        "leg-a-yes": _book(0.60, 0.61, bid_size=5000.0, ask_size=5000.0, ts=now),
        "leg-b-yes": _book(0.405, 0.415, bid_size=5000.0, ask_size=5000.0, ts=now),
    }

    fast_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=fast_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=False,
    )
    broad_payload = runner._evaluate_neg_risk_family_path_for_audit(
        event_group=event_group,
        books_by_token=broad_books,
        cycle_started=now,
        source_tags=["productive_history"],
        naturally_selected=True,
        audit_forced=False,
    )
    parity = runner._build_neg_risk_family_parity_payload(
        fast_path_result=fast_payload,
        broad_path_result=broad_payload,
    )

    assert fast_payload["qualified"] is False
    assert broad_payload["qualified"] is False
    assert parity["failure_class"] == "true_market_deterioration"


def test_watched_family_parity_metrics_emit_correctly() -> None:
    runner = _build_runner()
    runner._record_neg_risk_family_parity(
        {
            "family_key": "family-a",
            "source_tags": ["productive_history"],
            "expected_slugs": ["a", "b"],
            "fast_path_result": {"qualification_rejection_reasons": ["NET_PROFIT_BELOW_THRESHOLD"]},
            "broad_path_result": {"qualification_rejection_reasons": []},
            "qualification_outcome_differs": True,
            "failure_class": "incomplete_family_input",
            "metric_deltas": {},
        }
    )
    runner._record_neg_risk_family_parity(
        {
            "family_key": "family-b",
            "source_tags": ["audit_watchlist"],
            "expected_slugs": ["c", "d"],
            "fast_path_result": {"qualification_rejection_reasons": ["EDGE_BELOW_THRESHOLD"]},
            "broad_path_result": {"qualification_rejection_reasons": ["EDGE_BELOW_THRESHOLD"]},
            "qualification_outcome_differs": False,
            "failure_class": "edge_vanished_for_real",
            "metric_deltas": {},
        }
    )

    summary = runner._summarize_neg_risk_family_parity()

    assert summary["watched_families_audited"] == 2
    assert summary["watched_families_fast_vs_broad_mismatch_count"] == 1
    assert summary["watched_families_incomplete_input_count"] == 1
    assert summary["watched_families_net_profit_failure_count"] == 2


def test_family_promoted_to_primary_watchlist_now() -> None:
    runner = _build_runner()
    report = runner._build_neg_risk_productive_family_economics_report.__get__(runner)
    runner._neg_risk_family_parity_payloads = [
        {
            "family_key": "family-productive",
            "source_tags": ["history_productive"],
            "qualification_outcome_differs": False,
            "failure_class": "unclear",
            "fast_path_result": {
                "qualified": True,
                "gross_edge_cents": 0.05,
                "net_edge_cents": 0.03,
                "expected_net_profit_usd": 0.75,
                "required_depth_usd": 10.0,
                "available_depth_usd": 100.0,
                "max_spread_observed": 0.02,
                "partial_fill_risk_score": 0.0,
                "non_atomic_execution_risk_score": 0.2,
                "qualification_rejection_reasons": [],
            },
            "broad_path_result": {},
        }
    ]
    row = report()[0]
    assert row["economics_class"] == "still_productive_now"
    assert row["watchlist_recommendation"] == "viable_now"


def test_family_downgraded_when_edge_vanished_now() -> None:
    runner = _build_runner()
    row = runner._build_neg_risk_productive_family_economics_report.__get__(runner)
    runner._neg_risk_family_parity_payloads = [
        {
            "family_key": "family-dead",
            "source_tags": ["history_productive"],
            "qualification_outcome_differs": False,
            "failure_class": "edge_vanished_for_real",
            "fast_path_result": {
                "qualified": False,
                "gross_edge_cents": 0.01,
                "net_edge_cents": 0.0,
                "expected_net_profit_usd": 0.0,
                "required_depth_usd": 10.0,
                "available_depth_usd": 200.0,
                "max_spread_observed": 0.02,
                "partial_fill_risk_score": 0.0,
                "non_atomic_execution_risk_score": 0.2,
                "qualification_rejection_reasons": ["EDGE_BELOW_THRESHOLD", "NET_PROFIT_BELOW_THRESHOLD"],
            },
            "broad_path_result": {},
        }
    ]
    report_row = row()[0]
    assert report_row["economics_class"] == "edge_vanished_now"
    assert report_row["watchlist_recommendation"] == "downgrade"


def test_family_classified_as_near_miss_watchlist_now() -> None:
    runner = _build_runner()
    runner._neg_risk_family_parity_payloads = [
        {
            "family_key": "family-depth",
            "source_tags": ["history_near_miss"],
            "qualification_outcome_differs": False,
            "failure_class": "concentration_depth_degradation",
            "fast_path_result": {
                "qualified": False,
                "gross_edge_cents": 0.05,
                "net_edge_cents": 0.04,
                "expected_net_profit_usd": 0.8,
                "required_depth_usd": 100.0,
                "available_depth_usd": 20.0,
                "max_spread_observed": 0.02,
                "partial_fill_risk_score": 0.8,
                "non_atomic_execution_risk_score": 0.2,
                "qualification_rejection_reasons": ["INSUFFICIENT_DEPTH"],
            },
            "broad_path_result": {},
        }
    ]
    row = runner._build_neg_risk_productive_family_economics_report()[0]
    assert row["economics_class"] == "depth_insufficient_now"
    assert row["watchlist_recommendation"] == "near_miss_now"


def test_family_classified_as_spread_too_wide_now() -> None:
    runner = _build_runner()
    runner._neg_risk_family_parity_payloads = [
        {
            "family_key": "family-spread",
            "source_tags": ["history_near_miss"],
            "qualification_outcome_differs": False,
            "failure_class": "unclear",
            "fast_path_result": {
                "qualified": False,
                "gross_edge_cents": 0.05,
                "net_edge_cents": 0.04,
                "expected_net_profit_usd": 0.9,
                "required_depth_usd": 10.0,
                "available_depth_usd": 200.0,
                "max_spread_observed": 0.12,
                "partial_fill_risk_score": 0.0,
                "non_atomic_execution_risk_score": 0.2,
                "qualification_rejection_reasons": ["SPREAD_TOO_WIDE"],
            },
            "broad_path_result": {},
        }
    ]
    row = runner._build_neg_risk_productive_family_economics_report()[0]
    assert row["economics_class"] == "spread_too_wide_now"
    assert row["watchlist_recommendation"] == "near_miss_now"


def test_family_parked_when_unclear_and_not_productive() -> None:
    runner = _build_runner()
    runner._neg_risk_family_parity_payloads = [
        {
            "family_key": "family-park",
            "source_tags": ["history_raw_positive"],
            "qualification_outcome_differs": False,
            "failure_class": "unclear",
            "fast_path_result": {
                "qualified": False,
                "gross_edge_cents": 0.01,
                "net_edge_cents": 0.0,
                "expected_net_profit_usd": 0.0,
                "required_depth_usd": 10.0,
                "available_depth_usd": 100.0,
                "max_spread_observed": 0.03,
                "partial_fill_risk_score": 0.2,
                "non_atomic_execution_risk_score": 0.2,
                "qualification_rejection_reasons": [],
            },
            "broad_path_result": {},
        }
    ]
    row = runner._build_neg_risk_productive_family_economics_report()[0]
    assert row["economics_class"] == "unclear"
    assert row["watchlist_recommendation"] == "park"


def test_refreshed_watchlist_summary_emits_correctly() -> None:
    runner = _build_runner()
    report = [
        {"family_key": "family-primary", "watchlist_recommendation": "viable_now"},
        {"family_key": "family-near", "watchlist_recommendation": "near_miss_now"},
        {"family_key": "family-marginal", "watchlist_recommendation": "economically_marginal_now"},
        {"family_key": "family-down", "watchlist_recommendation": "downgrade"},
        {"family_key": "family-park", "watchlist_recommendation": "park"},
    ]
    summary = runner._summarize_neg_risk_selector_refresh(report)
    grouped = runner._build_neg_risk_selector_refresh_watchlist(report)
    assert summary["selector_refresh_families_audited"] == 5
    assert summary["selector_refresh_currently_viable_count"] == 1
    assert summary["selector_refresh_near_miss_count"] == 1
    assert summary["selector_refresh_marginal_count"] == 1
    assert summary["selector_refresh_downgrade_count"] == 1
    assert summary["selector_refresh_park_count"] == 1
    assert [row["family_key"] for row in grouped["viable_now"]] == ["family-primary"]
    assert [row["family_key"] for row in grouped["near_miss_now"]] == ["family-near"]
    assert [row["family_key"] for row in grouped["economically_marginal_now"]] == ["family-marginal"]
    assert [row["family_key"] for row in grouped["downgrade"]] == ["family-down"]
    assert [row["family_key"] for row in grouped["park"]] == ["family-park"]


def test_selector_refresh_expands_beyond_watchlist_to_recent_near_miss_family() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_family_audit_mode_enabled = True
    runner.config.market_data.neg_risk_family_audit_watchlist = ["watched-family"]
    runner.config.market_data.neg_risk_family_audit_budget = 1
    runner.config.market_data.neg_risk_selector_refresh_budget = 2
    runner.market_universe.family_entries["family-near"] = FamilyEntry(
        family_slug="family-near",
        seeded_productive=False,
        productive_outcome_count=0,
        seeded_recent_qualified=False,
        qualified_count=0,
        last_qualified_at=None,
        seeded_recent_near_miss=True,
        near_miss_count=1,
        last_near_miss_at=datetime.now(timezone.utc),
        raw_signal_count=1,
        last_raw_signal_at=datetime.now(timezone.utc),
        active=True,
    )
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    event_groups = [
        _event_group("watched-family", ["watched-leg"]),
        _event_group("family-near", ["near-leg-a", "near-leg-b"]),
    ]

    with patch.object(runner.market_universe, "select_neg_risk_groups_for_recompute", return_value=([], {"families_considered": 0})):
        selected, metrics, _naturally_selected, audit_forced_keys = runner._select_neg_risk_event_groups_for_scan(
            event_groups=event_groups,
            changed_market_slugs=set(),
            cycle_started=now,
        )

    assert {group["event_slug"] for group in selected} == {"watched-family", "family-near"}
    assert "watched-family" in audit_forced_keys
    assert "family-near" in audit_forced_keys
    assert metrics["families_considered"] >= 2


def test_slug_normalization_mismatch_still_matches_watchlist() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_family_audit_watchlist = ["HOW_MANY_FED_RATE_CUTS_IN_2026"]
    watched_group = _event_group(
        "how-many-fed-rate-cuts-in-2026",
        ["will-4-fed-rate-cuts-happen-in-2026", "will-5-fed-rate-cuts-happen-in-2026"],
    )

    assert runner._neg_risk_audit_watchlist_match(watched_group) is True


def test_family_key_mismatch_reconciles_via_market_slug() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_family_audit_watchlist = ["will-4-fed-rate-cuts-happen-in-2026"]
    watched_group = _event_group(
        "how-many-fed-rate-cuts-in-2026",
        ["will-4-fed-rate-cuts-happen-in-2026", "will-5-fed-rate-cuts-happen-in-2026"],
    )

    match = runner._neg_risk_match_watch_descriptor(
        watched_group,
        runner._neg_risk_watchlist_descriptors()[0],
    )

    assert match is not None
    assert match["family_key"] == "how-many-fed-rate-cuts-in-2026"
    assert match["matched_slugs_count"] == 1


def test_productive_watchlist_entry_reconciles_to_discovered_event_group() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_family_audit_watchlist = []
    runner.market_universe.seed_productive_markets(["will-4-fed-rate-cuts-happen-in-2026"])
    watched_group = _event_group(
        "how-many-fed-rate-cuts-in-2026",
        ["will-4-fed-rate-cuts-happen-in-2026", "will-5-fed-rate-cuts-happen-in-2026"],
    )

    assert runner._neg_risk_audit_watchlist_match(watched_group) is True


def test_forced_family_injection_causes_family_consideration() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_family_audit_mode_enabled = True
    runner.config.market_data.neg_risk_family_audit_watchlist = ["will-4-fed-rate-cuts-happen-in-2026"]
    runner.config.market_data.neg_risk_family_audit_budget = 2
    now = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    current_groups = [_event_group("unrelated-family", ["other-market"])]
    raw_fed_event = _raw_neg_risk_event(
        "how-many-fed-rate-cuts-in-2026",
        ["will-4-fed-rate-cuts-happen-in-2026", "will-5-fed-rate-cuts-happen-in-2026"],
    )

    with patch("src.runtime.runner.fetch_events", return_value=[raw_fed_event]):
        reconciled_groups, diagnostics = runner._reconcile_neg_risk_watchlist_groups(current_groups)

    assert len(reconciled_groups) == 2
    assert any(group["event_slug"] == "how-many-fed-rate-cuts-in-2026" for group in reconciled_groups)
    assert any(
        diagnostic["match_status"] == "bounded_rescan_injected"
        and diagnostic["matched_family_key"] == "how-many-fed-rate-cuts-in-2026"
        for diagnostic in diagnostics
    )

    selected, metrics, _natural_keys, audit_forced_keys = runner._select_neg_risk_event_groups_for_scan(
        event_groups=reconciled_groups,
        changed_market_slugs=set(),
        cycle_started=now,
    )

    assert any(group["event_slug"] == "how-many-fed-rate-cuts-in-2026" for group in selected)
    assert "how-many-fed-rate-cuts-in-2026" in audit_forced_keys
    assert metrics["families_considered"] >= 1


def test_unmatched_watchlist_diagnostics_emit_correctly() -> None:
    runner = _build_runner()
    runner.config.market_data.neg_risk_family_audit_watchlist = ["ghost-family-market"]
    current_groups = [_event_group("balance-of-power-2026-midterms", ["2026-balance-of-power-d-senate-d-house-949"])]

    with patch("src.runtime.runner.fetch_events", return_value=[]):
        _reconciled_groups, diagnostics = runner._reconcile_neg_risk_watchlist_groups(current_groups)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["watchlist_family_key"] == "ghost-family-market"
    assert diagnostic["matched_slugs_count"] == 0
    assert diagnostic["mismatch_reason"] == "no_active_neg_risk_family_match_found"
    assert diagnostic["match_status"] == "unmatched"
