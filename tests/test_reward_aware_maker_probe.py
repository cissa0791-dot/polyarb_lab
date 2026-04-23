"""
tests/test_reward_aware_maker_probe.py

Regression tests for reward_aware_single_market_maker_probe.

Coverage:
  1. empty_rewarded_universe        — discovery returns nothing → zero counts
  2. fee_disabled_filtering         — fee-disabled markets are excluded
  3. no_reward_filtering            — fee-enabled but no reward → excluded
  4. basic_ev_decomposition         — known inputs → correct EV components
  5. positive_raw_candidate_path    — EV > 0 → POSITIVE_RAW_EV classification
  6. rejection_path                 — EV <= 0 → NEGATIVE_RAW_EV + codes
  7. summary_report_generation      — full batch → summary fields populated

Hard constraints:
  - No network calls. All Gamma API calls mocked.
  - No order submission. No mainline imports.
  - Tests must pass without any live data.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup (allow running with pytest from repo root)
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _TESTS_DIR.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.discovery import (
    RawRewardedMarket,
    _is_fee_enabled_rewarded,
    _extract_daily_rate,
    discover_fee_enabled_rewarded_markets,
    discovery_summary,
)
from research_lines.reward_aware_maker_probe.modules.ev_model import (
    MarketEVResult,
    evaluate_market_ev,
    evaluate_batch,
    build_ev_summary,
    ECON_POSITIVE_RAW_EV,
    ECON_NEGATIVE_RAW_EV,
    ECON_REJECTED_NO_BOOK,
    ECON_REJECTED_NO_REWARD,
    ECON_REJECTED_SPREAD_TOO_WIDE,
    RC_NO_USABLE_BOOK,
    RC_REWARD_METADATA_MISSING,
    RC_SPREAD_EXCEEDS_REWARD_MAX,
    RC_NON_POSITIVE_NET_EV,
    MAKER_REBATE_CONTRIBUTION,
)
from research_lines.reward_aware_maker_probe.modules.paper_logger import (
    ProbeResult,
    log_probe,
    make_probe_id,
    load_probe_index,
    load_latest_probe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_market(
    market_slug: str = "test-market",
    event_slug: str = "test-event",
    best_bid: float = 0.45,
    best_ask: float = 0.55,
    reward_daily_rate_usdc: float = 50.0,
    rewards_min_size: float = 10.0,
    rewards_max_spread_cents: float = 5.0,  # 5 cents → 0.05 in price space
    fees_enabled: bool = True,
) -> RawRewardedMarket:
    return RawRewardedMarket(
        market_id="mkt-001",
        market_slug=market_slug,
        event_slug=event_slug,
        event_id="evt-001",
        category="crypto",
        question="Will X happen?",
        yes_token_id="tok-yes",
        no_token_id="tok-no",
        fees_enabled=fees_enabled,
        enable_orderbook=True,
        best_bid=best_bid,
        best_ask=best_ask,
        rewards_min_size=rewards_min_size,
        rewards_max_spread_cents=rewards_max_spread_cents,
        reward_daily_rate_usdc=reward_daily_rate_usdc,
        clob_rewards_raw=[{"rewardsDailyRate": reward_daily_rate_usdc}],
        volume_24hr=1000.0,
        liquidity=5000.0,
        fetched_at=datetime.now(timezone.utc),
    )


def _gamma_market_dict(
    fees_enabled: bool = True,
    enable_orderbook: bool = True,
    daily_rate: float = 50.0,
    min_size: float = 10.0,
    max_spread: float = 5.0,
    best_bid: float = 0.45,
    best_ask: float = 0.55,
) -> dict[str, Any]:
    """Build a minimal Gamma market payload dict."""
    return {
        "id": "mkt-001",
        "slug": "test-market",
        "question": "Will X happen?",
        "active": True,
        "closed": False,
        "fees_enabled": fees_enabled,
        "feesEnabled": fees_enabled,
        "enable_orderbook": enable_orderbook,
        "enableOrderBook": enable_orderbook,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "rewardsMinSize": min_size,
        "rewardsMaxSpread": max_spread,
        "clobRewards": [{"rewardsDailyRate": daily_rate}],
        "clobTokenIds": ["tok-yes", "tok-no"],
    }


def _gamma_event_with_market(market_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "evt-001",
        "slug": "test-event",
        "category": "crypto",
        "active": True,
        "closed": False,
        "markets": [market_dict],
    }


# ---------------------------------------------------------------------------
# Test 1: Empty rewarded universe
# ---------------------------------------------------------------------------

def test_empty_rewarded_universe():
    """Discovery returning no markets → summary shows zeros, EV batch is empty."""
    empty: list[RawRewardedMarket] = []
    disc = discovery_summary(empty)
    assert disc["fee_enabled_rewarded_market_count"] == 0
    assert disc["with_usable_book"] == 0

    ev_results = evaluate_batch(empty)
    assert ev_results == []

    summary = build_ev_summary(empty, ev_results)
    assert summary["rewarded_market_count"] == 0
    assert summary["positive_raw_maker_candidates"] == 0
    assert summary["best_raw_candidate"] is None


# ---------------------------------------------------------------------------
# Test 2: Fee-disabled filtering
# ---------------------------------------------------------------------------

def test_fee_disabled_filtering():
    """Markets with fees_enabled=False must be excluded by _is_fee_enabled_rewarded."""
    fee_disabled = _gamma_market_dict(fees_enabled=False)
    assert not _is_fee_enabled_rewarded(fee_disabled)

    fee_enabled = _gamma_market_dict(fees_enabled=True)
    assert _is_fee_enabled_rewarded(fee_enabled)


def test_fee_disabled_excluded_from_discovery():
    """CLOB market with fees_enabled=False → no markets discovered."""
    mkt = _gamma_market_dict(fees_enabled=False)

    with patch(
        "research_lines.reward_aware_maker_probe.modules.discovery._fetch_all_clob_rewards_markets",
        return_value=[mkt],
    ):
        markets = discover_fee_enabled_rewarded_markets()
    assert markets == []


# ---------------------------------------------------------------------------
# Test 3: No-reward filtering
# ---------------------------------------------------------------------------

def test_no_reward_filtering():
    """Fee-enabled markets with zero reward rate are excluded."""
    no_reward = _gamma_market_dict(fees_enabled=True, daily_rate=0.0)
    assert not _is_fee_enabled_rewarded(no_reward)

    no_min_size = _gamma_market_dict(fees_enabled=True, min_size=0.0)
    assert not _is_fee_enabled_rewarded(no_min_size)

    no_max_spread = _gamma_market_dict(fees_enabled=True, max_spread=0.0)
    assert not _is_fee_enabled_rewarded(no_max_spread)


def test_no_reward_excluded_from_discovery():
    """Zero-reward market is discovered but EV model classifies it as REJECTED_NO_REWARD."""
    mkt = _gamma_market_dict(fees_enabled=True, daily_rate=0.0)

    with patch(
        "research_lines.reward_aware_maker_probe.modules.discovery._fetch_all_clob_rewards_markets",
        return_value=[mkt],
    ), patch(
        "research_lines.reward_aware_maker_probe.modules.discovery._fetch_clob_book",
        return_value=(0.45, 0.55),
    ):
        markets = discover_fee_enabled_rewarded_markets()

    # Discovery no longer gates on reward rate; EV model handles that classification.
    assert len(markets) == 1
    result = evaluate_market_ev(markets[0])
    assert result.economics_class == ECON_REJECTED_NO_REWARD


# ---------------------------------------------------------------------------
# Test 4: Basic EV decomposition
# ---------------------------------------------------------------------------

def test_basic_ev_decomposition():
    """
    Known inputs → verify all EV component signs and formula.

    Setup:
      best_bid=0.45, best_ask=0.55 → spread=0.10, mid=0.50
      rewards_max_spread_cents=20.0 → 0.20 price → spread 0.10 <= 0.20 → FITS
      rewards_min_size=10.0 shares
      reward_daily_rate_usdc=50.0

    Expected (probe model parameters):
      quoted_spread = 0.10
      quote_size = 10.0
      fill_prob = 0.20 (base only, probe mode)

      spread_capture = 10.0 * 0.10 * 0.20 * 0.5 = 0.10
      reward_contribution = 50.0 * 0.05 * 1.0 = 2.50
      rebate = 0.0
      adverse_sel = 10.0 * 0.10 * 0.30 = 0.30
      inventory = 10.0 * 0.10 * 0.20 = 0.20
      net_ev = 0.10 + 2.50 + 0.0 - 0.30 - 0.20 = 2.10
    """
    mkt = _make_raw_market(
        best_bid=0.45,
        best_ask=0.55,
        reward_daily_rate_usdc=50.0,
        rewards_min_size=10.0,
        rewards_max_spread_cents=20.0,
    )
    result = evaluate_market_ev(mkt)

    assert result.quoted_spread == pytest.approx(0.10, abs=1e-6)
    assert result.midpoint == pytest.approx(0.50, abs=1e-6)
    assert result.estimated_spread_capture == pytest.approx(0.10, abs=1e-6)
    assert result.estimated_reward_contribution == pytest.approx(2.50, abs=1e-6)
    assert result.estimated_maker_rebate_contribution == pytest.approx(MAKER_REBATE_CONTRIBUTION, abs=1e-9)
    assert result.adverse_selection_penalty == pytest.approx(0.30, abs=1e-6)
    assert result.inventory_penalty == pytest.approx(0.20, abs=1e-6)
    assert result.reward_adjusted_raw_ev == pytest.approx(2.10, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 5: Positive raw candidate path
# ---------------------------------------------------------------------------

def test_positive_raw_candidate_path():
    """Market with sufficient reward rate → POSITIVE_RAW_EV classification."""
    mkt = _make_raw_market(
        best_bid=0.48,
        best_ask=0.52,
        reward_daily_rate_usdc=200.0,    # high reward
        rewards_min_size=5.0,
        rewards_max_spread_cents=10.0,   # 0.10 price → spread 0.04 fits
    )
    result = evaluate_market_ev(mkt)
    assert result.economics_class == ECON_POSITIVE_RAW_EV
    assert result.reward_adjusted_raw_ev > 0.0
    assert result.rejection_reason_codes == []
    assert result.watchlist_recommendation == "RESEARCH_CANDIDATE"


# ---------------------------------------------------------------------------
# Test 6: Rejection path — no book, no reward, spread too wide, net negative
# ---------------------------------------------------------------------------

def test_rejection_no_book():
    """Market with no usable book (best_bid=None) → REJECTED_NO_BOOK."""
    mkt = _make_raw_market()
    mkt.best_bid = None
    result = evaluate_market_ev(mkt)
    assert result.economics_class == ECON_REJECTED_NO_BOOK
    assert RC_NO_USABLE_BOOK in result.rejection_reason_codes
    assert result.reward_adjusted_raw_ev == 0.0


def test_rejection_crossed_book():
    """Crossed book (bid >= ask) → REJECTED_NO_BOOK."""
    mkt = _make_raw_market(best_bid=0.60, best_ask=0.40)
    result = evaluate_market_ev(mkt)
    assert result.economics_class == ECON_REJECTED_NO_BOOK
    assert RC_NO_USABLE_BOOK in result.rejection_reason_codes


def test_rejection_no_reward():
    """Market with zero reward rate → REJECTED_NO_REWARD."""
    mkt = _make_raw_market(reward_daily_rate_usdc=0.0)
    result = evaluate_market_ev(mkt)
    assert result.economics_class == ECON_REJECTED_NO_REWARD
    assert RC_REWARD_METADATA_MISSING in result.rejection_reason_codes


def test_rejection_spread_too_wide():
    """Current spread > rewardsMaxSpread → quotes at max spread instead of rejecting."""
    # spread = 0.55 - 0.45 = 0.10 price
    # max_spread_cents = 5.0 → 0.05 price → market is wide, quote at 0.05
    mkt = _make_raw_market(
        best_bid=0.45,
        best_ask=0.55,
        rewards_max_spread_cents=5.0,  # 0.05 price limit
    )
    result = evaluate_market_ev(mkt)
    # Wide markets are now handled by quoting inside the spread; not rejected.
    assert result.economics_class != ECON_REJECTED_SPREAD_TOO_WIDE
    assert round(result.quoted_spread, 6) == round(5.0 / 100.0, 6)


def test_rejection_net_negative_ev():
    """Market with very small reward → negative net EV → NEGATIVE_RAW_EV."""
    # High min_size → large penalties, tiny reward → net negative
    mkt = _make_raw_market(
        best_bid=0.45,
        best_ask=0.55,
        reward_daily_rate_usdc=0.10,    # near-zero reward
        rewards_min_size=100.0,         # large position → large penalties
        rewards_max_spread_cents=20.0,
    )
    result = evaluate_market_ev(mkt)
    assert result.economics_class == ECON_NEGATIVE_RAW_EV
    assert RC_NON_POSITIVE_NET_EV in result.rejection_reason_codes
    assert result.reward_adjusted_raw_ev < 0.0


# ---------------------------------------------------------------------------
# Test 7: Summary / report generation
# ---------------------------------------------------------------------------

def test_summary_report_generation():
    """Batch with mix of positive and negative → summary fields correct."""
    positive = _make_raw_market(
        market_slug="pos-market",
        reward_daily_rate_usdc=200.0,
        rewards_min_size=5.0,
        best_bid=0.48,
        best_ask=0.52,
        rewards_max_spread_cents=10.0,
    )
    negative = _make_raw_market(
        market_slug="neg-market",
        reward_daily_rate_usdc=0.05,
        rewards_min_size=100.0,
        best_bid=0.45,
        best_ask=0.55,
        rewards_max_spread_cents=20.0,
    )
    markets = [positive, negative]
    results = evaluate_batch(markets)
    summary = build_ev_summary(markets, results)

    assert summary["fee_enabled_rewarded_market_count"] == 2
    assert summary["rewarded_market_count"] == 2
    assert summary["websocket_books_collected"] == 0
    assert "reward_aware_single_market_maker" in summary["raw_candidates_by_family"]
    assert summary["positive_raw_maker_candidates"] >= 1
    assert summary["best_raw_candidate"] == "pos-market"


def test_summary_all_positive_pool_empty():
    """All-rejected batch → positive_raw_maker_candidates=0, best_raw_candidate=None."""
    no_book = _make_raw_market(market_slug="no-book")
    no_book.best_bid = None
    results = evaluate_batch([no_book])
    summary = build_ev_summary([no_book], results)
    assert summary["positive_raw_maker_candidates"] == 0
    assert summary["best_raw_candidate"] is None


def test_probe_result_serialization_to_dict():
    """ProbeResult.to_dict() produces the required output shape."""
    mkt = _make_raw_market(
        reward_daily_rate_usdc=100.0,
        rewards_min_size=10.0,
        best_bid=0.48,
        best_ask=0.52,
        rewards_max_spread_cents=10.0,
    )
    results = evaluate_batch([mkt])
    probe = ProbeResult(
        probe_id="RAMM_probe_test",
        probe_timestamp=datetime.now(timezone.utc),
        raw_markets=[mkt],
        ev_results=results,
        probe_config={"gamma_host": "https://gamma-api.polymarket.com"},
    )
    d = probe.to_dict()

    # Required top-level fields
    assert "probe_id" in d
    assert "probe_timestamp" in d
    assert "probe_version" in d
    assert "summary" in d
    assert "markets" in d

    # Required summary fields
    s = d["summary"]
    assert "rewarded_market_count" in s
    assert "fee_enabled_rewarded_market_count" in s
    assert "websocket_books_collected" in s
    assert "raw_candidates_by_family" in s
    assert "positive_raw_maker_candidates" in s
    assert "best_raw_candidate" in s

    # Required per-market fields
    assert len(d["markets"]) == 1
    m = d["markets"][0]
    required_market_fields = [
        "market_slug", "event_slug", "category", "fees_enabled",
        "reward_config_summary", "best_bid", "best_ask", "midpoint",
        "tick_size", "quoted_spread",
        "estimated_spread_capture", "estimated_reward_contribution",
        "estimated_maker_rebate_contribution",
        "inventory_penalty", "adverse_selection_penalty",
        "reward_adjusted_raw_ev", "economics_class",
        "rejection_reason_codes", "watchlist_recommendation",
    ]
    for f in required_market_fields:
        assert f in m, f"Missing required field: {f}"


def test_paper_logger_writes_and_reads(tmp_path: Path):
    """log_probe writes files; load_probe_index and load_latest_probe read them back."""
    mkt = _make_raw_market()
    results = evaluate_batch([mkt])
    probe = ProbeResult(
        probe_id="RAMM_probe_test",
        probe_timestamp=datetime.now(timezone.utc),
        raw_markets=[mkt],
        ev_results=results,
        probe_config={},
    )
    probe_path = log_probe(probe, output_dir=tmp_path)

    assert probe_path.exists()
    assert (tmp_path / "latest_probe.json").exists()
    assert (tmp_path / "probe_index.json").exists()

    index = load_probe_index(tmp_path)
    assert len(index) == 1
    assert index[0]["probe_id"] == "RAMM_probe_test"

    latest = load_latest_probe(tmp_path)
    assert latest is not None
    assert latest["probe_id"] == "RAMM_probe_test"
    assert "markets" in latest


# ---------------------------------------------------------------------------
# Test: _extract_daily_rate helper
# ---------------------------------------------------------------------------

def test_extract_daily_rate_sums_all_entries():
    market = {
        "clobRewards": [
            {"rewardsDailyRate": 30.0},
            {"rewardsDailyRate": 20.0},
        ]
    }
    assert _extract_daily_rate(market) == pytest.approx(50.0, abs=1e-9)


def test_extract_daily_rate_empty():
    assert _extract_daily_rate({}) == pytest.approx(0.0, abs=1e-9)
    assert _extract_daily_rate({"clobRewards": []}) == pytest.approx(0.0, abs=1e-9)
