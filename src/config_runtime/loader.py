from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config_runtime.models import RuntimeConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {file_path}")
    return payload


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value if item is not None]
    return value


def _normalize_legacy_settings(raw: dict[str, Any]) -> dict[str, Any]:
    if any(key in raw for key in ("market_data", "opportunity", "paper", "risk", "monitoring", "storage", "execution")):
        return raw

    normalized = _drop_none(
        {
            "market_data": {
                "gamma_host": raw.get("gamma_host"),
                "clob_host": raw.get("clob_host"),
                "market_limit": raw.get("market_limit"),
                "scan_interval_sec": raw.get("scan_interval_sec"),
                "stale_book_sec": raw.get("stale_book_sec"),
                "discovery_refresh_interval_sec": raw.get("discovery_refresh_interval_sec"),
                "backstop_full_rescan_interval_sec": raw.get("backstop_full_rescan_interval_sec"),
                "backstop_refresh_market_budget": raw.get("backstop_refresh_market_budget"),
                "force_refresh_productive_families_every_n_cycles": raw.get("force_refresh_productive_families_every_n_cycles"),
                "hot_refresh_interval_sec": raw.get("hot_refresh_interval_sec"),
                "warm_refresh_interval_sec": raw.get("warm_refresh_interval_sec"),
                "cold_refresh_interval_sec": raw.get("cold_refresh_interval_sec"),
                "hot_market_count": raw.get("hot_market_count"),
                "warm_market_count": raw.get("warm_market_count"),
                "cold_market_count": raw.get("cold_market_count"),
                "near_miss_retry_interval_sec": raw.get("near_miss_retry_interval_sec"),
                "neg_risk_family_due_refresh_interval_sec": raw.get("neg_risk_family_due_refresh_interval_sec"),
                "neg_risk_family_backstop_every_n_cycles": raw.get("neg_risk_family_backstop_every_n_cycles"),
                "neg_risk_family_backstop_budget": raw.get("neg_risk_family_backstop_budget"),
                "neg_risk_family_audit_mode_enabled": raw.get("neg_risk_family_audit_mode_enabled"),
                "neg_risk_family_audit_watchlist": raw.get("neg_risk_family_audit_watchlist"),
                "neg_risk_family_audit_budget": raw.get("neg_risk_family_audit_budget"),
                "neg_risk_selector_refresh_budget": raw.get("neg_risk_selector_refresh_budget"),
                "neg_risk_condition_monitor_mode_enabled": raw.get("neg_risk_condition_monitor_mode_enabled"),
                "neg_risk_condition_monitor_watchlist": raw.get("neg_risk_condition_monitor_watchlist"),
                "neg_risk_watchlist_reconciliation_event_limit": raw.get("neg_risk_watchlist_reconciliation_event_limit"),
                "recompute_midpoint_delta_cents": raw.get("recompute_midpoint_delta_cents"),
                "recompute_spread_delta_cents": raw.get("recompute_spread_delta_cents"),
                "recompute_top_depth_delta_ratio": raw.get("recompute_top_depth_delta_ratio"),
                "recompute_inventory_delta_shares": raw.get("recompute_inventory_delta_shares"),
                "no_orderbook_negative_cache_ttl_sec": raw.get("no_orderbook_negative_cache_ttl_sec"),
                "invalid_token_retry_interval_sec": raw.get("invalid_token_retry_interval_sec"),
                "enable_hot_tier_websocket": raw.get("enable_hot_tier_websocket"),
                "hot_tier_websocket_poll_timeout_sec": raw.get("hot_tier_websocket_poll_timeout_sec"),
                "hot_tier_websocket_stale_sec": raw.get("hot_tier_websocket_stale_sec"),
                "discovery_use_simplified_markets": raw.get("discovery_use_simplified_markets"),
            },
            "opportunity": {
                "min_edge_cents": raw.get("min_edge_cents"),
                "fee_buffer_cents": raw.get("fee_buffer_cents"),
                "slippage_buffer_cents": raw.get("slippage_buffer_cents"),
                "vwap_depth_cap": raw.get("vwap_depth_cap"),
                "min_depth_multiple": raw.get("min_depth_multiple"),
                "max_spread_cents": raw.get("max_spread_cents"),
                "max_latency_ms": raw.get("max_latency_ms"),
                "min_net_profit_usd": raw.get("min_net_profit_usd"),
                "max_partial_fill_risk": raw.get("max_partial_fill_risk"),
                "max_non_atomic_risk": raw.get("max_non_atomic_risk"),
                "min_absolute_leg_depth_usd": raw.get("min_absolute_leg_depth_usd"),
                "max_single_leg_bid": raw.get("max_single_leg_bid"),
                "min_sized_notional_usd": raw.get("min_sized_notional_usd"),
            },
            "paper": {
                "starting_cash": raw.get("starting_cash"),
                "max_notional_per_arb": raw.get("max_notional_per_arb"),
                "take_profit_usd": raw.get("take_profit_usd"),
                "stop_loss_usd": raw.get("stop_loss_usd"),
                "max_holding_sec": raw.get("max_holding_sec"),
                "flatten_on_run_end": raw.get("flatten_on_run_end"),
                "edge_decay_bid_delta": raw.get("edge_decay_bid_delta"),
                "basket_dominance_threshold": raw.get("basket_dominance_threshold"),
                "basket_drawdown_exit_threshold": raw.get("basket_drawdown_exit_threshold"),
                "basket_unrealized_pnl_floor": raw.get("basket_unrealized_pnl_floor"),
                "idle_hold_release_check_sec": raw.get("idle_hold_release_check_sec"),
                "idle_hold_release_max_repricing_events": raw.get("idle_hold_release_max_repricing_events"),
                "idle_hold_release_max_abs_unrealized_pnl": raw.get("idle_hold_release_max_abs_unrealized_pnl"),
                "idle_hold_release_max_drawdown": raw.get("idle_hold_release_max_drawdown"),
            },
            "monitoring": {
                "log_level": raw.get("log_level"),
            },
            "storage": {
                "sqlite_url": raw.get("sqlite_url"),
            },
            "execution": {
                "maker_quote_min_expected_net_edge_cents": raw.get("maker_quote_min_expected_net_edge_cents"),
                "maker_quote_max_fair_value_drift_cents": raw.get("maker_quote_max_fair_value_drift_cents"),
                "maker_quote_max_age_sec": raw.get("maker_quote_max_age_sec"),
                "maker_quote_inventory_soft_limit_shares": raw.get("maker_quote_inventory_soft_limit_shares"),
                "maker_quote_inventory_hard_limit_shares": raw.get("maker_quote_inventory_hard_limit_shares"),
                "maker_quote_inventory_skew_cents": raw.get("maker_quote_inventory_skew_cents"),
            },
            "risk": {
                "max_open_positions": raw.get("max_open_positions"),
            },
        }
    )
    # Preserve maker_mm section if present as a nested dict in the flat-format file.
    if "maker_mm" in raw and isinstance(raw["maker_mm"], dict):
        normalized["maker_mm"] = raw["maker_mm"]
    return normalized


def load_runtime_config(settings_path: str | Path = "config/settings.yaml") -> RuntimeConfig:
    raw_settings = _read_yaml(settings_path)
    normalized = _normalize_legacy_settings(raw_settings)
    return RuntimeConfig.model_validate(normalized)
