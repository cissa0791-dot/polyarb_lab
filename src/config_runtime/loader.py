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

    return _drop_none(
        {
            "market_data": {
                "gamma_host": raw.get("gamma_host"),
                "clob_host": raw.get("clob_host"),
                "market_limit": raw.get("market_limit"),
                "scan_interval_sec": raw.get("scan_interval_sec"),
                "stale_book_sec": raw.get("stale_book_sec"),
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
            },
            "paper": {
                "starting_cash": raw.get("starting_cash"),
                "max_notional_per_arb": raw.get("max_notional_per_arb"),
                "take_profit_usd": raw.get("take_profit_usd"),
                "stop_loss_usd": raw.get("stop_loss_usd"),
                "max_holding_sec": raw.get("max_holding_sec"),
                "flatten_on_run_end": raw.get("flatten_on_run_end"),
            },
            "monitoring": {
                "log_level": raw.get("log_level"),
            },
            "storage": {
                "sqlite_url": raw.get("sqlite_url"),
            },
            "execution": {},
            "risk": {},
        }
    )


def load_runtime_config(settings_path: str | Path = "config/settings.yaml") -> RuntimeConfig:
    raw_settings = _read_yaml(settings_path)
    normalized = _normalize_legacy_settings(raw_settings)
    return RuntimeConfig.model_validate(normalized)
