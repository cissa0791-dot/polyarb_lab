from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MarketDataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gamma_host: str = "https://gamma-api.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"
    market_limit: int = 200
    scan_interval_sec: float = 2.0
    stale_book_sec: float = 3.0


class OpportunityConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_edge_cents: float = 0.03
    fee_buffer_cents: float = 0.01
    slippage_buffer_cents: float = 0.01
    vwap_depth_cap: float = 100.0
    min_depth_multiple: float = 3.0
    max_spread_cents: float = 0.08
    max_latency_ms: int = 1500
    min_net_profit_usd: float = 0.50
    max_partial_fill_risk: float = 0.65
    max_non_atomic_risk: float = 0.60


class PaperConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    starting_cash: float = 10000.0
    max_notional_per_arb: float = 100.0
    cancel_unfilled_after_sec: float = 5.0
    default_order_ttl_sec: float = 10.0
    take_profit_usd: float = 1.0
    stop_loss_usd: float = 1.0
    max_holding_sec: float = 300.0
    flatten_on_run_end: bool = False


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    max_order_notional_usd: float = 100.0
    max_market_exposure_usd: float = 250.0
    max_total_exposure_usd: float = 750.0
    max_daily_loss_usd: float = 50.0
    max_consecutive_losses: int = 4
    max_open_positions: int = 8
    min_liquidity_usd: float = 30.0
    min_score: float = 65.0
    whitelist_markets: list[str] = Field(default_factory=list)
    require_human_confirmation_for_live: bool = True
    halt_on_data_errors: bool = True


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    log_level: str = "INFO"
    json_logs: bool = True
    emit_console_summary: bool = True


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sqlite_url: str = "sqlite:///data/processed/paper.db"
    raw_snapshot_retention_days: int = 30


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "paper"
    live_enabled: bool = False
    dry_run: bool = True
    max_live_order_usd: float = 25.0


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    opportunity: OpportunityConfig = Field(default_factory=OpportunityConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
