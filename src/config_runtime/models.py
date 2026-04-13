from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MarketDataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gamma_host: str = "https://gamma-api.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"
    market_limit: int = 200
    scan_interval_sec: float = 2.0
    stale_book_sec: float = 3.0
    discovery_refresh_interval_sec: float = 60.0
    backstop_full_rescan_interval_sec: float = 90.0
    backstop_refresh_market_budget: int = 48
    force_refresh_productive_families_every_n_cycles: int = 2
    hot_refresh_interval_sec: float = 2.0
    warm_refresh_interval_sec: float = 8.0
    cold_refresh_interval_sec: float = 30.0
    hot_market_count: int = 24
    warm_market_count: int = 64
    cold_market_count: int = 200
    near_miss_retry_interval_sec: float = 20.0
    neg_risk_family_due_refresh_interval_sec: float = 20.0
    neg_risk_family_backstop_every_n_cycles: int = 2
    neg_risk_family_backstop_budget: int = 8
    neg_risk_family_audit_mode_enabled: bool = False
    neg_risk_family_audit_watchlist: list[str] = Field(default_factory=list)
    neg_risk_family_audit_budget: int = 8
    neg_risk_selector_refresh_budget: int = 12
    neg_risk_condition_monitor_mode_enabled: bool = False
    neg_risk_condition_monitor_watchlist: list[str] = Field(default_factory=list)
    neg_risk_watchlist_reconciliation_event_limit: int = 500
    recompute_midpoint_delta_cents: float = 0.01
    recompute_spread_delta_cents: float = 0.01
    recompute_top_depth_delta_ratio: float = 0.25
    recompute_inventory_delta_shares: float = 10.0
    no_orderbook_negative_cache_ttl_sec: float = 300.0
    invalid_token_retry_interval_sec: float = 900.0
    enable_hot_tier_websocket: bool = False
    hot_tier_websocket_poll_timeout_sec: float = 0.25
    hot_tier_websocket_stale_sec: float = 5.0
    discovery_use_simplified_markets: bool = True


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
    # Absolute per-leg depth floor: each leg must have at least this many USD of
    # liquidity available on the correct side of the book before a candidate is
    # considered executable.  0.0 disables the check (backward-compatible default).
    min_absolute_leg_depth_usd: float = 0.0
    # Single-leg concentration ceiling: baskets where any individual leg's top-of-book
    # bid exceeds this threshold are rejected.  Eliminates dominant-probability-outcome
    # baskets where one leg absorbs an outsized share of notional.  1.0 disables (default).
    max_single_leg_bid: float = 1.0
    # Post-sizing viability floor: the notional produced by the SizingEngine must
    # be at least this large or the candidate is rejected before reaching the paper
    # broker.  0.0 disables the check (backward-compatible default).
    min_sized_notional_usd: float = 0.0


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
    inter_cycle_reset: bool = False
    edge_decay_bid_delta: float = 0.0
    # Basket-level dominance gate for EDGE_DECAY exits.
    # 0.0 = gate disabled (all EDGE_DECAY exits proceed unchanged).
    # > 0.0 = gate active: exit confirmed only if trigger IS dominant loss leg
    #         OR dominant_loss_leg_share >= this threshold (Path A),
    #         OR basket deterioration overrides via Path B thresholds.
    basket_dominance_threshold: float = 0.0
    # Path B override: confirm exit if basket has drawn down >= this many USD from peak.
    # 0.0 = not used.
    basket_drawdown_exit_threshold: float = 0.0
    # Path B override: confirm exit if basket unrealized PnL <= this value (USD, negative).
    # 0.0 = not used.
    basket_unrealized_pnl_floor: float = 0.0
    # Idle-hold early release for clearly inert baskets.
    # 0.0 = disabled; > 0.0 = earliest basket age checkpoint for release eligibility.
    idle_hold_release_check_sec: float = 0.0
    # Maximum total basket repricing events allowed after entry for idle release.
    idle_hold_release_max_repricing_events: int = 1
    # Basket abs(unrealized PnL) must stay at or below this value (USD).
    idle_hold_release_max_abs_unrealized_pnl: float = 0.01
    # Basket peak-to-current drawdown must stay at or below this value (USD).
    idle_hold_release_max_drawdown: float = 0.01


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
    maker_quote_min_expected_net_edge_cents: float = 0.01
    maker_quote_max_fair_value_drift_cents: float = 0.03
    maker_quote_max_age_sec: float = 45.0
    maker_quote_inventory_soft_limit_shares: float = 100.0
    maker_quote_inventory_hard_limit_shares: float = 150.0
    maker_quote_inventory_skew_cents: float = 0.01


class MakerMMConfig(BaseModel):
    """Config for the MAKER_REWARDED_EVENT_MM_V1 strategy family.

    These values serve as the config-level defaults for the maker-MM scan.
    They are overridden at run-level by matching experiment-context keys
    (maker_mm_cohort, maker_mm_min_edge, maker_mm_g6_margin), which allows
    campaign presets and ad-hoc calls to override without touching this file.
    """
    model_config = ConfigDict(extra="ignore")

    cohort: list[str] = Field(default_factory=lambda: [
        "next-prime-minister-of-hungary",
        "netanyahu-out-before-2027",
        "balance-of-power-2026-midterms",
        "next-james-bond-actor-635",
    ])
    min_edge_cents: float = 0.005        # G5: path-level edge threshold (maker-MM only)
    g6_margin: float = 1.25             # G6: headroom factor over rewards_min_size
    default_notional_usd: float = 100.0  # G6: base notional cap floor


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    opportunity: OpportunityConfig = Field(default_factory=OpportunityConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    maker_mm: MakerMMConfig = Field(default_factory=MakerMMConfig)
