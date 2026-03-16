from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OrderMode(str, Enum):
    PAPER = "paper"
    DRY_RUN = "dry_run"
    LIVE = "live"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class RiskStatus(str, Enum):
    APPROVED = "approved"
    BLOCKED = "blocked"
    CAPPED = "capped"
    HALTED = "halted"


class PositionState(str, Enum):
    OPENED = "opened"
    PARTIALLY_REDUCED = "partially_reduced"
    CLOSED = "closed"
    FORCE_CLOSED = "force_closed"
    EXPIRED = "expired"


class RejectionReason(str, Enum):
    EDGE_BELOW_THRESHOLD = "EDGE_BELOW_THRESHOLD"
    NET_PROFIT_BELOW_THRESHOLD = "NET_PROFIT_BELOW_THRESHOLD"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    PARTIAL_FILL_RISK_TOO_HIGH = "PARTIAL_FILL_RISK_TOO_HIGH"
    NON_ATOMIC_RISK_TOO_HIGH = "NON_ATOMIC_RISK_TOO_HIGH"
    POSITION_LIMIT_EXCEEDED = "POSITION_LIMIT_EXCEEDED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    CONSECUTIVE_LOSS_LIMIT = "CONSECUTIVE_LOSS_LIMIT"
    DATA_STALE = "DATA_STALE"
    MARKET_NOT_ALLOWED = "MARKET_NOT_ALLOWED"
    INVALID_ORDERBOOK = "INVALID_ORDERBOOK"
    EMPTY_BIDS = "EMPTY_BIDS"
    EMPTY_ASKS = "EMPTY_ASKS"
    CROSSED_BOOK = "CROSSED_BOOK"
    NO_TOUCH_DEPTH = "NO_TOUCH_DEPTH"
    MALFORMED_PRICE_LEVEL = "MALFORMED_PRICE_LEVEL"
    NON_MONOTONIC_BOOK = "NON_MONOTONIC_BOOK"
    MISSING_ORDERBOOK = "MISSING_ORDERBOOK"
    ORDERBOOK_FETCH_FAILED = "ORDERBOOK_FETCH_FAILED"
    ORDER_SIZE_LIMIT = "ORDER_SIZE_LIMIT"
    MARKET_EXPOSURE_LIMIT = "MARKET_EXPOSURE_LIMIT"
    TOTAL_EXPOSURE_LIMIT = "TOTAL_EXPOSURE_LIMIT"
    RISK_ENGINE_ERROR = "RISK_ENGINE_ERROR"
    EXECUTION_SIMULATION_FAILED = "EXECUTION_SIMULATION_FAILED"
    PAPER_ORDER_REJECTED = "PAPER_ORDER_REJECTED"


class Severity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BookLevel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    price: float
    size: float


class Market(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_id: str
    slug: str
    question: str | None = None
    outcomes: list[str] = Field(default_factory=list)
    token_ids: list[str] = Field(default_factory=list)
    active: bool = True
    close_time: str | None = None
    tags: list[str] = Field(default_factory=list)


class MarketPair(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_slug: str
    yes_token_id: str
    no_token_id: str
    question: str | None = None


class OrderBookSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token_id: str
    market_slug: str | None = None
    bids: list[BookLevel] = Field(default_factory=list)
    asks: list[BookLevel] = Field(default_factory=list)
    source_ts: datetime
    ingest_ts: datetime
    request_latency_ms: int = 0

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


class OpportunityCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_id: str
    candidate_id: str
    kind: str
    market_slugs: list[str]
    gross_edge_cents: float
    fee_estimate_cents: float
    slippage_estimate_cents: float
    latency_penalty_cents: float = 0.0
    expected_payout: float = 0.0
    target_notional_usd: float
    estimated_depth_usd: float
    score: float
    repeatable: bool = False
    book_age_ms: int = 0
    estimated_net_profit_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime

    @property
    def net_edge_cents(self) -> float:
        return self.gross_edge_cents - self.fee_estimate_cents - self.slippage_estimate_cents - self.latency_penalty_cents


class RiskDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    status: RiskStatus
    approved_notional_usd: float
    reason_codes: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    halt_trading: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class OrderIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent_id: str
    candidate_id: str
    mode: OrderMode
    market_slug: str
    token_id: str
    position_id: str | None = None
    side: str
    order_type: OrderType
    size: float
    limit_price: float | None = None
    max_notional_usd: float
    max_slippage_cents: float = 0.0
    expires_at: datetime | None = None
    ts: datetime


class ExecutionReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent_id: str
    position_id: str | None = None
    status: OrderStatus
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    fee_paid_usd: float = 0.0
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class PositionSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position_id: str | None = None
    candidate_id: str | None = None
    symbol: str
    shares: float = 0.0
    avg_price: float = 0.0
    market_slug: str | None = None
    state: PositionState = PositionState.OPENED
    opened_ts: datetime | None = None
    closed_ts: datetime | None = None
    entry_cost_usd: float = 0.0
    exit_proceeds_usd: float = 0.0
    fees_paid_usd: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    marked_value_usd: float = 0.0


class AccountSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cash: float
    frozen_cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    positions: list[PositionSnapshot] = Field(default_factory=list)
    exposure_by_market: dict[str, float] = Field(default_factory=dict)
    open_positions: int = 0
    ts: datetime


class SystemEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_type: str
    severity: Severity
    message: str
    entity_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class PositionMark(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position_id: str
    candidate_id: str | None = None
    symbol: str
    market_slug: str
    state: PositionState
    shares: float
    avg_entry_price: float
    source_bid: float | None = None
    source_ask: float | None = None
    mark_price: float
    mark_method: str = "best_bid"
    marked_value_usd: float
    remaining_entry_cost_usd: float
    unrealized_pnl_usd: float
    age_sec: float
    ts: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExitSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position_id: str
    candidate_id: str | None = None
    symbol: str
    market_slug: str
    reason_code: str
    force_exit: bool = False
    expected_exit_price: float | None = None
    expected_unrealized_pnl_usd: float | None = None
    age_sec: float | None = None
    ts: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position_id: str
    candidate_id: str | None = None
    symbol: str
    market_slug: str
    state: PositionState
    entry_cost_usd: float
    exit_proceeds_usd: float
    fees_paid_usd: float
    realized_pnl_usd: float
    holding_duration_sec: float
    opened_ts: datetime
    closed_ts: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RejectionEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    candidate_id: str | None = None
    stage: str
    reason_code: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    started_ts: datetime
    ended_ts: datetime
    markets_scanned: int = 0
    snapshots_stored: int = 0
    candidates_generated: int = 0
    risk_accepted: int = 0
    risk_rejected: int = 0
    near_miss_candidates: int = 0
    paper_orders_created: int = 0
    fills: int = 0
    partial_fills: int = 0
    cancellations: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    system_errors: int = 0
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    top_markets_by_candidates: dict[str, int] = Field(default_factory=dict)
    top_opportunity_types: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
