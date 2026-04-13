from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models import OpportunityCandidate, RiskDecision


class StrategyFamily(str, Enum):
    SINGLE_MARKET_MISPRICING = "single_market_mispricing"
    SINGLE_MARKET_TOUCH_MISPRICING = "single_market_touch_mispricing"
    CROSS_MARKET_CONSTRAINT = "cross_market_constraint"
    CROSS_MARKET_GROSS_CONSTRAINT = "cross_market_gross_constraint"
    CROSS_MARKET_EXECUTION_GROSS_CONSTRAINT = "cross_market_execution_gross_constraint"
    NEG_RISK_REBALANCING = "neg_risk_rebalancing"
    MAKER_REWARDED_EVENT_MM_V1 = "maker_rewarded_event_mm_v1"
    POLITICAL_BINARY_CONSTRAINT_PAPER = "political_binary_constraint_paper"
    REBALANCING = "rebalancing"
    EXTERNAL_BELIEF = "external_belief"


class CandidateLeg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token_id: str
    market_slug: str
    action: str
    side: str
    required_shares: float
    best_price: float | None = None
    vwap_price: float | None = None
    spread_cents: float | None = None
    available_shares: float = 0.0
    available_notional_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_id: str
    strategy_family: StrategyFamily
    candidate_id: str
    kind: str
    detection_name: str
    market_slugs: list[str]
    gross_edge_cents: float
    expected_payout: float = 0.0
    target_notional_usd: float
    target_shares: float
    gross_profit_usd: float = 0.0
    est_fill_cost_usd: float = 0.0
    execution_mode: str = "paper_eligible"
    research_only: bool = False
    legs: list[CandidateLeg] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime

    def to_legacy_opportunity(self):
        from src.core.models import ArbOpportunity

        return ArbOpportunity(
            kind=self.kind,
            name=self.detection_name,
            edge_cents=self.gross_edge_cents,
            gross_profit=self.gross_profit_usd,
            est_fill_cost=self.est_fill_cost_usd,
            est_payout=self.expected_payout,
            notional=self.target_notional_usd,
            details=self.metadata,
            ts=self.ts,
        )


class ExecutableCandidate(OpportunityCandidate):
    model_config = ConfigDict(extra="ignore")

    strategy_family: StrategyFamily
    execution_mode: str = "paper_eligible"
    research_only: bool = False
    qualification_reason_codes: list[str] = Field(default_factory=list)
    qualification_metadata: dict[str, Any] = Field(default_factory=dict)
    expected_gross_profit_usd: float = 0.0
    expected_fee_usd: float = 0.0
    expected_slippage_usd: float = 0.0
    pair_vwap: float = 0.0
    required_depth_usd: float = 0.0
    available_depth_usd: float = 0.0
    required_shares: float = 0.0
    available_shares: float = 0.0
    partial_fill_risk_score: float = 0.0
    non_atomic_execution_risk_score: float = 0.0
    legs: list[CandidateLeg] = Field(default_factory=list)


class RankedOpportunity(ExecutableCandidate):
    model_config = ConfigDict(extra="ignore")

    strategy_tag: str
    ranking_score: float
    sizing_hint_usd: float
    sizing_hint_shares: float
    capital_efficiency: float = 0.0
    expected_profit_usd: float = 0.0
    quality_score: float = 0.0


class LiveTradableOpportunity(BaseModel):
    """Explicit gate type: an ExecutableCandidate that has passed the risk
    decision layer and is approved for live broker submission.

    Signal pipeline:
        raw_signal (RawCandidate)
            → executable_opportunity (ExecutableCandidate)
            → live_tradable_opportunity (LiveTradableOpportunity)   ← here
            → OrderIntent (domain/models.py)
    """

    model_config = ConfigDict(extra="ignore")

    candidate: ExecutableCandidate
    risk_decision: RiskDecision
    approved_for_live: bool


class QualificationPassReason:
    """String constants naming each gate that explicitly passed on a qualified candidate.

    Populated only for passed decisions — rejected decisions carry reason_codes instead.
    Gate-by-gate visibility lets downstream code and audit reports explain *why* a
    candidate was accepted, not just that it was.
    """

    EDGE_SUFFICIENT = "EDGE_SUFFICIENT"
    DEPTH_SUFFICIENT = "DEPTH_SUFFICIENT"
    NET_PROFIT_SUFFICIENT = "NET_PROFIT_SUFFICIENT"
    PARTIAL_FILL_RISK_OK = "PARTIAL_FILL_RISK_OK"
    NON_ATOMIC_RISK_OK = "NON_ATOMIC_RISK_OK"
    ABSOLUTE_DEPTH_OK = "ABSOLUTE_DEPTH_OK"
    SINGLE_LEG_CONCENTRATION_OK = "SINGLE_LEG_CONCENTRATION_OK"


class QualificationDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_candidate: RawCandidate
    passed: bool
    reason_codes: list[str] = Field(default_factory=list)
    pass_reason_codes: list[str] = Field(default_factory=list)
    executable_candidate: ExecutableCandidate | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class QualificationShortlistEntry(BaseModel):
    """Key metrics for a single candidate that passed all qualification gates.

    Produced by QualificationAuditor.report() — one entry per passed decision.
    Surfaces the numbers that justified passage so a scan log can answer:
    "why did this candidate make the shortlist?"
    """

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    strategy_family: str
    market_slugs: list[str]
    pass_reason_codes: list[str]
    gross_edge_cents: float
    net_edge_cents: float
    expected_net_profit_usd: float
    required_depth_usd: float
    available_depth_usd: float
    partial_fill_risk_score: float
    non_atomic_execution_risk_score: float
    ts: datetime


class QualificationFunnelReport(BaseModel):
    """Per-run qualification funnel summary produced by QualificationAuditor.

    Captures the full pipeline from raw candidates evaluated → passed shortlist,
    with per-gate rejection counts so it is clear which gates are filtering most.
    """

    model_config = ConfigDict(extra="ignore")

    run_id: str
    evaluated: int
    passed: int
    rejected: int
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    shortlist: list[QualificationShortlistEntry] = Field(default_factory=list)
    ts: datetime
