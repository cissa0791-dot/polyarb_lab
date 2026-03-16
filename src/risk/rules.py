from __future__ import annotations

from dataclasses import dataclass

from src.config_runtime.models import ExecutionConfig, OpportunityConfig, RiskConfig
from src.domain.models import AccountSnapshot, OpportunityCandidate, RejectionReason


@dataclass(frozen=True)
class RuleResult:
    passed: bool
    code: str
    message: str


def check_score(candidate: OpportunityCandidate, risk: RiskConfig) -> RuleResult:
    passed = candidate.score >= risk.min_score
    return RuleResult(passed, RejectionReason.EDGE_BELOW_THRESHOLD.value, f"score={candidate.score:.2f} min={risk.min_score:.2f}")


def check_net_profit(candidate: OpportunityCandidate, opportunity: OpportunityConfig) -> RuleResult:
    passed = candidate.estimated_net_profit_usd >= opportunity.min_net_profit_usd
    return RuleResult(
        passed,
        RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value,
        f"net_profit={candidate.estimated_net_profit_usd:.4f} min={opportunity.min_net_profit_usd:.4f}",
    )


def check_depth(candidate: OpportunityCandidate, risk: RiskConfig) -> RuleResult:
    passed = candidate.estimated_depth_usd >= max(candidate.target_notional_usd, risk.min_liquidity_usd)
    return RuleResult(
        passed,
        RejectionReason.INSUFFICIENT_DEPTH.value,
        f"depth={candidate.estimated_depth_usd:.4f} target={candidate.target_notional_usd:.4f}",
    )


def check_book_age(candidate: OpportunityCandidate, opportunity: OpportunityConfig) -> RuleResult:
    passed = candidate.book_age_ms <= opportunity.max_latency_ms
    return RuleResult(
        passed,
        RejectionReason.DATA_STALE.value,
        f"book_age_ms={candidate.book_age_ms} max={opportunity.max_latency_ms}",
    )


def check_market_whitelist(candidate: OpportunityCandidate, risk: RiskConfig) -> RuleResult:
    if not risk.whitelist_markets:
        return RuleResult(True, RejectionReason.MARKET_NOT_ALLOWED.value, "whitelist disabled")
    candidate_markets = set(candidate.market_slugs)
    passed = candidate_markets.issubset(set(risk.whitelist_markets))
    return RuleResult(passed, RejectionReason.MARKET_NOT_ALLOWED.value, f"markets={sorted(candidate_markets)}")


def check_order_size(candidate: OpportunityCandidate, risk: RiskConfig, execution: ExecutionConfig) -> RuleResult:
    max_allowed = min(risk.max_order_notional_usd, execution.max_live_order_usd if execution.live_enabled else risk.max_order_notional_usd)
    passed = candidate.target_notional_usd <= max_allowed
    return RuleResult(
        passed,
        RejectionReason.ORDER_SIZE_LIMIT.value,
        f"notional={candidate.target_notional_usd:.4f} max={max_allowed:.4f}",
    )


def check_daily_loss(account: AccountSnapshot, risk: RiskConfig) -> RuleResult:
    passed = account.daily_pnl > -abs(risk.max_daily_loss_usd)
    return RuleResult(
        passed,
        RejectionReason.DAILY_LOSS_LIMIT.value,
        f"daily_pnl={account.daily_pnl:.4f} limit={-abs(risk.max_daily_loss_usd):.4f}",
    )


def check_consecutive_losses(account: AccountSnapshot, risk: RiskConfig) -> RuleResult:
    passed = account.consecutive_losses < risk.max_consecutive_losses
    return RuleResult(
        passed,
        RejectionReason.CONSECUTIVE_LOSS_LIMIT.value,
        f"consecutive_losses={account.consecutive_losses} max={risk.max_consecutive_losses}",
    )


def check_open_positions(account: AccountSnapshot, risk: RiskConfig) -> RuleResult:
    passed = account.open_positions < risk.max_open_positions
    return RuleResult(
        passed,
        RejectionReason.POSITION_LIMIT_EXCEEDED.value,
        f"open_positions={account.open_positions} max={risk.max_open_positions}",
    )


def check_market_exposure(candidate: OpportunityCandidate, account: AccountSnapshot, risk: RiskConfig) -> RuleResult:
    current = sum(account.exposure_by_market.get(slug, 0.0) for slug in candidate.market_slugs)
    passed = current + candidate.target_notional_usd <= risk.max_market_exposure_usd
    return RuleResult(
        passed,
        RejectionReason.MARKET_EXPOSURE_LIMIT.value,
        f"current={current:.4f} requested={candidate.target_notional_usd:.4f} max={risk.max_market_exposure_usd:.4f}",
    )


def check_total_exposure(candidate: OpportunityCandidate, account: AccountSnapshot, risk: RiskConfig) -> RuleResult:
    current = sum(account.exposure_by_market.values())
    passed = current + candidate.target_notional_usd <= risk.max_total_exposure_usd
    return RuleResult(
        passed,
        RejectionReason.TOTAL_EXPOSURE_LIMIT.value,
        f"current={current:.4f} requested={candidate.target_notional_usd:.4f} max={risk.max_total_exposure_usd:.4f}",
    )
