from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.config_runtime.models import OpportunityConfig, PaperConfig
from src.domain.models import AccountSnapshot, RejectionReason
from src.opportunity.models import RankedOpportunity


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class SizingDecision:
    notional_usd: float
    shares: float
    reason: str
    metadata: dict[str, float]
    # viable=False means the sizing engine rejected the candidate outright.
    # The runner's existing  `if sizing.notional_usd <= 1e-9`  guard catches
    # this without any runner-side changes: a non-viable decision always has
    # notional_usd=0.0 and shares=0.0.
    viable: bool = True
    rejection_reason: Optional[str] = None


class SizingEngine:
    def size(self, candidate: RankedOpportunity, account: AccountSnapshot) -> SizingDecision:
        raise NotImplementedError


class DepthCappedSizer(SizingEngine):
    def __init__(self, paper_config: PaperConfig, opportunity_config: OpportunityConfig):
        self.paper_config = paper_config
        self.opportunity_config = opportunity_config

    def size(self, candidate: RankedOpportunity, account: AccountSnapshot) -> SizingDecision:
        depth_cap = candidate.available_depth_usd / max(self.opportunity_config.min_depth_multiple, 1.0)
        max_budget = min(
            candidate.target_notional_usd,
            self.paper_config.max_notional_per_arb,
            max(0.0, account.cash - account.frozen_cash),
            depth_cap,
        )
        quality_multiplier = _clamp(candidate.quality_score / 100.0, lower=0.25, upper=0.80)
        risk_multiplier = _clamp(1.0 - max(candidate.partial_fill_risk_score, candidate.non_atomic_execution_risk_score), lower=0.20, upper=1.0)
        sized_notional = round(max_budget * quality_multiplier * risk_multiplier, 6)
        if candidate.target_notional_usd <= 1e-9 or candidate.required_shares <= 1e-9:
            shares = 0.0
        else:
            shares = round(candidate.required_shares * (sized_notional / candidate.target_notional_usd), 6)

        # Post-sizing viability gate.  If the resulting position is smaller than
        # the configured floor it cannot realistically execute at any useful size.
        # Returning notional_usd=0.0 / shares=0.0 triggers the runner's existing
        # `sizing.notional_usd <= 1e-9` guard, which discards the candidate and
        # continues to the next market — no runner edits required.
        min_sized = self.opportunity_config.min_sized_notional_usd
        if min_sized > 0.0 and 0.0 < sized_notional < min_sized:
            return SizingDecision(
                notional_usd=0.0,
                shares=0.0,
                viable=False,
                rejection_reason=RejectionReason.SIZED_NOTIONAL_TOO_SMALL.value,
                reason="sized_notional_below_minimum",
                metadata={
                    "depth_cap": round(depth_cap, 6),
                    "quality_multiplier": round(quality_multiplier, 6),
                    "risk_multiplier": round(risk_multiplier, 6),
                    "sized_notional_computed": sized_notional,
                    "min_sized_notional_usd": min_sized,
                },
            )

        return SizingDecision(
            notional_usd=sized_notional,
            shares=shares,
            reason="depth_capped_baseline",
            metadata={
                "depth_cap": round(depth_cap, 6),
                "quality_multiplier": round(quality_multiplier, 6),
                "risk_multiplier": round(risk_multiplier, 6),
            },
        )
