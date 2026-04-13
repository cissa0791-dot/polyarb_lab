"""
reward_aware_single_market_maker_executable_conversion_line
polyarb_lab / research_line / audit-only

Executable viability audit for POSITIVE_RAW_EV candidates confirmed by
the multi-cycle confirmation phase.

Takes MarketEVResult (from ev_model.py) + RawRewardedMarket (from discovery.py)
and applies four execution-layer gates to determine which raw-positive
candidates are also executable-positive.

No API calls. No order submission. No state. Pure computation.

Gates:
  E1 EC_CAPITAL_INTENSIVE  — quote_capital_usd > CAPITAL_THRESHOLD_USD
  E2 EC_NEAR_RESOLUTION    — midpoint < NEAR_RESOLUTION_LOW or > NEAR_RESOLUTION_HIGH
  E3 EC_ROC_TOO_LOW        — raw_ev / quote_capital_usd < EV_ROC_MIN
  E4 EC_REWARD_TOO_LOW     — estimated_reward_contribution < REWARD_FLOOR_USDC
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .discovery import RawRewardedMarket
from .ev_model import ECON_POSITIVE_RAW_EV, MarketEVResult

# ---------------------------------------------------------------------------
# Audit parameters — recalibrated for reward-aware market-making economics
# ---------------------------------------------------------------------------

# E1: Maximum acceptable single-side quote capital (shares × midpoint).
#
# REDESIGN (capital_efficiency_attack):
#   Previous threshold was $50 — calibrated as a single-trade conservative limit.
#   This incorrectly rejected markets where rewards_min_size × midpoint > $50 even
#   though the daily reward income (up to $15+/day) makes the capital deployment
#   highly efficient.
#
#   Root cause: high-reward markets necessarily have larger min sizes (to prevent
#   reward gaming). A rewards_min_size=200 shares at midpoint=0.60 = $120 capital,
#   which earns potentially $15/day reward share — a 12.5% daily ROC. The $50 gate
#   blocked this without examining the return.
#
#   New threshold: $200 — the minimum practical market-making deployment. This covers
#   buffer inventory through 1-2 adverse fills without constant rebalancing, which
#   is the actual operational minimum for continuous reward-eligible quoting.
CAPITAL_THRESHOLD_USD = 200.0

# E2: Probability extremes indicating near-resolution.
# Markets below 0.03 or above 0.97 are effectively resolved; reward program
# participation is likely ending and adverse-selection risk is asymmetric.
NEAR_RESOLUTION_LOW = 0.03
NEAR_RESOLUTION_HIGH = 0.97

# E3: Minimum return on capital (raw_ev / quote_capital_usd).
#
# REDESIGN (capital_efficiency_attack):
#   Previous threshold was 0.5% — calibrated assuming raw_ev is a per-fill metric.
#   This is incorrect: raw_ev = estimated_spread_capture (per-fill) +
#   estimated_reward_contribution (daily). The reward term dominates for the 13
#   positive candidates, making raw_ev effectively a daily income figure. Dividing
#   daily income by capital gives a daily ROC; 0.5% daily = 182% APY, which is
#   far too strict for an initial executable gate.
#
#   New threshold: 0.1% — daily ROC floor. At 0.1%/day = 36.5% APY, this still
#   screens out markets where reward income is negligible relative to capital.
#   The E1 redesign now handles the absolute capital floor; E3 handles relative
#   income adequacy once E1 passes.
EV_ROC_MIN = 0.001

# E4: Minimum daily reward share contribution.
# Below $0.05/day the reward is operationally negligible — not worth quoting
# exclusively for the reward.
REWARD_FLOOR_USDC = 0.05

# Informational threshold: minimum daily reward ROC
# reward_roc_daily = estimated_reward_contribution / quote_capital_usd
# Not a hard gate — surfaced in per-slug output for analysis.
REWARD_ROC_DAILY_INFO = 0.001  # 0.1%/day from reward alone

# ---------------------------------------------------------------------------
# Verdict + rejection codes
# ---------------------------------------------------------------------------

EXEC_POSITIVE = "EXECUTABLE_POSITIVE"
EXEC_REJECTED  = "EXECUTABLE_REJECTED"

EC_CAPITAL_INTENSIVE = "EC_CAPITAL_INTENSIVE"
EC_NEAR_RESOLUTION   = "EC_NEAR_RESOLUTION"
EC_ROC_TOO_LOW       = "EC_ROC_TOO_LOW"
EC_REWARD_TOO_LOW    = "EC_REWARD_TOO_LOW"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExecutableAuditResult:
    """Full executable audit result for one POSITIVE_RAW_EV candidate."""
    market_slug: str
    raw_ev: float
    net_ev: float                              # same as raw_ev — no execution cost model yet
    reward_rate_daily_usdc: float
    quoted_spread: Optional[float]
    midpoint: Optional[float]
    quote_size_shares: float
    quote_capital_usd: Optional[float]         # quote_size × midpoint
    ev_roc: Optional[float]                    # raw_ev / quote_capital_usd (daily ROC proxy)
    reward_roc_daily: Optional[float]          # reward_contribution / quote_capital_usd (reward-only daily ROC)
    estimated_reward_contribution: float
    executable_verdict: str
    rejection_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "market_slug": self.market_slug,
            "raw_ev": self.raw_ev,
            "net_ev": self.net_ev,
            "reward_rate_daily_usdc": self.reward_rate_daily_usdc,
            "quoted_spread": self.quoted_spread,
            "midpoint": self.midpoint,
            "quote_size_shares": self.quote_size_shares,
            "quote_capital_usd": self.quote_capital_usd,
            "ev_roc": round(self.ev_roc, 6) if self.ev_roc is not None else None,
            "reward_roc_daily": round(self.reward_roc_daily, 6) if self.reward_roc_daily is not None else None,
            "estimated_reward_contribution": self.estimated_reward_contribution,
            "executable_verdict": self.executable_verdict,
            "rejection_codes": self.rejection_codes,
        }


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit_executable(
    ev_result: MarketEVResult,
    market: RawRewardedMarket,
) -> ExecutableAuditResult:
    """Apply all four executable gates to one POSITIVE_RAW_EV candidate."""
    rejection_codes: list[str] = []

    midpoint = ev_result.midpoint
    quoted_spread = ev_result.quoted_spread
    quote_size = market.rewards_min_size

    # Derived metrics
    quote_capital_usd: Optional[float] = None
    ev_roc: Optional[float] = None
    reward_roc_daily: Optional[float] = None
    if midpoint is not None and midpoint > 0.0:
        quote_capital_usd = round(quote_size * midpoint, 4)
        if quote_capital_usd > 0.0:
            ev_roc = round(ev_result.reward_adjusted_raw_ev / quote_capital_usd, 6)
            reward_roc_daily = round(ev_result.estimated_reward_contribution / quote_capital_usd, 6)

    # Gate E1: capital-intensive
    if quote_capital_usd is not None and quote_capital_usd > CAPITAL_THRESHOLD_USD:
        rejection_codes.append(EC_CAPITAL_INTENSIVE)

    # Gate E2: near resolution
    if midpoint is not None and (
        midpoint < NEAR_RESOLUTION_LOW or midpoint > NEAR_RESOLUTION_HIGH
    ):
        rejection_codes.append(EC_NEAR_RESOLUTION)

    # Gate E3: return on capital too low
    if ev_roc is not None and ev_roc < EV_ROC_MIN:
        rejection_codes.append(EC_ROC_TOO_LOW)

    # Gate E4: reward contribution too low
    if ev_result.estimated_reward_contribution < REWARD_FLOOR_USDC:
        rejection_codes.append(EC_REWARD_TOO_LOW)

    verdict = EXEC_POSITIVE if not rejection_codes else EXEC_REJECTED

    return ExecutableAuditResult(
        market_slug=ev_result.market_slug,
        raw_ev=ev_result.reward_adjusted_raw_ev,
        net_ev=ev_result.reward_adjusted_raw_ev,
        reward_rate_daily_usdc=ev_result.reward_config_summary.get("reward_daily_rate_usdc", 0.0),
        quoted_spread=quoted_spread,
        midpoint=midpoint,
        quote_size_shares=quote_size,
        quote_capital_usd=quote_capital_usd,
        ev_roc=ev_roc,
        reward_roc_daily=reward_roc_daily,
        estimated_reward_contribution=ev_result.estimated_reward_contribution,
        executable_verdict=verdict,
        rejection_codes=rejection_codes,
    )


def audit_batch(
    ev_results: list[MarketEVResult],
    markets: list[RawRewardedMarket],
) -> list[ExecutableAuditResult]:
    """Audit all POSITIVE_RAW_EV results. One output per raw-positive candidate."""
    market_by_slug = {m.market_slug: m for m in markets}
    results: list[ExecutableAuditResult] = []
    for ev_r in ev_results:
        if ev_r.economics_class != ECON_POSITIVE_RAW_EV:
            continue
        market = market_by_slug.get(ev_r.market_slug)
        if market is None:
            continue
        results.append(audit_executable(ev_r, market))
    return results


def build_audit_summary(results: list[ExecutableAuditResult]) -> dict:
    survivors = [r for r in results if r.executable_verdict == EXEC_POSITIVE]
    rejected  = [r for r in results if r.executable_verdict == EXEC_REJECTED]

    all_codes: list[str] = []
    for r in rejected:
        all_codes.extend(r.rejection_codes)
    code_counts = dict(Counter(all_codes))

    return {
        "persistent_raw_positive_count": len(results),
        "persistent_executable_positive_count": len(survivors),
        "top_executable_survivors": [
            r.to_dict()
            for r in sorted(survivors, key=lambda x: x.raw_ev, reverse=True)
        ],
        "rejected_slugs": [
            {
                "slug": r.market_slug,
                "rejection_codes": r.rejection_codes,
                "raw_ev": r.raw_ev,
                "midpoint": r.midpoint,
                "quote_capital_usd": r.quote_capital_usd,
                "ev_roc": r.ev_roc,
                "estimated_reward_contribution": r.estimated_reward_contribution,
            }
            for r in rejected
        ],
        "rejection_code_counts": code_counts,
    }
