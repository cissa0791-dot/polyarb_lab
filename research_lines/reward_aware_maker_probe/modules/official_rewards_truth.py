"""
reward_aware_official_rewards_truth_line
polyarb_lab / research_line / validation-only

Official reward truth integration for the 4 confirmed survivors.

Data sources (all unauthenticated, public CLOB API):
  Primary:  GET /rewards/markets/{condition_id}
    Returns: market_competitiveness, rewards_config[]{rate_per_day, start_date,
             end_date, total_days, asset_address}, rewards_max_spread,
             rewards_min_size
  Fallback: /rewards/markets/multi pagination (already used in discovery)

market_competitiveness:
  Official CLOB metric representing total qualifying competition in the reward
  window for this market.  Higher = more competition = smaller our share.
  Used to estimate implied_share = min_size / (market_competitiveness + min_size).
  NOTE: unit semantics not officially documented; interpreted as total eligible
  quoting score in same units as rewards_min_size (shares).  This is an
  INFERENCE from endpoint semantics, not a confirmed spec.

Auth-required endpoints (returning 405 without credentials):
  /rewards/percentages        — user's actual reward % for a market
  /rewards/user-shares        — user reward breakdown per market
  /rewards/market-scores      — per-market scoring state
  /rewards/scores             — order-level scoring status
  /rewards/history            — historical reward payments
  /rewards/payments           — payment detail
  These would provide ground-truth user reward % if a wallet address and
  L1/L2 CLOB auth signature were available.  Without auth, market_competitiveness
  is the best available official signal.

No order submission.  No mainline imports.  Read-only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .discovery import RawRewardedMarket, _safe_float, _safe_str, CLOB_HOST
from .ev_model import REWARD_POOL_SHARE_FRACTION, MarketEVResult

logger = logging.getLogger(__name__)

CLOB_REWARDS_MARKET_PATH = "/rewards/markets"   # + "/{condition_id}"

# Tolerance for FAIR band (±30% relative)
FAIR_TOLERANCE = 0.30

# If market_competitiveness is zero but rewards_config is active, market is uncontested.
# We flag this separately — not the same as empty book window.
UNCONTESTED_LABEL   = "UNCONTESTED_OFFICIAL"
NO_REWARDS_LABEL    = "NO_ACTIVE_REWARDS"
UNVERIFIABLE_LABEL  = "UNVERIFIABLE"

# Auth endpoint probe targets (record response code only — no body needed)
AUTH_PROBE_PATHS = [
    "/rewards/percentages",
    "/rewards/user-shares",
    "/rewards/market-scores",
    "/rewards/scores",
    "/rewards/history",
    "/rewards/payments",
]


# ---------------------------------------------------------------------------
# Official config dataclass
# ---------------------------------------------------------------------------

@dataclass
class OfficialRewardConfig:
    """
    Official reward configuration for one market, fetched from CLOB
    /rewards/markets/{condition_id}.

    Fields are directly from the API response — not computed or proxied.
    """
    condition_id: str
    market_slug: str
    fetch_ok: bool
    fetched_at: datetime

    # Official competition metric (higher = more competition)
    market_competitiveness: Optional[float]

    # Official reward parameters (from endpoint, not proxy)
    official_rewards_min_size: Optional[float]
    official_rewards_max_spread: Optional[float]       # in cents (raw from endpoint)

    # Reward program details (from rewards_config array — may be empty)
    rewards_configs_raw: list[dict[str, Any]] = field(default_factory=list)
    official_rate_per_day: Optional[float] = None       # sum of rate_per_day across active configs
    reward_asset_address: Optional[str] = None          # USDC contract on Polygon
    reward_start_date: Optional[str] = None
    reward_end_date: Optional[str] = None
    reward_total_days: Optional[int] = None
    has_active_reward_config: bool = False

    # Auth probe results (populated separately)
    auth_probe_results: dict[str, int] = field(default_factory=dict)  # path -> HTTP status

    # Discrepancy flags vs discovery
    rate_discrepancy: bool = False           # official vs discovery rate differ
    spread_discrepancy: bool = False
    size_discrepancy: bool = False


def fetch_official_reward_config(
    host: str,
    condition_id: str,
    client: httpx.Client,
) -> OfficialRewardConfig:
    """
    Fetch official reward config for one market from /rewards/markets/{condition_id}.

    Parses: market_competitiveness, rewards_config[], rewards_min_size,
    rewards_max_spread.
    """
    fetched_at = datetime.now(timezone.utc)
    empty = OfficialRewardConfig(
        condition_id=condition_id,
        market_slug="",
        fetch_ok=False,
        fetched_at=fetched_at,
        market_competitiveness=None,
        official_rewards_min_size=None,
        official_rewards_max_spread=None,
    )
    if not condition_id:
        return empty

    url = f"{host.rstrip('/')}{CLOB_REWARDS_MARKET_PATH}/{condition_id}"
    try:
        resp = client.get(url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        markets = payload.get("data") or []
        if not markets:
            return empty
        m = markets[0] if isinstance(markets, list) else markets

        # Parse rewards_config array
        configs_raw = m.get("rewards_config") or []
        active_configs = [
            c for c in configs_raw
            if isinstance(c, dict) and _safe_float(c.get("rate_per_day", 0)) > 0
        ]
        total_rate = sum(_safe_float(c.get("rate_per_day", 0)) for c in active_configs)

        # Primary config details (first active entry)
        first_cfg = active_configs[0] if active_configs else {}

        return OfficialRewardConfig(
            condition_id=condition_id,
            market_slug=_safe_str(m.get("market_slug")),
            fetch_ok=True,
            fetched_at=fetched_at,
            market_competitiveness=_safe_float(m.get("market_competitiveness"), default=None),  # type: ignore[arg-type]
            official_rewards_min_size=_safe_float(m.get("rewards_min_size"), default=None),  # type: ignore[arg-type]
            official_rewards_max_spread=_safe_float(m.get("rewards_max_spread"), default=None),  # type: ignore[arg-type]
            rewards_configs_raw=configs_raw,
            official_rate_per_day=round(total_rate, 4) if total_rate > 0 else None,
            reward_asset_address=_safe_str(first_cfg.get("asset_address", "")),
            reward_start_date=_safe_str(first_cfg.get("start_date", "")),
            reward_end_date=_safe_str(first_cfg.get("end_date", "")),
            reward_total_days=int(first_cfg["total_days"]) if "total_days" in first_cfg else None,
            has_active_reward_config=len(active_configs) > 0,
        )
    except Exception as exc:
        logger.debug("official_reward_config fetch failed cid=%s: %s", condition_id, exc)
        return empty


def probe_auth_endpoints(
    host: str,
    client: httpx.Client,
) -> dict[str, int]:
    """
    Probe auth-required reward endpoints and record HTTP response codes.
    Returns {path: status_code}.
    200 = public (no auth needed)
    401/403/405 = auth required
    404 = endpoint does not exist
    """
    results: dict[str, int] = {}
    for path in AUTH_PROBE_PATHS:
        url = f"{host.rstrip('/')}{path}"
        try:
            resp = client.get(url, timeout=5)
            results[path] = resp.status_code
        except Exception:
            results[path] = -1
    return results


# ---------------------------------------------------------------------------
# Per-survivor official truth analysis
# ---------------------------------------------------------------------------

@dataclass
class OfficialTruthAnalysis:
    """
    Official reward truth analysis for one survivor.

    Combines official CLOB data with model assumptions to produce:
      - Official competition-based implied share estimate
      - Break-even share required for positive EV
      - Discrepancy report vs discovery proxy
      - Share assumption verdict using official data (not book-depth proxy)
    """
    slug: str                        # target slug (may be truncated)
    full_slug: str                   # actual market slug from universe
    condition_id: str

    # From discovery (proxy source-of-truth up to now)
    discovery_rate_per_day: float
    discovery_min_size: float
    discovery_max_spread_cents: float
    model_share_fraction: float = REWARD_POOL_SHARE_FRACTION
    model_reward_contribution: float = 0.0

    # Official config (from /rewards/markets/{condition_id})
    official: Optional[OfficialRewardConfig] = None

    # Official implied share computation
    # implied_share_official = min_size / (market_competitiveness + min_size)
    # market_competitiveness units: inferred as same scale as min_size (shares)
    implied_share_official: Optional[float] = None
    implied_reward_contribution_official: Optional[float] = None
    share_bias_official: Optional[float] = None     # implied - model (+ = conservative)

    # Break-even share required for positive EV
    # At what share fraction does reward_contribution exactly offset fill penalties?
    # break_even_share = |fill_net_at_min_size| / rate_per_day
    # If break_even_share < model_share → model is robust to share compression
    # If break_even_share > model_share → model only works with higher-than-assumed share
    break_even_share: Optional[float] = None
    break_even_reward_contribution: Optional[float] = None
    model_is_robust: Optional[bool] = None      # True if model share > break-even share

    # Discrepancies
    rate_discrepancy_pct: Optional[float] = None     # (official - discovery) / discovery
    spread_discrepancy_pct: Optional[float] = None
    size_discrepancy_pct: Optional[float] = None
    identity_mismatch: bool = False     # market slug found doesn't match survivor slug

    # Verdict
    share_assumption_verdict: str = UNVERIFIABLE_LABEL
    share_assumption_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "full_slug": self.full_slug,
            "condition_id": self.condition_id,
            "discovery_rate_per_day": self.discovery_rate_per_day,
            "discovery_min_size": self.discovery_min_size,
            "discovery_max_spread_cents": self.discovery_max_spread_cents,
            "model_share_fraction": self.model_share_fraction,
            "model_reward_contribution": self.model_reward_contribution,
            "official_market_competitiveness": (
                self.official.market_competitiveness if self.official else None
            ),
            "official_rate_per_day": (
                self.official.official_rate_per_day if self.official else None
            ),
            "official_has_active_rewards": (
                self.official.has_active_reward_config if self.official else None
            ),
            "official_rewards_min_size": (
                self.official.official_rewards_min_size if self.official else None
            ),
            "official_rewards_max_spread_cents": (
                self.official.official_rewards_max_spread if self.official else None
            ),
            "implied_share_official": self.implied_share_official,
            "implied_reward_contribution_official": self.implied_reward_contribution_official,
            "share_bias_official": self.share_bias_official,
            "break_even_share": self.break_even_share,
            "model_is_robust": self.model_is_robust,
            "rate_discrepancy_pct": self.rate_discrepancy_pct,
            "identity_mismatch": self.identity_mismatch,
            "share_assumption_verdict": self.share_assumption_verdict,
            "share_assumption_notes": self.share_assumption_notes,
            "auth_probe_results": (
                self.official.auth_probe_results if self.official else {}
            ),
        }


def build_official_truth_analysis(
    slug: str,
    market: RawRewardedMarket,
    ev_result: Optional[MarketEVResult],
    official: OfficialRewardConfig,
) -> OfficialTruthAnalysis:
    """
    Build OfficialTruthAnalysis for one survivor using official CLOB data.

    market_competitiveness interpretation:
      Treated as total eligible quoting score in units comparable to min_size.
      implied_share = min_size / (market_competitiveness + min_size)
      This is an INFERRED interpretation — not a confirmed spec.
    """
    model_contrib = round(market.reward_daily_rate_usdc * REWARD_POOL_SHARE_FRACTION, 6)

    analysis = OfficialTruthAnalysis(
        slug=slug,
        full_slug=market.market_slug,
        condition_id=market.market_id,
        discovery_rate_per_day=market.reward_daily_rate_usdc,
        discovery_min_size=market.rewards_min_size,
        discovery_max_spread_cents=market.rewards_max_spread_cents,
        model_share_fraction=REWARD_POOL_SHARE_FRACTION,
        model_reward_contribution=model_contrib,
        official=official,
    )

    # Identity check: does the official market_slug match what we expect?
    if official.fetch_ok and official.market_slug:
        expected_full = market.market_slug
        if official.market_slug != expected_full:
            analysis.identity_mismatch = True

    # Discrepancies vs discovery
    if official.fetch_ok:
        if official.official_rate_per_day is not None and market.reward_daily_rate_usdc > 0:
            analysis.rate_discrepancy_pct = round(
                (official.official_rate_per_day - market.reward_daily_rate_usdc)
                / market.reward_daily_rate_usdc * 100.0, 2
            )
        if official.official_rewards_max_spread is not None and market.rewards_max_spread_cents > 0:
            analysis.spread_discrepancy_pct = round(
                (official.official_rewards_max_spread - market.rewards_max_spread_cents)
                / market.rewards_max_spread_cents * 100.0, 2
            )
        if official.official_rewards_min_size is not None and market.rewards_min_size > 0:
            analysis.size_discrepancy_pct = round(
                (official.official_rewards_min_size - market.rewards_min_size)
                / market.rewards_min_size * 100.0, 2
            )

    # Official implied share from market_competitiveness
    competitiveness = official.market_competitiveness if official.fetch_ok else None
    min_size = market.rewards_min_size
    rate = market.reward_daily_rate_usdc

    if not official.fetch_ok:
        analysis.share_assumption_verdict = UNVERIFIABLE_LABEL
        analysis.share_assumption_notes = "Official config fetch failed."
        return analysis

    if not official.has_active_reward_config:
        analysis.share_assumption_verdict = NO_REWARDS_LABEL
        analysis.share_assumption_notes = (
            "rewards_config is empty — this market has no active reward program "
            "according to the official CLOB endpoint. "
            "Discovery may have matched a different market variant. "
            "This survivor should be removed from the active pool."
        )
        return analysis

    if competitiveness is None:
        analysis.share_assumption_verdict = UNVERIFIABLE_LABEL
        analysis.share_assumption_notes = "market_competitiveness field missing from endpoint response."
        return analysis

    # Compute implied share
    if competitiveness <= 0:
        # market_competitiveness = 0: no measured competition
        # Could mean: uncontested, or metric not yet populated
        analysis.implied_share_official = 1.0
        analysis.implied_reward_contribution_official = round(rate * 1.0, 4)
        analysis.share_bias_official = round(1.0 - REWARD_POOL_SHARE_FRACTION, 6)
        analysis.share_assumption_verdict = UNCONTESTED_LABEL
        analysis.share_assumption_notes = (
            "market_competitiveness = 0. Market appears uncontested by this metric. "
            "Implied share = 100% but this may reflect: (a) no qualifying makers yet, "
            "(b) metric not yet populated, or (c) dead market. "
            "Do not treat as confirmed reward capture without further evidence."
        )
    else:
        implied = min_size / (competitiveness + min_size)
        analysis.implied_share_official = round(implied, 6)
        analysis.implied_reward_contribution_official = round(rate * implied, 6)
        analysis.share_bias_official = round(implied - REWARD_POOL_SHARE_FRACTION, 6)

        # Verdict
        fair_low  = REWARD_POOL_SHARE_FRACTION * (1 - FAIR_TOLERANCE)   # 0.035
        fair_high = REWARD_POOL_SHARE_FRACTION * (1 + FAIR_TOLERANCE)   # 0.065
        if implied >= fair_high:
            analysis.share_assumption_verdict = "CONSERVATIVE"
            analysis.share_assumption_notes = (
                f"Official: market_competitiveness={competitiveness:.2f}, min_size={min_size:.0f}sh. "
                f"implied_share = {min_size:.0f}/({competitiveness:.2f}+{min_size:.0f}) = {implied:.2%}. "
                f"Exceeds model 5% by {analysis.share_bias_official:.2%}. "
                f"Model UNDERSTATES reward contribution."
            )
        elif implied >= fair_low:
            analysis.share_assumption_verdict = "FAIR"
            analysis.share_assumption_notes = (
                f"Official: market_competitiveness={competitiveness:.2f}, min_size={min_size:.0f}sh. "
                f"implied_share = {implied:.2%}, within ±{FAIR_TOLERANCE:.0%} of model 5%. "
                f"Model is approximately correct."
            )
        else:
            analysis.share_assumption_verdict = "OPTIMISTIC"
            analysis.share_assumption_notes = (
                f"Official: market_competitiveness={competitiveness:.2f}, min_size={min_size:.0f}sh. "
                f"implied_share = {min_size:.0f}/({competitiveness:.2f}+{min_size:.0f}) = {implied:.2%}. "
                f"BELOW model 5% by {abs(analysis.share_bias_official):.2%}. "
                f"Model OVERSTATES reward contribution."
            )

    # Break-even share analysis
    # How much share do we need for reward_contribution to exceed fill penalties?
    # From EV model: raw_ev = reward_contribution + fill_net
    # fill_net = spread_capture - adverse_selection - inventory (negative at probe calibration)
    # At break-even: reward_contribution = |fill_net| (i.e. raw_ev = 0 from reward alone)
    # break_even_share = |fill_net| / rate_per_day
    if ev_result is not None:
        fill_net = (
            ev_result.estimated_spread_capture
            - ev_result.adverse_selection_penalty
            - ev_result.inventory_penalty
        )
        if fill_net < 0 and rate > 0:
            break_even = abs(fill_net) / rate
            analysis.break_even_share = round(break_even, 6)
            analysis.break_even_reward_contribution = round(rate * break_even, 6)
            # Is model share sufficient to cover fill penalties?
            analysis.model_is_robust = REWARD_POOL_SHARE_FRACTION > break_even

    return analysis
