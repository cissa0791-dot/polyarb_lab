"""
cohort_router.py — Saturation-aware cohort router for trial_entry_scan.

Goal
----
Before the CLOB fetch step, score and filter candidates so the scan
budget is allocated across diverse, high-information event families
instead of repeatedly spending on saturated same-day sports cohorts.

Insertion point
---------------
Between the 3-stage pre-filter and process_candidates() in
scan_events() and scan_slice() in trial_entry_scan.py:

    candidates = _router.filter_candidates(candidates, now_utc)

Hard constraints
----------------
- No Track A mutation.
- No live trading, no network calls.
- Errors never propagate to the caller (broad except at the boundary).
- Backward compatible: CohortRouter(enabled=False) is a no-op pass-through.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("polyarb.scanner.cohort_router")

# Import lazily to avoid circular dependency risk; function used only in scoring.
def _get_reward_bonus(m: dict) -> float:
    try:
        from src.research.reward_eval import reward_routing_bonus
        return reward_routing_bonus(m)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Event family labels
# ---------------------------------------------------------------------------

FAMILY_SAME_DAY_SPORTS    = "same_day_sports_props"
FAMILY_MULTI_DAY_SPORTS   = "multi_day_sports_futures"
FAMILY_ELECTION           = "election"
FAMILY_POLITICAL_BINARY   = "political_binary"
FAMILY_CRYPTO_DATE_TARGET = "crypto_date_target"
FAMILY_MACRO              = "macro_economic"
FAMILY_MISC               = "misc"

ALL_FAMILIES = [
    FAMILY_SAME_DAY_SPORTS,
    FAMILY_MULTI_DAY_SPORTS,
    FAMILY_ELECTION,
    FAMILY_POLITICAL_BINARY,
    FAMILY_CRYPTO_DATE_TARGET,
    FAMILY_MACRO,
    FAMILY_MISC,
]


# ---------------------------------------------------------------------------
# Slug / payload classification patterns
# ---------------------------------------------------------------------------

_SPORTS_RE = re.compile(
    r"\b(nba|nfl|nhl|mlb|ncaa|cbb|cfb|mls|epl|ufc|mma|boxing|"
    r"lol|esport|dota|csgo|valorant|rugby|soccer|hockey|"
    r"basketball|football|baseball|tennis|golf|nascar|f1|racing)\b",
    re.IGNORECASE,
)

_ELECTION_RE = re.compile(
    r"\b(election|elect|presidential|senator|congress|parliament|"
    r"primaries|caucus|governor|ballot|referendum|impeach)\b",
    re.IGNORECASE,
)

_POLITICAL_RE = re.compile(
    r"\b(trump|biden|harris|pelosi|mcconnell|zelensky|putin|"
    r"xi-jinping|netanyahu|modi|macron|will-the-us|will-congress|"
    r"will-senate|will-house|will-president|white-house|"
    r"supreme-court|cabinet|administration|sanctions)\b",
    re.IGNORECASE,
)

_CRYPTO_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|doge|xrp|"
    r"coinbase|binance|hit-\d+k|reach-\d+k|above-\d+k|below-\d+k)\b",
    re.IGNORECASE,
)

_MACRO_RE = re.compile(
    r"\b(fed|federal-reserve|rate-hike|rate-cut|inflation|cpi|gdp|"
    r"jobs|unemployment|recession|ecb|fomc|interest-rate|treasury|"
    r"debt-ceiling|payroll)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def end_date_horizon_hours(m: dict, now_utc: dt.datetime) -> Optional[float]:
    """Return hours to expiry or None if unparseable."""
    for key in ("endDate", "end_date_iso", "endDateIso"):
        end_str = m.get(key) or ""
        if end_str:
            try:
                end = dt.datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=dt.timezone.utc)
                if now_utc.tzinfo is None:
                    now_utc = now_utc.replace(tzinfo=dt.timezone.utc)
                return (end - now_utc).total_seconds() / 3600.0
            except Exception:
                pass
    return None


def classify_family(m: dict, now_utc: dt.datetime) -> str:
    """
    Classify a market dict into one of the seven event families.
    Uses slug patterns and horizon heuristic. No network calls.
    """
    slug = str(m.get("slug") or "").lower()
    hours = end_date_horizon_hours(m, now_utc)

    if _CRYPTO_RE.search(slug):
        return FAMILY_CRYPTO_DATE_TARGET

    if _MACRO_RE.search(slug):
        return FAMILY_MACRO

    if _ELECTION_RE.search(slug):
        return FAMILY_ELECTION

    if _POLITICAL_RE.search(slug):
        return FAMILY_POLITICAL_BINARY

    if _SPORTS_RE.search(slug):
        # Same-day = expires within 24h
        if hours is not None and hours <= 24.0:
            return FAMILY_SAME_DAY_SPORTS
        return FAMILY_MULTI_DAY_SPORTS

    return FAMILY_MISC


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

#: Per-family maximum candidates admitted to the CLOB fetch step per round.
DEFAULT_FAMILY_QUOTA: dict[str, int] = {
    FAMILY_SAME_DAY_SPORTS   : 3,
    FAMILY_MULTI_DAY_SPORTS  : 8,
    FAMILY_ELECTION          : 10,
    FAMILY_POLITICAL_BINARY  : 8,
    FAMILY_CRYPTO_DATE_TARGET: 5,
    FAMILY_MACRO             : 5,
    FAMILY_MISC              : 5,
}

#: Absolute cap on candidates passed to process_candidates() per round.
DEFAULT_MAX_TOTAL = 30

#: Cooldown thresholds: a slug is hard-suppressed once it exceeds BOTH.
DEFAULT_COOLDOWN_BOTH98_MIN = 5   # both_98_rounds >=
DEFAULT_COOLDOWN_CONSEC_MIN = 5   # consecutive_rounds_seen >=


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def score_candidate(
    m: dict,
    family: str,
    horizon_hours: Optional[float],
    plateau_state: Optional[dict],
    cooldown_both98: int = DEFAULT_COOLDOWN_BOTH98_MIN,
    cooldown_consec: int = DEFAULT_COOLDOWN_CONSEC_MIN,
) -> float:
    """
    Score one candidate. Higher = more budget-worthy.
    Returns -1e6 for hard-suppressed (cooled-down) slugs.
    """
    # --- Hard cooldown gate ---
    if plateau_state:
        both98 = plateau_state.get("both_98_rounds", 0) or 0
        consec = plateau_state.get("consecutive_rounds_seen", 0) or 0
        if both98 >= cooldown_both98 and consec >= cooldown_consec:
            return -1e6   # suppress entirely

    score = 0.0

    # Horizon bonus (longer remaining life = more information potential)
    if horizon_hours is not None:
        if horizon_hours > 168:    # > 7 days
            score += 30.0
        elif horizon_hours > 72:   # > 3 days
            score += 20.0
        elif horizon_hours > 24:   # > 1 day
            score += 10.0
        # <= 24h: 0

    # Family base score (reflects structural probability of executable edge)
    _family_base = {
        FAMILY_ELECTION          : 25.0,
        FAMILY_MACRO             : 20.0,
        FAMILY_POLITICAL_BINARY  : 18.0,
        FAMILY_CRYPTO_DATE_TARGET: 15.0,
        FAMILY_MULTI_DAY_SPORTS  : 12.0,
        FAMILY_MISC              : 8.0,
        FAMILY_SAME_DAY_SPORTS   : 2.0,
    }
    score += _family_base.get(family, 5.0)

    # Volume bonus (log-scale; already post-prefilter so vol > MIN_VOLUME_USD)
    try:
        vol = float(m.get("volume") or 0)
        if vol > 100_000:
            score += 15.0
        elif vol > 10_000:
            score += 8.0
        elif vol > 1_000:
            score += 3.0
    except Exception:
        pass

    # Soft saturation penalty (plateau state)
    if plateau_state:
        both98 = plateau_state.get("both_98_rounds", 0) or 0
        consec = plateau_state.get("consecutive_rounds_seen", 0) or 0
        score -= min(both98 * 4.0, 40.0)   # up to -40 for 10+ both_98 rounds
        score -= min(consec * 2.0, 20.0)   # up to -20 for 10+ consecutive

    # Reward routing bonus (from Gamma payload reward fields)
    score += _get_reward_bonus(m)

    return score


# ---------------------------------------------------------------------------
# RouterConfig
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """All knobs for the cohort router. Explicit and small."""
    family_quota    : dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_FAMILY_QUOTA)
    )
    max_total       : int  = DEFAULT_MAX_TOTAL
    cooldown_both98 : int  = DEFAULT_COOLDOWN_BOTH98_MIN
    cooldown_consec : int  = DEFAULT_COOLDOWN_CONSEC_MIN
    enabled         : bool = True


# ---------------------------------------------------------------------------
# CohortRouter
# ---------------------------------------------------------------------------

class CohortRouter:
    """
    Scores and filters a candidate list before the CLOB fetch step.

    Usage in trial_entry_scan.py (after both_98_out / pre-filter step):

        _router = CohortRouter(_tracker)
        # inside scan_events() / scan_slice(), before process_candidates():
        candidates = _router.filter_candidates(candidates, now_utc)

    When config.enabled is False, filter_candidates() is a pass-through.
    Errors are caught at the boundary; original list returned on failure.
    """

    def __init__(
        self,
        plateau_tracker=None,
        config: Optional[RouterConfig] = None,
    ) -> None:
        self._tracker = plateau_tracker
        self._cfg     = config or RouterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_candidates(
        self,
        candidates: list,
        now_utc: Optional[dt.datetime] = None,
    ) -> list:
        """
        Score, filter, and quota-allocate candidates.
        Returns filtered list ordered by score desc within each family quota.

        If router is disabled or candidates is empty, returns original list.
        Errors are caught and logged; original list is returned on any failure.
        """
        if not self._cfg.enabled or not candidates:
            return candidates
        try:
            return self._filter_inner(candidates, now_utc)
        except Exception as exc:
            logger.warning(
                "CohortRouter.filter_candidates failed (non-fatal): %s", exc
            )
            return candidates

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter_inner(
        self, candidates: list, now_utc: Optional[dt.datetime]
    ) -> list:
        if now_utc is None:
            now_utc = dt.datetime.now(dt.timezone.utc)

        plateau_map = self._load_plateau()

        # Score every candidate
        scored: list[tuple[float, str, dict]] = []
        for m in candidates:
            slug    = str(m.get("slug") or "?")
            family  = classify_family(m, now_utc)
            hours   = end_date_horizon_hours(m, now_utc)
            plateau = plateau_map.get(slug)
            s       = score_candidate(
                m, family, hours, plateau,
                self._cfg.cooldown_both98,
                self._cfg.cooldown_consec,
            )
            scored.append((s, family, m))

        # Descending score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Quota + total cap pass
        family_counts: dict[str, int] = {f: 0 for f in ALL_FAMILIES}
        selected:     list[dict] = []
        suppressed:   list[str]  = []
        quota_capped: list[str]  = []

        for s, family, m in scored:
            slug = str(m.get("slug") or "?")

            if s <= -1e5:
                suppressed.append(slug)
                continue

            quota = self._cfg.family_quota.get(family, 3)
            if family_counts[family] >= quota:
                quota_capped.append(f"{slug[:30]}({family})")
                continue

            if len(selected) >= self._cfg.max_total:
                break

            selected.append(m)
            family_counts[family] += 1

        # Telemetry line (always printed so it's visible in scan output)
        active_families = {f: c for f, c in family_counts.items() if c > 0}
        print(
            f"  [router] in={len(candidates)}  "
            f"suppressed={len(suppressed)}  "
            f"quota_capped={len(quota_capped)}  "
            f"out={len(selected)}  "
            f"families={active_families}"
        )
        if suppressed:
            shown = ", ".join(suppressed[:6])
            tail  = " ..." if len(suppressed) > 6 else ""
            print(f"  [router] cooled_down: {shown}{tail}")
        if quota_capped:
            shown = ", ".join(quota_capped[:4])
            tail  = " ..." if len(quota_capped) > 4 else ""
            print(f"  [router] quota_capped: {shown}{tail}")

        logger.info(
            "CohortRouter: in=%d suppressed=%d quota_capped=%d out=%d families=%s",
            len(candidates), len(suppressed), len(quota_capped),
            len(selected), active_families,
        )
        return selected

    def _load_plateau(self) -> dict[str, dict]:
        """
        Return slug → plateau_state dict from Both98PlateauTracker.
        Returns empty dict on any failure (non-fatal).
        """
        if self._tracker is None:
            return {}
        try:
            # Both98PlateauTracker loads its state lazily; _load_state() is
            # idempotent after first call.
            self._tracker._load_state()
            return dict(self._tracker._state)
        except Exception as exc:
            logger.debug("CohortRouter._load_plateau failed: %s", exc)
            return {}
