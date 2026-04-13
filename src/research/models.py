"""
Research pipeline data models — three explicit stages.

Stage 1  RawCandidate        detected anomaly — passed pre-filters, book not yet fetched
Stage 2  ExecutableCandidate  paper executable — book fetched, feasibility assessed
Stage 3  RankedOpportunity    ranked candidate — features computed, composite score assigned

These are research/paper-only constructs. Not connected to Track A execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Stage 1 — Detected anomaly
# ---------------------------------------------------------------------------

@dataclass
class RawCandidate:
    """
    Market passed event-level pre-filters (negRisk, structural slug, outcomePrices).
    CLOB book not yet fetched. Cost: zero CLOB API calls.
    """
    slug:        str
    p_yes:       float          # outcomePrices[0]
    p_no:        float          # outcomePrices[1]
    volume_usd:  float
    end_date:    Optional[str]
    yes_id:      str
    no_id:       str
    min_size:    float          # orderMinSize from Gamma payload
    scanned_at:  str            # ISO timestamp


# ---------------------------------------------------------------------------
# Stage 2 — Paper executable opportunity
# ---------------------------------------------------------------------------

@dataclass
class ExecutableCandidate:
    """
    CLOB book fetched. Execution feasibility assessed against trial-spec thresholds.
    Still paper-only — is_executable is a research label, not a live order trigger.
    """
    # From RawCandidate
    slug:        str
    p_yes:       float
    p_no:        float
    volume_usd:  float
    end_date:    Optional[str]
    yes_id:      str
    no_id:       str
    min_size:    float
    scanned_at:  str

    # From CLOB book
    yes_ask:          float
    no_ask:           float
    edge:             float       # 1 - yes_ask - no_ask

    # Depth within DEPTH_BAND of best ask
    yes_depth_shares: float
    no_depth_shares:  float
    yes_book_levels:  int         # ask price levels present in book response
    no_book_levels:   int

    # Feasibility labels (research use only)
    is_executable:    bool        # cleared depth + spread thresholds
    skip_reason:      str         # "" if executable, reason string otherwise


# ---------------------------------------------------------------------------
# Stage 3 — Ranked deployable-looking opportunity
# ---------------------------------------------------------------------------

@dataclass
class RankedOpportunity:
    """
    Full feature set computed. Composite score assigned. Rank assigned within run.
    Output artifact for the research report.
    """
    candidate: ExecutableCandidate

    # Logit-space features
    logit_p_yes:    float   # logit(p_yes from outcomePrices) — prior in log-odds space
    logit_ask_yes:  float   # logit(yes_ask) — market ask in log-odds space
    logit_spread:   float   # logit_ask_yes - logit_p_yes: ask premium over prior in logit space

    # Uncertainty
    uncertainty:    float   # p_yes * (1 - p_yes), peaks at 0.25 when p=0.50

    # Spread / depth
    spread_cents:             float
    spread_over_edge_ratio:   float
    depth_imbalance:          float   # (yes_depth - no_depth) / (total + 1)

    # Fragility
    fragility_score:     float   # 0..1, higher = more fragile

    # Persistence / belief-vol (cold-start = 0 on first run)
    persistence_rounds:  int
    belief_vol_proxy:    float   # stdev of p_yes across snapshots (naive) or theory σ_b

    # Theory-pipeline outputs (optional; populated when pipeline registry is active)
    sigma_b_theory:  float = 0.0    # EWMA σ_b from pipeline (theory-backed, logit units)
    p_filtered:      float = 0.0    # Kalman-filtered belief probability
    is_jump:         bool  = False  # EM separator jump classification

    # Composite
    composite_score:  float = 0.0
    rank:             int   = 0
    explanation:      str   = ""    # per-component score breakdown

    def to_dict(self) -> dict:
        c = self.candidate
        return {
            "rank":                    self.rank,
            "slug":                    c.slug,
            "composite_score":         round(self.composite_score, 4),
            # Raw market state
            "p_yes_prior":             round(c.p_yes, 4),
            "p_no_prior":              round(c.p_no, 4),
            "yes_ask":                 round(c.yes_ask, 4),
            "no_ask":                  round(c.no_ask, 4),
            "edge_cents":              round(c.edge * 100, 3),
            "volume_usd":              round(c.volume_usd, 2),
            # Logit features
            "logit_p_yes":             round(self.logit_p_yes, 4),
            "logit_ask_yes":           round(self.logit_ask_yes, 4),
            "logit_spread":            round(self.logit_spread, 4),
            # Uncertainty
            "uncertainty":             round(self.uncertainty, 4),
            # Spread / depth
            "spread_cents":            round(self.spread_cents, 3),
            "spread_over_edge_ratio":  round(self.spread_over_edge_ratio, 4),
            "yes_depth_shares":        round(c.yes_depth_shares, 1),
            "no_depth_shares":         round(c.no_depth_shares, 1),
            "depth_imbalance":         round(self.depth_imbalance, 4),
            "yes_book_levels":         c.yes_book_levels,
            "no_book_levels":          c.no_book_levels,
            # Fragility / persistence / belief-vol
            "fragility_score":         round(self.fragility_score, 4),
            "persistence_rounds":      self.persistence_rounds,
            "belief_vol_proxy":        round(self.belief_vol_proxy, 6),
            "sigma_b_theory":          round(self.sigma_b_theory, 6),
            "p_filtered":              round(self.p_filtered, 4),
            "is_jump":                 self.is_jump,
            # Feasibility labels
            "is_executable":           c.is_executable,
            "skip_reason":             c.skip_reason,
            # Meta
            "end_date":                c.end_date,
            "min_size":                c.min_size,
            "scanned_at":              c.scanned_at,
            "explanation":             self.explanation,
        }
