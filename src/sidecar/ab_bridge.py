"""
A+B Sidecar Bridge — observation-only layer.

When Track A emits a qualified candidate, this bridge computes Track B
features for that same candidate and stores a joined A+B observation record.

HARD CONSTRAINTS:
  - Track A is frozen and unchanged.
  - No thresholds, gates, or order logic are modified.
  - B score is stored as ranking/explanation signal ONLY.
  - B score does NOT trigger, suppress, or influence any Track A action.
  - All exceptions are caught internally; errors never propagate into Track A.

Storage:
  Writes to a separate SQLite file (ab_sidecar.db) to keep Track A tables
  untouched. The table ab_bridge_records can be joined to Track A's
  opportunity_candidates on candidate_id.

Feature extraction:
  Leg prices are read from Track A RawCandidate.legs (read-only access).
  p_yes_proxy is derived as (1 - no_ask) — an implied prior from the NO side.
  Track B feature computation delegates entirely to src.research.features.

In-memory rolling history:
  Persistence and belief-vol features require per-slug snapshot history.
  This state is ephemeral (process lifetime) and mirrors belief_ranker.py
  behaviour on a warm session.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.sql import insert

from src.research.features import (
    logit_features,
    uncertainty,
    spread_features,
    depth_features,
    fragility_score,
    persistence_rounds,
    belief_vol_proxy,
    composite_score_and_explanation,
)

logger = logging.getLogger("polyarb.sidecar.ab_bridge")

_DEFAULT_MIN_SIZE = 5.0          # Polymarket standard minimum order size (shares)
_MAX_HISTORY_SNAPSHOTS = 20      # rolling window per slug, matches belief_ranker.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sidecar_db_url(track_a_url: str) -> str:
    """Derive the sidecar SQLite URL from Track A's URL.

    sqlite:///data/processed/paper.db  →  sqlite:///data/processed/ab_sidecar.db
    """
    if "/" in track_a_url:
        prefix = track_a_url.rsplit("/", 1)[0]
        return f"{prefix}/ab_sidecar.db"
    return "sqlite:///data/processed/ab_sidecar.db"


def _extract_leg_prices(raw_candidate: Any) -> tuple[float | None, float | None, float, float]:
    """Read YES/NO ask prices and available depths from Track A RawCandidate.legs.

    Legs are matched by the `side` attribute ("YES" / "NO").  The leg's
    `best_price` is the best ask at detection time.  `available_shares` is
    the liquidity available on that side.

    Returns (yes_ask, no_ask, yes_depth, no_depth).
    Prices are None when the corresponding leg is absent.
    """
    yes_ask: float | None = None
    no_ask: float | None = None
    yes_depth: float = 0.0
    no_depth: float = 0.0

    for leg in getattr(raw_candidate, "legs", []):
        side = str(getattr(leg, "side", "")).upper()
        price = getattr(leg, "best_price", None)
        avail = float(getattr(leg, "available_shares", 0.0) or 0.0)

        if side == "YES":
            if price is not None:
                yes_ask = float(price)
            yes_depth = avail
        elif side == "NO":
            if price is not None:
                no_ask = float(price)
            no_depth = avail

    return yes_ask, no_ask, yes_depth, no_depth


# ---------------------------------------------------------------------------
# Main sidecar class
# ---------------------------------------------------------------------------

class ABSidecar:
    """Observation-only A+B bridge.

    Instantiate once inside ResearchRunner.__init__ and call observe() at
    every Track A candidate emission point.  The sidecar writes to its own
    SQLite table and never touches Track A state.

    Example (runner.py, minimal insertion):

        # __init__:
        self._ab_sidecar = ABSidecar(self.config.storage.sqlite_url)

        # after each self.store.save_candidate(candidate):
        self._ab_sidecar.observe(raw_candidate, candidate)
    """

    def __init__(self, track_a_sqlite_url: str) -> None:
        db_url = _sidecar_db_url(track_a_sqlite_url)
        if db_url.startswith("sqlite:///") and ":memory:" not in db_url:
            Path(db_url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(db_url, future=True)
        meta = MetaData()

        self._table = Table(
            "ab_bridge_records",
            meta,
            # --- identity ---
            Column("id",                        Integer, primary_key=True),
            Column("candidate_id",              String(128), nullable=False),
            Column("strategy_id",               String(64),  nullable=False),
            Column("strategy_family",           String(64),  nullable=False),
            Column("kind",                      String(64),  nullable=False),
            Column("market_slugs_json",         Text,        nullable=False),
            # --- Track A signals (observation, not copied into B logic) ---
            Column("a_gross_edge_cents",        Float,       nullable=False),
            Column("a_target_notional_usd",     Float,       nullable=False),
            # --- derived market state (read from A legs, used only for B scoring) ---
            Column("b_yes_ask",                 Float),
            Column("b_no_ask",                  Float),
            Column("b_p_yes_proxy",             Float),
            # --- Track B feature families ---
            Column("b_logit_p_yes",             Float),
            Column("b_logit_ask_yes",           Float),
            Column("b_logit_spread",            Float),
            Column("b_uncertainty",             Float),
            Column("b_spread_cents",            Float),
            Column("b_spread_over_edge_ratio",  Float),
            Column("b_depth_imbalance",         Float),
            Column("b_total_depth",             Float),
            Column("b_fragility",               Float),
            Column("b_persistence_rounds",      Integer),
            Column("b_belief_vol_proxy",        Float),
            # --- Track B composite output (ranking + explanation signals only) ---
            Column("b_score",                   Float,       nullable=False),
            Column("b_explanation",             Text,        nullable=False),
            Column("b_features_json",           Text,        nullable=False),
            Column("ts",                        DateTime,    nullable=False),
        )

        # Co-create both98_plateau so the runner's startup guarantees it exists.
        # Both98PlateauTracker uses the same DB; its own __init__ also calls
        # create_all — both are idempotent (CREATE TABLE IF NOT EXISTS semantics).
        Table(
            "both98_plateau",
            meta,
            Column("id",                      Integer, primary_key=True),
            Column("slug",                    String(256), nullable=False, unique=True),
            Column("total_rounds_seen",       Integer,     nullable=False, default=0),
            Column("both_98_rounds",          Integer,     nullable=False, default=0),
            Column("executable_rounds",       Integer,     nullable=False, default=0),
            Column("consecutive_rounds_seen", Integer,     nullable=False, default=0),
            Column("latest_edge_cents",       Float),
            Column("latest_persistence",      Integer),
            Column("latest_b_score",          Float),
            Column("plateau_flag",            Integer,     nullable=False, default=0),
            Column("last_seen_ts",            DateTime,    nullable=False),
        )

        meta.create_all(self.engine)

        # Ephemeral per-slug rolling history for persistence / belief-vol features.
        # Mirrors the snapshot state maintained by belief_ranker.py in-process.
        self._history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_scan_record(self, rec: dict, ts: Any) -> None:
        """Observe a viable_record dict produced by trial_entry_scan.process_candidates().

        Expected keys: slug, yes_ask, no_ask, edge (float), min_size (float).
        Depth is not available in this path; fragility uses depth=0.

        Errors are swallowed so they never affect the scan loop.
        """
        try:
            self._observe_scan_record_inner(rec, ts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ab_bridge.observe_scan_record failed (non-fatal): %s", exc)

    def _observe_scan_record_inner(self, rec: dict, ts: Any) -> None:
        import hashlib

        slug     = str(rec.get("slug", "?"))
        yes_ask  = float(rec["yes_ask"])
        no_ask   = float(rec["no_ask"])
        edge     = float(rec.get("edge", 1.0 - yes_ask - no_ask))
        min_size = float(rec.get("min_size", _DEFAULT_MIN_SIZE))

        # Stable candidate_id: slug + timestamp string (no UUID randomness so
        # repeated sidecar runs don't explode the table with duplicate slugs)
        ts_str       = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        candidate_id = hashlib.md5(f"scan:{slug}:{ts_str}".encode()).hexdigest()

        edge_cents  = edge * 100.0
        p_yes_proxy = float(max(0.01, min(0.99, 1.0 - no_ask)))

        lf  = logit_features(p_yes_proxy, yes_ask)
        sf  = spread_features(yes_ask, no_ask, edge)
        df  = depth_features(0.0, 0.0)   # depth not available from scan path
        u   = uncertainty(p_yes_proxy)
        fg  = fragility_score(yes_ask, no_ask, 0.0, 0.0, min_size)
        per = persistence_rounds(slug, self._history)
        bv  = belief_vol_proxy(slug, self._history)

        b_score, b_explanation = composite_score_and_explanation(
            edge                   = edge,
            uncertainty_val        = u,
            spread_over_edge_ratio = sf["spread_over_edge_ratio"],
            fragility              = fg,
            persistence            = per,
            belief_vol             = bv,
        )

        self._update_history(slug, p_yes_proxy, edge, ts_str)

        b_features = {
            **lf, **sf, **df,
            "uncertainty":        round(u,   6),
            "fragility":          round(fg,  6),
            "persistence_rounds": per,
            "belief_vol_proxy":   round(bv,  8),
            "p_yes_proxy":        round(p_yes_proxy, 6),
            "yes_ask":            round(yes_ask,     6),
            "no_ask":             round(no_ask,      6),
        }

        self._write_record(
            candidate_id=candidate_id,
            strategy_id="trial_entry_scan",
            strategy_family="single_market_mispricing",
            kind="single_market",
            market_slugs=[slug],
            gross_edge_cents=edge_cents,
            target_notional=0.0,
            yes_ask=yes_ask,
            no_ask=no_ask,
            p_yes_proxy=p_yes_proxy,
            logit_p_yes=lf.get("logit_p_yes"),
            logit_ask_yes=lf.get("logit_ask_yes"),
            logit_spread=lf.get("logit_spread"),
            b_uncertainty=u,
            spread_cents=sf.get("spread_cents"),
            spread_over_edge_ratio=sf.get("spread_over_edge_ratio"),
            depth_imbalance=df.get("depth_imbalance"),
            total_depth=df.get("total_depth"),
            fragility=fg,
            persistence=per,
            belief_vol=bv,
            b_score=b_score,
            b_explanation=b_explanation,
            b_features=b_features,
            ts=ts,
        )

    def observe(self, raw_candidate: Any, qualified: Any) -> None:
        """Compute Track B features for a Track A candidate and store the joined record.

        This method is the sole public entry point.  It wraps all logic in a
        broad exception handler so that sidecar failures never propagate into
        Track A execution.

        Args:
            raw_candidate: src.opportunity.models.RawCandidate — the raw signal
                           emitted by Track A strategy detection.
            qualified:     ExecutableCandidate or RankedOpportunity — the
                           qualified/ranked Track A candidate (used for metadata
                           only; no execution fields are read).
        """
        try:
            self._observe_inner(raw_candidate, qualified)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ab_bridge.observe failed (non-fatal, Track A unaffected): %s", exc)

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _observe_inner(self, raw_candidate: Any, qualified: Any) -> None:
        # -- identity fields --
        candidate_id    = str(raw_candidate.candidate_id)
        strategy_id     = str(raw_candidate.strategy_id)
        sf_val          = getattr(raw_candidate, "strategy_family", "")
        strategy_family = sf_val.value if hasattr(sf_val, "value") else str(sf_val)
        kind            = str(raw_candidate.kind)
        market_slugs    = list(raw_candidate.market_slugs)
        primary_slug    = market_slugs[0] if market_slugs else ""
        gross_edge_cents = float(raw_candidate.gross_edge_cents)
        target_notional  = float(raw_candidate.target_notional_usd)
        ts = getattr(raw_candidate, "ts", None) or datetime.now(timezone.utc)

        # -- extract market state from Track A legs (read-only) --
        yes_ask, no_ask, yes_depth, no_depth = _extract_leg_prices(raw_candidate)

        # If leg prices are unavailable we can still write a partial record,
        # but Track B features will be null.  This keeps the join intact.
        if yes_ask is None or no_ask is None:
            self._write_record(
                candidate_id=candidate_id,
                strategy_id=strategy_id,
                strategy_family=strategy_family,
                kind=kind,
                market_slugs=market_slugs,
                gross_edge_cents=gross_edge_cents,
                target_notional=target_notional,
                yes_ask=None, no_ask=None, p_yes_proxy=None,
                logit_p_yes=None, logit_ask_yes=None, logit_spread=None,
                b_uncertainty=None,
                spread_cents=None, spread_over_edge_ratio=None,
                depth_imbalance=None, total_depth=None,
                fragility=None, persistence=0, belief_vol=0.0,
                b_score=0.0,
                b_explanation="no_leg_prices",
                b_features={},
                ts=ts,
            )
            return

        edge = gross_edge_cents / 100.0
        # p_yes_proxy: implied prior from the NO-side ask.
        # Clamped to [0.01, 0.99] for logit stability.
        p_yes_proxy = float(max(0.01, min(0.99, 1.0 - no_ask)))

        # -- compute Track B feature families --
        lf  = logit_features(p_yes_proxy, yes_ask)
        sf  = spread_features(yes_ask, no_ask, edge)
        df  = depth_features(yes_depth, no_depth)
        u   = uncertainty(p_yes_proxy)
        fg  = fragility_score(yes_ask, no_ask, yes_depth, no_depth, _DEFAULT_MIN_SIZE)
        per = persistence_rounds(primary_slug, self._history)
        bv  = belief_vol_proxy(primary_slug, self._history)

        b_score, b_explanation = composite_score_and_explanation(
            edge                   = edge,
            uncertainty_val        = u,
            spread_over_edge_ratio = sf["spread_over_edge_ratio"],
            fragility              = fg,
            persistence            = per,
            belief_vol             = bv,
        )

        # Update rolling history AFTER scoring so this cycle's data feeds
        # the NEXT cycle's persistence/belief-vol computation.
        self._update_history(
            slug   = primary_slug,
            p_yes  = p_yes_proxy,
            edge   = edge,
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        )

        b_features = {
            **lf,
            **sf,
            **df,
            "uncertainty":        round(u,   6),
            "fragility":          round(fg,  6),
            "persistence_rounds": per,
            "belief_vol_proxy":   round(bv,  8),
            "p_yes_proxy":        round(p_yes_proxy, 6),
            "yes_ask":            round(yes_ask,     6),
            "no_ask":             round(no_ask,      6),
            "yes_depth":          round(yes_depth,   4),
            "no_depth":           round(no_depth,    4),
        }

        self._write_record(
            candidate_id=candidate_id,
            strategy_id=strategy_id,
            strategy_family=strategy_family,
            kind=kind,
            market_slugs=market_slugs,
            gross_edge_cents=gross_edge_cents,
            target_notional=target_notional,
            yes_ask=yes_ask,
            no_ask=no_ask,
            p_yes_proxy=p_yes_proxy,
            logit_p_yes=lf.get("logit_p_yes"),
            logit_ask_yes=lf.get("logit_ask_yes"),
            logit_spread=lf.get("logit_spread"),
            b_uncertainty=u,
            spread_cents=sf.get("spread_cents"),
            spread_over_edge_ratio=sf.get("spread_over_edge_ratio"),
            depth_imbalance=df.get("depth_imbalance"),
            total_depth=df.get("total_depth"),
            fragility=fg,
            persistence=per,
            belief_vol=bv,
            b_score=b_score,
            b_explanation=b_explanation,
            b_features=b_features,
            ts=ts,
        )

    def _update_history(self, slug: str, p_yes: float, edge: float, ts_str: str) -> None:
        entry = {"p_yes": p_yes, "edge": edge, "ts": ts_str}
        snaps = self._history.get(slug, [])
        snaps.append(entry)
        self._history[slug] = snaps[-_MAX_HISTORY_SNAPSHOTS:]

    def _write_record(
        self,
        candidate_id: str,
        strategy_id: str,
        strategy_family: str,
        kind: str,
        market_slugs: list,
        gross_edge_cents: float,
        target_notional: float,
        yes_ask: float | None,
        no_ask: float | None,
        p_yes_proxy: float | None,
        logit_p_yes: float | None,
        logit_ask_yes: float | None,
        logit_spread: float | None,
        b_uncertainty: float | None,
        spread_cents: float | None,
        spread_over_edge_ratio: float | None,
        depth_imbalance: float | None,
        total_depth: float | None,
        fragility: float | None,
        persistence: int,
        belief_vol: float,
        b_score: float,
        b_explanation: str,
        b_features: dict,
        ts: Any,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(self._table).values(
                candidate_id              = candidate_id,
                strategy_id               = strategy_id,
                strategy_family           = strategy_family,
                kind                      = kind,
                market_slugs_json         = json.dumps(market_slugs),
                a_gross_edge_cents        = gross_edge_cents,
                a_target_notional_usd     = target_notional,
                b_yes_ask                 = yes_ask,
                b_no_ask                  = no_ask,
                b_p_yes_proxy             = p_yes_proxy,
                b_logit_p_yes             = logit_p_yes,
                b_logit_ask_yes           = logit_ask_yes,
                b_logit_spread            = logit_spread,
                b_uncertainty             = b_uncertainty,
                b_spread_cents            = spread_cents,
                b_spread_over_edge_ratio  = spread_over_edge_ratio,
                b_depth_imbalance         = depth_imbalance,
                b_total_depth             = total_depth,
                b_fragility               = fragility,
                b_persistence_rounds      = persistence,
                b_belief_vol_proxy        = belief_vol,
                b_score                   = b_score,
                b_explanation             = b_explanation,
                b_features_json           = json.dumps(b_features),
                ts                        = ts,
            ))
