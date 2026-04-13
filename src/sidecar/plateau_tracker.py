"""
Both98PlateauTracker — observation-only both_98 saturation / plateau detector.

Tracks slugs that repeatedly appear with yes_ask >= 0.98 AND no_ask >= 0.98
across rounds of belief_ranker.py.  Writes to a single SQLite table
(both98_plateau) in the sidecar DB.  One row per slug, upserted each round.

HARD CONSTRAINTS:
  - Observation only.  No Track A policy, gate, or threshold is read or written.
  - Errors never propagate to the caller (broad except in record_round).
  - Does not import from src.runtime.*, src.live.*, src.paper.*, src.risk.*.

Plateau rule
------------
plateau_flag = 1  iff ALL of:
  1. both_98_rounds >= 3    (seen as both_98 at least 3 rounds)
  2. consecutive_rounds_seen >= 3  (no gap in the last 3 rounds)
  3. executable_rounds == 0  (never cleared book thresholds to executable)

Conservative and simple.  Requires sustained saturation with zero execution
signal.

Usage (belief_ranker.py)
------------------------
    tracker = Both98PlateauTracker("sqlite:///data/processed/ab_sidecar.db")
    # inside one_round():
    tracker.record_round(both_98_records, ranked)
    # both_98_records: list of dicts from build_executable_candidates()
    # ranked: list of RankedOpportunity (for b_score lookup)

Inspect:
    python -c "
    import sqlite3, json
    rows = sqlite3.connect('data/processed/ab_sidecar.db').execute(
        'SELECT slug,both_98_rounds,consecutive_rounds_seen,executable_rounds,'
        'latest_edge_cents,latest_b_score,plateau_flag,last_seen_ts '
        'FROM both98_plateau WHERE plateau_flag=1 ORDER BY both_98_rounds DESC'
    ).fetchall(); [print(r) for r in rows]
    "
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, select
from sqlalchemy.sql import insert, update

logger = logging.getLogger("polyarb.sidecar.plateau_tracker")


class Both98PlateauTracker:
    """Tracks both_98 slug saturation across belief_ranker rounds.

    State is kept in-memory (per process) and persisted to SQLite after each
    round.  The in-memory dict is the source of truth for consecutive_rounds
    counting; SQLite is the persistent log.
    """

    # Plateau thresholds — simple, explicit, easy to audit.
    PLATEAU_BOTH98_MIN   = 3   # both_98_rounds >= this
    PLATEAU_CONSEC_MIN   = 3   # consecutive_rounds_seen >= this
    PLATEAU_EXEC_MAX     = 0   # executable_rounds <= this (must be zero)

    def __init__(self, db_url: str) -> None:
        self.engine = create_engine(db_url, future=True)
        meta = MetaData()

        self._table = Table(
            "both98_plateau",
            meta,
            Column("id",                    Integer, primary_key=True),
            Column("slug",                  String(256), nullable=False, unique=True),
            Column("total_rounds_seen",     Integer,     nullable=False, default=0),
            Column("both_98_rounds",        Integer,     nullable=False, default=0),
            Column("executable_rounds",     Integer,     nullable=False, default=0),
            Column("consecutive_rounds_seen", Integer,   nullable=False, default=0),
            Column("latest_edge_cents",     Float),
            Column("latest_persistence",    Integer),
            Column("latest_b_score",        Float),
            Column("plateau_flag",          Integer,     nullable=False, default=0),
            Column("last_seen_ts",          DateTime,    nullable=False),
        )

        meta.create_all(self.engine)

        # In-memory state per slug.  Keys: same as column names above.
        # Populated lazily from SQLite on first access per slug.
        self._state: dict[str, dict] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_round(self, both_98_records: list[dict], ranked: list) -> None:
        """Update plateau state for all both_98 slugs seen this round.

        Args:
            both_98_records: list of dicts, each with keys:
                slug, yes_ask, no_ask, edge_cents
                Produced by belief_ranker.build_executable_candidates().
            ranked: list of RankedOpportunity objects from rank_candidates().
                Used for b_score / persistence lookup when slug was scored
                (only relevant when --include-both-98 is active).
        """
        try:
            self._record_round_inner(both_98_records, ranked)
        except Exception as exc:  # noqa: BLE001
            logger.warning("plateau_tracker.record_round failed (non-fatal): %s", exc)

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load existing rows from SQLite into in-memory state (once per process)."""
        if self._loaded:
            return
        stmt = select(self._table)
        with self.engine.connect() as conn:
            for row in conn.execute(stmt):
                r = dict(row._mapping)
                self._state[r["slug"]] = r
        self._loaded = True

    def _record_round_inner(self, both_98_records: list[dict], ranked: list) -> None:
        if not both_98_records:
            return

        self._load_state()

        # Build b_score lookup from ranked (slug → (composite_score, persistence_rounds))
        b_lookup: dict[str, tuple[float, int]] = {}
        for r in ranked:
            try:
                slug  = r.candidate.slug
                score = float(getattr(r, "composite_score", 0.0) or 0.0)
                pers  = int(getattr(r, "persistence_rounds", 0) or 0)
                b_lookup[slug] = (score, pers)
            except Exception:
                continue

        ts_now = datetime.now(timezone.utc).replace(tzinfo=None)  # store as naive UTC

        for rec in both_98_records:
            slug       = rec["slug"]
            edge_cents = float(rec.get("edge_cents", 0.0))
            b_score, persistence = b_lookup.get(slug, (None, None))

            state = self._state.get(slug)
            if state is None:
                state = {
                    "slug":                   slug,
                    "total_rounds_seen":      0,
                    "both_98_rounds":         0,
                    "executable_rounds":      0,
                    "consecutive_rounds_seen": 0,
                    "latest_edge_cents":      None,
                    "latest_persistence":     None,
                    "latest_b_score":         None,
                    "plateau_flag":           0,
                    "last_seen_ts":           None,
                    "_in_db":                 False,
                }
                self._state[slug] = state

            state["total_rounds_seen"]       += 1
            state["both_98_rounds"]          += 1
            state["consecutive_rounds_seen"] += 1
            state["latest_edge_cents"]        = round(edge_cents, 4)
            if persistence is not None:
                state["latest_persistence"]   = persistence
            if b_score is not None:
                state["latest_b_score"]       = round(b_score, 6)
            state["last_seen_ts"]             = ts_now

            # Plateau rule (explicit, all three conditions)
            state["plateau_flag"] = int(
                state["both_98_rounds"]         >= self.PLATEAU_BOTH98_MIN
                and state["consecutive_rounds_seen"] >= self.PLATEAU_CONSEC_MIN
                and state["executable_rounds"]   <= self.PLATEAU_EXEC_MAX
            )

            self._upsert(state)

    def _upsert(self, state: dict) -> None:
        slug     = state["slug"]
        in_db    = state.get("_in_db", True)
        row_vals = {k: v for k, v in state.items() if not k.startswith("_")}

        with self.engine.begin() as conn:
            if not in_db:
                conn.execute(insert(self._table).values(**row_vals))
                state["_in_db"] = True
            else:
                conn.execute(
                    update(self._table)
                    .where(self._table.c.slug == slug)
                    .values(**row_vals)
                )
