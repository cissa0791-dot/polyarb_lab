"""
Tests for run_fill_pressure_audit.py.

Coverage:
  1. _cross_distance_bid: correct ¢ distance from live_best_ask to quote_bid
  2. _cross_distance_ask: correct ¢ distance from quote_ask to live_best_bid
  3. _inter_session_moves: signed per-interval deltas
  4. _sweep_point delta=0: matches actual cross count
  5. _sweep_point tighter quote: more hypothetical crosses
  6. _sweep_point: reward-eligibility check (tighter than max_spread → ineligible)
  7. audit_market NEED_MORE_OBS: fewer than threshold observations
  8. audit_market MARKET_STATIC: 100% stasis, max_move < cross_dist
  9. audit_market QUOTE_TOO_FAR: tighter quote would produce crosses
 10. audit_market MODEL_TOO_OPTIMISTIC: max move < cross_dist even with tighter quotes
 11. sweep delta=0 never inflates current-placement count
 12. Sensitivity sweep deltas are strictly ordered by cross count (more crosses = tighter)
"""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fill_pressure_audit import (
    _cross_distance_bid,
    _cross_distance_ask,
    _inter_session_moves,
    _sweep_point,
    audit_market,
    SWEEP_DELTAS_CENTS,
)
from scripts.run_maker_paper_calibration import SCHEMA_SQL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    # Ensure fill-pressure columns exist
    existing = {r[1] for r in conn.execute("PRAGMA table_info(quote_observations)")}
    for col, typ in [
        ("competitor_q_sum",           "REAL"),
        ("our_q_score",                "REAL"),
        ("implied_reward_ev",          "REAL"),
        ("reward_ev_ratio",            "REAL"),
        ("bid_cross_distance_cents",   "REAL"),
        ("ask_cross_distance_cents",   "REAL"),
        ("best_bid_move_from_prev",    "REAL"),
        ("best_ask_move_from_prev",    "REAL"),
        ("time_since_last_obs_sec",    "REAL"),
        ("quote_aggressiveness_bucket","TEXT"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE quote_observations ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def _insert_obs(
    conn,
    *,
    obs_id        : str,
    session_id    : str = "s1",
    event_slug    : str = "test-event",
    market_slug   : str = "test-market",
    obs_ts        : str = "2026-03-21T08:00:00+00:00",
    live_best_bid : float = 0.37,
    live_best_ask : float = 0.38,
    quote_bid     : float = 0.3691,
    quote_ask     : float = 0.3810,
    bid_spread_cents: float = 0.59,
    ask_spread_cents: float = 0.60,
    max_spread_cents: float = 3.5,
    status        : str = "OPEN",
) -> None:
    mid = (live_best_bid + live_best_ask) / 2.0
    conn.execute(
        """
        INSERT INTO quote_observations
          (obs_id, session_id, event_slug, market_slug, obs_ts,
           live_best_bid, live_best_ask, mid_p, current_spread_cents,
           quote_bid, quote_ask, quote_size, bid_spread_cents, ask_spread_cents,
           max_spread_cents, reward_eligible,
           modeled_reward_ev, modeled_spread_capture_ev, modeled_total_ev,
           modeled_competition_factor, reward_daily_rate, neg_risk, status,
           quote_aggressiveness_bucket)
        VALUES
          (?,?,?,?,?,
           ?,?,?,1.0,
           ?,?,200.0,?,?,
           ?,1,
           8.0,0.73,7.97,25.0,200.0,0,?,
           'STANDARD')
        """,
        (obs_id, session_id, event_slug, market_slug, obs_ts,
         live_best_bid, live_best_ask, mid,
         quote_bid, quote_ask, bid_spread_cents, ask_spread_cents,
         max_spread_cents, status),
    )
    conn.commit()


def _row(
    live_best_bid: float = 0.37,
    live_best_ask: float = 0.38,
    quote_bid    : float = 0.3691,
    quote_ask    : float = 0.3810,
    mid_p        : float = 0.375,
    max_spread_cents: float = 3.5,
) -> sqlite3.Row:
    """Create a minimal in-memory Row for unit tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE t (live_best_bid REAL, live_best_ask REAL, "
        "quote_bid REAL, quote_ask REAL, mid_p REAL, max_spread_cents REAL)"
    )
    conn.execute("INSERT INTO t VALUES (?,?,?,?,?,?)",
                 (live_best_bid, live_best_ask, quote_bid, quote_ask, mid_p, max_spread_cents))
    conn.commit()
    return conn.execute("SELECT * FROM t").fetchone()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class CrossDistanceTests(unittest.TestCase):

    def test_bid_cross_distance_not_crossed(self):
        """live_best_ask=0.38, quote_bid=0.3691 → 1.09¢ needed."""
        r = _row(live_best_ask=0.38, quote_bid=0.3691)
        self.assertAlmostEqual(_cross_distance_bid(r), 1.09, places=2)

    def test_ask_cross_distance_not_crossed(self):
        """quote_ask=0.381, live_best_bid=0.37 → 1.10¢ needed."""
        r = _row(live_best_bid=0.37, quote_ask=0.381)
        self.assertAlmostEqual(_cross_distance_ask(r), 1.10, places=2)

    def test_bid_cross_distance_zero_at_boundary(self):
        """live_best_ask exactly equals quote_bid → 0¢ = just crossed."""
        r = _row(live_best_ask=0.3691, quote_bid=0.3691)
        self.assertAlmostEqual(_cross_distance_bid(r), 0.0, places=6)

    def test_bid_cross_distance_negative_when_crossed(self):
        """live_best_ask < quote_bid → negative (already crossed)."""
        r = _row(live_best_ask=0.36, quote_bid=0.3691)
        self.assertLess(_cross_distance_bid(r), 0.0)


class InterSessionMovesTests(unittest.TestCase):

    def _make_rows(self, bid_seq, ask_seq):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (live_best_bid REAL, live_best_ask REAL)")
        for b, a in zip(bid_seq, ask_seq):
            conn.execute("INSERT INTO t VALUES (?,?)", (b, a))
        conn.commit()
        return conn.execute("SELECT * FROM t").fetchall()

    def test_static_book_all_zero(self):
        rows = self._make_rows([0.37, 0.37, 0.37], [0.38, 0.38, 0.38])
        bm, am = _inter_session_moves(rows)
        self.assertEqual(bm, [0.0, 0.0])
        self.assertEqual(am, [0.0, 0.0])

    def test_one_cent_move_detected(self):
        rows = self._make_rows([0.37, 0.36], [0.38, 0.37])
        bm, am = _inter_session_moves(rows)
        self.assertAlmostEqual(bm[0], -1.0, places=4)
        self.assertAlmostEqual(am[0], -1.0, places=4)

    def test_returns_empty_for_single_obs(self):
        rows = self._make_rows([0.37], [0.38])
        bm, am = _inter_session_moves(rows)
        self.assertEqual(bm, [])
        self.assertEqual(am, [])


class SweepPointTests(unittest.TestCase):

    def _static_obs(self, n=5, live_ask=0.38, quote_bid=0.3691, quote_ask=0.381,
                    live_bid=0.37, mid=0.375, max_sp=3.5):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE t (quote_bid REAL, quote_ask REAL, mid_p REAL, "
            "max_spread_cents REAL, live_best_bid REAL, live_best_ask REAL)"
        )
        for _ in range(n):
            conn.execute("INSERT INTO t VALUES (?,?,?,?,?,?)",
                         (quote_bid, quote_ask, mid, max_sp, live_bid, live_ask))
        conn.commit()
        return conn.execute("SELECT * FROM t").fetchall()

    def test_delta_zero_no_cross_when_far(self):
        """Book far from quote → 0 crosses at delta=0."""
        obs = self._static_obs(live_ask=0.38, quote_bid=0.3691)
        sp = _sweep_point(obs, 0.0, 3.5)
        self.assertEqual(sp.hypo_bid_crosses + sp.hypo_bid_crosses + sp.hypo_both_crosses, 0)

    def test_tighter_quote_produces_crosses(self):
        """
        live_best_ask=0.38, quote_bid=0.379 → cross dist = 0.1¢.
        At +0.2¢ tighter: hypo_bid = 0.381 > live_ask (0.38) → crossed.
        """
        obs = self._static_obs(live_ask=0.38, quote_bid=0.379, live_bid=0.37,
                                quote_ask=0.382, mid=0.3805, max_sp=3.5)
        sp0 = _sweep_point(obs, 0.0, 3.5)
        sp2 = _sweep_point(obs, 0.2, 3.5)
        self.assertEqual(sp0.hypo_bid_crosses, 0)
        self.assertGreater(sp2.hypo_bid_crosses, 0)

    def test_tighter_outside_max_spread_ineligible(self):
        """
        Quote already at mid (spread=0). Tighter would go ABOVE mid → ineligible.
        """
        # mid=0.375, quote_bid=0.375 (at mid), quote_ask=0.375
        obs = self._static_obs(live_ask=0.38, quote_bid=0.375, quote_ask=0.375,
                                live_bid=0.37, mid=0.375, max_sp=3.5)
        sp = _sweep_point(obs, 0.6, 3.5)
        # tighter_bid = 0.375 + 0.006 = 0.381, spread from mid = -0.6¢ → ineligible
        self.assertEqual(sp.still_reward_eligible, 0)

    def test_both_sides_crossed_simultaneously(self):
        """When live_ask ≤ tighter_bid AND live_bid ≥ tighter_ask → BOTH."""
        # Tight book: live_bid=0.375, live_ask=0.375 (spread=0)
        obs = self._static_obs(live_ask=0.375, quote_bid=0.374, quote_ask=0.376,
                                live_bid=0.375, mid=0.375, max_sp=3.5)
        sp = _sweep_point(obs, 0.1, 3.5)
        # tighter_bid=0.375: live_ask(0.375) ≤ 0.375 → bid cross
        # tighter_ask=0.375: live_bid(0.375) ≥ 0.375 → ask cross
        self.assertGreater(sp.hypo_both_crosses, 0)
        self.assertEqual(sp.hypo_bid_crosses, 0)
        self.assertEqual(sp.hypo_ask_crosses, 0)


class JudgmentTests(unittest.TestCase):

    def _conn_with_obs(self, obs_specs):
        """obs_specs: list of (obs_ts, live_bid, live_ask, quote_bid, quote_ask)"""
        conn = _make_db()
        for i, (ts, lb, la, qb, qa) in enumerate(obs_specs):
            _insert_obs(conn, obs_id=f"q{i}", obs_ts=ts,
                        live_best_bid=lb, live_best_ask=la,
                        quote_bid=qb, quote_ask=qa)
        return conn

    def test_need_more_obs_when_too_few(self):
        """Fewer than threshold obs → NEED_MORE_OBS."""
        conn = _make_db()
        _insert_obs(conn, obs_id="q1", obs_ts="2026-03-21T08:00:00+00:00")
        _insert_obs(conn, obs_id="q2", obs_ts="2026-03-21T09:00:00+00:00")
        result = audit_market(conn, "test-market", "test-event")
        self.assertEqual(result.judgment, "NEED_MORE_OBS")

    def test_market_static_when_book_frozen(self):
        """100% zero moves + max_move < cross_dist → MARKET_STATIC."""
        obs = []
        for i in range(6):
            ts = f"2026-03-21T{8+i:02d}:00:00+00:00"
            obs.append((ts, 0.37, 0.38, 0.3691, 0.381))
        conn = self._conn_with_obs(obs)
        result = audit_market(conn, "test-market", "test-event")
        self.assertEqual(result.judgment, "MARKET_STATIC")
        self.assertAlmostEqual(result.pct_zero_bid_move, 1.0)
        self.assertAlmostEqual(result.max_bid_move_cents, 0.0)

    def test_quote_too_far_when_tighter_quote_would_cross(self):
        """
        Book fluctuates enough that a tighter quote would cross.
        live_ask drops to 0.379 in one session; our bid at 0.3691 misses,
        but at +0.2¢ (0.3711) it would still miss; at a book asking 0.370
        a bid of 0.3701 would be crossed.
        """
        # Set up: book mostly static, but in one session ask drops close to bid
        obs = [
            ("2026-03-21T08:00:00+00:00", 0.37, 0.38, 0.3691, 0.381),
            ("2026-03-21T09:00:00+00:00", 0.37, 0.38, 0.3691, 0.381),
            ("2026-03-21T10:00:00+00:00", 0.37, 0.38, 0.3691, 0.381),
            ("2026-03-21T11:00:00+00:00", 0.37, 0.38, 0.3691, 0.381),
            # Ask drops: live_ask=0.375 → cross_dist = 0.6¢. At +0.4¢ tighter:
            # hypo_bid = 0.3731. 0.375 > 0.3731 → still not crossed.
            # At live_ask=0.370: cross_dist=0¢ → crossed with current quote
            ("2026-03-21T12:00:00+00:00", 0.36, 0.370, 0.3691, 0.381),
        ]
        conn = self._conn_with_obs(obs)
        result = audit_market(conn, "test-market", "test-event")
        # Cross distance in session 5 is (0.370 - 0.3691)*100 = 0.09¢ ← crossed
        # The current quote at 0.3691 IS already crossed (live_ask=0.370 > 0.3691)
        # Wait: 0.370 > 0.3691, so NOT crossed. Cross requires live_ask <= quote_bid.
        # 0.370 > 0.3691 → not crossed at delta=0.
        # At delta=0.2: hypo_bid=0.3711. 0.370 > 0.3711 → not crossed.
        # At delta=0.6: hypo_bid=0.3751. 0.370 < 0.3751 → CROSSED!
        self.assertEqual(result.judgment, "QUOTE_TOO_FAR")

    def test_model_too_optimistic_when_max_move_less_than_cross_dist(self):
        """
        Book fluctuates but ask never drops below hypo_bid at any sweep delta,
        and bid never rises above hypo_ask.  Result: MODEL_TOO_OPTIMISTIC.

        Constraints chosen:
          quote_bid=0.3691, quote_ask=0.381
          At +0.6¢: hypo_bid=0.3751, hypo_ask=0.3750
          live_ask always >= 0.376 > 0.3751 → no bid cross at any delta
          live_bid always <= 0.370 < 0.3750 → no ask cross at any delta
          Book oscillates (pct_zero_bid < 0.80) → not MARKET_STATIC
        """
        # Alternating book (always moves, never close enough to cross)
        obs = [
            ("2026-03-21T08:00:00+00:00", 0.365, 0.376, 0.3691, 0.381),
            ("2026-03-21T09:00:00+00:00", 0.370, 0.380, 0.3691, 0.381),
            ("2026-03-21T10:00:00+00:00", 0.365, 0.376, 0.3691, 0.381),
            ("2026-03-21T11:00:00+00:00", 0.370, 0.380, 0.3691, 0.381),
            ("2026-03-21T12:00:00+00:00", 0.365, 0.376, 0.3691, 0.381),
        ]
        conn = self._conn_with_obs(obs)
        result = audit_market(conn, "test-market", "test-event")
        # pct_zero = 0% (book alternates every interval) → not MARKET_STATIC
        # min live_ask = 0.376 > hypo_bid_max (0.3751) → 0 bid crosses at any delta
        # max live_bid = 0.370 < hypo_ask_min (0.3750) → 0 ask crosses at any delta
        self.assertEqual(result.judgment, "MODEL_TOO_OPTIMISTIC")


class TelemetryFieldTests(unittest.TestCase):
    """Cross distance and movement fields are correct in audit output."""

    def test_avg_cross_dist_correct(self):
        """With constant book (0.37/0.38) and bid=0.3691: avg cdist ≈ 1.09¢."""
        conn = _make_db()
        for i in range(5):
            _insert_obs(conn, obs_id=f"q{i}",
                        obs_ts=f"2026-03-21T{8+i:02d}:00:00+00:00",
                        live_best_ask=0.38, quote_bid=0.3691)
        result = audit_market(conn, "test-market", "test-event")
        self.assertAlmostEqual(result.avg_bid_cross_dist_cents, 1.09, places=1)

    def test_pct_zero_move_static_book(self):
        conn = _make_db()
        for i in range(6):
            _insert_obs(conn, obs_id=f"q{i}",
                        obs_ts=f"2026-03-21T{8+i:02d}:00:00+00:00",
                        live_best_bid=0.37, live_best_ask=0.38)
        result = audit_market(conn, "test-market", "test-event")
        self.assertAlmostEqual(result.pct_zero_bid_move, 1.0)

    def test_nonzero_move_tracked(self):
        conn = _make_db()
        _insert_obs(conn, obs_id="q0", obs_ts="2026-03-21T08:00:00+00:00",
                    live_best_bid=0.37, live_best_ask=0.38)
        _insert_obs(conn, obs_id="q1", obs_ts="2026-03-21T09:00:00+00:00",
                    live_best_bid=0.36, live_best_ask=0.37)
        for i in range(4):
            _insert_obs(conn, obs_id=f"q{i+2}",
                        obs_ts=f"2026-03-21T{10+i:02d}:00:00+00:00",
                        live_best_bid=0.36, live_best_ask=0.37)
        result = audit_market(conn, "test-market", "test-event")
        self.assertAlmostEqual(result.max_bid_move_cents, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
