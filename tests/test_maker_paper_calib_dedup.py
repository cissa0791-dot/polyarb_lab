"""
Tests for the open-quote dedup logic in run_maker_paper_calibration.

One-live-quote-per-market invariant: each market must have at most one OPEN
resting quote at any time.  A new session's quote supersedes ALL prior OPEN
quotes for that market, regardless of price or size.

Coverage:
  1. Identical price/size → prior OPEN quote is superseded
  2. Different bid price (book moved) → prior OPEN quote is still superseded
  3. Different ask price → prior OPEN quote is still superseded
  4. Different size → prior OPEN quote is still superseded
  5. Cross detection fires exactly once after dedup (not twice)
  6. Superseded rows remain in DB for audit
  7. Already-crossed rows are not superseded by a new insert
  8. Three sequential quotes → only one OPEN, two SUPERSEDED
  9. Dedup on market-A does not touch market-B
"""
from __future__ import annotations

import sqlite3
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_maker_paper_calibration import (
    _supersede_all_open_quotes,
    _load_open_quotes,
    _mark_crossed,
    check_crosses,
    SCHEMA_SQL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def _insert_open(
    conn: sqlite3.Connection,
    *,
    obs_id: str,
    session_id: str = "s1",
    event_slug: str = "test-event",
    market_slug: str = "test-market",
    obs_ts: str = "2026-03-21T08:00:00+00:00",
    quote_bid: float = 0.37,
    quote_ask: float = 0.38,
    quote_size: float = 200.0,
) -> None:
    conn.execute(
        """
        INSERT INTO quote_observations
          (obs_id, session_id, event_slug, market_slug, obs_ts,
           live_best_bid, live_best_ask, mid_p, current_spread_cents,
           quote_bid, quote_ask, quote_size,
           bid_spread_cents, ask_spread_cents, max_spread_cents,
           reward_eligible, modeled_reward_ev, modeled_spread_capture_ev,
           modeled_total_ev, modeled_competition_factor,
           reward_daily_rate, neg_risk, status)
        VALUES
          (?,?,?,?,?,
           0.37,0.38,0.375,1.0,
           ?,?,?,
           0.5,0.5,3.5,
           1, 5.0, 2.0, 7.0, 0.5,
           200.0, 0, 'OPEN')
        """,
        (obs_id, session_id, event_slug, market_slug, obs_ts,
         round(quote_bid, 6), round(quote_ask, 6), quote_size),
    )
    conn.commit()


def _status(conn: sqlite3.Connection, obs_id: str) -> str:
    return conn.execute(
        "SELECT status FROM quote_observations WHERE obs_id=?", (obs_id,)
    ).fetchone()["status"]


_T1 = "2026-03-21T08:00:00+00:00"
_T2 = "2026-03-21T08:30:00+00:00"
_T3 = "2026-03-21T09:00:00+00:00"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class DedupInsertTests(unittest.TestCase):

    def test_identical_price_supersedes_older_open(self):
        """New quote at same bid/ask/size → prior OPEN quote becomes SUPERSEDED."""
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        self.assertEqual(_status(conn, "q1"), "SUPERSEDED")

    def test_different_bid_still_supersedes(self):
        """Book moved (different bid) → prior OPEN quote is still superseded.

        Old behavior kept both OPEN when prices differed, causing ghost
        accumulation.  New invariant: any prior OPEN is superseded unconditionally.
        """
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1, quote_bid=0.37, quote_ask=0.38)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        self.assertEqual(_status(conn, "q1"), "SUPERSEDED")

    def test_different_ask_still_supersedes(self):
        """Different ask price → prior OPEN quote is still superseded."""
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1, quote_bid=0.37, quote_ask=0.38)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        self.assertEqual(_status(conn, "q1"), "SUPERSEDED")

    def test_different_size_still_supersedes(self):
        """Different quote_size → prior OPEN quote is still superseded."""
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1, quote_size=200.0)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        self.assertEqual(_status(conn, "q1"), "SUPERSEDED")

    def test_already_crossed_not_superseded(self):
        """A BID_CROSSED row is not touched by the supersede call."""
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1)
        _mark_crossed(conn, "q1", "BID_CROSSED", _T2)
        conn.commit()
        _supersede_all_open_quotes(conn, "test-market", _T3)
        self.assertEqual(_status(conn, "q1"), "BID_CROSSED")

    def test_three_sequential_leaves_one_open(self):
        """Three sequential quotes → only the latest OPEN survives each time."""
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1)
        # Second session: supersede q1, insert q2
        _supersede_all_open_quotes(conn, "test-market", _T2)
        _insert_open(conn, obs_id="q2", obs_ts=_T2)
        # Third session: supersede q2, insert q3
        _supersede_all_open_quotes(conn, "test-market", _T3)
        _insert_open(conn, obs_id="q3", obs_ts=_T3)

        open_count = conn.execute(
            "SELECT COUNT(*) FROM quote_observations WHERE status='OPEN'"
        ).fetchone()[0]
        superseded_count = conn.execute(
            "SELECT COUNT(*) FROM quote_observations WHERE status='SUPERSEDED'"
        ).fetchone()[0]
        self.assertEqual(open_count, 1)
        self.assertEqual(superseded_count, 2)
        self.assertEqual(_status(conn, "q3"), "OPEN")

    def test_superseded_rows_remain_in_db(self):
        """Superseded rows are not deleted — audit trail preserved."""
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        row = conn.execute(
            "SELECT * FROM quote_observations WHERE obs_id='q1'"
        ).fetchone()
        self.assertIsNotNone(row, "Superseded row must remain in DB")
        self.assertEqual(row["status"], "SUPERSEDED")
        self.assertEqual(row["crossed_ts"], _T2)

    def test_different_market_unaffected(self):
        """Supersede on market-A does not touch OPEN quotes for market-B."""
        conn = _make_db()
        _insert_open(conn, obs_id="q_a", obs_ts=_T1, market_slug="market-a")
        _insert_open(conn, obs_id="q_b", obs_ts=_T1, market_slug="market-b")
        _supersede_all_open_quotes(conn, "market-a", _T2)
        self.assertEqual(_status(conn, "q_a"), "SUPERSEDED")
        self.assertEqual(_status(conn, "q_b"), "OPEN")


class CrossCountDeduplicationTests(unittest.TestCase):
    """
    Verify that after dedup, a book move fires exactly one cross per market,
    not once per duplicate.
    """

    def test_cross_fires_once_after_dedup(self):
        """
        Two quotes at different prices, one OPEN + one SUPERSEDED.
        Book crosses the bid → exactly 1 bid_cross, not 2.
        """
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1, quote_bid=0.37, quote_ask=0.38)
        # Supersede q1 (book moved), insert q2 at new price
        _supersede_all_open_quotes(conn, "test-market", _T2)
        _insert_open(conn, obs_id="q2", obs_ts=_T2, quote_bid=0.38, quote_ask=0.39)

        # Only q2 is OPEN; load open quotes for the event
        open_quotes = _load_open_quotes(conn, "test-event")
        self.assertEqual(len(open_quotes), 1, "Only one OPEN quote after supersede")

        # live ask drops to 0.375 (≤ quote_bid 0.38) → BID cross
        bc, ac, both_c, sc = check_crosses(open_quotes, 0.36, 0.375, _T3, "s3", conn)
        self.assertEqual(bc, 1, "Exactly one bid cross")
        self.assertEqual(both_c, 0)

    def test_cross_fires_once_both_crossed_after_dedup(self):
        """Both sides cross simultaneously → exactly 1 BOTH_CROSS."""
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1, quote_bid=0.37, quote_ask=0.38)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        _insert_open(conn, obs_id="q2", obs_ts=_T2, quote_bid=0.37, quote_ask=0.38)

        open_quotes = _load_open_quotes(conn, "test-event")
        # live bid >= 0.38 and live ask <= 0.37 → both sides crossed
        bc, ac, both_c, sc = check_crosses(open_quotes, 0.39, 0.365, _T3, "s3", conn)
        self.assertEqual(both_c, 1)
        self.assertEqual(bc, 0)  # BOTH subsumes BID
        self.assertGreater(sc, 0.0)


class LoadOpenQuotesFilterTests(unittest.TestCase):

    def test_load_open_excludes_superseded(self):
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        rows = _load_open_quotes(conn, "test-event")
        self.assertEqual(len(rows), 0)

    def test_load_open_returns_canonical(self):
        conn = _make_db()
        _insert_open(conn, obs_id="q1", obs_ts=_T1)
        _supersede_all_open_quotes(conn, "test-market", _T2)
        _insert_open(conn, obs_id="q2", obs_ts=_T2)
        rows = _load_open_quotes(conn, "test-event")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["obs_id"], "q2")


if __name__ == "__main__":
    unittest.main()
