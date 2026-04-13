"""
run_maker_paper_calibration.py — Maker-MM paper deployment calibration.

Purpose
-------
Calibrates the maker-first EV model against observable market behaviour by
repeatedly quoting target markets in paper mode, recording each quote placement,
and checking on subsequent runs whether the book crossed our posted price.

This is NOT a taker simulation.  PaperBroker.submit_limit_order() does point-in-
time fill checking (fills BUY if current_ask <= limit_price), which almost never
triggers for a maker bid below mid.  For maker-MM, fill rate must be measured by
tracking whether the live book subsequently crosses our resting quote price.

What this script does
---------------------
1. Fetch live Gamma data for the target events (Phase 1: Hungary; Phase 2: Netanyahu)
2. Compute EV plan per market using the wide-scan EV model
3. Compute reservation quote via maker_scan_quote_planner.plan_quote()
4. Fetch live CLOB books for each target market
5. Log a QUOTE_OBSERVATION record per market (maker_paper_calib.db)
6. Load OPEN_QUOTE records from the previous run; check if the current book
   crossed them (simulated fill observation)
7. Log FILL_OBSERVATION records where crossed
8. Print a per-session calibration summary with modeled vs observed EV

What this does NOT do
---------------------
- Does not place real orders (paper only, no credentials needed)
- Does not mutate paper.db (Track A is isolated)
- Does not track reward share (impossible in paper mode; flagged as open gap)

Isolation
---------
All persistence goes to: data/processed/maker_paper_calib.db
Separate from paper.db.  Safe to delete and restart.

Usage
-----
Phase 1 (Hungary only):
    python scripts/run_maker_paper_calibration.py

Phase 2 (add Netanyahu):
    python scripts/run_maker_paper_calibration.py --events netanyahu-out-before-2027

Calibration cycle (scheduled):
    # Run every 30 minutes to build fill frequency sample
    python scripts/run_maker_paper_calibration.py --loop --interval-sec 1800

Output
------
data/processed/maker_paper_calib.db  — SQLite telemetry store
data/reports/maker_paper_calib_latest.json — last session summary
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger_calib = logging.getLogger("polyarb.maker_paper_calib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.gamma import fetch_events, fetch_markets
from src.ingest.clob import ReadOnlyClob
from src.intelligence.market_intelligence import build_event_market_registry
from src.scanner.maker_scan_quote_planner import plan_quote, MakerQuotePlan
from scripts.run_wide_maker_scan import (
    compute_maker_mm_ev,
    estimate_implied_reward_ev,
    q_score_per_side,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST  = "https://clob.polymarket.com"

CALIB_DB   = ROOT / "data" / "processed" / "maker_paper_calib.db"
REPORT_DIR = ROOT / "data" / "reports"

# Phase 1 anchor events only
DEFAULT_EVENTS = [
    "next-prime-minister-of-hungary",
]

# Sizing caps (paper)
# ---------------------------------------------------------------------------
# Hungary: rewards_min_size=200, neg_risk=True, pair_cost~$200 → cap at 200 shares
# Netanyahu: rewards_min_size=100, neg_risk=False, cap at 100 shares
# Default: use rewards_min_size from market payload, floor at 20
QUOTE_SIZE_OVERRIDE: dict[str, float] = {
    "next-prime-minister-of-hungary":       200.0,
    "netanyahu-out-before-2027":            100.0,
    "nba-mvp-694":                          100.0,
    "balance-of-power-2026-midterms":        20.0,
    "presidential-election-winner-2028":    200.0,
    "republican-presidential-nominee-2028": 200.0,
    "democratic-presidential-nominee-2028": 200.0,
}

# Capital exposure caps per event (USDC).  Paper simulation only.
CAPITAL_CAP_PER_EVENT: dict[str, float] = {
    "next-prime-minister-of-hungary":       400.0,   # 2 markets × 200sh
    "netanyahu-out-before-2027":            100.0,   # 1 market × 100sh × mid_p~$0.40
    "nba-mvp-694":                          200.0,   # 2 markets
    "balance-of-power-2026-midterms":       120.0,   # 3 markets × 20sh
    "presidential-election-winner-2028":    600.0,   # 3 markets × 200sh
    "republican-presidential-nominee-2028": 800.0,   # 4 markets × 200sh (cluster)
    "democratic-presidential-nominee-2028": 800.0,   # correlated to above
}
DEFAULT_CAPITAL_CAP = 200.0

# Per-session max open positions per event (guards against correlated fills)
MAX_POSITIONS_PER_EVENT = 2   # even in neg_risk events, limit simultaneous quote fills


# ---------------------------------------------------------------------------
# Telemetry schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS quote_observations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    obs_id         TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    event_slug     TEXT NOT NULL,
    market_slug    TEXT NOT NULL,
    obs_ts         TEXT NOT NULL,

    -- Live book state at quote time
    live_best_bid  REAL NOT NULL,
    live_best_ask  REAL NOT NULL,
    mid_p          REAL NOT NULL,
    current_spread_cents REAL NOT NULL,

    -- Reservation quote (from maker_scan_quote_planner)
    quote_bid      REAL NOT NULL,
    quote_ask      REAL NOT NULL,
    quote_size     REAL NOT NULL,
    bid_spread_cents REAL NOT NULL,   -- |mid - bid| * 100
    ask_spread_cents REAL NOT NULL,   -- |ask - mid| * 100
    max_spread_cents REAL NOT NULL,
    reward_eligible INTEGER NOT NULL, -- 1 if quote within reward bounds

    -- Modeled EV components (competition_factor-based, assumes ~4% pool share)
    modeled_reward_ev         REAL NOT NULL,
    modeled_spread_capture_ev REAL NOT NULL,
    modeled_total_ev          REAL NOT NULL,
    modeled_competition_factor REAL NOT NULL,
    reward_daily_rate         REAL NOT NULL,
    neg_risk                  INTEGER NOT NULL,

    -- Conservative / book-implied reward EV
    -- Uses live CLOB book depth Q-scores as competitor pool (lower bound on share)
    competitor_q_sum          REAL,          -- sum of Q-scores of book orders within max_spread
    our_q_score               REAL,          -- our two-sided Q-score
    implied_reward_ev         REAL,          -- daily_rate * our_q / (our_q + competitor_q)
    reward_ev_ratio           REAL,          -- implied_reward_ev / modeled_reward_ev

    -- Fill-pressure / cross-distance telemetry
    bid_cross_distance_cents  REAL,   -- (live_best_ask - quote_bid)*100; how far book must fall to fill bid
    ask_cross_distance_cents  REAL,   -- (quote_ask - live_best_bid)*100; how far book must rise to fill ask
    best_bid_move_from_prev   REAL,   -- signed cents change in best_bid vs prior obs (NULL for first)
    best_ask_move_from_prev   REAL,   -- signed cents change in best_ask vs prior obs (NULL for first)
    time_since_last_obs_sec   REAL,   -- seconds since last obs for this market (NULL for first)
    quote_aggressiveness_bucket TEXT, -- TIGHT (<0.5¢) / STANDARD (0.5–1.5¢) / WIDE (>1.5¢)

    -- Observation status: OPEN until crossed or expired
    status  TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN | BID_CROSSED | ASK_CROSSED | BOTH_CROSSED | EXPIRED | SUPERSEDED
    crossed_ts TEXT
);

CREATE TABLE IF NOT EXISTS fill_observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    obs_id           TEXT NOT NULL,          -- FK to quote_observations.obs_id
    session_id       TEXT NOT NULL,          -- session that detected the cross
    event_slug       TEXT NOT NULL,
    market_slug      TEXT NOT NULL,
    fill_side        TEXT NOT NULL,          -- BID | ASK | BOTH
    quote_price      REAL NOT NULL,          -- our resting quote price that was crossed
    crossing_price   REAL NOT NULL,          -- book price that crossed ours
    time_in_queue_sec REAL,                  -- seconds since original quote placement
    spread_captured_usdc REAL,              -- non-null only if BOTH crossed in same session
    obs_ts           TEXT NOT NULL,
    detected_ts      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                    TEXT NOT NULL,
    session_ts                    TEXT NOT NULL,
    event_slugs                   TEXT NOT NULL,  -- JSON list
    markets_observed              INTEGER NOT NULL,
    reward_eligible_quotes        INTEGER NOT NULL,
    open_quotes_loaded            INTEGER NOT NULL,
    bid_crosses_detected          INTEGER NOT NULL,
    ask_crosses_detected          INTEGER NOT NULL,
    both_crosses_detected         INTEGER NOT NULL,
    realized_spread_capture_usdc  REAL NOT NULL,
    modeled_total_ev_sum          REAL NOT NULL,
    actual_vs_model_ratio         REAL,           -- NULL until >= 3 fill obs exist
    reward_gap_note               TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_MIGRATION_SQL = """
ALTER TABLE quote_observations ADD COLUMN competitor_q_sum  REAL;
ALTER TABLE quote_observations ADD COLUMN our_q_score       REAL;
ALTER TABLE quote_observations ADD COLUMN implied_reward_ev REAL;
ALTER TABLE quote_observations ADD COLUMN reward_ev_ratio   REAL;
"""


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    # Idempotent migration: add new columns to existing DB
    existing = {r[1] for r in conn.execute("PRAGMA table_info(quote_observations)")}
    for col, typ in [
        # Implied-reward fields (added in prior round)
        ("competitor_q_sum",           "REAL"),
        ("our_q_score",                "REAL"),
        ("implied_reward_ev",          "REAL"),
        ("reward_ev_ratio",            "REAL"),
        # Fill-pressure / cross-distance telemetry
        ("bid_cross_distance_cents",   "REAL"),   # (live_best_ask - quote_bid)*100
        ("ask_cross_distance_cents",   "REAL"),   # (quote_ask - live_best_bid)*100
        ("best_bid_move_from_prev",    "REAL"),   # signed ¢ change vs previous obs
        ("best_ask_move_from_prev",    "REAL"),   # signed ¢ change vs previous obs
        ("time_since_last_obs_sec",    "REAL"),   # seconds since prior obs for this market
        ("quote_aggressiveness_bucket","TEXT"),   # TIGHT / STANDARD / WIDE
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE quote_observations ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def _prev_obs_for_market(conn: sqlite3.Connection, market_slug: str):
    """Return the most recent quote_observation row for this market, or None."""
    return conn.execute(
        """SELECT live_best_bid, live_best_ask, obs_ts
             FROM quote_observations
            WHERE market_slug = ?
            ORDER BY obs_ts DESC
            LIMIT 1""",
        (market_slug,),
    ).fetchone()


def _aggressiveness_bucket(bid_spread_cents: float) -> str:
    """Categorise quote aggressiveness by distance of bid (or ask) from mid."""
    if bid_spread_cents < 0.5:
        return "TIGHT"
    if bid_spread_cents < 1.5:
        return "STANDARD"
    return "WIDE"


def _book_competitor_q(book, mid_p: float, max_spread_cents: float) -> float:
    """
    Sum Q-scores of all resting book orders within the reward spread zone.

    Uses q_score_per_side() for each bid and ask level.  Treats every order
    as a competing MM — this is a conservative upper bound on competition.
    Position-taker orders resting in the book inflate this estimate, so the
    true implied pool share is likely higher than what this implies.

    book: CLOB book object with .bids and .asks (price, size attributes)
    """
    if book is None:
        return 0.0
    total = 0.0
    for level in list(book.bids) + list(book.asks):
        dist_cents = abs(level.price - mid_p) * 100.0
        total += q_score_per_side(max_spread_cents, dist_cents, level.size)
    return total


def _insert_quote_obs(conn: sqlite3.Connection, r: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO quote_observations
          (obs_id, session_id, event_slug, market_slug, obs_ts,
           live_best_bid, live_best_ask, mid_p, current_spread_cents,
           quote_bid, quote_ask, quote_size, bid_spread_cents, ask_spread_cents,
           max_spread_cents, reward_eligible,
           modeled_reward_ev, modeled_spread_capture_ev, modeled_total_ev,
           modeled_competition_factor, reward_daily_rate, neg_risk,
           competitor_q_sum, our_q_score, implied_reward_ev, reward_ev_ratio,
           bid_cross_distance_cents, ask_cross_distance_cents,
           best_bid_move_from_prev, best_ask_move_from_prev,
           time_since_last_obs_sec, quote_aggressiveness_bucket,
           status)
        VALUES
          (:obs_id, :session_id, :event_slug, :market_slug, :obs_ts,
           :live_best_bid, :live_best_ask, :mid_p, :current_spread_cents,
           :quote_bid, :quote_ask, :quote_size, :bid_spread_cents, :ask_spread_cents,
           :max_spread_cents, :reward_eligible,
           :modeled_reward_ev, :modeled_spread_capture_ev, :modeled_total_ev,
           :modeled_competition_factor, :reward_daily_rate, :neg_risk,
           :competitor_q_sum, :our_q_score, :implied_reward_ev, :reward_ev_ratio,
           :bid_cross_distance_cents, :ask_cross_distance_cents,
           :best_bid_move_from_prev, :best_ask_move_from_prev,
           :time_since_last_obs_sec, :quote_aggressiveness_bucket,
           'OPEN')
        """,
        r,
    )


def _insert_fill_obs(conn: sqlite3.Connection, r: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO fill_observations
          (obs_id, session_id, event_slug, market_slug, fill_side,
           quote_price, crossing_price, time_in_queue_sec,
           spread_captured_usdc, obs_ts, detected_ts)
        VALUES
          (:obs_id, :session_id, :event_slug, :market_slug, :fill_side,
           :quote_price, :crossing_price, :time_in_queue_sec,
           :spread_captured_usdc, :obs_ts, :detected_ts)
        """,
        r,
    )


def _load_open_quotes(conn: sqlite3.Connection, event_slug: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM quote_observations WHERE status='OPEN' AND event_slug=?",
        (event_slug,),
    ).fetchall()


def _supersede_all_open_quotes(
    conn        : sqlite3.Connection,
    market_slug : str,
    now_ts      : str,
) -> int:
    """
    Mark ALL OPEN quotes for this market as SUPERSEDED before inserting a new
    quote observation.

    One-live-quote-per-market invariant: each market must have at most one OPEN
    resting quote at any time.  When a new session posts a quote, any prior OPEN
    quote — regardless of price or size — is no longer the active resting quote
    and must be retired.

    The old price-match-gated implementation only superseded identical
    (bid, ask, size) rows.  When the book moved between sessions the new quote
    had different prices, the WHERE found zero rows, and the stale OPEN quote
    accumulated forever.  After N sessions with any book movement, each market
    had N OPEN rows: ghost inflation of open-quote counts and invalid cross
    detection against stale price levels.

    Returns the number of rows updated.

    Auditability: superseded rows remain in quote_observations with
    status='SUPERSEDED' and crossed_ts=now_ts.  They are excluded from
    _load_open_quotes (which filters status='OPEN') and from cross detection,
    but are still queryable for session history.
    """
    cur = conn.execute(
        """
        UPDATE quote_observations
           SET status = 'SUPERSEDED', crossed_ts = ?
         WHERE market_slug = ?
           AND status      = 'OPEN'
        """,
        (now_ts, market_slug),
    )
    return cur.rowcount


def _mark_crossed(conn: sqlite3.Connection, obs_id: str, status: str, crossed_ts: str) -> None:
    conn.execute(
        "UPDATE quote_observations SET status=?, crossed_ts=? WHERE obs_id=?",
        (status, crossed_ts, obs_id),
    )


def _insert_session_summary(conn: sqlite3.Connection, r: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO session_summaries
          (session_id, session_ts, event_slugs, markets_observed,
           reward_eligible_quotes, open_quotes_loaded,
           bid_crosses_detected, ask_crosses_detected, both_crosses_detected,
           realized_spread_capture_usdc, modeled_total_ev_sum,
           actual_vs_model_ratio, reward_gap_note)
        VALUES
          (:session_id, :session_ts, :event_slugs, :markets_observed,
           :reward_eligible_quotes, :open_quotes_loaded,
           :bid_crosses_detected, :ask_crosses_detected, :both_crosses_detected,
           :realized_spread_capture_usdc, :modeled_total_ev_sum,
           :actual_vs_model_ratio, :reward_gap_note)
        """,
        r,
    )


# ---------------------------------------------------------------------------
# Market fetching
# ---------------------------------------------------------------------------

def fetch_eligible_markets(
    gamma_host: str,
    event_slugs: list[str],
) -> list[dict[str, Any]]:
    """Fetch Gamma registry and return reward-eligible markets for target events."""
    events  = fetch_events(gamma_host, limit=500)
    markets = fetch_markets(gamma_host, limit=500)
    registry = build_event_market_registry(events, markets)

    slug_set = set(event_slugs)
    result: list[dict[str, Any]] = []

    for event in registry.get("events", []):
        if event.get("slug") not in slug_set:
            continue
        for m in event.get("markets", []):
            if not m.get("is_binary_yes_no") or not m.get("enable_orderbook"):
                continue
            rewards = m.get("clob_rewards") or []
            rate = sum(float(r.get("rewardsDailyRate", 0) or 0) for r in rewards)
            if rate <= 0:
                continue
            min_size = float(m.get("rewards_min_size") or 0)
            max_spread = float(m.get("rewards_max_spread") or 0)
            if min_size <= 0 or max_spread <= 0:
                continue
            best_bid = float(m.get("best_bid") or 0)
            best_ask = float(m.get("best_ask") or 0)
            if best_bid <= 0 or best_ask <= best_bid:
                continue
            yes_token = m.get("yes_token_id")
            no_token  = m.get("no_token_id")
            if not yes_token or not no_token:
                continue

            ev_slug = str(event.get("slug") or "")
            result.append({
                "event_slug":        ev_slug,
                "market_slug":       str(m.get("slug") or ""),
                "question":          m.get("question"),
                "yes_token_id":      str(yes_token),
                "no_token_id":       str(no_token),
                "best_bid":          best_bid,
                "best_ask":          best_ask,
                "rewards_min_size":  min_size,
                "rewards_max_spread": max_spread,
                "reward_daily_rate": rate,
                "fees_enabled":      bool(m.get("fees_enabled")),
                "neg_risk":          bool(m.get("neg_risk")),
                "volume_num":        float(m.get("volume_num") or 0),
            })

    return result


# ---------------------------------------------------------------------------
# Fill cross detection
# ---------------------------------------------------------------------------

def check_crosses(
    open_quotes: list[sqlite3.Row],
    live_best_bid: float,
    live_best_ask: float,
    now_ts: str,
    session_id: str,
    conn: sqlite3.Connection,
) -> tuple[int, int, int, float]:
    """
    Check if the current live book crossed any open resting quotes.

    A BID cross occurs when the live best ask dropped to <= our quote_bid.
    An ASK cross occurs when the live best bid rose to >= our quote_ask.

    Returns: (bid_crosses, ask_crosses, both_crosses, spread_captured_usdc)
    """
    bid_crosses = ask_crosses = both_crosses = 0
    spread_captured = 0.0

    for row in open_quotes:
        obs_id     = row["obs_id"]
        obs_ts_str = row["obs_ts"]
        quote_bid  = row["quote_bid"]
        quote_ask  = row["quote_ask"]
        quote_size = row["quote_size"]
        event_slug = row["event_slug"]
        market_slug = row["market_slug"]

        # Time since placement
        try:
            placed_ts = datetime.fromisoformat(obs_ts_str)
            now_dt    = datetime.fromisoformat(now_ts)
            tiq_sec   = (now_dt - placed_ts).total_seconds()
        except Exception:
            tiq_sec = None

        bid_crossed = (live_best_ask <= quote_bid) and (quote_bid > 0)
        ask_crossed = (live_best_bid >= quote_ask) and (quote_ask > 0)

        if bid_crossed and ask_crossed:
            both_crosses += 1
            spread_cap = (quote_ask - quote_bid) * quote_size
            spread_captured += spread_cap
            _mark_crossed(conn, obs_id, "BOTH_CROSSED", now_ts)
            _insert_fill_obs(conn, {
                "obs_id":                obs_id,
                "session_id":            session_id,
                "event_slug":            event_slug,
                "market_slug":           market_slug,
                "fill_side":             "BOTH",
                "quote_price":           round((quote_bid + quote_ask) / 2, 6),
                "crossing_price":        round((live_best_ask + live_best_bid) / 2, 6),
                "time_in_queue_sec":     tiq_sec,
                "spread_captured_usdc":  round(spread_cap, 6),
                "obs_ts":                obs_ts_str,
                "detected_ts":           now_ts,
            })

        elif bid_crossed:
            bid_crosses += 1
            _mark_crossed(conn, obs_id, "BID_CROSSED", now_ts)
            _insert_fill_obs(conn, {
                "obs_id":                obs_id,
                "session_id":            session_id,
                "event_slug":            event_slug,
                "market_slug":           market_slug,
                "fill_side":             "BID",
                "quote_price":           quote_bid,
                "crossing_price":        live_best_ask,
                "time_in_queue_sec":     tiq_sec,
                "spread_captured_usdc":  None,
                "obs_ts":                obs_ts_str,
                "detected_ts":           now_ts,
            })

        elif ask_crossed:
            ask_crosses += 1
            _mark_crossed(conn, obs_id, "ASK_CROSSED", now_ts)
            _insert_fill_obs(conn, {
                "obs_id":                obs_id,
                "session_id":            session_id,
                "event_slug":            event_slug,
                "market_slug":           market_slug,
                "fill_side":             "ASK",
                "quote_price":           quote_ask,
                "crossing_price":        live_best_bid,
                "time_in_queue_sec":     tiq_sec,
                "spread_captured_usdc":  None,
                "obs_ts":                obs_ts_str,
                "detected_ts":           now_ts,
            })

    return bid_crosses, ask_crosses, both_crosses, spread_captured


# ---------------------------------------------------------------------------
# Main calibration session
# ---------------------------------------------------------------------------

def run_calibration_session(
    event_slugs : list[str],
    gamma_host  : str,
    clob_host   : str,
    conn        : sqlite3.Connection,
) -> dict[str, Any]:
    session_id = f"mcal_{uuid.uuid4().hex[:12]}"
    now_ts     = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*70}")
    print(f"MAKER PAPER CALIBRATION  session={session_id}")
    print(f"events: {event_slugs}")
    print(f"{'='*70}")

    # 1. Fetch eligible markets
    print("[1/4] Fetching Gamma data...")
    markets = fetch_eligible_markets(gamma_host, event_slugs)
    print(f"      Found {len(markets)} eligible markets across {len(event_slugs)} event(s)")

    if not markets:
        print("      No eligible markets. Exiting session.")
        return {"session_id": session_id, "markets_observed": 0}

    # 2. Fetch live CLOB books for all markets (YES token only for bid/ask reference)
    print("[2/4] Fetching live CLOB books...")
    clob = ReadOnlyClob(clob_host)
    yes_token_ids = [m["yes_token_id"] for m in markets]
    books = clob.prefetch_books(yes_token_ids[:100], max_workers=8)
    print(f"      Fetched {len(books)} orderbooks")

    # 3. Update live prices from CLOB; compute competitor Q for each market
    for m in markets:
        book = books.get(m["yes_token_id"])
        if book and book.bids and book.asks:
            m["best_bid"] = book.bids[0].price
            m["best_ask"] = book.asks[0].price
            m["live_book"] = True
        else:
            m["live_book"] = False
        # Store raw book for competitor-Q computation in per-market loop
        m["_book"] = book

    # 4. Per-market: check previous open quotes, place new quote observation
    print("[3/4] Checking crosses + placing quote observations...")

    total_bid_crosses     = 0
    total_ask_crosses     = 0
    total_both_crosses    = 0
    total_spread_cap      = 0.0
    total_eligible        = 0
    total_open_loaded     = 0
    total_modeled_ev      = 0.0
    total_implied_ev      = 0.0
    markets_observed      = 0

    for m in markets:
        slug       = m["market_slug"]
        ev_slug    = m["event_slug"]
        best_bid   = m["best_bid"]
        best_ask   = m["best_ask"]

        if best_ask <= best_bid or best_bid <= 0:
            print(f"  SKIP {slug}: invalid book ({best_bid}/{best_ask})")
            continue

        # Check open quotes from previous sessions
        open_quotes = _load_open_quotes(conn, ev_slug)
        # Filter to this specific market
        market_open = [q for q in open_quotes if q["market_slug"] == slug]
        total_open_loaded += len(market_open)

        bc, ac, both_c, sc = check_crosses(
            market_open, best_bid, best_ask, now_ts, session_id, conn
        )
        total_bid_crosses  += bc
        total_ask_crosses  += ac
        total_both_crosses += both_c
        total_spread_cap   += sc

        if bc or ac or both_c:
            print(f"  CROSS {slug}: bid={bc} ask={ac} both={both_c} spread_cap=${sc:.4f}")

        # Compute EV plan
        ev = compute_maker_mm_ev(
            best_bid              = best_bid,
            best_ask              = best_ask,
            rewards_min_size      = m["rewards_min_size"],
            rewards_max_spread_cents = m["rewards_max_spread"],
            reward_daily_rate     = m["reward_daily_rate"],
            volume_num            = m.get("volume_num", 0),
        )

        # Reservation quote via maker_scan_quote_planner
        quote_size = QUOTE_SIZE_OVERRIDE.get(ev_slug, max(m["rewards_min_size"], 20.0))
        # Override market quote_size in dict for plan_quote
        m_for_plan = dict(m)
        m_for_plan["rewards_min_size"] = quote_size
        plan: MakerQuotePlan = plan_quote(m_for_plan)

        if ev["total_ev"] <= 0:
            print(f"  SKIP {slug}: total_ev={ev['total_ev']:.4f} <= 0")
            continue

        eligible = 1 if plan.eligible else 0
        total_eligible += eligible
        total_modeled_ev += ev["total_ev"]
        markets_observed += 1

        obs_id = f"qo_{uuid.uuid4().hex[:16]}"
        mid_p  = (best_bid + best_ask) / 2.0
        current_spread_cents = (best_ask - best_bid) * 100.0

        # Conservative / book-implied reward EV
        # Compute competitor Q from live CLOB book depth (orders within max_spread zone)
        max_sp_cents  = m["rewards_max_spread"]
        competitor_q  = _book_competitor_q(m.get("_book"), mid_p, max_sp_cents)
        our_q         = ev["our_q_score"]
        implied_rew   = estimate_implied_reward_ev(m["reward_daily_rate"], our_q, competitor_q)
        rew_ratio     = (implied_rew / ev["reward_ev"]) if ev["reward_ev"] > 0 else None
        total_implied_ev += implied_rew

        # Fill-pressure telemetry: cross distance and book movement
        bid_cross_dist = round((best_ask - plan.quote_bid) * 100.0, 4)
        ask_cross_dist = round((plan.quote_ask - best_bid) * 100.0, 4)
        bucket         = _aggressiveness_bucket(plan.bid_spread_cents)

        prev_obs = _prev_obs_for_market(conn, slug)
        if prev_obs is not None:
            bid_move = round((best_bid - float(prev_obs["live_best_bid"])) * 100.0, 4)
            ask_move = round((best_ask - float(prev_obs["live_best_ask"])) * 100.0, 4)
            try:
                prev_dt   = datetime.fromisoformat(str(prev_obs["obs_ts"]))
                now_dt    = datetime.fromisoformat(now_ts)
                tiq_sec   = round((now_dt - prev_dt).total_seconds(), 1)
            except Exception:
                tiq_sec = None
        else:
            bid_move = None
            ask_move = None
            tiq_sec  = None

        # One-live-quote-per-market: supersede ALL prior OPEN quotes for this
        # market before inserting the new one.  Price/size are irrelevant — any
        # prior resting quote is stale once a new session posts a fresh quote.
        superseded_n = _supersede_all_open_quotes(conn, slug, now_ts)
        if superseded_n:
            logger_calib.debug(
                "Superseded %d stale OPEN quote(s) for %s",
                superseded_n, slug,
            )

        _insert_quote_obs(conn, {
            "obs_id":                     obs_id,
            "session_id":                 session_id,
            "event_slug":                 ev_slug,
            "market_slug":                slug,
            "obs_ts":                     now_ts,
            "live_best_bid":              round(best_bid, 6),
            "live_best_ask":              round(best_ask, 6),
            "mid_p":                      round(mid_p, 6),
            "current_spread_cents":       round(current_spread_cents, 3),
            "quote_bid":                  round(plan.quote_bid, 6),
            "quote_ask":                  round(plan.quote_ask, 6),
            "quote_size":                 quote_size,
            "bid_spread_cents":           round(plan.bid_spread_cents, 3),
            "ask_spread_cents":           round(plan.ask_spread_cents, 3),
            "max_spread_cents":           round(plan.max_spread_cents, 3),
            "reward_eligible":            eligible,
            "modeled_reward_ev":          round(ev["reward_ev"], 6),
            "modeled_spread_capture_ev":  round(ev["spread_capture_ev"], 6),
            "modeled_total_ev":           round(ev["total_ev"], 6),
            "modeled_competition_factor": round(ev["competition_factor"], 4),
            "reward_daily_rate":          m["reward_daily_rate"],
            "neg_risk":                   1 if m["neg_risk"] else 0,
            "competitor_q_sum":              round(competitor_q, 2),
            "our_q_score":                   round(our_q, 2),
            "implied_reward_ev":             round(implied_rew, 6),
            "reward_ev_ratio":               round(rew_ratio, 4) if rew_ratio is not None else None,
            "bid_cross_distance_cents":      bid_cross_dist,
            "ask_cross_distance_cents":      ask_cross_dist,
            "best_bid_move_from_prev":       bid_move,
            "best_ask_move_from_prev":       ask_move,
            "time_since_last_obs_sec":       tiq_sec,
            "quote_aggressiveness_bucket":   bucket,
        })

        print(
            f"  OBS  {slug[:50]}"
            f"  bid={plan.quote_bid:.4f}  ask={plan.quote_ask:.4f}"
            f"  elig={'Y' if eligible else 'N'}"
            f"  ev=${ev['total_ev']:.3f}"
            f"  reward_mod=${ev['reward_ev']:.3f}  reward_impl=${implied_rew:.3f}"
        )

    # 5. Session summary
    print("[4/4] Writing session summary...")

    # Compute actual vs modeled ratio (only meaningful once fill data exists)
    total_fills_all = conn.execute(
        "SELECT COUNT(*) FROM fill_observations"
    ).fetchone()[0]
    actual_vs_model = None
    if total_fills_all >= 3:
        # Very rough: realized spread capture total / sum of modeled_spread_capture_ev
        realized_total = conn.execute(
            "SELECT COALESCE(SUM(spread_captured_usdc),0) FROM fill_observations WHERE fill_side='BOTH'"
        ).fetchone()[0]
        modeled_total  = conn.execute(
            "SELECT COALESCE(SUM(modeled_spread_capture_ev),0) FROM quote_observations"
        ).fetchone()[0]
        if modeled_total > 0:
            actual_vs_model = round(realized_total / modeled_total, 4)

    reward_gap_note = (
        "REWARD_UNOBSERVABLE: reward share requires live on-chain order posting; "
        "paper mode captures fill frequency and spread capture only"
    )
    # Realism band: spread_capture_ev is unaffected; reward component is bounded
    # below by implied_reward_ev (book-depth Q-score estimate)
    spread_cap_sum = sum(
        row[0]
        for row in conn.execute(
            "SELECT modeled_spread_capture_ev FROM quote_observations "
            "WHERE session_id=?", (session_id,)
        )
    )
    implied_total_ev = spread_cap_sum + total_implied_ev
    avg_ratio = (total_implied_ev / total_modeled_ev * markets_observed
                 if total_modeled_ev > 0 and markets_observed > 0
                 else None)

    summary = {
        "session_id":                   session_id,
        "session_ts":                   now_ts,
        "event_slugs":                  json.dumps(event_slugs),
        "markets_observed":             markets_observed,
        "reward_eligible_quotes":       total_eligible,
        "open_quotes_loaded":           total_open_loaded,
        "bid_crosses_detected":         total_bid_crosses,
        "ask_crosses_detected":         total_ask_crosses,
        "both_crosses_detected":        total_both_crosses,
        "realized_spread_capture_usdc": round(total_spread_cap, 6),
        "modeled_total_ev_sum":         round(total_modeled_ev, 6),
        "actual_vs_model_ratio":        actual_vs_model,
        "reward_gap_note":              reward_gap_note,
    }
    _insert_session_summary(conn, summary)
    conn.commit()

    # Console output
    print(f"\n{'='*70}")
    print(f"SESSION SUMMARY  {session_id}")
    print(f"{'='*70}")
    print(f"  Markets observed:            {markets_observed}")
    print(f"  Reward-eligible quotes:      {total_eligible}")
    print(f"  Open quotes loaded (prev):   {total_open_loaded}")
    print(f"  Bid crosses detected:        {total_bid_crosses}")
    print(f"  Ask crosses detected:        {total_ask_crosses}")
    print(f"  BOTH crosses (spread cap):   {total_both_crosses}")
    print(f"  Realized spread capture:     ${total_spread_cap:.4f}")
    print()
    print(f"  EV REALISM BAND (per session, USDC/day):")
    print(f"    modeled_total_ev           ${total_modeled_ev:.4f}  [competition_factor=25× assumed]")
    print(f"    implied_total_ev (lower)   ${implied_total_ev:.4f}  [book-depth Q-score competitor]")
    if avg_ratio is not None:
        pct = avg_ratio * 100
        print(f"    reward_ev_ratio            {avg_ratio:.3f}  ({pct:.1f}% of modeled reward)")
    print(f"    spread_capture_ev          ${spread_cap_sum:.4f}  [unaffected by reward uncertainty]")
    print()
    if actual_vs_model is not None:
        print(f"  Actual/Model ratio:          {actual_vs_model:.4f}  ← CALIBRATION SIGNAL")
    else:
        print(f"  Actual/Model ratio:          N/A (need >= 3 fill observations)")
    print(f"\n  NOTE: {reward_gap_note}")
    print(f"\n  DB: {CALIB_DB}")
    print(f"{'='*70}")

    return summary


# ---------------------------------------------------------------------------
# Cumulative calibration report
# ---------------------------------------------------------------------------

def print_cumulative_stats(conn: sqlite3.Connection) -> None:
    """Print running calibration statistics across all sessions."""
    total_obs = conn.execute("SELECT COUNT(*) FROM quote_observations").fetchone()[0]
    total_fills = conn.execute("SELECT COUNT(*) FROM fill_observations").fetchone()[0]
    bid_fills = conn.execute(
        "SELECT COUNT(*) FROM fill_observations WHERE fill_side IN ('BID','BOTH')"
    ).fetchone()[0]
    ask_fills = conn.execute(
        "SELECT COUNT(*) FROM fill_observations WHERE fill_side IN ('ASK','BOTH')"
    ).fetchone()[0]
    both_fills = conn.execute(
        "SELECT COUNT(*) FROM fill_observations WHERE fill_side='BOTH'"
    ).fetchone()[0]
    realized_sc = conn.execute(
        "SELECT COALESCE(SUM(spread_captured_usdc),0) FROM fill_observations WHERE fill_side='BOTH'"
    ).fetchone()[0]
    total_sessions = conn.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]

    bid_rate  = bid_fills  / total_obs if total_obs > 0 else 0.0
    ask_rate  = ask_fills  / total_obs if total_obs > 0 else 0.0
    both_rate = both_fills / total_obs if total_obs > 0 else 0.0

    print(f"\n{'='*70}")
    print("CUMULATIVE CALIBRATION STATS")
    print(f"{'='*70}")
    print(f"  Sessions run:               {total_sessions}")
    print(f"  Total quote observations:   {total_obs}")
    print(f"  Total fill events:          {total_fills}")
    print(f"  Bid cross rate:             {bid_rate:.3f}  (target: model fill_prob_per_side)")
    print(f"  Ask cross rate:             {ask_rate:.3f}")
    print(f"  Both cross rate:            {both_rate:.4f}  (target: both_fill_prob in model)")
    print(f"  Realized spread captured:   ${realized_sc:.4f}")
    print(f"{'='*70}")

    # Per-event breakdown
    events = conn.execute(
        "SELECT DISTINCT event_slug FROM quote_observations"
    ).fetchall()
    if events:
        print("\n  Per-event fill rates:")
        for row in events:
            ev = row[0]
            n_obs   = conn.execute(
                "SELECT COUNT(*) FROM quote_observations WHERE event_slug=?", (ev,)
            ).fetchone()[0]
            n_fills = conn.execute(
                "SELECT COUNT(*) FROM fill_observations WHERE event_slug=? AND fill_side='BOTH'", (ev,)
            ).fetchone()[0]
            rate = n_fills / n_obs if n_obs > 0 else 0.0
            print(f"    {ev[:45]:<45}  obs={n_obs}  both_fills={n_fills}  rate={rate:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Maker-MM paper calibration run")
    p.add_argument(
        "--events", nargs="+", default=DEFAULT_EVENTS,
        help="Event slugs to calibrate (default: Hungary only)"
    )
    p.add_argument("--gamma-host", default=GAMMA_HOST)
    p.add_argument("--clob-host",  default=CLOB_HOST)
    p.add_argument("--loop",       action="store_true", help="Run continuously")
    p.add_argument("--interval-sec", type=int, default=1800,
                   help="Loop interval in seconds (default: 30 min)")
    p.add_argument("--stats",      action="store_true",
                   help="Print cumulative stats from DB and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = _open_db(CALIB_DB)

    if args.stats:
        print_cumulative_stats(conn)
        conn.close()
        return

    print("\nMAKER PAPER CALIBRATION")
    print(f"DB: {CALIB_DB}")
    print(f"Phase 1 events: {args.events}")
    print(
        "\nSizing caps (paper, per event):"
        f"\n  Hungary:   200 shares, cap=${CAPITAL_CAP_PER_EVENT.get('next-prime-minister-of-hungary')}"
        f"\n  Netanyahu: 100 shares, cap=${CAPITAL_CAP_PER_EVENT.get('netanyahu-out-before-2027')}"
    )
    print(
        "\nKnown gap: REWARD_SHARE_UNOBSERVABLE in paper mode."
        "\n  Reward payouts require real on-chain orders. This script calibrates:"
        "\n    - fill frequency (bid/ask cross rate)"
        "\n    - spread capture (BOTH crosses)"
        "\n    - quote placement validity (reward-eligible flag)"
        "\n  Reward EV calibration requires live paper trading at min_size."
    )

    if args.loop:
        print(f"\nLoop mode: every {args.interval_sec}s. Ctrl-C to stop.")
        try:
            while True:
                summary = run_calibration_session(
                    args.events, args.gamma_host, args.clob_host, conn
                )
                # Write latest report
                report_path = REPORT_DIR / "maker_paper_calib_latest.json"
                report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print_cumulative_stats(conn)
                print(f"\nSleeping {args.interval_sec}s...")
                time.sleep(args.interval_sec)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        summary = run_calibration_session(
            args.events, args.gamma_host, args.clob_host, conn
        )
        report_path = REPORT_DIR / "maker_paper_calib_latest.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print_cumulative_stats(conn)

    conn.close()


if __name__ == "__main__":
    main()
