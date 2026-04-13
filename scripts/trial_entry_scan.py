"""
trial_entry_scan.py  v3 — multi-mode discovery + near-miss telemetry + WS follow
Read-only scanner. No orders. No DB writes.

Discovery modes (--mode):
  slice   – rotate through (order, offset) windows of /markets  [default, fallback]
  events  – harvest candidates from /events nested markets using outcomePrices pre-filter
  both    – run events first, then one slice round

Near-miss telemetry:
  All viable markets are printed with their edge and distance-to-threshold,
  even when edge < min_edge. Regimes: HOPELESS (<−0.10), COLD, WARM, HOT, HIT.

Optional real-time follow (--follow):
  After a discovery scan, subscribes via WebSocket to the top-N candidate markets
  and prints live edge updates. Read-only. Exits on Ctrl-C.
"""

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import pathlib
import re
import time

import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import httpx
import websockets

from src.sidecar.ab_bridge import ABSidecar
from src.sidecar.plateau_tracker import Both98PlateauTracker
from src.scanner.cohort_router import CohortRouter, RouterConfig
from src.research.reward_eval import (
    evaluate_fitness as _eval_reward_fitness,
    format_reward_line as _fmt_reward,
    parse_reward_config as _parse_reward_config,
    reward_routing_bonus as _reward_routing_bonus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"
WS    = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

SLICE_ROTATION = [
    ("volume24hr", 0),
    ("liquidity",  0),
    ("volume24hr", 50),
    ("liquidity",  50),
    ("volume",     0),
    ("volume24hr", 100),
    ("liquidity",  100),
    ("volume",     50),
    ("startDate",  0),
    ("startDate",  50),
]

DEFAULT_STATE_FILE = pathlib.Path(__file__).parent.parent / ".scan_state.json"

EXPIRY_HORIZON_HOURS = 6
MIN_VOLUME_USD       = 100.0

# outcomePrices pre-filter: exclude near-certain outcomes before any CLOB fetch
OUTCOME_PRICE_MIN = 0.05
OUTCOME_PRICE_MAX = 0.95


# ---------------------------------------------------------------------------
# Regime classification for near-miss telemetry
# ---------------------------------------------------------------------------

def regime(edge: float, min_edge: float) -> str:
    if edge >= min_edge:                  return "HIT ✓"
    gap = min_edge - edge
    if edge < -0.10:                      return "HOPELESS"
    if gap <= 0.005:                      return "HOT"     # within 0.5¢
    if gap <= 0.015:                      return "WARM"    # within 1.5¢
    return "COLD"


# ---------------------------------------------------------------------------
# Rotation state helpers
# ---------------------------------------------------------------------------

def load_state(path: pathlib.Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"slice_index": 0, "last_slug_hash": None, "round": 0}


def save_state(path: pathlib.Path, state: dict):
    path.write_text(json.dumps(state, indent=2))


def slug_set_hash(slugs: list) -> str:
    return hashlib.md5("|".join(sorted(slugs)).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pre-filter helpers
# ---------------------------------------------------------------------------

def prefilter_reason(m: dict, now_utc: dt.datetime) -> str:
    """Non-empty string = skip this market before CLOB fetch."""
    if not m.get("active", True):
        return "inactive"
    for key in ("endDate", "end_date_iso", "endDateIso"):
        end_str = m.get(key) or ""
        if end_str:
            try:
                end = dt.datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=dt.timezone.utc)
                now_aware = now_utc.replace(tzinfo=dt.timezone.utc)
                if end <= now_aware + dt.timedelta(hours=EXPIRY_HORIZON_HOURS):
                    return f"expires {end_str[:16]}"
            except Exception:
                pass
            break
    vol = float(m.get("volume") or 0)
    if vol < MIN_VOLUME_USD:
        return f"low vol ${vol:.0f}"
    return ""


# ---------------------------------------------------------------------------
# Structural slug pre-filter
# ---------------------------------------------------------------------------

# Patterns that identify multi-outcome derivative families:
#   - game total/spread lines (sports)
#   - commodity and crypto threshold ladders
#   - esports per-game and handicap derivatives
#   - deadline ladder rungs (same event, N resolution dates)
_STRUCTURAL_SLUG_RE = re.compile(
    r"(-total-\d+pt"
    r"|-spread-(home|away)-\d+"
    r"|-(hit-high|hit-low)-\d+"
    r"|-(reach|dip-to|above)-\d+"
    r"|-game\d+$"
    r"|-game-handicap-"
    r"|-total-games-"
    r"|-by-(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)-\d+)",
    re.IGNORECASE,
)


def slug_is_structural_derivative(slug: str) -> bool:
    """Return True if slug matches a known non-standalone structural family."""
    return bool(_STRUCTURAL_SLUG_RE.search(slug))


def outcome_prices_viable(m: dict) -> tuple:
    """
    Use outcomePrices (cheap, already in payload) as first-pass filter.
    Returns (viable: bool, p_yes: float, p_no: float).
    """
    op_raw = m.get("outcomePrices")
    if not op_raw:
        return True, None, None     # no data → don't exclude, let CLOB decide
    try:
        op = json.loads(op_raw) if isinstance(op_raw, str) else list(op_raw)
        if len(op) < 2:
            return True, None, None
        p_yes = float(op[0])
        p_no  = float(op[1])
        if p_yes < OUTCOME_PRICE_MIN or p_yes > OUTCOME_PRICE_MAX:
            return False, p_yes, p_no
        if p_no < OUTCOME_PRICE_MIN or p_no > OUTCOME_PRICE_MAX:
            return False, p_yes, p_no
        return True, p_yes, p_no
    except Exception:
        return True, None, None


# ---------------------------------------------------------------------------
# CLOB book fetch + edge computation
# ---------------------------------------------------------------------------

def fetch_book_edge(yes_id: str, no_id: str) -> tuple:
    """
    Returns (yes_ask, no_ask, edge) or raises on failure.
    Raises ValueError if book is empty or both asks >= 0.98.
    """
    yb = httpx.get(f"{CLOB}/book", params={"token_id": yes_id}, timeout=5).json()
    nb = httpx.get(f"{CLOB}/book", params={"token_id": no_id}, timeout=5).json()
    ya = yb.get("asks") or []
    na = nb.get("asks") or []
    if not ya or not na:
        raise ValueError("empty book")
    yes_ask = float(ya[0]["price"])
    no_ask  = float(na[0]["price"])
    if yes_ask >= 0.98 and no_ask >= 0.98:
        raise ValueError("both_98")
    return yes_ask, no_ask, 1.0 - yes_ask - no_ask


# ---------------------------------------------------------------------------
# Core scan logic (shared by both modes)
# ---------------------------------------------------------------------------

def process_candidates(
    candidates: list,       # list of market dicts with clobTokenIds
    min_edge: float,
    target_usd: float,
    verbose: bool = True,
    both_98_out: list | None = None,
    reward_mode: bool = False,   # print reward fitness telemetry when True
) -> tuple:
    """
    Fetch CLOB books for candidates, print near-miss telemetry, return (hits, viable_records).
    viable_records: all markets that cleared book fetch (for WS follow).
    """
    now_utc = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    stats   = dict(no_book=0, both_98=0, viable=0)
    hits    = []
    viable_records = []

    for i, m in enumerate(candidates, 1):
        raw_ids = m.get("clobTokenIds") or []
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except Exception:
                continue
        if len(raw_ids) < 2:
            continue

        slug     = str(m.get("slug") or "?")
        min_size = float(m.get("orderMinSize") or 5)
        yes_id, no_id = str(raw_ids[0]), str(raw_ids[1])
        _reward_cfg = _parse_reward_config(m) if reward_mode else None

        if verbose:
            print(f"  BOOK [{i:3d}] {slug[:58]}")

        try:
            yes_ask, no_ask, edge = fetch_book_edge(yes_id, no_id)
        except ValueError as e:
            if "empty" in str(e):
                stats["no_book"] += 1
            else:
                stats["both_98"] += 1
                if both_98_out is not None:
                    both_98_out.append({"slug": slug, "edge_cents": -98.0})
            continue
        except Exception:
            stats["no_book"] += 1
            continue

        stats["viable"] += 1
        reg = regime(edge, min_edge)
        gap = min_edge - edge

        # Near-miss telemetry line
        print(
            f"  {reg:<9}  {slug[:50]:<50}  "
            f"YES={yes_ask:.4f} NO={no_ask:.4f}  "
            f"edge={edge*100:+.2f}¢  gap={gap*100:+.2f}¢"
        )

        # Reward fitness telemetry (only when --rewards active)
        if reward_mode and _reward_cfg is not None:
            fitness = _eval_reward_fitness(yes_ask, no_ask, min_size, _reward_cfg)
            line = _fmt_reward(slug, fitness)
            if line:
                print(line)

        rec = dict(
            slug=slug, edge=edge, yes_ask=yes_ask, no_ask=no_ask,
            yes_id=yes_id, no_id=no_id, min_size=min_size,
        )
        viable_records.append(rec)

        if edge >= min_edge:
            buy_side  = "YES" if yes_ask <= no_ask else "NO"
            buy_token = yes_id if buy_side == "YES" else no_id
            buy_ask   = yes_ask if buy_side == "YES" else no_ask
            shares    = max(int(min_size), int(target_usd / buy_ask))
            hit = dict(
                edge=edge, slug=slug, buy_side=buy_side,
                buy_token=buy_token, buy_ask=buy_ask,
                shares=shares, min_size=min_size,
                yes_ask=yes_ask, no_ask=no_ask,
            )
            hits.append(hit)

    print(
        f"\n  ── clob stats: no_book={stats['no_book']}  "
        f"both_98={stats['both_98']}  viable={stats['viable']}  hits={len(hits)}"
    )

    if hits:
        best = sorted(hits, key=lambda x: x["edge"], reverse=True)[0]
        print("\nBEST ENTRY:")
        print(f"  --token        {best['buy_token']}")
        print(f"  --ask          {best['buy_ask']}")
        print(f"  --min-size     {best['shares']}")
        print(f"  --market-slug  {best['slug']}")
        print(f"  --buy-side     {best['buy_side']}")
        print(f"  --edge         {best['edge']*100:.2f}¢")

    return hits, viable_records


# ---------------------------------------------------------------------------
# Discovery mode: events
# ---------------------------------------------------------------------------

def scan_events(
    min_edge: float,
    target_usd: float,
    events_limit: int = 20,
    verbose: bool = True,
    both_98_out: list | None = None,
    router: "CohortRouter | None" = None,
    reward_mode: bool = False,
) -> tuple:
    """
    Fetch /events, extract nested markets, apply outcomePrices pre-filter,
    then fetch CLOB books for viable candidates only.
    Returns (hits, viable_records, all_slugs).
    """
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}] mode=events  events_limit={events_limit}  min_edge={min_edge}")

    try:
        resp = httpx.get(
            f"{GAMMA}/events",
            params={
                "limit":     events_limit,
                "active":    "true",
                "closed":    "false",
                "order":     "volume24hr",
                "ascending": "false",
            },
            timeout=20,
        )
        events = resp.json()
        if not isinstance(events, list):
            events = events.get("events", [])
    except Exception as e:
        print(f"  ERROR fetching events: {e}")
        return [], [], []

    now_utc = dt.datetime.now(dt.timezone.utc)
    all_slugs  = []
    candidates = []
    skipped_outcome     = 0
    skipped_prefilter   = 0
    skipped_negrisk     = 0
    skipped_structural  = 0

    for ev in events:
        if ev.get("negRisk") or ev.get("enableNegRisk"):
            skipped_negrisk += len(ev.get("markets", []))
            if verbose:
                print(f"  SKIP(negRisk) event={ev.get('slug', ev.get('id', '?'))!r}  "
                      f"markets_dropped={len(ev.get('markets', []))}")
            continue
        for m in ev.get("markets", []):
            slug = str(m.get("slug") or "?")
            all_slugs.append(slug)

            # Stage 1: outcome_prices pre-filter (free — in payload)
            viable, p_yes, p_no = outcome_prices_viable(m)
            if not viable:
                skipped_outcome += 1
                if verbose:
                    print(f"  SKIP(prices) {slug[:55]}  p_yes={p_yes} p_no={p_no}")
                continue

            # Stage 2: expiry / volume / active pre-filter
            reason = prefilter_reason(m, now_utc)
            if reason:
                skipped_prefilter += 1
                if verbose:
                    print(f"  SKIP({reason}) {slug[:55]}")
                continue

            # Stage 3: structural slug pre-filter — exclude non-standalone families
            if slug_is_structural_derivative(slug):
                skipped_structural += 1
                if verbose:
                    print(f"  SKIP(structural) {slug[:55]}")
                continue

            candidates.append(m)

    print(
        f"  Events={len(events)}  total_markets={len(all_slugs)+skipped_negrisk}  "
        f"skipped_negRisk={skipped_negrisk}  skipped_prices={skipped_outcome}  "
        f"skipped_prefilter={skipped_prefilter}  skipped_structural={skipped_structural}  "
        f"candidates_for_clob={len(candidates)}"
    )

    if not candidates:
        print("  No candidates survived pre-filter — nothing to book-check.")
        return [], [], all_slugs

    # Cohort router: score + filter before CLOB fetch (no-op when disabled)
    if router is not None:
        candidates = router.filter_candidates(candidates, now_utc)
        if not candidates:
            print("  [router] All candidates suppressed — nothing to book-check.")
            return [], [], all_slugs

    hits, viable_records = process_candidates(
        candidates, min_edge, target_usd, verbose,
        both_98_out=both_98_out, reward_mode=reward_mode,
    )
    return hits, viable_records, all_slugs


# ---------------------------------------------------------------------------
# Discovery mode: slice rotation (v2, preserved)
# ---------------------------------------------------------------------------

def fetch_markets_slice(order: str, offset: int, limit: int) -> list:
    resp = httpx.get(
        f"{GAMMA}/markets",
        params={
            "limit":     limit,
            "offset":    offset,
            "active":    "true",
            "closed":    "false",
            "order":     order,
            "ascending": "false",
        },
        timeout=20,
    )
    data = resp.json()
    if not isinstance(data, list):
        data = data.get("markets", [])
    return data


def scan_slice(
    limit: int,
    min_edge: float,
    target_usd: float,
    order: str = "volume24hr",
    offset: int = 0,
    verbose: bool = True,
    both_98_out: list | None = None,
    router: "CohortRouter | None" = None,
    reward_mode: bool = False,
) -> tuple:
    """Returns (hits, viable_records, all_slugs)."""
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}] mode=slice  order={order!r}  offset={offset}  limit={limit}  min_edge={min_edge}")

    try:
        markets = fetch_markets_slice(order, offset, limit)
    except Exception as e:
        print(f"  ERROR fetching slice: {e}")
        return [], [], []

    now_utc    = dt.datetime.now(dt.timezone.utc)
    all_slugs  = []
    candidates = []
    stats      = dict(no_pair=0, prefiltered=0, outcome_skipped=0, structural=0)

    for i, m in enumerate(markets, 1):
        raw_ids = m.get("clobTokenIds") or []
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except Exception:
                stats["no_pair"] += 1
                continue
        if len(raw_ids) < 2:
            stats["no_pair"] += 1
            continue

        slug = str(m.get("slug") or "?")
        all_slugs.append(slug)

        viable, p_yes, p_no = outcome_prices_viable(m)
        if not viable:
            stats["outcome_skipped"] += 1
            if verbose:
                print(f"  SKIP(prices) [{i:2d}] {slug[:50]}  p_yes={p_yes} p_no={p_no}")
            continue

        reason = prefilter_reason(m, now_utc)
        if reason:
            stats["prefiltered"] += 1
            if verbose:
                print(f"  SKIP({reason}) [{i:2d}] {slug[:55]}")
            continue

        if slug_is_structural_derivative(slug):
            stats["structural"] += 1
            if verbose:
                print(f"  SKIP(structural) [{i:2d}] {slug[:55]}")
            continue

        candidates.append(m)

    print(
        f"  Markets={len(markets)}  no_pair={stats['no_pair']}  "
        f"outcome_skipped={stats['outcome_skipped']}  prefiltered={stats['prefiltered']}  "
        f"structural={stats['structural']}  candidates_for_clob={len(candidates)}"
    )

    if not candidates:
        print("  No candidates survived pre-filter.")
        return [], [], all_slugs

    # Cohort router: score + filter before CLOB fetch (no-op when disabled)
    if router is not None:
        candidates = router.filter_candidates(candidates, now_utc)
        if not candidates:
            print("  [router] All candidates suppressed — nothing to book-check.")
            return [], [], all_slugs

    hits, viable_records = process_candidates(
        candidates, min_edge, target_usd, verbose,
        both_98_out=both_98_out, reward_mode=reward_mode,
    )
    return hits, viable_records, all_slugs


def run_slice_rotate(
    args,
    state_path: pathlib.Path,
    both_98_out: list | None = None,
    router: "CohortRouter | None" = None,
    reward_mode: bool = False,
) -> tuple:
    state        = load_state(state_path)
    idx          = state["slice_index"] % len(SLICE_ROTATION)
    order, offset = SLICE_ROTATION[idx]
    prev_hash    = state.get("last_slug_hash")

    hits, viable_records, slugs = scan_slice(
        args.limit, args.min_edge, args.target_usd, order, offset,
        both_98_out=both_98_out, router=router, reward_mode=reward_mode,
    )

    current_hash = slug_set_hash(slugs) if slugs else None
    if current_hash and current_hash == prev_hash:
        print(f"  [WARN] Slice identical to previous round (hash={current_hash[:8]}). Advancing.")

    next_idx = (idx + 1) % len(SLICE_ROTATION)
    state.update(
        slice_index=next_idx,
        last_slug_hash=current_hash,
        round=state.get("round", 0) + 1,
    )
    save_state(state_path, state)
    nxt_o, nxt_off = SLICE_ROTATION[next_idx]
    print(f"  [rotate] round={state['round']}  used={order!r}+{offset}  next={nxt_o!r}+{nxt_off}")
    return hits, viable_records


# ---------------------------------------------------------------------------
# WebSocket real-time follow
# ---------------------------------------------------------------------------

async def ws_follow(viable_records: list, min_edge: float, top_n: int = 10):
    """
    Subscribe to Polymarket market channel for top-N candidates by edge proximity.
    Prints live edge updates. Read-only. Exits on Ctrl-C or after 5 minutes.
    """
    if not viable_records:
        print("  [follow] No viable records to monitor.")
        return

    # Rank by closeness to min_edge (HOT first, then WARM)
    ranked = sorted(viable_records, key=lambda r: abs(min_edge - r["edge"]))[:top_n]

    token_ids = []
    print(f"\n  [follow] Subscribing to {len(ranked)} candidates via WebSocket:")
    for r in ranked:
        gap = (min_edge - r["edge"]) * 100
        print(f"    {r['slug'][:55]}  edge={r['edge']*100:+.2f}¢  gap={gap:+.2f}¢")
        token_ids.append(r["yes_id"])
        token_ids.append(r["no_id"])

    # Build a lookup: token_id → (slug, side)
    book_state = {}   # yes_id → yes_ask,  no_id → no_ask
    meta       = {}   # yes_id → {slug, yes_id, no_id}
    for r in ranked:
        book_state[r["yes_id"]] = r["yes_ask"]
        book_state[r["no_id"]]  = r["no_ask"]
        meta[r["yes_id"]] = r
        meta[r["no_id"]]  = r

    subscribe_msg = json.dumps({"assets_ids": token_ids, "type": "Market"})
    deadline = time.time() + 300   # 5-minute cap

    print(f"\n  [follow] Monitoring live. Ctrl-C to stop. Auto-exits in 5 min.\n")

    try:
        async with websockets.connect(WS, ping_interval=20) as ws:
            await ws.send(subscribe_msg)
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    continue

                try:
                    payload = json.loads(raw)
                except Exception:
                    continue

                # Polymarket WS delivers either a single dict or a list of dicts
                msgs = payload if isinstance(payload, list) else [payload]

                asset_id = None
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue

                    # Ignore keepalives / non-book messages
                    event_type = msg.get("event_type") or ""
                    asset_id   = msg.get("asset_id") or ""

                    if event_type == "book":
                        asks = msg.get("asks") or []
                        if asks and asset_id in book_state:
                            book_state[asset_id] = float(asks[0]["price"])

                    elif event_type == "price_change":
                        changes = msg.get("changes") or []
                        for ch in changes:
                            if ch.get("side") == "SELL" and asset_id in book_state:
                                book_state[asset_id] = float(ch["price"])

                    else:
                        continue

                    # Recompute edge for affected market
                    if asset_id not in meta:
                        continue
                    r      = meta[asset_id]
                    yes_a  = book_state.get(r["yes_id"], r["yes_ask"])
                    no_a   = book_state.get(r["no_id"],  r["no_ask"])
                    edge   = 1.0 - yes_a - no_a
                    reg    = regime(edge, min_edge)
                    ts     = dt.datetime.now().isoformat(timespec="seconds")
                    gap    = (min_edge - edge) * 100
                    print(
                        f"  [{ts}] {reg:<9}  {r['slug'][:45]:<45}  "
                        f"YES={yes_a:.4f} NO={no_a:.4f}  "
                        f"edge={edge*100:+.2f}¢  gap={gap:+.2f}¢"
                    )

    except KeyboardInterrupt:
        print("\n  [follow] Stopped by user.")
    except Exception as e:
        print(f"\n  [follow] WebSocket error: {e}")

    print("  [follow] Session ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trial entry scanner v3 — multi-mode discovery + WS follow (read-only)"
    )
    parser.add_argument("--limit",         type=int,   default=50,
                        help="Markets per slice (slice mode)")
    parser.add_argument("--events-limit",  type=int,   default=20,
                        help="Events to fetch (events mode)")
    parser.add_argument("--min-edge",      type=float, default=0.03)
    parser.add_argument("--target-usd",    type=float, default=10.0)
    # Discovery mode
    parser.add_argument("--mode",          type=str,   default="both",
                        choices=["slice", "events", "both"],
                        help="Discovery mode: slice | events | both  (default: both)")
    # Slice-specific
    parser.add_argument("--order",         type=str,   default=None,
                        help="Force slice order key (volume24hr|liquidity|volume|startDate)")
    parser.add_argument("--offset",        type=int,   default=None,
                        help="Force slice offset (0, 50, 100, …)")
    parser.add_argument("--rotate",        action="store_true",
                        help="Auto-rotate slice mode across runs (saves .scan_state.json)")
    parser.add_argument("--state-file",    type=str,   default=str(DEFAULT_STATE_FILE))
    # Loop / follow
    parser.add_argument("--loop",          action="store_true")
    parser.add_argument("--interval-sec",  type=int,   default=60)
    parser.add_argument("--follow",        action="store_true",
                        help="After scan, subscribe to top candidates via WebSocket")
    parser.add_argument("--follow-top",    type=int,   default=10,
                        help="Number of top candidates to follow via WS (default: 10)")
    # Cohort router
    parser.add_argument("--no-router",     action="store_true",
                        help="Disable cohort router (pass all pre-filter survivors to CLOB fetch)")
    # Reward-aware mode
    parser.add_argument("--rewards",       action="store_true",
                        help="Print reward fitness telemetry (rate, spread, size eligibility) per candidate")
    # Misc
    parser.add_argument("--show-slices",   action="store_true",
                        help="Print slice rotation table and exit")
    args = parser.parse_args()

    if args.show_slices:
        print("SLICE_ROTATION:")
        for i, (o, off) in enumerate(SLICE_ROTATION):
            print(f"  [{i:2d}]  order={o!r}  offset={off}")
        return

    state_path = pathlib.Path(args.state_file)
    _sidecar = ABSidecar("sqlite:///data/processed/paper.db")
    _tracker = Both98PlateauTracker("sqlite:///data/processed/ab_sidecar.db")
    _router  = CohortRouter(
        plateau_tracker=_tracker,
        config=RouterConfig(enabled=not args.no_router),
    )

    def one_round():
        all_viable   = []
        both_98_seen = []
        round_ts     = dt.datetime.now(dt.timezone.utc)

        if args.mode in ("events", "both"):
            hits_e, viable_e, _ = scan_events(
                args.min_edge, args.target_usd, args.events_limit,
                both_98_out=both_98_seen,
                router=_router,
                reward_mode=args.rewards,
            )
            all_viable.extend(viable_e)

        if args.mode in ("slice", "both"):
            if args.rotate:
                hits_s, viable_s = run_slice_rotate(
                    args, state_path, both_98_out=both_98_seen,
                    router=_router, reward_mode=args.rewards,
                )
            else:
                order  = args.order  or "volume24hr"
                offset = args.offset or 0
                hits_s, viable_s, _ = scan_slice(
                    args.limit, args.min_edge, args.target_usd, order, offset,
                    both_98_out=both_98_seen,
                    router=_router,
                    reward_mode=args.rewards,
                )
            all_viable.extend(viable_s)

        # A+B sidecar: write viable records and plateau observations
        for rec in all_viable:
            _sidecar.observe_scan_record(rec, round_ts)
        _tracker.record_round(both_98_seen, [])

        if args.follow and all_viable:
            # Deduplicate by slug
            seen  = set()
            dedup = []
            for r in all_viable:
                if r["slug"] not in seen:
                    seen.add(r["slug"])
                    dedup.append(r)
            asyncio.run(ws_follow(dedup, args.min_edge, args.follow_top))

    if not args.loop:
        one_round()
        return

    try:
        while True:
            one_round()
            print(f"\nSleeping {args.interval_sec}s…\n")
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
