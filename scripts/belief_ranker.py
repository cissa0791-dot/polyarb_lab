"""
belief_ranker.py — Track B / Phase 2/3
Belief-aware opportunity ranker for Polymarket prediction markets.

Pipeline:
  Stage 1  RawCandidate        — event pre-filters (negRisk, structural slug, outcomePrices)
  Stage 2  ExecutableCandidate — CLOB book fetch, depth, feasibility labeling
  Stage 3  RankedOpportunity   — logit + uncertainty + depth + fragility + persistence + belief-vol

Output:
  data/research/ranked_<timestamp>.json   full ranked report
  data/research/ranked_latest.json        overwritten each run
  stdout                                   ranked table + explanation lines

Hard constraints:
  - read-only: no orders, no DB writes to Track A tables
  - does not import from src.live.*, src.runtime.*, src.storage.*
  - does not modify Track A scanner or trial policy
  - branch-safe: safe to run on branch/belief_ranker_research_v1

Usage:
  py -3 scripts/belief_ranker.py
  py -3 scripts/belief_ranker.py --events-limit 50 --min-edge -0.10 --top 20
  py -3 scripts/belief_ranker.py --loop --interval-sec 120
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import statistics
import sys
import time

import httpx

# ---------------------------------------------------------------------------
# Research pipeline imports (Track B only)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.research.models   import RawCandidate, ExecutableCandidate, RankedOpportunity
from src.sidecar.plateau_tracker import Both98PlateauTracker
from src.research.features import (
    safe_logit, logit_features, uncertainty,
    spread_features, depth_near_ask, depth_features,
    fragility_score, persistence_rounds, belief_vol_proxy,
    composite_score_and_explanation,
)
from src.research.pipeline import BeliefPipelineRegistry
from src.research.state_estimation import R_from_spread

# ---------------------------------------------------------------------------
# API endpoints (same as Track A scanner, no import dependency)
# ---------------------------------------------------------------------------

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

# ---------------------------------------------------------------------------
# Pre-filter constants (duplicated from Track A — no cross-import)
# ---------------------------------------------------------------------------

OUTCOME_PRICE_MIN    = 0.05
OUTCOME_PRICE_MAX    = 0.95
MIN_VOLUME_USD       = 100.0
EXPIRY_HORIZON_HOURS = 6
DEPTH_BAND           = 0.05    # shares within this many ticks of best ask count as depth
MIN_DEPTH_MULTIPLE   = 3.0     # minimum depth = MIN_DEPTH_MULTIPLE * min_size

# Structural slug filter (same patterns as Track A scanner, independent copy)
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

# ---------------------------------------------------------------------------
# Snapshot persistence state
# ---------------------------------------------------------------------------

STATE_PATH    = pathlib.Path(__file__).parent.parent / "data" / "research" / "ranker_snapshots.json"
MAX_SNAPSHOTS = 20   # keep last N snapshots per slug
PIPELINE_PATH = pathlib.Path(__file__).parent.parent / "data" / "research" / "pipeline_state.json"


def load_snapshot_history() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_snapshot_history(history: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_pipeline_state() -> BeliefPipelineRegistry:
    if PIPELINE_PATH.exists():
        try:
            import json as _json
            d = _json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
            return BeliefPipelineRegistry.from_dict(d)
        except Exception:
            pass
    return BeliefPipelineRegistry()


def save_pipeline_state(registry: BeliefPipelineRegistry) -> None:
    import json as _json
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_PATH.write_text(_json.dumps(registry.to_dict(), indent=2), encoding="utf-8")


def update_history(history: dict, slug: str, p_yes: float, edge: float, ts: str) -> None:
    entry = {"p_yes": p_yes, "edge": edge, "ts": ts}
    snaps = history.get(slug, [])
    snaps.append(entry)
    history[slug] = snaps[-MAX_SNAPSHOTS:]   # keep rolling window


# ---------------------------------------------------------------------------
# Stage 1 — Pre-filter and build RawCandidates
# ---------------------------------------------------------------------------

def fetch_raw_candidates(events_limit: int, verbose: bool) -> tuple[list[RawCandidate], dict]:
    """
    Fetch /events and apply pre-filters. Returns (raw_candidates, filter_stats).
    Zero CLOB calls at this stage.
    """
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}]  stage=raw_candidates  events_limit={events_limit}")

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
        return [], {}

    now_utc = dt.datetime.now(dt.timezone.utc)
    stats   = dict(total=0, negrisk=0, prices=0, prefilter=0, structural=0, passed=0)
    raw     = []

    for ev in events:
        if ev.get("negRisk") or ev.get("enableNegRisk"):
            dropped = len(ev.get("markets", []))
            stats["negrisk"] += dropped
            stats["total"]   += dropped
            continue

        for m in ev.get("markets", []):
            stats["total"] += 1
            slug = str(m.get("slug") or "?")

            # outcomePrices filter
            op_raw = m.get("outcomePrices")
            if not op_raw:
                stats["prices"] += 1
                continue
            try:
                op    = json.loads(op_raw) if isinstance(op_raw, str) else list(op_raw)
                p_yes = float(op[0])
                p_no  = float(op[1])
            except Exception:
                stats["prices"] += 1
                continue
            if p_yes < OUTCOME_PRICE_MIN or p_yes > OUTCOME_PRICE_MAX:
                stats["prices"] += 1
                continue
            if p_no < OUTCOME_PRICE_MIN or p_no > OUTCOME_PRICE_MAX:
                stats["prices"] += 1
                continue

            # Expiry / volume / active filter
            skip = _prefilter_reason(m, now_utc)
            if skip:
                stats["prefilter"] += 1
                continue

            # Structural slug filter
            if _STRUCTURAL_SLUG_RE.search(slug):
                stats["structural"] += 1
                continue

            # Token IDs
            raw_ids = m.get("clobTokenIds") or []
            if isinstance(raw_ids, str):
                try:
                    raw_ids = json.loads(raw_ids)
                except Exception:
                    continue
            if len(raw_ids) < 2:
                continue

            stats["passed"] += 1
            raw.append(RawCandidate(
                slug       = slug,
                p_yes      = p_yes,
                p_no       = p_no,
                volume_usd = float(m.get("volume") or 0),
                end_date   = (m.get("endDate") or m.get("end_date_iso") or m.get("endDateIso")),
                yes_id     = str(raw_ids[0]),
                no_id      = str(raw_ids[1]),
                min_size   = float(m.get("orderMinSize") or 5),
                scanned_at = ts,
            ))

    print(
        f"  total={stats['total']}  negRisk={stats['negrisk']}  prices={stats['prices']}  "
        f"prefilter={stats['prefilter']}  structural={stats['structural']}  "
        f"raw_candidates={stats['passed']}"
    )
    return raw, stats


def _prefilter_reason(m: dict, now_utc: dt.datetime) -> str:
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
                    return f"expires_soon"
            except Exception:
                pass
            break
    if float(m.get("volume") or 0) < MIN_VOLUME_USD:
        return "low_vol"
    return ""


# ---------------------------------------------------------------------------
# Stage 2 — CLOB book fetch and feasibility labeling
# ---------------------------------------------------------------------------

def build_executable_candidates(
    raw: list[RawCandidate],
    min_edge: float,
    verbose: bool,
    include_both_98: bool = False,
    both_98_out: list | None = None,
) -> list[ExecutableCandidate]:
    """
    Fetch CLOB books for each RawCandidate. Compute edge and depth. Label feasibility.
    """
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}]  stage=executable_candidates  raw={len(raw)}")

    results = []
    stats   = dict(no_book=0, both_98=0, passed=0, executable=0)

    for i, rc in enumerate(raw, 1):
        if verbose:
            print(f"  BOOK [{i:3d}] {rc.slug[:60]}")

        try:
            yb = httpx.get(f"{CLOB}/book", params={"token_id": rc.yes_id}, timeout=5).json()
            nb = httpx.get(f"{CLOB}/book", params={"token_id": rc.no_id},  timeout=5).json()
        except Exception:
            stats["no_book"] += 1
            continue

        ya = yb.get("asks") or []
        na = nb.get("asks") or []
        if not ya or not na:
            stats["no_book"] += 1
            continue

        yes_ask = float(ya[0]["price"])
        no_ask  = float(na[0]["price"])

        if yes_ask >= 0.98 and no_ask >= 0.98:
            stats["both_98"] += 1
            if both_98_out is not None:
                both_98_out.append({
                    "slug":       rc.slug,
                    "yes_ask":    yes_ask,
                    "no_ask":     no_ask,
                    "edge_cents": round((1.0 - yes_ask - no_ask) * 100.0, 4),
                })
            if not include_both_98:
                continue
            # Research mode: include with both_98 flag so ranker can score and explain it
            skip_forced = "both_98"
        else:
            skip_forced = None

        edge           = 1.0 - yes_ask - no_ask
        yes_depth      = depth_near_ask(ya, yes_ask, DEPTH_BAND)
        no_depth       = depth_near_ask(na, no_ask,  DEPTH_BAND)
        required_depth = rc.min_size * MIN_DEPTH_MULTIPLE

        # Feasibility check (paper label — not a live order trigger)
        if skip_forced:
            skip = skip_forced
        elif edge < min_edge:
            skip = f"edge={edge*100:.2f}¢<{min_edge*100:.2f}¢"
        elif yes_depth < required_depth or no_depth < required_depth:
            skip = f"depth_short yes={yes_depth:.0f} no={no_depth:.0f} req={required_depth:.0f}"
        else:
            skip = ""

        stats["passed"] += 1
        if not skip:
            stats["executable"] += 1

        results.append(ExecutableCandidate(
            slug             = rc.slug,
            p_yes            = rc.p_yes,
            p_no             = rc.p_no,
            volume_usd       = rc.volume_usd,
            end_date         = rc.end_date,
            yes_id           = rc.yes_id,
            no_id            = rc.no_id,
            min_size         = rc.min_size,
            scanned_at       = rc.scanned_at,
            yes_ask          = yes_ask,
            no_ask           = no_ask,
            edge             = edge,
            yes_depth_shares = yes_depth,
            no_depth_shares  = no_depth,
            yes_book_levels  = len(ya),
            no_book_levels   = len(na),
            is_executable    = (skip == ""),
            skip_reason      = skip,
        ))

    print(
        f"  no_book={stats['no_book']}  both_98={stats['both_98']}  "
        f"passed={stats['passed']}  executable={stats['executable']}"
    )
    return results


# ---------------------------------------------------------------------------
# Stage 3 — Feature computation and ranking
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: list[ExecutableCandidate],
    history: dict,
    pipeline_results: dict = None,
) -> list[RankedOpportunity]:
    """
    Compute full feature set and composite score for each candidate. Return ranked list.
    """
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}]  stage=ranked_opportunities  candidates={len(candidates)}")

    ranked = []
    for c in candidates:
        lf  = logit_features(c.p_yes, c.yes_ask)
        sf  = spread_features(c.yes_ask, c.no_ask, c.edge)
        df  = depth_features(c.yes_depth_shares, c.no_depth_shares)
        u   = uncertainty(c.p_yes)
        fg  = fragility_score(c.yes_ask, c.no_ask, c.yes_depth_shares, c.no_depth_shares, c.min_size)
        per = persistence_rounds(c.slug, history)
        if pipeline_results and c.slug in pipeline_results:
            pr = pipeline_results[c.slug]
            bv = pr.sigma_b        # theory-backed EWMA σ_b
            sigma_b_t  = pr.sigma_b
            p_filt     = pr.p_filtered
            jump_flag  = pr.is_jump
        else:
            bv = belief_vol_proxy(c.slug, history)
            sigma_b_t  = 0.0
            p_filt     = c.p_yes
            jump_flag  = False

        score, expl = composite_score_and_explanation(
            edge                  = c.edge,
            uncertainty_val       = u,
            spread_over_edge_ratio= sf["spread_over_edge_ratio"],
            fragility             = fg,
            persistence           = per,
            belief_vol            = bv,
        )

        ranked.append(RankedOpportunity(
            candidate              = c,
            logit_p_yes            = lf["logit_p_yes"],
            logit_ask_yes          = lf["logit_ask_yes"],
            logit_spread           = lf["logit_spread"],
            uncertainty            = u,
            spread_cents           = sf["spread_cents"],
            spread_over_edge_ratio = sf["spread_over_edge_ratio"],
            depth_imbalance        = df["depth_imbalance"],
            fragility_score        = fg,
            persistence_rounds     = per,
            belief_vol_proxy       = bv,
            sigma_b_theory         = sigma_b_t,
            p_filtered             = p_filt,
            is_jump                = jump_flag,
            composite_score        = score,
            rank                   = 0,   # assigned after sort
            explanation            = expl,
        ))

    ranked.sort(key=lambda r: r.composite_score, reverse=True)
    for i, r in enumerate(ranked, 1):
        r.rank = i

    return ranked


# ---------------------------------------------------------------------------
# Saturation detection
# ---------------------------------------------------------------------------

# Thresholds are explicit constants — conservative by design.
_SAT_MIN_PERSIST   = 20      # persistence_rounds must be pinned at max window
_SAT_EDGE_WINDOW   = 10      # look at last N edge snapshots for drift
_SAT_MIN_SNAPS     = 5       # require at least this many recent snaps
_SAT_EDGE_STDEV    = 0.001   # edge stdev below this → effectively frozen


def _is_saturated(r: RankedOpportunity, history: dict) -> bool:
    """Return True when this candidate is a frozen both_98 cohort with no new signal.

    All three conditions must hold:
      1. persist >= 20       — pinned at history-window max; no new discovery
      2. not is_executable   — never cleared CLOB thresholds in any round
      3. edge stdev < 0.001  — edge is frozen over the last 10 rounds

    Conservative: a candidate escapes saturation the moment any condition
    fails (e.g. persistence drops after a gap, or it becomes executable).
    """
    if r.persistence_rounds < _SAT_MIN_PERSIST:
        return False
    if r.candidate.is_executable:
        return False
    snaps = history.get(r.candidate.slug, [])
    recent = [s.get("edge", 0.0) for s in snaps[-_SAT_EDGE_WINDOW:] if isinstance(s, dict)]
    if len(recent) < _SAT_MIN_SNAPS:
        return False
    try:
        return statistics.stdev(recent) < _SAT_EDGE_STDEV
    except statistics.StatisticsError:
        return False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "research"


def emit_report(
    ranked: list[RankedOpportunity],
    top: int,
    history: dict | None = None,
    show_saturated: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "total_ranked": len(ranked),
        "opportunities": [r.to_dict() for r in ranked],
    }

    out_path    = OUTPUT_DIR / f"ranked_{ts_file}.json"
    latest_path = OUTPUT_DIR / "ranked_latest.json"

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  report → {out_path.name}  (also → ranked_latest.json)")

    # Split into active (novel) and saturated cohorts.
    # JSON report always contains all candidates; only the display table filters.
    if history is not None and not show_saturated:
        active     = [r for r in ranked if not _is_saturated(r, history)]
        n_suppressed = len(ranked) - len(active)
    else:
        active       = ranked
        n_suppressed = 0

    if n_suppressed:
        print(f"  [{n_suppressed} saturated plateau cohort(s) suppressed "
              f"— use --include-saturated to show]")

    display = active[:top]
    if not display:
        print("  No non-saturated candidates to display.")
        return

    mode_tag = "" if not show_saturated else "  [--include-saturated: all cohorts shown]"
    print(f"\n{'':=<120}")
    print(f"  {'#':>3}  {'slug':<52} {'edge¢':>6} {'p_yes':>6} {'u':>6} {'logit_spr':>10} "
          f"{'frag':>5} {'pers':>5} {'bvol':>7} {'score':>8}  {'exec':>5}{mode_tag}")
    print(f"{'':=<120}")

    for r in display:
        c = r.candidate
        exec_flag = "YES" if c.is_executable else "---"
        print(
            f"  {r.rank:>3}  {c.slug[:52]:<52} {c.edge*100:>+6.2f} {c.p_yes:>6.3f} "
            f"{r.uncertainty:>6.4f} {r.logit_spread:>+10.4f} "
            f"{r.fragility_score:>5.3f} {r.persistence_rounds:>5d} {r.belief_vol_proxy:>7.5f} "
            f"{r.composite_score:>+8.4f}  {exec_flag:>5}"
        )

    print(f"{'':=<120}")
    print(f"\n  Score explanations (top {min(5, len(display))}):")
    for r in display[:5]:
        print(f"  [{r.rank}] {r.candidate.slug[:50]}")
        print(f"      {r.explanation}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def one_round(args: argparse.Namespace, history: dict, registry: BeliefPipelineRegistry = None, tracker: Both98PlateauTracker | None = None) -> None:
    raw, _        = fetch_raw_candidates(args.events_limit, args.verbose)
    both_98_seen: list = []
    candidates    = build_executable_candidates(raw, args.min_edge, args.verbose, args.include_both_98, both_98_out=both_98_seen)

    # Update snapshot history + run pipeline for all candidates
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pipeline_results = {}
    for c in candidates:
        update_history(history, c.slug, c.p_yes, c.edge, ts)
        if registry is not None:
            R = R_from_spread(c.yes_ask, c.no_ask)
            pr = registry.step(c.slug, c.p_yes, R_override=R, timestamp=ts)
            pipeline_results[c.slug] = pr
    save_snapshot_history(history)
    if registry is not None:
        save_pipeline_state(registry)

    ranked = rank_candidates(candidates, history, pipeline_results)
    emit_report(ranked, args.top, history=history, show_saturated=args.include_saturated)
    if tracker is not None:
        tracker.record_round(both_98_seen, ranked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="belief_ranker.py — Track B Phase 2/3 / read-only research ranker"
    )
    parser.add_argument("--events-limit",  type=int,   default=50)
    parser.add_argument("--min-edge",      type=float, default=-0.10,
                        help="Include candidates with edge >= this value (default -0.10 = all viable)")
    parser.add_argument("--top",           type=int,   default=15,
                        help="Number of ranked candidates to display in table")
    parser.add_argument("--loop",          action="store_true")
    parser.add_argument("--interval-sec",  type=int,   default=120)
    parser.add_argument("--verbose",          action="store_true")
    parser.add_argument("--include-both-98", action="store_true",
                        help="Research mode: include both_98 candidates with max fragility penalty")
    parser.add_argument("--include-saturated", action="store_true",
                        help="Show saturated plateau cohorts in ranked table (default: suppressed)")
    args = parser.parse_args()

    history  = load_snapshot_history()
    registry = load_pipeline_state()
    tracker  = Both98PlateauTracker("sqlite:///data/processed/ab_sidecar.db")

    if not args.loop:
        one_round(args, history, registry, tracker=tracker)
        return

    print(f"  [belief_ranker] loop mode: interval={args.interval_sec}s  Ctrl-C to stop")
    try:
        while True:
            one_round(args, history, registry, tracker=tracker)
            print(f"\n  sleeping {args.interval_sec}s …\n")
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("\n  [belief_ranker] stopped.")


if __name__ == "__main__":
    main()
