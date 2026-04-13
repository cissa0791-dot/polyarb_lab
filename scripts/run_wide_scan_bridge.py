"""
Minimum bridge: routes one wide_scan ROBUST_PAPER_MM candidate (Hungary PM neg-risk pair)
through the existing downstream validation stack, including paper execution and persistence.

Resolves:
  G1 - token_id: fetched fresh from Gamma registry for each Hungary market.
  G2 - live books: fetched from CLOB using token IDs before qualification.
  G3 - RawCandidate: plan dict transformed into RawCandidate + CandidateLeg[].
  G5 - config adapter: min_edge_cents=0.005 for maker-MM reward-driven candidates.
  G6 - config adapter: max_order_notional_usd=250 for rewards_min_size=200.

Downstream components used UNCHANGED:
  ExecutionFeasibilityEvaluator.qualify()  [src/opportunity/qualification.py]
  OpportunityRanker.rank()                 [src/opportunity/qualification.py]
  DepthCappedSizer.size()                  [src/sizing/engine.py]
  RiskManager.evaluate()                   [src/risk/manager.py]
  PaperBroker.submit_limit_order()         [src/paper/broker.py]
  Ledger (in-memory)                       [src/paper/ledger.py]
  ResearchStore (SQLite)                   [src/storage/event_store.py]

Persistence target: data/processed/bridge_test.db (isolated from paper.db)

Usage:
    py -3 scripts/run_wide_scan_bridge.py
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.models import ExecutionConfig, OpportunityConfig, PaperConfig, RiskConfig
from src.domain.models import AccountSnapshot, OrderIntent, OrderMode, OrderType, RiskStatus
from src.ingest.clob import ReadOnlyClob
from src.ingest.gamma import fetch_events, fetch_markets
from src.intelligence.market_intelligence import build_event_market_registry
from src.opportunity.models import CandidateLeg, RawCandidate, StrategyFamily
from src.opportunity.qualification import ExecutionFeasibilityEvaluator, OpportunityRanker
from src.paper.broker import PaperBroker
from src.paper.ledger import Ledger
from src.risk.manager import RiskManager
from src.sizing.engine import DepthCappedSizer
from src.storage.event_store import ResearchStore

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST  = "https://clob.polymarket.com"
HUNGARY_EVENT_SLUG = "next-prime-minister-of-hungary"
OUT_DIR    = ROOT / "data" / "reports"
# Isolated bridge DB — does not touch the main paper.db
BRIDGE_DB_URL = f"sqlite:///{ROOT / 'data' / 'processed' / 'bridge_test.db'}"


# ---------------------------------------------------------------------------
# G1: Locate Hungary markets in the Gamma registry and extract token IDs
# ---------------------------------------------------------------------------

def fetch_hungary_market_candidates(
    gamma_host: str,
    event_slug: str,
) -> list[dict[str, Any]]:
    """
    Fetch Gamma registry and return eligible Hungary PM markets with token IDs.
    Eligible = is_binary_yes_no, enable_orderbook, reward_daily_rate > 0,
               rewards_min_size > 0, rewards_max_spread > 0,
               best_bid > 0, best_ask > best_bid, yes_token_id present.
    """
    print(f"  [G1] Fetching Gamma registry for event '{event_slug}'...")
    events = fetch_events(gamma_host, limit=500)
    markets = fetch_markets(gamma_host, limit=500)
    registry = build_event_market_registry(events, markets)

    candidates: list[dict[str, Any]] = []
    for event in registry.get("events", []):
        if event.get("slug") != event_slug:
            continue
        for m in event.get("markets", []):
            if not m.get("is_binary_yes_no") or not m.get("enable_orderbook"):
                continue
            rewards = m.get("clob_rewards") or []
            reward_rate = sum(float(r.get("rewardsDailyRate", 0) or 0) for r in rewards)
            if reward_rate <= 0:
                continue
            min_size  = float(m.get("rewards_min_size") or 0)
            max_spread = float(m.get("rewards_max_spread") or 0)
            if min_size <= 0 or max_spread <= 0:
                continue
            best_bid = float(m.get("best_bid") or 0)
            best_ask = float(m.get("best_ask") or 0)
            if best_bid <= 0 or best_ask <= best_bid:
                continue
            yes_token_id = m.get("yes_token_id")
            no_token_id  = m.get("no_token_id")
            if not yes_token_id or not no_token_id:
                continue

            candidates.append({
                "event_slug":        event_slug,
                "event_title":       event.get("title"),
                "market_slug":       m.get("slug"),
                "question":          m.get("question"),
                "yes_token_id":      str(yes_token_id),
                "no_token_id":       str(no_token_id),
                "best_bid":          best_bid,
                "best_ask":          best_ask,
                "rewards_min_size":  min_size,
                "rewards_max_spread": max_spread,
                "reward_daily_rate": reward_rate,
                "fees_enabled":      bool(m.get("fees_enabled")),
                "neg_risk":          bool(m.get("neg_risk")),
                "volume_num":        float(m.get("volume_num") or 0),
            })

    print(f"  [G1] Found {len(candidates)} eligible Hungary market(s) with token IDs.")
    return candidates


# ---------------------------------------------------------------------------
# G2: Fetch live CLOB orderbooks for both YES and NO token IDs
# ---------------------------------------------------------------------------

def fetch_live_books(
    clob_host: str,
    yes_token_id: str,
    no_token_id: str,
) -> tuple[object | None, object | None]:
    """Fetch live CLOB orderbooks for a YES/NO token pair. Returns (yes_book, no_book)."""
    print(f"  [G2] Fetching live books for YES={yes_token_id[:12]}... NO={no_token_id[:12]}...")
    clob = ReadOnlyClob(clob_host)
    books = clob.prefetch_books([yes_token_id, no_token_id], max_workers=2)
    yes_book = books.get(yes_token_id)
    no_book  = books.get(no_token_id)
    yes_depth = len(yes_book.asks) if yes_book else 0
    no_depth  = len(no_book.asks)  if no_book  else 0
    print(f"  [G2] YES book: {yes_depth} ask levels | NO book: {no_depth} ask levels")
    return yes_book, no_book


# ---------------------------------------------------------------------------
# EV computation (same formula as run_wide_maker_scan.py)
# ---------------------------------------------------------------------------

def compute_ev(m: dict[str, Any]) -> dict[str, Any]:
    best_bid  = m["best_bid"]
    best_ask  = m["best_ask"]
    min_size  = m["rewards_min_size"]
    max_spread_cents = m["rewards_max_spread"]
    reward_rate = m["reward_daily_rate"]
    volume_num  = m.get("volume_num", 0)

    current_spread = best_ask - best_bid
    max_spread     = max_spread_cents / 100.0
    midpoint       = (best_bid + best_ask) / 2.0
    our_half_spread = min(current_spread / 2.0, max_spread / 2.0)
    our_half_spread = max(our_half_spread, 0.005)
    quote_bid  = round(max(0.01, midpoint - our_half_spread), 4)
    quote_ask  = round(min(0.99, midpoint + our_half_spread), 4)
    quote_spread = quote_ask - quote_bid
    quote_size = max(min_size, 20.0)

    distance_from_mid = our_half_spread * 100
    v = max_spread_cents
    s = distance_from_mid
    q_per_side = ((v - s) / v) ** 2 * quote_size if v > 0 and s < v else 0.0
    our_q_score = q_per_side * 2
    competition_factor = min(30.0, max(10.0, 10.0 + reward_rate / 20.0))
    estimated_total_q  = our_q_score * competition_factor
    reward_share = reward_rate * (our_q_score / estimated_total_q) if estimated_total_q > 0 else 0.0

    volume_factor   = min(1.0, volume_num / 50000.0) if volume_num > 0 else 0.1
    tightness_factor = max(0.0, 1.0 - (distance_from_mid / max_spread_cents))
    fill_prob_per_side = min(0.90, max(0.05, 0.10 + 0.50 * volume_factor + 0.30 * tightness_factor))
    both_fill_prob = fill_prob_per_side ** 2 * 0.5
    spread_capture_ev = both_fill_prob * quote_spread * quote_size
    reward_ev = reward_share

    adverse_cost  = fill_prob_per_side * 0.15 * current_spread * 2 * quote_size
    inventory_cost = fill_prob_per_side * (1 - fill_prob_per_side) * 2 * current_spread * quote_size * 0.5
    cancel_cost    = 0.001
    total_ev = round(spread_capture_ev + reward_ev - adverse_cost - inventory_cost - cancel_cost, 6)

    return {
        "quote_bid":          quote_bid,
        "quote_ask":          quote_ask,
        "quote_spread":       round(quote_spread, 6),
        "quote_size":         quote_size,
        "midpoint":           round(midpoint, 4),
        "spread_capture_ev":  round(spread_capture_ev, 6),
        "reward_ev":          round(reward_ev, 6),
        "adverse_cost":       round(adverse_cost, 6),
        "inventory_cost":     round(inventory_cost, 6),
        "cancel_cost":        round(cancel_cost, 6),
        "total_ev":           total_ev,
    }


# ---------------------------------------------------------------------------
# G3: Transform market metadata + EV into RawCandidate + CandidateLeg[]
# ---------------------------------------------------------------------------

def build_raw_candidate(
    m: dict[str, Any],
    ev: dict[str, Any],
    yes_token_id: str,
    no_token_id: str,
) -> RawCandidate:
    """
    Construct a RawCandidate from wide_scan plan data.

    expected_payout encoding for neg-risk maker-MM:
      - Redemption value of position:  quote_size × $1.00 (neg-risk: YES+NO=$1)
      - Net daily EV from market making: total_ev
      - Total: quote_size + total_ev
    This makes payout_per_share = 1.0 + total_ev/quote_size,
    and pair_vwap (live books, neg-risk) ≈ 1.0, so:
      gross_edge_cents ≈ total_ev / quote_size (per-share daily maker reward).
    Qualification arithmetic is valid under this encoding.
    """
    quote_size   = ev["quote_size"]
    total_ev     = ev["total_ev"]
    market_slug  = m["market_slug"]

    # For non-neg-risk: estimate pair cost from live bid/ask rather than assuming 1.0.
    # For neg-risk: YES + NO = $1 by construction.
    pair_cost_per_unit = 1.0 if m.get("neg_risk") else (m["best_ask"] * 2)

    legs = [
        CandidateLeg(
            token_id=yes_token_id,
            market_slug=market_slug,
            action="BUY",
            side="YES",
            required_shares=quote_size,
        ),
        CandidateLeg(
            token_id=no_token_id,
            market_slug=market_slug,
            action="BUY",
            side="NO",
            required_shares=quote_size,
        ),
    ]

    return RawCandidate(
        strategy_id="wide_scan_bridge",
        strategy_family=StrategyFamily.MAKER_REWARDED_EVENT_MM_V1,
        candidate_id=f"wsb_{market_slug[:32]}_{uuid.uuid4().hex[:8]}",
        kind="maker_mm_quote",
        detection_name=f"wide_scan_bridge:{m['event_slug']}:{market_slug}",
        market_slugs=[market_slug],
        gross_edge_cents=round(total_ev / max(quote_size, 1e-9), 6),
        expected_payout=quote_size * pair_cost_per_unit + total_ev,
        target_notional_usd=quote_size * pair_cost_per_unit,
        target_shares=quote_size,
        execution_mode="paper_eligible",
        research_only=False,
        legs=legs,
        metadata={
            "event_slug":        m["event_slug"],
            "event_title":       m.get("event_title"),
            "question":          m.get("question"),
            "neg_risk":          m.get("neg_risk"),
            "reward_daily_rate": m["reward_daily_rate"],
            "rewards_min_size":  m["rewards_min_size"],
            "rewards_max_spread": m["rewards_max_spread"],
            "wide_scan_ev":      ev,
            "bridge_note": (
                "expected_payout = quote_size*pair_cost + total_ev; "
                "encodes redemption + maker reward for qualification arithmetic"
            ),
        },
        ts=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Paper execution + persistence (Stage 5 + 6)
# ---------------------------------------------------------------------------

def run_paper_and_persist(
    ranked: Any,
    risk_decision: Any,
    sizing: Any,
    books_by_token: dict[str, Any],
    db_url: str,
) -> dict[str, Any]:
    """
    Submit paper orders for each leg and persist all records to bridge_test.db.

    Uses:
      PaperBroker.submit_limit_order()  [src/paper/broker.py]   — UNCHANGED
      Ledger                             [src/paper/ledger.py]   — UNCHANGED
      ResearchStore                      [src/storage/]          — UNCHANGED

    limit_price per leg = leg.best_price (live best ask at qualification time).
    order size per leg  = sizing.shares (depth-capped, quality/risk adjusted).
    """
    now = datetime.now(timezone.utc)
    ledger = Ledger(cash=10000.0)
    broker = PaperBroker(ledger, fee_rate=0.0, auto_cancel_unfilled=True)
    store  = ResearchStore(db_url)

    # Persist candidate and risk decision before orders
    store.save_candidate(ranked)
    store.save_risk_decision(risk_decision)

    order_records: list[dict[str, Any]] = []
    all_filled = True

    for leg in ranked.legs:
        # limit_price: live best ask (BUY action). Falls back to book lookup.
        limit_price: float | None = leg.best_price
        if limit_price is None:
            book = books_by_token.get(leg.token_id)
            limit_price = book.asks[0].price if (book and book.asks) else None
        if limit_price is None or limit_price <= 0:
            order_records.append({
                "token_id":   leg.token_id,
                "side":       leg.action,
                "status":     "SKIPPED",
                "reason":     "no valid limit_price from book",
            })
            all_filled = False
            continue

        intent = OrderIntent(
            intent_id=str(uuid.uuid4()),
            candidate_id=ranked.candidate_id,
            mode=OrderMode.PAPER,
            market_slug=leg.market_slug,
            token_id=leg.token_id,
            side=leg.action,
            order_type=OrderType.LIMIT,
            size=sizing.shares,
            limit_price=limit_price,
            max_notional_usd=round(sizing.shares * limit_price, 6),
            ts=now,
        )

        book = books_by_token.get(leg.token_id)
        report = broker.submit_limit_order(intent, book)

        store.save_order_intent(intent)
        store.save_execution_report(report)

        if report.status.value != "filled":
            all_filled = False

        order_records.append({
            "token_id":       leg.token_id,
            "side":           leg.action,
            "size_requested": sizing.shares,
            "limit_price":    limit_price,
            "fill_status":    report.status.value,
            "filled_size":    report.filled_size,
            "avg_fill_price": report.avg_fill_price,
            "fee_paid_usd":   report.fee_paid_usd,
        })

    # Post-execution account snapshot
    post_snap = ledger.snapshot(ts=now)
    store.save_account_snapshot(post_snap)

    # Count rows written to DB for verification
    from sqlalchemy import select, func as sqlfunc
    row_counts: dict[str, int] = {}
    tables = {
        "opportunity_candidates": store.opportunity_candidates,
        "risk_decisions":         store.risk_decisions,
        "order_intents":          store.order_intents,
        "execution_reports":      store.execution_reports,
        "account_snapshots":      store.account_snapshots,
    }
    with store.engine.begin() as conn:
        for name, table in tables.items():
            row_counts[name] = conn.execute(
                select(sqlfunc.count()).select_from(table)
            ).scalar()

    store.close()

    open_positions = [
        {
            "symbol":     pos.symbol,
            "shares":     round(pos.remaining_shares, 6),
            "avg_price":  round(pos.avg_entry_price, 6),
            "entry_cost": round(pos.entry_cost_usd, 6),
            "state":      pos.state.value,
        }
        for pos in ledger.get_open_positions()
    ]

    any_partial = any(o.get("fill_status") == "partial" for o in order_records)
    if all_filled:
        verdict = "FULL_PAPER_FILL_AND_PERSISTED"
    elif any_partial:
        verdict = "PARTIAL_FILL_AND_PERSISTED"
    else:
        verdict = "UNFILLED_AND_PERSISTED"

    return {
        "execution": {
            "all_legs_filled":    all_filled,
            "orders":             order_records,
            "post_cash":          round(post_snap.cash, 4),
            "post_frozen":        round(post_snap.frozen_cash, 4),
            "open_positions":     open_positions,
            "open_position_count": post_snap.open_positions,
        },
        "persistence": {
            "db_url":    db_url,
            "row_counts": row_counts,
        },
        "final_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main bridge validation
# ---------------------------------------------------------------------------

def validate_one_market(
    m: dict[str, Any],
    opp_cfg: OpportunityConfig,
    risk_cfg: RiskConfig,
    exec_cfg: ExecutionConfig,
    paper_cfg: PaperConfig,
    account: AccountSnapshot,
) -> dict[str, Any]:
    """Run one Hungary PM market through the full downstream validation stack."""
    result: dict[str, Any] = {
        "market_slug": m["market_slug"],
        "question":    m.get("question"),
        "neg_risk":    m.get("neg_risk"),
        "yes_token_id": m["yes_token_id"],
        "no_token_id":  m["no_token_id"],
        "stages": {},
    }

    # G1 already resolved — token IDs are in `m`
    result["stages"]["G1_token_ids"] = {
        "status": "RESOLVED",
        "yes_token_id": m["yes_token_id"],
        "no_token_id":  m["no_token_id"],
    }

    # G2: fetch live books
    yes_book, no_book = fetch_live_books(CLOB_HOST, m["yes_token_id"], m["no_token_id"])
    if yes_book is None or no_book is None:
        result["stages"]["G2_live_books"] = {
            "status": "FAILED",
            "reason": "one or both books missing from CLOB fetch",
            "yes_book_fetched": yes_book is not None,
            "no_book_fetched":  no_book  is not None,
        }
        result["final_verdict"] = "BLOCKED_AT_G2"
        return result

    yes_spread = (yes_book.asks[0].price - yes_book.bids[0].price) if yes_book.asks and yes_book.bids else None
    no_spread  = (no_book.asks[0].price  - no_book.bids[0].price)  if no_book.asks  and no_book.bids  else None
    result["stages"]["G2_live_books"] = {
        "status":    "RESOLVED",
        "yes_best_bid":  yes_book.bids[0].price if yes_book.bids else None,
        "yes_best_ask":  yes_book.asks[0].price if yes_book.asks else None,
        "yes_ask_levels": len(yes_book.asks),
        "yes_bid_levels": len(yes_book.bids),
        "no_best_bid":   no_book.bids[0].price  if no_book.bids  else None,
        "no_best_ask":   no_book.asks[0].price  if no_book.asks  else None,
        "no_ask_levels":  len(no_book.asks),
        "no_bid_levels":  len(no_book.bids),
        "yes_spread":    round(yes_spread, 4) if yes_spread is not None else None,
        "no_spread":     round(no_spread, 4)  if no_spread  is not None else None,
    }

    # Update m with live prices for EV computation
    if yes_book.bids and yes_book.asks:
        m["best_bid"] = yes_book.bids[0].price
        m["best_ask"] = yes_book.asks[0].price

    # Compute EV with live prices
    ev = compute_ev(m)
    result["ev"] = {k: v for k, v in ev.items()}

    if ev["total_ev"] <= 0:
        result["stages"]["G3_raw_candidate"] = {
            "status": "SKIPPED",
            "reason": f"total_ev={ev['total_ev']} <= 0 after live book update",
        }
        result["final_verdict"] = "NEGATIVE_EV_AFTER_LIVE_UPDATE"
        return result

    # G3: build RawCandidate
    raw = build_raw_candidate(m, ev, m["yes_token_id"], m["no_token_id"])
    result["stages"]["G3_raw_candidate"] = {
        "status":          "RESOLVED",
        "candidate_id":    raw.candidate_id,
        "strategy_family": raw.strategy_family.value,
        "target_shares":   raw.target_shares,
        "target_notional_usd": raw.target_notional_usd,
        "expected_payout": raw.expected_payout,
        "gross_edge_cents": raw.gross_edge_cents,
        "expected_payout_encoding": (
            f"quote_size({raw.target_shares}) * pair_cost(1.0) + total_ev({ev['total_ev']}) "
            f"= {raw.expected_payout:.4f}"
        ),
    }

    books_by_token = {
        m["yes_token_id"]: yes_book,
        m["no_token_id"]:  no_book,
    }

    # Stage 1: Qualification (unchanged from production)
    print(f"  [qualify] Running ExecutionFeasibilityEvaluator.qualify()...")
    evaluator = ExecutionFeasibilityEvaluator(opp_cfg)
    qual_decision = evaluator.qualify(raw, books_by_token)

    result["stages"]["qualification"] = {
        "passed":       qual_decision.passed,
        "reason_codes": qual_decision.reason_codes,
        "metadata": {
            k: v for k, v in (qual_decision.metadata or {}).items()
            if k not in ("legs",)  # omit large nested lists from report
        },
    }

    if not qual_decision.passed:
        result["final_verdict"] = "BLOCKED_AT_QUALIFICATION"
        result["qualification_blocker"] = qual_decision.reason_codes
        return result

    executable = qual_decision.executable_candidate

    # Stage 2: Ranking (unchanged from production)
    print(f"  [rank]    Running OpportunityRanker.rank()...")
    ranker = OpportunityRanker(opp_cfg)
    ranked = ranker.rank(executable)

    result["stages"]["ranking"] = {
        "passed":           True,
        "quality_score":    ranked.quality_score,
        "ranking_score":    ranked.ranking_score,
        "capital_efficiency": ranked.capital_efficiency,
        "expected_profit_usd": ranked.expected_profit_usd,
        "sizing_hint_usd":  ranked.sizing_hint_usd,
        "sizing_hint_shares": ranked.sizing_hint_shares,
    }

    # Stage 3: Sizing (unchanged from production)
    print(f"  [size]    Running DepthCappedSizer.size()...")
    sizer = DepthCappedSizer(paper_cfg, opp_cfg)
    sizing = sizer.size(ranked, account)

    result["stages"]["sizing"] = {
        "passed":       True,
        "notional_usd": sizing.notional_usd,
        "shares":       sizing.shares,
        "reason":       sizing.reason,
        "metadata":     sizing.metadata,
    }

    # Stage 4: Risk evaluation (unchanged from production)
    print(f"  [risk]    Running RiskManager.evaluate()...")
    risk_mgr = RiskManager(risk_cfg, opp_cfg, exec_cfg)
    risk_decision = risk_mgr.evaluate(ranked, account)

    risk_checks = risk_decision.metadata.get("checks", [])
    result["stages"]["risk"] = {
        "status":      risk_decision.status.value,
        "approved":    risk_decision.status == RiskStatus.APPROVED,
        "approved_notional_usd": risk_decision.approved_notional_usd,
        "reason_codes": risk_decision.reason_codes,
        "checks": [
            {"code": c["code"], "passed": c["passed"], "message": c["message"]}
            for c in risk_checks
        ],
    }

    if risk_decision.status != RiskStatus.APPROVED:
        result["final_verdict"] = "BLOCKED_AT_RISK"
        result["paper_execution_eligible"] = False
        result["risk_blocker"] = risk_decision.reason_codes
        return result

    # Stage 5+6: Paper execution and persistence
    print(f"  [paper]   Running PaperBroker + ResearchStore persistence...")
    paper_result = run_paper_and_persist(ranked, risk_decision, sizing, books_by_token, BRIDGE_DB_URL)
    result["stages"]["paper_execution"] = {
        "status": "FILLED" if paper_result["execution"]["all_legs_filled"] else "PARTIAL_OR_UNFILLED",
        **paper_result["execution"],
    }
    result["stages"]["persistence"] = {
        "status": "PERSISTED",
        **paper_result["persistence"],
    }
    result["final_verdict"] = paper_result["final_verdict"]
    result["paper_execution_eligible"] = True
    return result


def main() -> None:
    print("=" * 70)
    print("WIDE_SCAN BRIDGE VALIDATION — Hungary PM neg-risk pair")
    print("=" * 70)

    # Maker-MM strategy requires adapted thresholds vs arb defaults.
    # G5 finding: min_edge_cents=0.03 is calibrated for per-share arb edge.
    # Maker-MM net edge per share is ~0.01-0.02; profit comes from reward pool.
    # max_order_notional_usd and max_notional_per_arb raised to accommodate
    # Hungary rewards_min_size=200 (minimum quote size for reward eligibility).
    # settings.yaml is NOT changed — these are runtime overrides in the bridge only.
    opp_cfg   = OpportunityConfig(min_edge_cents=0.005)
    risk_cfg  = RiskConfig(max_order_notional_usd=250.0)
    exec_cfg  = ExecutionConfig()
    paper_cfg = PaperConfig(max_notional_per_arb=250.0)
    account   = AccountSnapshot(cash=10000.0, ts=datetime.now(timezone.utc))

    print("\n[Step 1] Resolving G1: fetching Hungary PM markets from Gamma...")
    candidates = fetch_hungary_market_candidates(GAMMA_HOST, HUNGARY_EVENT_SLUG)

    if not candidates:
        print("ERROR: No eligible Hungary PM markets found in Gamma registry.")
        sys.exit(1)

    print(f"\n[Step 2] Running bridge validation for {len(candidates)} market(s)...\n")
    results = []
    for i, m in enumerate(candidates, 1):
        print(f"--- Market {i}/{len(candidates)}: {m['market_slug']} ---")
        r = validate_one_market(m, opp_cfg, risk_cfg, exec_cfg, paper_cfg, account)
        results.append(r)
        verdict = r.get("final_verdict", "UNKNOWN")
        print(f"    Verdict: {verdict}\n")

    # Build report
    approved = [r for r in results if r.get("paper_execution_eligible")]
    report = {
        "report_type":    "wide_scan_bridge_validation",
        "generated_ts":   datetime.now(timezone.utc).isoformat(),
        "event_slug":     HUNGARY_EVENT_SLUG,
        "config": {
            "min_edge_cents":       opp_cfg.min_edge_cents,
            "max_spread_cents":     opp_cfg.max_spread_cents,
            "min_net_profit_usd":   opp_cfg.min_net_profit_usd,
            "min_depth_multiple":   opp_cfg.min_depth_multiple,
            "max_partial_fill_risk": opp_cfg.max_partial_fill_risk,
            "max_non_atomic_risk":   opp_cfg.max_non_atomic_risk,
            "min_score":            risk_cfg.min_score,
            "max_order_notional_usd": risk_cfg.max_order_notional_usd,
        },
        "summary": {
            "markets_validated": len(results),
            "paper_execution_eligible": len(approved),
            "blocked": len(results) - len(approved),
        },
        "markets": results,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "wide_scan_bridge_latest.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    print("=" * 70)
    print(f"BRIDGE VALIDATION RESULT")
    print("=" * 70)
    for r in results:
        print(f"  {r['market_slug']}")
        print(f"    Verdict: {r.get('final_verdict')}")
        if "stages" in r:
            for stage, detail in r["stages"].items():
                status = detail.get("status") or ("PASS" if detail.get("passed") or detail.get("approved") else "FAIL")
                codes  = detail.get("reason_codes") or []
                suffix = f"  <- {codes}" if codes else ""
                print(f"    [{stage}] {status}{suffix}")
                if stage == "paper_execution" and detail.get("orders"):
                    for o in detail["orders"]:
                        print(
                            f"      order: {o.get('side')} {o.get('size_requested')}sh "
                            f"@ {o.get('limit_price')} → "
                            f"{o.get('fill_status')} filled={o.get('filled_size')} "
                            f"avg={o.get('avg_fill_price')}"
                        )
                    print(
                        f"      post_cash=${detail.get('post_cash')}  "
                        f"open_positions={detail.get('open_position_count')}"
                    )
                if stage == "persistence" and detail.get("row_counts"):
                    print(f"      DB rows: {detail['row_counts']}")
    print(f"\nReport: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
