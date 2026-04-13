"""
Debug: why does EDGE_BELOW_THRESHOLD fire for all negRisk candidates?
Prints raw qualification numbers for the first 5 failing candidates.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.gamma import fetch_events
from src.ingest.clob import ReadOnlyClob
from src.scanner.neg_risk import build_eligible_neg_risk_event_groups
from src.strategies.opportunity_strategies import NegRiskRebalancingStrategy
from src.opportunity.qualification import VWAPCalculator, DepthAnalyzer

SETTINGS = ROOT / "config" / "settings.yaml"


def main():
    cfg = load_runtime_config(str(SETTINGS))
    gamma_host = cfg.market_data.gamma_host
    clob_host = cfg.market_data.clob_host
    max_notional = cfg.paper.max_notional_per_arb

    print(f"Config: min_edge_cents={cfg.opportunity.min_edge_cents}, "
          f"fee_buffer={cfg.opportunity.fee_buffer_cents}, "
          f"slip_buffer={cfg.opportunity.slippage_buffer_cents}, "
          f"min_net_profit={cfg.opportunity.min_net_profit_usd}, "
          f"max_notional={max_notional}", flush=True)

    events = fetch_events(gamma_host, limit=2000)
    event_groups = build_eligible_neg_risk_event_groups(events)

    token_ids = list({
        m["yes_token_id"]
        for g in event_groups
        for m in g.get("markets", [])
        if m.get("yes_token_id")
    })
    clob = ReadOnlyClob(clob_host)
    books: dict[str, object] = {}
    batch = 20
    for i in range(0, len(token_ids), batch):
        ids = token_ids[i: i + batch]
        try:
            for book in clob.get_books(ids):
                books[str(book.token_id)] = book
        except Exception:
            for tid in ids:
                try:
                    books[tid] = clob.get_book(tid)
                except Exception:
                    pass

    strategy = NegRiskRebalancingStrategy()
    calc = VWAPCalculator()

    shown = 0
    for g in event_groups:
        if shown >= 5:
            break
        raw, audit = strategy.detect_with_audit(g, books, max_notional)
        if raw is None:
            continue

        print(f"\n=== {g.get('event_slug')} ({len(raw.legs)} legs) ===")
        print(f"  gross_edge_cents={raw.gross_edge_cents}, target_shares={raw.target_shares}, "
              f"expected_payout={raw.expected_payout}, est_fill_cost={raw.est_fill_cost_usd}")

        pair_vwap_ask = 0.0
        pair_vwap_bid = 0.0
        for leg in raw.legs:
            book = books.get(leg.token_id)
            if book is None:
                print(f"  leg {leg.token_id[:12]}: NO BOOK")
                continue
            asks = getattr(book, "asks", [])
            bids = getattr(book, "bids", [])
            best_ask = float(asks[0].price) if asks else None
            best_bid = float(bids[0].price) if bids else None
            spread = None
            if best_ask is not None and best_bid is not None:
                spread = round(best_ask - best_bid, 6)
            fill_ask = calc.estimate_buy(asks, leg.required_shares, spread)
            fill_bid = calc.estimate_sell(bids, leg.required_shares, spread)
            print(f"  leg {leg.token_id[:12]}: best_bid={best_bid}, best_ask={best_ask}, "
                  f"spread={spread}, req_shares={leg.required_shares}, "
                  f"vwap_ask={fill_ask.vwap_price}, vwap_bid={fill_bid.vwap_price}")
            pair_vwap_ask += float(fill_ask.vwap_price or 0.0)
            pair_vwap_bid += float(fill_bid.vwap_price or 0.0)

        payout_per_share = raw.expected_payout / raw.target_shares if raw.target_shares > 1e-9 else 0.0
        gross_edge_ask = max(0.0, payout_per_share - pair_vwap_ask)
        print(f"  pair_vwap_ask={round(pair_vwap_ask,6)}, pair_vwap_bid={round(pair_vwap_bid,6)}")
        print(f"  payout_per_share={payout_per_share}, gross_edge_at_ask={round(gross_edge_ask,6)}")
        print(f"  net_edge_at_ask={round(gross_edge_ask - cfg.opportunity.fee_buffer_cents - cfg.opportunity.slippage_buffer_cents, 6)}")
        print(f"  min_edge_cents={cfg.opportunity.min_edge_cents}")
        print(f"  EDGE_PASSES: {gross_edge_ask - cfg.opportunity.fee_buffer_cents - cfg.opportunity.slippage_buffer_cents >= cfg.opportunity.min_edge_cents}")
        shown += 1

if __name__ == "__main__":
    main()
