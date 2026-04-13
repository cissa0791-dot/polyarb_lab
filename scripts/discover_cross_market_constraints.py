from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.core.fees import total_buffer_cents
from src.ingest.gamma import fetch_markets
from src.runtime.constraint_mining import (
    build_constraints_document,
    discover_cross_market_constraints,
    limit_constraints,
    rank_constraints_by_current_execution,
    rank_constraints_by_current_relation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover simple cross-market leq constraints from active Polymarket markets.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--market-limit", type=int, default=1000, help="How many active markets to scan before mining relations.")
    parser.add_argument("--out-path", required=True, help="Where to write the discovered constraints YAML.")
    parser.add_argument("--disable-single-market", action="store_true", help="Write single_market.enabled=false in the generated constraints file.")
    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument("--rank-current-relation", action="store_true", help="Score discovered constraints by current relation gap using live YES asks.")
    rank_group.add_argument("--rank-current-execution", action="store_true", help="Score discovered constraints by current execution-side best-ask edge using the same legs the gross cross-market family would buy.")
    parser.add_argument("--max-constraints", type=int, default=None, help="Keep only the top-N discovered constraints after optional ranking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    markets = fetch_markets(config.market_data.gamma_host, args.market_limit)
    discovered = discover_cross_market_constraints(markets)
    selected = discovered
    if args.rank_current_relation:
        selected = rank_constraints_by_current_relation(
            discovered,
            markets,
            clob_host=config.market_data.clob_host,
            total_buffer_cents=total_buffer_cents(config.opportunity.model_dump()),
        )
    elif args.rank_current_execution:
        selected = rank_constraints_by_current_execution(
            discovered,
            markets,
            clob_host=config.market_data.clob_host,
            total_buffer_cents=total_buffer_cents(config.opportunity.model_dump()),
        )
    selected = limit_constraints(selected, args.max_constraints)
    document = build_constraints_document(
        markets,
        single_market_enabled=not args.disable_single_market,
        constraints=selected,
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    counts_by_rule: dict[str, int] = {}
    for item in selected:
        counts_by_rule[item.discovery_rule] = counts_by_rule.get(item.discovery_rule, 0) + 1

    print(
        json.dumps(
            {
                "market_limit": args.market_limit,
                "markets_fetched": len(markets),
                "constraints_discovered": len(discovered),
                "constraints_selected": len(selected),
                "rank_current_relation": args.rank_current_relation,
                "rank_current_execution": args.rank_current_execution,
                "max_constraints": args.max_constraints,
                "counts_by_rule": counts_by_rule,
                "top_constraints_preview": [
                    {
                        "name": item.name,
                        "discovery_rule": item.discovery_rule,
                        "current_relation_rank": item.current_relation_rank,
                        "current_execution_rank": item.current_execution_rank,
                        "relation_gap": item.relation_gap,
                        "edge_after_buffer": item.edge_after_buffer,
                        "execution_pair_best_ask_cost": item.execution_pair_best_ask_cost,
                        "execution_best_ask_edge_cents": item.execution_best_ask_edge_cents,
                    }
                    for item in selected[:10]
                ],
                "out_path": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
