from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.replay import ReplayConfig, parse_book_snapshots, replay_quote_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay reward-MM quotes from live orderbook snapshots.")
    parser.add_argument(
        "--snapshots",
        type=str,
        default=str(ROOT / "data" / "reports" / "live_orderbook_snapshots.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "reports" / "replay_report_latest.json"),
    )
    parser.add_argument("--order-size", type=float, default=5.0)
    parser.add_argument("--latency-ms", type=int, default=500)
    parser.add_argument("--quote-bid-offset", type=float, default=0.0)
    parser.add_argument("--quote-ask-offset", type=float, default=0.0)
    parser.add_argument("--max-inventory-shares", type=float, default=50.0)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_replay_report(rows: Iterable[dict[str, Any]], config: ReplayConfig | None = None) -> dict[str, Any]:
    cfg = config or ReplayConfig()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "orderbook_snapshot":
            continue
        slug = str(row.get("market_slug") or "")
        if not slug:
            continue
        if _float_or_none(row.get("best_bid")) is None or _float_or_none(row.get("best_ask")) is None:
            continue
        groups[slug].append(row)

    market_reports: list[dict[str, Any]] = []
    for slug, group_rows in groups.items():
        snapshots = parse_book_snapshots(group_rows)
        if len(snapshots) < 2:
            market_reports.append(
                {
                    "market_slug": slug,
                    "snapshot_count": len(snapshots),
                    "suitability": "INSUFFICIENT_DATA",
                    "reason": "NEED_AT_LEAST_TWO_SNAPSHOTS",
                }
            )
            continue
        result = replay_quote_strategy(snapshots, cfg)
        payload = result.to_dict()
        adverse = _adverse_selection_after_fill(payload.get("fills") or [], snapshots[-1].midpoint)
        payload["adverse_selection_after_fill_usdc"] = round(adverse, 6)
        payload["best_quote_offsets"] = {
            "quote_bid_offset": cfg.quote_bid_offset,
            "quote_ask_offset": cfg.quote_ask_offset,
        }
        payload["suitability"] = _suitability(payload)
        market_reports.append(payload)

    replayed = [row for row in market_reports if row.get("snapshot_count", 0) >= 2]
    profitable = [row for row in replayed if _float(row.get("net_pnl_usdc")) > 0.0]
    return {
        "report_type": "live_orderbook_snapshot_replay",
        "market_count": len(groups),
        "replayed_market_count": len(replayed),
        "profitable_market_count": len(profitable),
        "total_net_pnl_usdc": round(sum(_float(row.get("net_pnl_usdc")) for row in replayed), 6),
        "total_adverse_selection_after_fill_usdc": round(
            sum(_float(row.get("adverse_selection_after_fill_usdc")) for row in replayed),
            6,
        ),
        "markets": sorted(market_reports, key=lambda row: _float(row.get("net_pnl_usdc")), reverse=True),
    }


def _adverse_selection_after_fill(fills: list[dict[str, Any]], final_midpoint: float) -> float:
    adverse = 0.0
    for fill in fills:
        side = str(fill.get("side") or "").upper()
        price = _float(fill.get("price"))
        size = _float(fill.get("size"))
        if side == "BUY":
            adverse += max(0.0, price - final_midpoint) * size
        elif side == "SELL":
            adverse += max(0.0, final_midpoint - price) * size
    return adverse


def _suitability(row: dict[str, Any]) -> str:
    fill_count = int(_float(row.get("fill_count")))
    net = _float(row.get("net_pnl_usdc"))
    adverse = _float(row.get("adverse_selection_after_fill_usdc"))
    if fill_count <= 0:
        return "INSUFFICIENT_DATA"
    if net > 0.0 and adverse <= max(net, 0.0):
        return "REWARD_MM_CANDIDATE"
    return "REWARD_MM_WATCH"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    cfg = ReplayConfig(
        order_size=args.order_size,
        quote_bid_offset=args.quote_bid_offset,
        quote_ask_offset=args.quote_ask_offset,
        latency_ms=args.latency_ms,
        max_inventory_shares=args.max_inventory_shares,
    )
    report = build_replay_report(load_jsonl(args.snapshots), cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("market_count", "replayed_market_count", "profitable_market_count", "total_net_pnl_usdc")}, indent=2))


if __name__ == "__main__":
    main()
