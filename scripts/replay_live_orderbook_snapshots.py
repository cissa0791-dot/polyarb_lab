from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.replay import ReplayConfig, parse_book_snapshots, replay_quote_strategy
from src.backtest.replay import _parse_ts


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
                    "simulated_fill_count": 0,
                    "net_pnl_usdc": 0.0,
                    "adverse_selection_30s_usdc": 0.0,
                    "adverse_selection_2m_usdc": 0.0,
                    "adverse_selection_5m_usdc": 0.0,
                    "recommended_quote_offset": 0.0,
                    "suitability": "NO_EVIDENCE",
                    "reason": "NEED_AT_LEAST_TWO_SNAPSHOTS",
                }
            )
            continue
        result = replay_quote_strategy(snapshots, cfg)
        payload = result.to_dict()
        adverse_30s = _adverse_selection_after_fill(payload.get("fills") or [], snapshots, 30)
        adverse_2m = _adverse_selection_after_fill(payload.get("fills") or [], snapshots, 120)
        adverse_5m = _adverse_selection_after_fill(payload.get("fills") or [], snapshots, 300)
        payload["simulated_fill_count"] = int(_float(payload.get("fill_count")))
        payload["adverse_selection_30s_usdc"] = round(adverse_30s, 6)
        payload["adverse_selection_2m_usdc"] = round(adverse_2m, 6)
        payload["adverse_selection_5m_usdc"] = round(adverse_5m, 6)
        payload["adverse_selection_after_fill_usdc"] = round(adverse_5m, 6)
        payload["recommended_quote_offset"] = _recommended_quote_offset(payload)
        payload["average_spread"] = round(_average_spread(snapshots), 6)
        payload["best_quote_offsets"] = {
            "quote_bid_offset": cfg.quote_bid_offset,
            "quote_ask_offset": cfg.quote_ask_offset,
        }
        payload["suitability"] = _suitability(payload, min_spread=cfg.min_spread)
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


def _adverse_selection_after_fill(fills: list[dict[str, Any]], snapshots: list[Any], window_seconds: int) -> float:
    adverse = 0.0
    for fill in fills:
        side = str(fill.get("side") or "").upper()
        price = _float(fill.get("price"))
        size = _float(fill.get("size"))
        midpoint = _future_midpoint(fill.get("ts"), snapshots, window_seconds)
        if side == "BUY":
            adverse += max(0.0, price - midpoint) * size
        elif side == "SELL":
            adverse += max(0.0, midpoint - price) * size
    return adverse


def _future_midpoint(fill_ts: Any, snapshots: list[Any], window_seconds: int) -> float:
    if not snapshots:
        return 0.0
    target = _parse_ts(str(fill_ts or "")) + timedelta(seconds=window_seconds)
    for snapshot in snapshots:
        if _parse_ts(snapshot.ts) >= target:
            return float(snapshot.midpoint)
    return float(snapshots[-1].midpoint)


def _average_spread(snapshots: list[Any]) -> float:
    if not snapshots:
        return 0.0
    return sum(max(0.0, snapshot.best_ask - snapshot.best_bid) for snapshot in snapshots) / len(snapshots)


def _recommended_quote_offset(row: dict[str, Any]) -> float:
    fill_count = int(_float(row.get("fill_count")))
    adverse = _float(row.get("adverse_selection_5m_usdc"))
    if fill_count <= 0:
        return 0.0
    if adverse > 0.0:
        return 0.001
    return 0.0


def _suitability(row: dict[str, Any], *, min_spread: float) -> str:
    fill_count = int(_float(row.get("fill_count")))
    net = _float(row.get("net_pnl_usdc"))
    adverse = _float(row.get("adverse_selection_5m_usdc"))
    avg_spread = _float(row.get("average_spread"))
    if avg_spread <= max(0.0, min_spread):
        return "AVOID_TOO_THIN"
    if fill_count <= 0:
        if int(_float(row.get("snapshot_count"))) <= 2:
            return "NO_EVIDENCE"
        return "AVOID_NO_FILL"
    if adverse > max(abs(net), 0.01):
        return "AVOID_ADVERSE_SELECTION"
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
