from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.clob_compat import ClobClient  # noqa: E402
from src.live.market_microstructure_readiness import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    build_market_microstructure_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_HEALTH = DEFAULT_REPORTS_DIR / "live_api_health_readonly_now.json"
DEFAULT_CANDIDATE = DEFAULT_REPORTS_DIR / "maker_engine_A_p0_exit_aware_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "live_market_microstructure_latest.json"
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_GAMMA_HOST = "https://gamma-api.polymarket.com"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only market microstructure quote sanity report.")
    parser.add_argument("--health-report", default=str(DEFAULT_HEALTH))
    parser.add_argument("--candidate-report", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--market-slug")
    parser.add_argument("--token-id")
    parser.add_argument("--quote-bid", type=float)
    parser.add_argument("--quote-ask", type=float)
    parser.add_argument("--quote-size", type=float)
    parser.add_argument("--tick-size", type=float)
    parser.add_argument("--best-bid", type=float)
    parser.add_argument("--best-ask", type=float)
    parser.add_argument("--best-bid-size", type=float)
    parser.add_argument("--best-ask-size", type=float)
    parser.add_argument("--rewards-min-size", type=float)
    parser.add_argument("--rewards-max-spread-cents", type=float)
    parser.add_argument("--clob-host", default=DEFAULT_CLOB_HOST)
    parser.add_argument("--gamma-host", default=DEFAULT_GAMMA_HOST)
    parser.add_argument(
        "--token-outcome",
        default="YES",
        choices=["YES", "NO"],
        help="Outcome token to resolve from Gamma when --token-id is omitted.",
    )
    parser.add_argument("--chain-id", type=int, default=137)
    parser.add_argument("--fetch-tick-size", action="store_true")
    parser.add_argument(
        "--fetch-orderbook",
        action="store_true",
        help="Read current CLOB top-of-book for the resolved token and include best bid/ask sizes.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    token_id = args.token_id
    token_resolution = {
        "token_id_source": "EXPLICIT_ARG" if token_id else None,
        "token_id_read_error": None,
        "token_outcome": args.token_outcome,
    }
    if not token_id and args.market_slug:
        token_id, token_resolution = resolve_token_id_from_gamma(
            gamma_host=args.gamma_host,
            market_slug=args.market_slug,
            outcome=args.token_outcome,
        )

    orderbook = fetch_orderbook_top(clob_host=args.clob_host, token_id=token_id) if args.fetch_orderbook and token_id else {}
    explicit = {
        "market_slug": args.market_slug,
        "token_id": token_id,
        "quote_bid": args.quote_bid if args.quote_bid is not None else orderbook.get("best_bid"),
        "quote_ask": args.quote_ask if args.quote_ask is not None else orderbook.get("best_ask"),
        "quote_size": args.quote_size,
        "tick_size": args.tick_size,
        "best_bid": args.best_bid if args.best_bid is not None else orderbook.get("best_bid"),
        "best_ask": args.best_ask if args.best_ask is not None else orderbook.get("best_ask"),
        "best_bid_size": args.best_bid_size if args.best_bid_size is not None else orderbook.get("best_bid_size"),
        "best_ask_size": args.best_ask_size if args.best_ask_size is not None else orderbook.get("best_ask_size"),
        "rewards_min_size": args.rewards_min_size,
        "rewards_max_spread_cents": args.rewards_max_spread_cents,
    }
    tick_reader = _tick_reader(args.clob_host, args.chain_id) if args.fetch_tick_size else None
    report = build_market_microstructure_report(
        candidate_report=_load_json(args.candidate_report),
        health_report=_load_json(args.health_report),
        explicit=explicit,
        tick_size_reader=tick_reader,
    )
    report.update(token_resolution)
    if orderbook:
        report["orderbook_source"] = orderbook.get("orderbook_source")
        report["orderbook_read_error"] = orderbook.get("orderbook_read_error")
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.candidate_report), Path(args.health_report)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "market_microstructure_readiness.py"],
        root=ROOT,
    )
    return report


def fetch_orderbook_top(*, clob_host: str, token_id: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"orderbook_source": "CLOB_BOOK_HTTP", "orderbook_read_error": None}
    try:
        response = httpx.get(f"{clob_host.rstrip('/')}/book", params={"token_id": str(token_id)}, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {**meta, "orderbook_read_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {**meta, "orderbook_read_error": "CLOB_BOOK_RESPONSE_NOT_OBJECT"}
    bids = _levels(payload.get("bids"))
    asks = _levels(payload.get("asks"))
    best_bid = max(bids, key=lambda item: item["price"]) if bids else None
    best_ask = min(asks, key=lambda item: item["price"]) if asks else None
    return {
        **meta,
        "best_bid": best_bid["price"] if best_bid else None,
        "best_bid_size": best_bid["size"] if best_bid else None,
        "best_ask": best_ask["price"] if best_ask else None,
        "best_ask_size": best_ask["size"] if best_ask else None,
    }


def _levels(value: Any) -> list[dict[str, float]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        price = _float(row.get("price"))
        size = _float(row.get("size"))
        if price is None or size is None:
            continue
        out.append({"price": price, "size": size})
    return out


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_token_id_from_gamma(
    *,
    gamma_host: str,
    market_slug: str,
    outcome: str = "YES",
) -> tuple[str | None, dict[str, Any]]:
    """Resolve a market outcome token from Gamma metadata by exact slug.

    This is a read-only metadata lookup. It exists to prevent final live-probe
    reports from depending on stale health artifacts solely to learn token_id.
    """

    outcome_key = str(outcome or "YES").strip().upper()
    meta = {
        "token_id_source": "GAMMA_MARKET_SLUG",
        "token_id_read_error": None,
        "token_outcome": outcome_key,
        "gamma_market_slug": market_slug,
    }
    try:
        response = httpx.get(
            f"{gamma_host.rstrip('/')}/markets",
            params={"slug": market_slug, "limit": 1, "active": "true", "closed": "false"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        meta["token_id_source"] = None
        meta["token_id_read_error"] = f"{type(exc).__name__}: {exc}"
        return None, meta

    markets = payload if isinstance(payload, list) else []
    matched = next((item for item in markets if isinstance(item, dict) and str(item.get("slug") or "") == market_slug), None)
    if not matched:
        meta["token_id_source"] = None
        meta["token_id_read_error"] = "GAMMA_MARKET_SLUG_NOT_FOUND"
        return None, meta

    outcomes = _json_list(matched.get("outcomes"))
    token_ids = _json_list(matched.get("clobTokenIds") or matched.get("clob_token_ids"))
    if len(outcomes) != len(token_ids) or not token_ids:
        meta["token_id_source"] = None
        meta["token_id_read_error"] = "GAMMA_MARKET_TOKEN_SHAPE_INVALID"
        return None, meta

    by_outcome = {str(name).strip().upper(): str(token_id) for name, token_id in zip(outcomes, token_ids)}
    token_id = by_outcome.get(outcome_key)
    if not token_id:
        meta["token_id_source"] = None
        meta["token_id_read_error"] = f"GAMMA_MARKET_OUTCOME_{outcome_key}_MISSING"
        return None, meta
    return token_id, meta


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _tick_reader(host: str, chain_id: int):
    try:
        client = ClobClient(host, chain_id=chain_id)
    except TypeError:
        client = ClobClient(host)

    def read(token_id: str) -> Any:
        return client.get_tick_size(str(token_id))

    return read


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == "MARKET_MICROSTRUCTURE_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
