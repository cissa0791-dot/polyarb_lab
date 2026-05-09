from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.one_time_auth_token import (  # noqa: E402
    DEFAULT_TOKEN_PATH,
    create_authorization_token,
    write_authorization_token,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a short-lived single-side BID probe token.")
    parser.add_argument("--market-slug")
    parser.add_argument("--max-live-risk-usdc", type=float)
    parser.add_argument("--quote-price", type=float)
    parser.add_argument("--quote-size", type=float)
    parser.add_argument("--hold-seconds", type=float)
    parser.add_argument(
        "--planner-report",
        help="Optional live_probe_planner report. When supplied, token parameters must match its recommended_plan.",
    )
    parser.add_argument(
        "--token-issuance-review-report",
        help=(
            "Optional B stability token issuance review report. When supplied, "
            "token parameters must match its token_binding_fields and the review must be READY."
        ),
    )
    parser.add_argument("--ttl-seconds", type=int)
    parser.add_argument("--out", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--confirm-create-token", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.confirm_create_token:
        print("Refusing to create token without --confirm-create-token", file=sys.stderr)
        return 2
    if args.planner_report and args.token_issuance_review_report:
        print(
            json.dumps(
                {"status": "TOKEN_CREATE_BLOCKED", "blockers": ["MULTIPLE_TOKEN_BINDING_SOURCES_PROVIDED"]},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    planner = _load_planner(args.planner_report) if args.planner_report else {}
    review = _load_json(args.token_issuance_review_report, "TOKEN_ISSUANCE_REVIEW") if args.token_issuance_review_report else {}
    params, blockers = _resolve_params(args, planner, review)
    if blockers:
        print(json.dumps({"status": "TOKEN_CREATE_BLOCKED", "blockers": blockers}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    token = create_authorization_token(
        market_slug=params["market_slug"],
        max_live_risk_usdc=params["max_live_risk_usdc"],
        quote_price=params["quote_price"],
        quote_size=params["quote_size"],
        hold_seconds=params.get("hold_seconds"),
        planner_hash=params.get("planner_hash"),
        planner_snapshot_ts=params.get("planner_snapshot_ts"),
        planner_expires_at=params.get("planner_expires_at"),
        ttl_seconds=params.get("ttl_seconds"),
    )
    path = write_authorization_token(token, args.out)
    result = {
        "token_file": str(path),
        "status": token["status"],
        "expires_at_utc": token["expires_at_utc"],
        "planner_hash": token.get("planner_hash"),
        "hold_seconds": token.get("hold_seconds"),
        "token_hash_prefix": token["token_hash"][:12],
        "can_submit_order": False,
        "live_order_sent": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


def _load_planner(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {"_load_error": "PLANNER_REPORT_MISSING"}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"_load_error": "PLANNER_REPORT_INVALID_JSON"}
    return payload if isinstance(payload, dict) else {"_load_error": "PLANNER_REPORT_NOT_OBJECT"}


def _load_json(path: str | None, label: str) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {"_load_error": f"{label}_MISSING"}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"_load_error": f"{label}_INVALID_JSON"}
    return payload if isinstance(payload, dict) else {"_load_error": f"{label}_NOT_OBJECT"}


def _resolve_params(
    args: argparse.Namespace,
    planner: dict[str, Any],
    review: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    review = review or {}
    plan = planner.get("recommended_plan") if isinstance(planner.get("recommended_plan"), dict) else {}
    review_binding = review.get("token_binding_fields") if isinstance(review.get("token_binding_fields"), dict) else {}
    if planner:
        if planner.get("_load_error"):
            blockers.append(str(planner["_load_error"]))
        if planner.get("status") != "LIVE_PROBE_PLAN_READY":
            blockers.append("PLANNER_REPORT_NOT_READY")
        if not plan:
            blockers.append("PLANNER_RECOMMENDED_PLAN_MISSING")
        if not planner.get("planner_hash"):
            blockers.append("PLANNER_HASH_MISSING")
        if _planner_expired(planner):
            blockers.append("PLANNER_REPORT_EXPIRED")
    if review:
        if review.get("_load_error"):
            blockers.append(str(review["_load_error"]))
        if review.get("status") != "B_STABILITY_TOKEN_ISSUANCE_REVIEW_READY":
            blockers.append("TOKEN_ISSUANCE_REVIEW_NOT_READY")
        if review.get("token_issuance_review_ready") is not True:
            blockers.append("TOKEN_ISSUANCE_REVIEW_READY_FLAG_FALSE")
        if review.get("token_created") is not False:
            blockers.append("TOKEN_ISSUANCE_REVIEW_TOKEN_CREATED_UNEXPECTED")
        if review.get("execution_authorized") is not False:
            blockers.append("TOKEN_ISSUANCE_REVIEW_EXECUTION_AUTHORIZED_UNEXPECTED")
        if review.get("can_submit_order") is not False:
            blockers.append("TOKEN_ISSUANCE_REVIEW_CAN_SUBMIT_ORDER_UNEXPECTED")
        if review.get("live_order_sent") is not False:
            blockers.append("TOKEN_ISSUANCE_REVIEW_LIVE_ORDER_SENT_UNEXPECTED")
        if review.get("blockers"):
            blockers.append("TOKEN_ISSUANCE_REVIEW_HAS_BLOCKERS")
        if not review_binding:
            blockers.append("TOKEN_ISSUANCE_REVIEW_BINDING_MISSING")

    params = {
        "market_slug": review_binding.get("market_slug") if review_binding else (plan.get("market_slug") if plan else args.market_slug),
        "max_live_risk_usdc": (
            review_binding.get("max_live_risk_usdc")
            if review_binding
            else (planner.get("max_live_risk_usdc") if planner else args.max_live_risk_usdc)
        ),
        "quote_price": review_binding.get("quote_price") if review_binding else (plan.get("quote_price") if plan else args.quote_price),
        "quote_size": review_binding.get("quote_size") if review_binding else (plan.get("quote_size") if plan else args.quote_size),
        "hold_seconds": review_binding.get("hold_seconds") if review_binding else (planner.get("hold_seconds") if planner else args.hold_seconds),
        "ttl_seconds": (
            args.ttl_seconds
            or review_binding.get("token_ttl_seconds")
            or planner.get("token_ttl_seconds")
            or 300
        ),
        "planner_hash": review_binding.get("planner_hash") if review_binding else (planner.get("planner_hash") if planner else None),
        "planner_snapshot_ts": (
            (review.get("planner_evidence") or {}).get("planner_snapshot_ts")
            if review_binding
            else (planner.get("planner_snapshot_ts") if planner else None)
        ),
        "planner_expires_at": (
            (review.get("planner_evidence") or {}).get("planner_expires_at")
            if review_binding
            else (planner.get("planner_expires_at") if planner else None)
        ),
    }
    required = ["market_slug", "max_live_risk_usdc", "quote_price", "quote_size"]
    for key in required:
        if params.get(key) in {None, ""}:
            blockers.append(f"{key.upper()}_MISSING")

    explicit_checks = {
        "market_slug": args.market_slug,
        "max_live_risk_usdc": args.max_live_risk_usdc,
        "quote_price": args.quote_price,
        "quote_size": args.quote_size,
        "hold_seconds": args.hold_seconds,
    }
    for key, explicit in explicit_checks.items():
        if not (planner or review_binding) or explicit in {None, ""}:
            continue
        planned = params.get(key)
        if isinstance(explicit, (int, float)) or isinstance(planned, (int, float)):
            if not _same_float(explicit, planned):
                blockers.append(f"EXPLICIT_{key.upper()}_MISMATCHES_PLANNER")
        elif explicit != planned:
            blockers.append(f"EXPLICIT_{key.upper()}_MISMATCHES_PLANNER")
    if review_binding and args.ttl_seconds is not None and not _same_float(args.ttl_seconds, review_binding.get("token_ttl_seconds")):
        blockers.append("EXPLICIT_TTL_SECONDS_MISMATCHES_TOKEN_ISSUANCE_REVIEW")
    return params, _unique(blockers)


def _planner_expired(planner: dict[str, Any]) -> bool:
    raw = planner.get("planner_expires_at")
    if raw in {None, ""}:
        return True
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc) <= datetime.now(timezone.utc)


def _same_float(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
