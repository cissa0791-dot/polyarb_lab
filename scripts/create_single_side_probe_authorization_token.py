from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    parser.add_argument("--market-slug", required=True)
    parser.add_argument("--max-live-risk-usdc", type=float, required=True)
    parser.add_argument("--quote-price", type=float, required=True)
    parser.add_argument("--quote-size", type=float, required=True)
    parser.add_argument("--ttl-seconds", type=int, default=300)
    parser.add_argument("--out", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--confirm-create-token", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.confirm_create_token:
        print("Refusing to create token without --confirm-create-token", file=sys.stderr)
        return 2
    token = create_authorization_token(
        market_slug=args.market_slug,
        max_live_risk_usdc=args.max_live_risk_usdc,
        quote_price=args.quote_price,
        quote_size=args.quote_size,
        ttl_seconds=args.ttl_seconds,
    )
    path = write_authorization_token(token, args.out)
    result = {
        "token_file": str(path),
        "status": token["status"],
        "expires_at_utc": token["expires_at_utc"],
        "token_hash_prefix": token["token_hash"][:12],
        "can_submit_order": False,
        "live_order_sent": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
