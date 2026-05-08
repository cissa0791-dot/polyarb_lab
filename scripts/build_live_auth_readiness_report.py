from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.auth_scope_validator import (  # noqa: E402
    CLOB_HOST,
    DEFAULT_CANCEL_BUFFER_USDC,
    DEFAULT_FEE_BUFFER_USDC,
    DEFAULT_MAX_LIVE_RISK_USDC,
    REPORT_SCHEMA_VERSION,
    build_auth_scope_readiness_report,
)
from src.live.env_file import load_env_file, resolve_env_file  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_DEPOSIT_WALLET_REPORT = DEFAULT_REPORTS_DIR / "deposit_wallet_readonly_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "live_auth_readiness_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only live auth scope readiness report.")
    parser.add_argument("--env-file", default=None, help="Optional dotenv-style live env file.")
    parser.add_argument("--host", default=CLOB_HOST)
    parser.add_argument("--deposit-wallet-report", default=str(DEFAULT_DEPOSIT_WALLET_REPORT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-live-risk-usdc", type=float, default=DEFAULT_MAX_LIVE_RISK_USDC)
    parser.add_argument("--fee-buffer-usdc", type=float, default=DEFAULT_FEE_BUFFER_USDC)
    parser.add_argument("--cancel-buffer-usdc", type=float, default=DEFAULT_CANCEL_BUFFER_USDC)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    resolved_env_file = resolve_env_file(args.env_file, root=ROOT)
    env_file_status = load_env_file(resolved_env_file)
    deposit_wallet_path = Path(args.deposit_wallet_report)
    report = build_auth_scope_readiness_report(
        env_file_status=env_file_status,
        deposit_wallet_report=_load_json(deposit_wallet_path),
        host=args.host,
        max_live_risk_usdc=args.max_live_risk_usdc,
        fee_buffer_usdc=args.fee_buffer_usdc,
        cancel_buffer_usdc=args.cancel_buffer_usdc,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[deposit_wallet_path],
        source_files=[Path(__file__), ROOT / "src" / "live" / "auth_scope_validator.py"],
        root=ROOT,
    )
    return report


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == "AUTH_SCOPE_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
