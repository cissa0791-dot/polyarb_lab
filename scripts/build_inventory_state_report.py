from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.env_file import load_env_file, resolve_env_file  # noqa: E402
from src.live.inventory_state_readiness import (  # noqa: E402
    CLOB_HOST,
    POSITIONS_API_URL,
    READY_STATUS,
    REPORT_SCHEMA_VERSION,
    collect_live_inventory_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_OUT = ROOT / "data" / "reports" / "inventory_state_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only inventory state report.")
    parser.add_argument("--env-file", default=None, help="Optional dotenv-style live env file.")
    parser.add_argument("--host", default=CLOB_HOST)
    parser.add_argument("--positions-api-url", default=POSITIONS_API_URL)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict:
    resolved_env_file = resolve_env_file(args.env_file, root=ROOT)
    env_file_status = load_env_file(resolved_env_file)
    report = collect_live_inventory_report(
        host=args.host,
        positions_api_url=args.positions_api_url,
        timeout_sec=args.timeout_sec,
    )
    report["env_file"] = env_file_status
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        source_files=[Path(__file__), ROOT / "src" / "live" / "inventory_state_readiness.py"],
        root=ROOT,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
