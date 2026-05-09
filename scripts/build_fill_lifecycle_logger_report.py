from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.fill_lifecycle_logger import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    read_fill_lifecycle_events,
    summarize_fill_lifecycle_events,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_LOG = ROOT / "data" / "reports" / "fill_lifecycle.jsonl"
DEFAULT_OUT = ROOT / "data" / "reports" / "fill_lifecycle_summary_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal fill lifecycle summary report.")
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    events, read_stats = read_fill_lifecycle_events(args.log)
    report = summarize_fill_lifecycle_events(events)
    report["read_stats"] = read_stats
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.log)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "fill_lifecycle_logger.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
