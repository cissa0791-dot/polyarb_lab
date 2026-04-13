"""Compact calibration replay entry point.

Runs ThresholdCalibrationService against a persisted SQLite research DB and
prints a per-parameter-set summary to stdout.  Optionally writes the full
CalibrationReport as JSON to --out.

Usage:
    py -3 scripts/run_calibration.py [--db PATH] [--settings PATH]
        [--experiment-label LABEL] [--experiment-id ID] [--out PATH]

DB path resolution (first match wins):
    1. --db argument
    2. config.storage.sqlite_url from --settings file

Exits non-zero on: missing DB file, unreadable settings, build failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run threshold calibration replay against a persisted research DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=None, metavar="PATH",
                        help="Path to SQLite DB. Defaults to config sqlite_url.")
    parser.add_argument("--settings", default="config/settings.yaml", metavar="PATH",
                        help="Runtime settings YAML. Default: config/settings.yaml")
    parser.add_argument("--experiment-label", default=None, metavar="LABEL",
                        help="Filter corpus to this experiment label.")
    parser.add_argument("--experiment-id", default=None, metavar="ID",
                        help="Filter corpus to this experiment ID.")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Optional path to write full CalibrationReport JSON.")
    return parser.parse_args()


def _resolve_db(args: argparse.Namespace) -> Path:
    if args.db:
        return Path(args.db)
    try:
        from src.config_runtime.loader import load_runtime_config
        from src.reporting.analytics import resolve_sqlite_path
        config = load_runtime_config(args.settings)
        return resolve_sqlite_path(config.storage.sqlite_url)
    except Exception as exc:
        print(f"ERROR: could not load settings from {args.settings!r}: {exc}", file=sys.stderr)
        sys.exit(1)


def _print_summary(report) -> None:
    print(f"db:               {report.db_path}")
    print(f"generated_ts:     {report.generated_ts.isoformat()}")
    print(f"record_count:     {report.record_count}")
    if report.experiment_label:
        print(f"experiment_label: {report.experiment_label}")
    if report.experiment_id:
        print(f"experiment_id:    {report.experiment_id}")
    print()

    for result in report.parameter_results:
        qual_pct = (
            round(result.qualified_count / result.total_records * 100, 1)
            if result.total_records > 0 else 0.0
        )
        print(f"  [{result.parameter_set_label}]")
        print(f"    total={result.total_records}  "
              f"qualified={result.qualified_count} ({qual_pct}%)  "
              f"rejected={result.rejected_count}  "
              f"near_miss={result.near_miss_count}")

        if result.qualified_by_family:
            families = "  ".join(
                f"{fam}:{count}" for fam, count in sorted(result.qualified_by_family.items())
            )
            print(f"    qualified_by_family: {families}")

        top_rejections = sorted(
            result.rejection_reason_counts.items(), key=lambda kv: -kv[1]
        )[:3]
        if top_rejections:
            reasons = "  ".join(f"{r}:{n}" for r, n in top_rejections)
            print(f"    top_rejections: {reasons}")
        print()


def main() -> None:
    args = parse_args()

    db_path = _resolve_db(args)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from src.reporting.calibration import ThresholdCalibrationService
    except ImportError as exc:
        print(f"ERROR: import failed: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        service = ThresholdCalibrationService(
            db_path=db_path,
            settings_path=args.settings,
        )
    except Exception as exc:
        print(f"ERROR: failed to initialise ThresholdCalibrationService: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        report = service.build_report(
            experiment_label=args.experiment_label,
            experiment_id=args.experiment_id,
        )
    except Exception as exc:
        print(f"ERROR: build_report failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        service.close()

    _print_summary(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"full report written to: {out_path}")


if __name__ == "__main__":
    main()
