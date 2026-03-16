from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.reporting.analytics import OfflineAnalyticsService, resolve_sqlite_path
from src.reporting.campaigns import build_collection_evidence_snapshot
from src.reporting.exporter import export_collection_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a collection-evidence snapshot for before/after campaign comparison.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--db-path", default=None, help="Optional explicit path to SQLite db. Defaults to config sqlite_url.")
    parser.add_argument("--snapshot-label", required=True, help="Human-readable label for the exported snapshot.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory for snapshot exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path) if args.db_path else resolve_sqlite_path(load_runtime_config(args.settings).storage.sqlite_url)
    service = OfflineAnalyticsService(db_path=db_path, settings_path=args.settings)
    try:
        report = service.build_report()
    finally:
        service.close()

    snapshot = build_collection_evidence_snapshot(report, snapshot_label=args.snapshot_label)
    written = export_collection_snapshot(snapshot, out_dir=args.out_dir)
    print(
        json.dumps(
            {
                "db_path": str(db_path),
                "snapshot_label": args.snapshot_label,
                "out_dir": str(Path(args.out_dir)),
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
