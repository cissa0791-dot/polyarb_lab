from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.reporting.analytics import resolve_sqlite_path
from src.reporting.calibration import ThresholdCalibrationService
from src.reporting.exporter import export_calibration_report
from src.reporting.models import CalibrationParameterSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export offline threshold calibration results from persisted research artifacts.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--db-path", default=None, help="Optional explicit path to SQLite db. Defaults to config sqlite_url.")
    parser.add_argument("--experiment-label", default=None, help="Optional experiment label filter.")
    parser.add_argument("--experiment-id", default=None, help="Optional experiment id filter.")
    parser.add_argument("--parameter-sets-json", default=None, help="Optional JSON file containing a list of parameter sets.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory for calibration exports.")
    return parser.parse_args()


def _load_parameter_sets(path: str | None, service: ThresholdCalibrationService) -> list[CalibrationParameterSet]:
    if path is None:
        return service.default_parameter_sets()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CalibrationParameterSet.model_validate(item) for item in payload]


def main() -> None:
    args = parse_args()
    if args.db_path:
        service = ThresholdCalibrationService(db_path=Path(args.db_path), settings_path=args.settings)
    else:
        sqlite_url = load_runtime_config(args.settings).storage.sqlite_url
        service = ThresholdCalibrationService(db_path=resolve_sqlite_path(sqlite_url), settings_path=args.settings)
    try:
        parameter_sets = _load_parameter_sets(args.parameter_sets_json, service)
        report = service.build_report(
            parameter_sets=parameter_sets,
            experiment_label=args.experiment_label,
            experiment_id=args.experiment_id,
        )
        written = export_calibration_report(report, out_dir=args.out_dir)
    finally:
        service.close()

    print(
        json.dumps(
            {
                "experiment_label": args.experiment_label,
                "experiment_id": args.experiment_id,
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
