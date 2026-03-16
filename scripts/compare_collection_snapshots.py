from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.campaigns import compare_collection_evidence_snapshots
from src.reporting.exporter import export_collection_comparison_report
from src.reporting.models import CollectionEvidenceSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two collection-evidence snapshots.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline snapshot JSON.")
    parser.add_argument("--current", required=True, help="Path to the post-campaign snapshot JSON.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory for comparison exports.")
    return parser.parse_args()


def _load_snapshot(path: str | Path) -> CollectionEvidenceSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CollectionEvidenceSnapshot.model_validate(payload)


def main() -> None:
    args = parse_args()
    baseline = _load_snapshot(args.baseline)
    current = _load_snapshot(args.current)
    comparison = compare_collection_evidence_snapshots(baseline, current)
    written = export_collection_comparison_report(comparison, out_dir=args.out_dir)
    print(
        json.dumps(
            {
                "baseline": str(Path(args.baseline)),
                "current": str(Path(args.current)),
                "out_dir": str(Path(args.out_dir)),
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
