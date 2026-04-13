from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.batch import BatchResearchRunner
from src.runtime.campaigns import load_campaign_manifest, save_campaign_manifest
from src.runtime.political_binary_runner import PoliticalBinaryPaperRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the narrow political binary constraint paper model.")
    parser.add_argument("--manifest", required=True, help="Path to a YAML or JSON campaign manifest.")
    parser.add_argument("--save-manifest", default=None, help="Optional path to save the resolved manifest.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument(
        "--constraints",
        default="config/political_constraint_rules.yaml",
        help="Path to the political constraint rules file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_campaign_manifest(args.manifest)
    saved_manifest_path = None
    if args.save_manifest:
        saved_manifest_path = save_campaign_manifest(manifest, args.save_manifest)
    batch = BatchResearchRunner(
        runner_factory=lambda: PoliticalBinaryPaperRunner(
            settings_path=args.settings,
            constraints_path=args.constraints,
        )
    )
    summary = batch.run_campaign(manifest)
    print(
        json.dumps(
            {
                "manifest_path": str(saved_manifest_path) if saved_manifest_path is not None else None,
                "summary": summary.model_dump(mode="json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

