from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.batch import BatchResearchRunner
from src.runtime.campaigns import (
    build_campaign_manifest_from_preset,
    list_campaign_presets,
    load_campaign_manifest,
    save_campaign_manifest,
)
from src.runtime.runner import ResearchRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a research campaign manifest using the existing paper-safe runner.")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--manifest", help="Path to a JSON or YAML campaign manifest.")
    source_group.add_argument("--preset", help="Built-in campaign preset name.")
    parser.add_argument("--list-presets", action="store_true", help="List built-in campaign presets and exit.")
    parser.add_argument("--campaign-label", default=None, help="Optional campaign label override when using a preset.")
    parser.add_argument("--cycles", type=int, default=None, help="Optional cycles override when using a preset.")
    parser.add_argument("--market-limit", type=int, default=None, help="Optional market limit override when using a preset.")
    parser.add_argument("--save-manifest", default=None, help="Optional path to save the resolved manifest before running.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--constraints", default="config/constraints.yaml", help="Path to constraints file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        payload = [
            {
                "preset_name": preset.preset_name,
                "description": preset.description,
                "target_strategy_families": preset.target_strategy_families,
                "target_parameter_sets": preset.target_parameter_sets,
                "cycles": preset.cycles,
                "market_limit": preset.market_limit,
            }
            for preset in list_campaign_presets()
        ]
        print(json.dumps(payload, indent=2))
        return

    if args.manifest:
        manifest = load_campaign_manifest(args.manifest)
    elif args.preset:
        manifest = build_campaign_manifest_from_preset(
            args.preset,
            campaign_label=args.campaign_label,
            cycles=args.cycles,
            market_limit=args.market_limit,
        )
    else:
        raise SystemExit("One of --manifest or --preset is required unless --list-presets is used.")

    saved_manifest_path = None
    if args.save_manifest:
        saved_manifest_path = save_campaign_manifest(manifest, args.save_manifest)
    batch = BatchResearchRunner(
        runner_factory=lambda: ResearchRunner(settings_path=args.settings, constraints_path=args.constraints)
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
