from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.batch import BatchResearchRunner
from src.runtime.runner import ResearchRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local research batch using the existing paper-safe runner.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--constraints", default="config/constraints.yaml", help="Path to constraints file.")
    parser.add_argument("--cycles", type=int, default=3, help="Number of run_once cycles to execute.")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Optional pause between cycles.")
    parser.add_argument("--experiment-label", default=None, help="Optional experiment label.")
    parser.add_argument("--parameter-set-label", default="runtime_default", help="Label recorded with this batch.")
    parser.add_argument("--market-limit", type=int, default=None, help="Optional market limit override for the batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = BatchResearchRunner(
        runner_factory=lambda: ResearchRunner(settings_path=args.settings, constraints_path=args.constraints)
    )
    summary = batch.run_batch(
        cycles=args.cycles,
        sleep_sec=args.sleep_sec,
        experiment_label=args.experiment_label,
        parameter_set_label=args.parameter_set_label,
        market_limit=args.market_limit,
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
