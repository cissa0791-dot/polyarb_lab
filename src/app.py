from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.runtime.runner import ResearchRunner

console = Console()


def main() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    console.print("[cyan]Starting research runner...[/cyan]")
    runner = ResearchRunner()
    runner.run_forever()


if __name__ == "__main__":
    main()
