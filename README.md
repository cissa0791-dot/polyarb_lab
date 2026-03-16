# polyarb_lab

A local PC research and paper-trading scaffold for Polymarket-style arbitrage detection.

## What it does
- Reads active YES/NO markets from Gamma
- Pulls order books from the Polymarket CLOB
- Detects single-market under-1 opportunities in paper mode
- Detects simple cross-market logical violations from `constraints.yaml`
- Stores opportunities in SQLite
- Does **not** place live orders

## Why no live trading
This project is intentionally limited to research, monitoring, and paper-trading. It is designed to help you test whether an apparent arbitrage survives order-book depth, slippage, and time.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.app
```

## Configure
Edit `config/settings.yaml` and `config/constraints.yaml`.

## Notes
- Gamma payloads change. The normalizer is defensive, but you may still need to adapt field handling.
- Cross-market detection is **detection only** in v1.
- Historical replay is scaffolded but not implemented.
- Architecture notes live in `docs/architecture.md`.

## Next steps
- Add markout measurement (5s / 30s / 5m)
- Add Telegram/email alerts
- Add richer multi-outcome normalization
- Add backtest replay from saved books
- Expand the new typed config, event store, and risk runner foundation
