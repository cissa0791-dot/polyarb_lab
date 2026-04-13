# polyarb_lab

> Status: Research and infrastructure project for Polymarket market scanning, paper trading, and execution diagnostics.

## Overview
polyarb_lab is a local research repository for studying Polymarket-style market structure, order book conditions, candidate qualification, and execution constraints.

The project is primarily used to test whether apparent paper opportunities remain executable after spread, depth, fees, queue competition, and runtime friction are taken into account.

## What it does
- Reads active YES/NO markets from Gamma
- Pulls order books from the Polymarket CLOB
- Detects paper candidates and cross-market logical violations
- Stores outputs in SQLite
- Supports research, diagnostics, and paper/live execution components depending on the active line

## Current scope
This repository is an engineering and research workspace, not a claim of proven production profitability.

Current work may include:
- market discovery and filtering
- order book and spread analysis
- paper execution
- live execution plumbing
- diagnostics, reporting, and audit scripts

## Why no default live trading
Live trading should only be enabled when runtime evidence supports it.  
Paper logic alone is not enough.  
This repository is structured to separate:
- paper edge
- executable edge
- operational blockers

## Setup
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -3 -m src.app