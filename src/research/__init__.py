"""
Track B research pipeline — belief-aware opportunity ranker.

Branch: branch/belief_ranker_research_v1
Status: research-only / paper sandbox

This module does NOT import from:
  - src.live.*          (execution / broker / signing)
  - src.runtime.*       (runner / campaigns)
  - src.storage.*       (Track A DB)
  - scripts/trial_entry_scan.py (Track A scanner)

No live trading. No DB writes to Track A tables. No order placement.
"""
