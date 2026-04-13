"""
Probe live markets for Champions League round progression slugs
and NBA win-implies-playoffs subject matching.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.gamma import fetch_markets

SETTINGS = ROOT / "config" / "settings.yaml"

_NBA_FINALS_RE = re.compile(r"^will-the-(?P<subject>.+)-win-the-2026-nba-finals$")
_NBA_PLAYOFF_RE = re.compile(r"^will-the-(?P<subject>.+)-make-the-nba-playoffs(?:-[0-9]+)?$")
_CL_WIN_RE = re.compile(r"^will-(?P<subject>.+)-win-the-202526-champions-league$")
_CL_ANY_RE = re.compile(r".*champions.?league.*")
_UCL_REACH_RE = re.compile(r".*(reach|advance|semi|final|knockout|quarter).*(?:champions|ucl|cl).*|.*(?:champions|ucl|cl).*(reach|advance|semi|final|knockout|quarter).*")

def main():
    cfg = load_runtime_config(str(SETTINGS))
    gamma_host = cfg.market_data.gamma_host
    print(f"Fetching 3000 markets...", flush=True)
    markets = fetch_markets(gamma_host, limit=3000)
    slugs = [str(m.get("slug", "")).strip().lower() for m in markets if m.get("slug")]

    # NBA subject matching check
    finals_subjects = set()
    playoff_subjects = set()
    for slug in slugs:
        m = _NBA_FINALS_RE.match(slug)
        if m:
            finals_subjects.add(m.group("subject"))
        m2 = _NBA_PLAYOFF_RE.match(slug)
        if m2:
            playoff_subjects.add(m2.group("subject"))

    matched = finals_subjects & playoff_subjects
    print(f"\n=== NBA Finals -> Playoffs ===")
    print(f"Finals subjects: {len(finals_subjects)}")
    print(f"Playoff subjects: {len(playoff_subjects)}")
    print(f"Matched subjects (both exist): {len(matched)}")
    for s in sorted(matched)[:10]:
        print(f"  {s}")

    # CL win subjects
    cl_win_subjects = set()
    for slug in slugs:
        m = _CL_WIN_RE.match(slug)
        if m:
            cl_win_subjects.add(m.group("subject"))

    print(f"\n=== Champions League Win subjects: {len(cl_win_subjects)} ===")
    for s in sorted(cl_win_subjects):
        print(f"  {s}")

    # CL round progression slugs (anything CL-related that isn't just win)
    print(f"\n=== CL progression slugs (non-win) ===")
    for slug in slugs:
        if _CL_ANY_RE.search(slug) and not _CL_WIN_RE.match(slug):
            print(f"  {slug}")

    # Any reach/advance/semifinal patterns
    print(f"\n=== All reach/advance/semifinal/final slugs ===")
    reach_pat = re.compile(r".*(reach|advance-to|advance-from|make-the|reach-the).*(final|semi|quarter|round).*")
    for slug in slugs:
        if reach_pat.search(slug):
            print(f"  {slug}")

if __name__ == "__main__":
    main()
