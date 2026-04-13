"""
Probe 3000 live markets for slug patterns beyond current 4 mining families.
Outputs counts by pattern family and up to 5 sample slugs each.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.gamma import fetch_markets

SETTINGS = ROOT / "config" / "settings.yaml"

# Known families (exclude from new discovery)
KNOWN_PATTERNS = [
    re.compile(r"^will-the-.+-win-the-2026-nba-finals$"),
    re.compile(r"^will-the-.+-win-the-nba-(eastern|western)-conference-finals$"),
    re.compile(r"^will-.+-win-the-2026-fifa-world-cup"),
    re.compile(r"^will-.+-qualify-for-the-2026-fifa-world-cup$"),
    re.compile(r"^will-.+-win-the-2028-us-presidential-election$"),
    re.compile(r"^will-.+-win-the-2028-(democratic|republican)-presidential-nomination"),
    re.compile(r"^will-the-.+-win-the-2026-nhl-stanley-cup$"),
    re.compile(r"^will-the-.+-make-the-nhl-playoffs$"),
]

# Candidate new patterns to probe for
PROBE_PATTERNS = {
    "nfl_super_bowl":       re.compile(r".*nfl.*super.?bowl.*|.*super.?bowl.*nfl.*"),
    "nfl_playoffs":         re.compile(r".*-make-the-nfl-playoffs$"),
    "nfl_afc_nfc":          re.compile(r".*win-the-(afc|nfc).*"),
    "nfl_division":         re.compile(r".*nfl.*(afc|nfc).*(north|south|east|west).*"),
    "mlb_world_series":     re.compile(r".*mlb.*world.?series.*|.*world.?series.*mlb.*"),
    "mlb_playoffs":         re.compile(r".*make.*mlb.*playoffs|.*mlb.*make.*playoffs"),
    "nba_playoffs":         re.compile(r".*make.*nba.*playoffs|.*nba.*make.*playoffs"),
    "fifa_semifinal":       re.compile(r".*fifa.*semi.?final.*|.*semi.?final.*world.?cup.*"),
    "fifa_final":           re.compile(r".*fifa.*final.*|.*world.?cup.*final.*"),
    "fifa_quarterfinal":    re.compile(r".*quarter.?final.*world.?cup|.*world.?cup.*quarter"),
    "fifa_advance_group":   re.compile(r".*advance.*group.*|.*group.*stage.*world.?cup"),
    "tennis_win":           re.compile(r".*win.*wimbledon|.*win.*us.?open|.*win.*french.?open|.*win.*australian.?open"),
    "tennis_reach":         re.compile(r".*reach.*wimbledon|.*reach.*us.?open|.*wimbledon.*final|.*wimbledon.*semi"),
    "golf_major":           re.compile(r".*win.*masters|.*win.*pga.?championship|.*win.*open.?championship"),
    "nba_mvp":              re.compile(r".*nba.*mvp|.*mvp.*nba"),
    "nba_allstar":          re.compile(r".*all.?star.*nba|.*nba.*all.?star"),
    "oscar":                re.compile(r".*oscar.*|.*academy.?award"),
    "grammy":               re.compile(r".*grammy.*"),
    "senate_election":      re.compile(r".*win.*senate.*election|.*senate.*2026|.*senate.*2028"),
    "governor":             re.compile(r".*win.*governor|.*governor.*2026|.*governor.*2028"),
    "nfl_division_winner":  re.compile(r".*win.*nfl.*(division|title)|.*nfl.*(north|south|east|west).*winner"),
    "mlb_pennant":          re.compile(r".*pennant|.*al.*champion|.*nl.*champion"),
    "mlb_ws_team":          re.compile(r".*win.*world.?series"),
    "sports_title_generic": re.compile(r".*win-the-2026-(nfl|mlb|mls|nba|wnba|mls)"),
    "champions_league":     re.compile(r".*champions.?league|.*ucl.*"),
    "premier_league":       re.compile(r".*premier.?league.*title|.*win.*premier.?league"),
    "euros":                re.compile(r".*euro.*2026|.*2026.*euros"),
    "march_madness":        re.compile(r".*ncaa|.*march.?madness|.*college.*basketball.*champion"),
    "cfb_natl_champ":       re.compile(r".*college.?football.*champion|.*cfp.*"),
    "crypto_price":         re.compile(r".*bitcoin.*\$|.*btc.*above|.*eth.*above"),
    "ufc_belt":             re.compile(r".*ufc.*champion|.*defend.*title.*ufc"),
    "boxing":               re.compile(r".*boxing.*champion|.*win.*fight"),
    "snooker_wc":           re.compile(r".*snooker.*world"),
    "formula1":             re.compile(r".*formula.?1|.*f1.*champion|.*win.*grand.?prix"),
    "horse_racing":         re.compile(r".*kentucky.?derby|.*triple.?crown|.*belmont|.*preakness"),
}

def is_known(slug: str) -> bool:
    return any(p.match(slug) for p in KNOWN_PATTERNS)

def main():
    cfg = load_runtime_config(str(SETTINGS))
    gamma_host = cfg.market_data.gamma_host
    print(f"Fetching 3000 markets from {gamma_host}...", flush=True)
    markets = fetch_markets(gamma_host, limit=3000)
    print(f"Fetched {len(markets)} markets.", flush=True)

    slugs = [str(m.get("slug", "")).strip().lower() for m in markets if m.get("slug")]
    print(f"Unique slugs: {len(slugs)}\n")

    hits: dict[str, list[str]] = defaultdict(list)
    for slug in slugs:
        if is_known(slug):
            continue
        for family, pat in PROBE_PATTERNS.items():
            if pat.search(slug):
                hits[family].append(slug)

    # Print results sorted by count
    for family, matched in sorted(hits.items(), key=lambda x: -len(x[1])):
        print(f"[{family}] {len(matched)} markets")
        for s in matched[:8]:
            print(f"    {s}")
        print()

if __name__ == "__main__":
    main()
