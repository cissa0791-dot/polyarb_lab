import sys
sys.path.insert(0, ".")

from src.storage.event_store import ResearchStore

s = ResearchStore("sqlite:///data/processed/paper.db")
rows = s.load_open_live_positions()

print("open live positions:", len(rows))
for r in rows:
    print(
        "pos=", r["position_id"][:16],
        "filled=", r["filled_size"],
        "price=", r["avg_fill_price"],
        "side=", r["side"],
    )

s.close()