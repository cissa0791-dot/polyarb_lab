import sqlite3, datetime

DB = "data/processed/maker_paper_calib.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

markets = conn.execute(
    "SELECT DISTINCT market_slug FROM quote_observations WHERE status='OPEN'"
).fetchall()

total_cleaned = 0
for row in markets:
    slug = row["market_slug"]
    latest = conn.execute(
        "SELECT obs_id FROM quote_observations WHERE market_slug=? AND status='OPEN' ORDER BY obs_ts DESC LIMIT 1",
        (slug,)
    ).fetchone()
    if latest is None:
        continue

    keep_id = latest["obs_id"]
    cur = conn.execute(
        "UPDATE quote_observations SET status='SUPERSEDED', crossed_ts=? WHERE market_slug=? AND status='OPEN' AND obs_id != ?",
        (now_ts, slug, keep_id)
    )
    total_cleaned += cur.rowcount
    print(f"{slug}: superseded {cur.rowcount} stale OPEN row(s), kept {keep_id}")

conn.commit()
conn.close()
print(f"Done. {total_cleaned} stale OPEN rows cleaned.")
