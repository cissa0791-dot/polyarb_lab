import sys
sys.path.insert(0, ".")

from sqlalchemy import create_engine, text

e = create_engine("sqlite:///data/processed/paper.db")

with e.connect() as c:
    ts = c.execute(text("""
        SELECT position_id, entry_cost_usd, exit_proceeds_usd, realized_pnl_usd
        FROM trade_summaries
    """)).fetchall()

    pc = c.execute(text("""
        SELECT position_id, ts
        FROM position_events
        WHERE event_type = 'position_closed'
    """)).fetchall()

print("trade_summaries:", len(ts), "rows")
for r in ts:
    print(" ", r[0][:16], "entry=$%.4f" % r[1], "exit=$%.4f" % r[2], "pnl=$%.4f" % r[3])

print("position_closed events:", len(pc))
for r in pc:
    print(" ", r[0][:16], r[1])