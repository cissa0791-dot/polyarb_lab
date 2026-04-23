from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import insert


class OpportunityStore:
    def __init__(self, sqlite_url: str):
        if sqlite_url.startswith("sqlite:///") and ":memory:" not in sqlite_url:
            Path(sqlite_url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(sqlite_url, future=True)
        self.meta = MetaData()
        self.opps = Table(
            "opportunities",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("kind", String(32), nullable=False),
            Column("name", String(128), nullable=False),
            Column("edge_cents", Float, nullable=False),
            Column("gross_profit", Float, nullable=False),
            Column("notional", Float, nullable=False),
            Column("ts", DateTime, nullable=False),
            Column("details_json", Text, nullable=False),
        )
        self.meta.create_all(self.engine)

    def save(self, opp):
        with self.engine.begin() as conn:
            conn.execute(insert(self.opps).values(
                kind=opp.kind,
                name=opp.name,
                edge_cents=opp.edge_cents,
                gross_profit=opp.gross_profit,
                notional=opp.notional,
                ts=opp.ts,
                details_json=json.dumps(opp.details, ensure_ascii=False),
            ))
