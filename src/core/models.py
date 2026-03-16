from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    token_id: str
    bids: List[BookLevel] = Field(default_factory=list)
    asks: List[BookLevel] = Field(default_factory=list)
    ts: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Instrument(BaseModel):
    token_id: str
    market_slug: str
    outcome: str
    side: str
    active: bool = True
    question: Optional[str] = None
    end_date: Optional[str] = None


class ArbOpportunity(BaseModel):
    kind: str
    name: str
    edge_cents: float
    gross_profit: float
    est_fill_cost: float
    est_payout: float
    notional: float
    details: Dict[str, Any]
    ts: datetime


class Fill(BaseModel):
    symbol: str
    side: str
    shares: float
    price: float
    ts: datetime


class Position(BaseModel):
    symbol: str
    shares: float = 0.0
    avg_price: float = 0.0


class MarketPair(BaseModel):
    market_slug: str
    yes_token_id: str
    no_token_id: str
    question: Optional[str] = None
