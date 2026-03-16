from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ConfidenceScore:
    value: float
    label: str = "model"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", max(0.0, min(1.0, float(self.value))))


@dataclass(frozen=True)
class BeliefSnapshot:
    source_id: str
    subject_id: str
    probability: float
    confidence: ConfidenceScore
    ts: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "probability", max(0.0, min(1.0, float(self.probability))))


class BeliefSource(ABC):
    source_id: str

    @abstractmethod
    def get_snapshot(self, subject_id: str) -> BeliefSnapshot | None:
        raise NotImplementedError


@dataclass(frozen=True)
class BeliefComparison:
    subject_id: str
    belief_probability: float
    market_probability: float
    discrepancy: float
    weighted_discrepancy: float
    confidence: float


class BeliefVsMarketComparator:
    def compare(self, belief: BeliefSnapshot, market_probability: float) -> BeliefComparison:
        bounded_market_probability = max(0.0, min(1.0, float(market_probability)))
        discrepancy = belief.probability - bounded_market_probability
        weighted = discrepancy * belief.confidence.value
        return BeliefComparison(
            subject_id=belief.subject_id,
            belief_probability=belief.probability,
            market_probability=bounded_market_probability,
            discrepancy=round(discrepancy, 6),
            weighted_discrepancy=round(weighted, 6),
            confidence=belief.confidence.value,
        )

