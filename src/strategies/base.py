from __future__ import annotations

from abc import ABC, abstractmethod

from src.opportunity.models import RawCandidate, StrategyFamily


class BaseOpportunityStrategy(ABC):
    strategy_id: str
    strategy_family: StrategyFamily

    @abstractmethod
    def detect(self, *args, **kwargs) -> RawCandidate | None:
        raise NotImplementedError

