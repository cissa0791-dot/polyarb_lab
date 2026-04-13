from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

import websockets

from src.config_runtime.models import MarketDataConfig
from src.core.models import MarketPair

MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class MarketTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class PairBookSnapshot:
    midpoint_sum: float | None
    max_spread: float | None
    top_depth: float | None
    inventory_abs: float
    source: str
    ts: datetime


@dataclass
class RecomputeDecision:
    triggered: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class UniverseEntry:
    pair: MarketPair
    tier: MarketTier = MarketTier.COLD
    active: bool = True
    event_slug: str | None = None
    event_title: str | None = None
    last_discovered_at: datetime | None = None
    last_refreshed_at: datetime | None = None
    last_evaluated_at: datetime | None = None
    last_raw_signal_at: datetime | None = None
    last_qualified_at: datetime | None = None
    last_near_miss_at: datetime | None = None
    last_productive_outcome_at: datetime | None = None
    raw_signal_count: int = 0
    qualified_count: int = 0
    near_miss_count: int = 0
    productive_outcome_count: int = 0
    seeded_recent_qualified: bool = False
    seeded_recent_near_miss: bool = False
    seeded_productive: bool = False
    websocket_update_count: int = 0
    last_websocket_update_at: datetime | None = None
    last_snapshot: PairBookSnapshot | None = None

    @property
    def market_slug(self) -> str:
        return self.pair.market_slug

    @property
    def token_ids(self) -> list[str]:
        return [self.pair.yes_token_id, self.pair.no_token_id]


@dataclass
class FamilyEntry:
    family_slug: str
    event_title: str | None = None
    market_slugs: set[str] = field(default_factory=set)
    active: bool = True
    last_recomputed_at: datetime | None = None
    last_raw_signal_at: datetime | None = None
    last_qualified_at: datetime | None = None
    last_near_miss_at: datetime | None = None
    last_productive_outcome_at: datetime | None = None
    raw_signal_count: int = 0
    qualified_count: int = 0
    near_miss_count: int = 0
    productive_outcome_count: int = 0
    seeded_recent_qualified: bool = False
    seeded_recent_near_miss: bool = False
    seeded_productive: bool = False


@dataclass
class RefreshPlan:
    hot: list[UniverseEntry] = field(default_factory=list)
    warm: list[UniverseEntry] = field(default_factory=list)
    cold: list[UniverseEntry] = field(default_factory=list)
    backstop: list[UniverseEntry] = field(default_factory=list)
    force_scan_market_slugs: set[str] = field(default_factory=set)

    def entries_by_tier(self) -> dict[str, list[UniverseEntry]]:
        return {
            MarketTier.HOT.value: list(self.hot),
            MarketTier.WARM.value: list(self.warm),
            MarketTier.COLD.value: list(self.cold),
            "backstop": list(self.backstop),
        }

    def all_entries(self) -> list[UniverseEntry]:
        return [*self.hot, *self.warm, *self.cold, *self.backstop]

    def all_market_slugs(self) -> set[str]:
        return {entry.market_slug for entry in self.all_entries()}


def _top_depth(book: object) -> float | None:
    bids = getattr(book, "bids", [])
    asks = getattr(book, "asks", [])
    if not bids and not asks:
        return None
    bid_size = float(bids[0].size) if bids else 0.0
    ask_size = float(asks[0].size) if asks else 0.0
    return round(bid_size + ask_size, 6)


def _midpoint(book: object) -> float | None:
    bids = getattr(book, "bids", [])
    asks = getattr(book, "asks", [])
    if not bids or not asks:
        return None
    bid = float(bids[0].price)
    ask = float(asks[0].price)
    if ask <= bid:
        return None
    return round((bid + ask) / 2.0, 6)


def _spread(book: object) -> float | None:
    bids = getattr(book, "bids", [])
    asks = getattr(book, "asks", [])
    if not bids or not asks:
        return None
    return round(max(0.0, float(asks[0].price) - float(bids[0].price)), 6)


def build_pair_snapshot(
    *,
    yes_book: object,
    no_book: object,
    inventory_abs: float,
    source: str,
    ts: datetime,
) -> PairBookSnapshot:
    yes_mid = _midpoint(yes_book)
    no_mid = _midpoint(no_book)
    midpoint_sum = None
    if yes_mid is not None and no_mid is not None:
        midpoint_sum = round(yes_mid + no_mid, 6)

    spreads = [value for value in (_spread(yes_book), _spread(no_book)) if value is not None]
    depths = [value for value in (_top_depth(yes_book), _top_depth(no_book)) if value is not None]
    return PairBookSnapshot(
        midpoint_sum=midpoint_sum,
        max_spread=max(spreads) if spreads else None,
        top_depth=min(depths) if depths else None,
        inventory_abs=round(float(inventory_abs), 6),
        source=source,
        ts=ts,
    )


class MarketUniverseManager:
    def __init__(self, config: MarketDataConfig) -> None:
        self.config = config
        self.entries: dict[str, UniverseEntry] = {}
        self.current_events: list[dict[str, Any]] = []
        self.current_markets: list[dict[str, Any]] = []
        self.current_pairs: list[MarketPair] = []
        self.last_discovery_ts: datetime | None = None
        self.last_backstop_ts: datetime | None = None
        self._active_market_slugs: set[str] = set()
        self.family_entries: dict[str, FamilyEntry] = {}
        self._pinned_productive_market_slugs: set[str] = set()
        self._seeded_recent_qualified_market_slugs: set[str] = set()
        self._seeded_recent_near_miss_market_slugs: set[str] = set()
        self._backstop_cursor: int = 0
        self._family_backstop_cursor: int = 0

    def reconfigure(self, config: MarketDataConfig) -> None:
        self.config = config

    def discovery_due(self, now: datetime) -> bool:
        if self.last_discovery_ts is None:
            return True
        return (now - self.last_discovery_ts).total_seconds() >= float(self.config.discovery_refresh_interval_sec)

    def update_discovery(
        self,
        *,
        events: list[dict[str, Any]],
        markets: list[dict[str, Any]],
        pairs: list[MarketPair],
        discovered_at: datetime,
    ) -> None:
        self.current_events = list(events)
        self.current_markets = list(markets)
        self.current_pairs = list(pairs)
        self.last_discovery_ts = discovered_at
        self._active_market_slugs = {pair.market_slug for pair in pairs}
        market_lookup = {
            str(market.get("slug") or market.get("market_slug") or market.get("id") or ""): market
            for market in markets
            if isinstance(market, dict)
        }

        for entry in self.entries.values():
            entry.active = False
        for family_entry in self.family_entries.values():
            family_entry.active = False
            family_entry.market_slugs.clear()

        for pair in pairs:
            entry = self.entries.get(pair.market_slug)
            if entry is None:
                entry = UniverseEntry(pair=pair)
                self.entries[pair.market_slug] = entry
            else:
                entry.pair = pair
            entry.active = True
            entry.last_discovered_at = discovered_at
            source_market = market_lookup.get(pair.market_slug, {})
            entry.event_slug = str(
                source_market.get("eventSlug")
                or source_market.get("event_slug")
                or source_market.get("eventId")
                or entry.event_slug
                or ""
            ) or None
            entry.event_title = str(
                source_market.get("eventTitle")
                or source_market.get("event_title")
                or entry.event_title
                or ""
            ) or None
            if pair.market_slug in self._seeded_recent_qualified_market_slugs and entry.last_qualified_at is None:
                entry.last_qualified_at = discovered_at
                entry.qualified_count = max(entry.qualified_count, 1)
                entry.seeded_recent_qualified = True
            if pair.market_slug in self._seeded_recent_near_miss_market_slugs and entry.last_near_miss_at is None:
                entry.last_near_miss_at = discovered_at
                entry.near_miss_count = max(entry.near_miss_count, 1)
                entry.seeded_recent_near_miss = True
            if pair.market_slug in self._pinned_productive_market_slugs:
                entry.last_productive_outcome_at = entry.last_productive_outcome_at or discovered_at
                entry.productive_outcome_count = max(entry.productive_outcome_count, 1)
                entry.seeded_productive = True
            family_slug = str(entry.event_slug or "").strip()
            if family_slug:
                family_entry = self._ensure_family_entry(family_slug, event_title=entry.event_title)
                family_entry.active = True
                family_entry.market_slugs.add(entry.market_slug)
                family_entry.seeded_recent_qualified = family_entry.seeded_recent_qualified or entry.seeded_recent_qualified
                family_entry.seeded_recent_near_miss = family_entry.seeded_recent_near_miss or entry.seeded_recent_near_miss
                family_entry.seeded_productive = family_entry.seeded_productive or entry.seeded_productive
                family_entry.last_raw_signal_at = _max_ts(family_entry.last_raw_signal_at, entry.last_raw_signal_at)
                family_entry.last_qualified_at = _max_ts(family_entry.last_qualified_at, entry.last_qualified_at)
                family_entry.last_near_miss_at = _max_ts(family_entry.last_near_miss_at, entry.last_near_miss_at)
                family_entry.last_productive_outcome_at = _max_ts(
                    family_entry.last_productive_outcome_at,
                    entry.last_productive_outcome_at,
                )

    def current_token_map(self) -> dict[str, str]:
        token_map: dict[str, str] = {}
        for entry in self.active_entries():
            token_map[entry.pair.yes_token_id] = entry.market_slug
            token_map[entry.pair.no_token_id] = entry.market_slug
        return token_map

    def active_entries(self) -> list[UniverseEntry]:
        return [entry for entry in self.entries.values() if entry.active]

    def active_pairs(self) -> list[MarketPair]:
        return [entry.pair for entry in self.active_entries()]

    def active_entry_for_slug(self, market_slug: str) -> UniverseEntry | None:
        entry = self.entries.get(market_slug)
        if entry is None or not entry.active:
            return None
        return entry

    def active_family_entries(self) -> list[FamilyEntry]:
        return [entry for entry in self.family_entries.values() if entry.active and entry.market_slugs]

    def tier_counts(self, now: datetime | None = None) -> dict[str, int]:
        if now is not None:
            self._reassign_tiers(now)
        counts = {MarketTier.HOT.value: 0, MarketTier.WARM.value: 0, MarketTier.COLD.value: 0}
        for entry in self.active_entries():
            counts[entry.tier.value] += 1
        return counts

    def all_hot_entries(self, now: datetime | None = None) -> list[UniverseEntry]:
        current_time = now or datetime.now(timezone.utc)
        self._reassign_tiers(current_time)
        return [entry for entry in self.active_entries() if entry.tier == MarketTier.HOT]

    def record_raw_signal(self, market_slugs: list[str], observed_at: datetime) -> None:
        touched_families: set[str] = set()
        for market_slug in market_slugs:
            entry = self.active_entry_for_slug(market_slug)
            if entry is None:
                continue
            entry.last_raw_signal_at = observed_at
            entry.raw_signal_count += 1
            family_slug = str(entry.event_slug or "").strip()
            if family_slug and family_slug not in touched_families:
                family_entry = self._ensure_family_entry(family_slug, event_title=entry.event_title)
                family_entry.last_raw_signal_at = observed_at
                family_entry.raw_signal_count += 1
                family_entry.market_slugs.add(entry.market_slug)
                touched_families.add(family_slug)

    def record_qualified_candidate(self, market_slugs: list[str], observed_at: datetime) -> None:
        touched_families: set[str] = set()
        for market_slug in market_slugs:
            entry = self.active_entry_for_slug(market_slug)
            if entry is None:
                continue
            entry.last_qualified_at = observed_at
            entry.qualified_count += 1
            family_slug = str(entry.event_slug or "").strip()
            if family_slug and family_slug not in touched_families:
                family_entry = self._ensure_family_entry(family_slug, event_title=entry.event_title)
                family_entry.last_qualified_at = observed_at
                family_entry.qualified_count += 1
                family_entry.market_slugs.add(entry.market_slug)
                touched_families.add(family_slug)

    def record_near_miss(self, market_slugs: list[str], observed_at: datetime) -> None:
        touched_families: set[str] = set()
        for market_slug in market_slugs:
            entry = self.active_entry_for_slug(market_slug)
            if entry is None:
                continue
            entry.last_near_miss_at = observed_at
            entry.near_miss_count += 1
            family_slug = str(entry.event_slug or "").strip()
            if family_slug and family_slug not in touched_families:
                family_entry = self._ensure_family_entry(family_slug, event_title=entry.event_title)
                family_entry.last_near_miss_at = observed_at
                family_entry.near_miss_count += 1
                family_entry.market_slugs.add(entry.market_slug)
                touched_families.add(family_slug)

    def record_productive_outcome(self, market_slug: str, observed_at: datetime) -> None:
        entry = self.active_entry_for_slug(market_slug)
        if entry is None:
            return
        entry.last_productive_outcome_at = observed_at
        entry.productive_outcome_count += 1
        entry.seeded_productive = True
        self._pinned_productive_market_slugs.add(market_slug)
        family_slug = str(entry.event_slug or "").strip()
        if family_slug:
            family_entry = self._ensure_family_entry(family_slug, event_title=entry.event_title)
            family_entry.last_productive_outcome_at = observed_at
            family_entry.productive_outcome_count += 1
            family_entry.seeded_productive = True
            family_entry.market_slugs.add(entry.market_slug)

    def seed_recent_qualified_markets(self, market_slugs: list[str], observed_at: datetime | None = None) -> None:
        ts = observed_at or datetime.now(timezone.utc)
        for market_slug in market_slugs:
            slug = str(market_slug or "")
            if not slug:
                continue
            self._seeded_recent_qualified_market_slugs.add(slug)
            entry = self.active_entry_for_slug(slug)
            if entry is None:
                continue
            entry.last_qualified_at = entry.last_qualified_at or ts
            entry.qualified_count = max(entry.qualified_count, 1)
            entry.seeded_recent_qualified = True

    def seed_recent_near_miss_markets(self, market_slugs: list[str], observed_at: datetime | None = None) -> None:
        ts = observed_at or datetime.now(timezone.utc)
        for market_slug in market_slugs:
            slug = str(market_slug or "")
            if not slug:
                continue
            self._seeded_recent_near_miss_market_slugs.add(slug)
            entry = self.active_entry_for_slug(slug)
            if entry is None:
                continue
            entry.last_near_miss_at = entry.last_near_miss_at or ts
            entry.near_miss_count = max(entry.near_miss_count, 1)
            entry.seeded_recent_near_miss = True

    def seed_productive_markets(self, market_slugs: list[str], observed_at: datetime | None = None) -> None:
        ts = observed_at or datetime.now(timezone.utc)
        for market_slug in market_slugs:
            slug = str(market_slug or "")
            if not slug:
                continue
            self._pinned_productive_market_slugs.add(slug)
            entry = self.active_entry_for_slug(slug)
            if entry is None:
                continue
            entry.last_productive_outcome_at = entry.last_productive_outcome_at or ts
            entry.productive_outcome_count = max(entry.productive_outcome_count, 1)
            entry.seeded_productive = True

    def record_websocket_updates(self, token_ids: list[str], observed_at: datetime) -> set[str]:
        token_map = self.current_token_map()
        changed_market_slugs: set[str] = set()
        for token_id in token_ids:
            market_slug = token_map.get(token_id)
            if not market_slug:
                continue
            entry = self.active_entry_for_slug(market_slug)
            if entry is None:
                continue
            entry.last_websocket_update_at = observed_at
            entry.websocket_update_count += 1
            changed_market_slugs.add(market_slug)
        return changed_market_slugs

    def select_refresh_plan(self, now: datetime, cycle_index: int = 0) -> RefreshPlan:
        self._reassign_tiers(now)
        hot = self._select_due_entries(now, MarketTier.HOT, int(self.config.hot_market_count))
        warm = self._select_due_entries(now, MarketTier.WARM, int(self.config.warm_market_count))
        cold = self._select_due_entries(now, MarketTier.COLD, int(self.config.cold_market_count))
        selected_market_slugs = {entry.market_slug for entry in [*hot, *warm, *cold]}
        backstop = self._select_backstop_entries(
            now,
            cycle_index=cycle_index,
            excluded_market_slugs=selected_market_slugs,
        )
        force_scan_market_slugs = {entry.market_slug for entry in hot}
        force_scan_market_slugs.update(entry.market_slug for entry in warm)
        force_scan_market_slugs.update(entry.market_slug for entry in backstop)
        return RefreshPlan(
            hot=hot,
            warm=warm,
            cold=cold,
            backstop=backstop,
            force_scan_market_slugs=force_scan_market_slugs,
        )

    def seeded_hot_count(self, now: datetime | None = None) -> int:
        if now is not None:
            self._reassign_tiers(now)
        return sum(
            1
            for entry in self.active_entries()
            if entry.tier == MarketTier.HOT
            and (entry.seeded_productive or entry.seeded_recent_qualified or entry.seeded_recent_near_miss)
        )

    def markets_with_any_signal(self) -> int:
        return sum(
            1
            for entry in self.active_entries()
            if (
                entry.raw_signal_count > 0
                or entry.qualified_count > 0
                or entry.near_miss_count > 0
                or entry.productive_outcome_count > 0
            )
        )

    def pinned_productive_market_slugs(self) -> set[str]:
        return {
            entry.market_slug
            for entry in self.active_entries()
            if entry.seeded_productive or entry.productive_outcome_count > 0
        }

    def remembered_watchlist_market_slugs(self) -> set[str]:
        return {
            *{str(slug) for slug in self._pinned_productive_market_slugs if str(slug)},
            *{str(slug) for slug in self._seeded_recent_qualified_market_slugs if str(slug)},
            *{str(slug) for slug in self._seeded_recent_near_miss_market_slugs if str(slug)},
        }

    def productive_event_slugs(self) -> set[str]:
        return {
            str(entry.event_slug)
            for entry in self.active_entries()
            if entry.event_slug and (entry.seeded_productive or entry.productive_outcome_count > 0)
        }

    def select_neg_risk_groups_for_recompute(
        self,
        *,
        event_groups: list[dict[str, Any]],
        changed_market_slugs: set[str] | None,
        now: datetime,
        cycle_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        family_descriptors: list[dict[str, Any]] = []
        selected_family_slugs: set[str] = set()
        selected_reasons: dict[str, set[str]] = {}

        for event_group in event_groups:
            family_slug = self._family_slug_for_group(event_group)
            market_slugs = {
                str(market.get("slug") or "").strip()
                for market in event_group.get("markets", [])
                if str(market.get("slug") or "").strip()
            }
            family_entry = self.family_entries.get(family_slug) if family_slug else None
            if family_entry is None and market_slugs:
                family_entry = self._family_entry_for_market_slugs(market_slugs)
            is_productive = bool(family_entry and (family_entry.seeded_productive or family_entry.productive_outcome_count > 0))
            is_recent_qualified = bool(
                family_entry and (family_entry.seeded_recent_qualified or family_entry.qualified_count > 0 or family_entry.last_qualified_at is not None)
            )
            is_recent_near_miss = bool(
                family_entry and (family_entry.seeded_recent_near_miss or family_entry.near_miss_count > 0 or family_entry.last_near_miss_at is not None)
            )
            is_watched = is_productive or is_recent_qualified or is_recent_near_miss
            changed = bool(changed_market_slugs and market_slugs and (market_slugs & changed_market_slugs))
            due = self._family_refresh_due(family_entry, now) and (is_watched or (family_entry is not None and family_entry.last_recomputed_at is not None))
            descriptor = {
                "event_group": event_group,
                "family_slug": family_slug,
                "market_slugs": market_slugs,
                "family_entry": family_entry,
                "is_productive": is_productive,
                "is_recent_qualified": is_recent_qualified,
                "is_recent_near_miss": is_recent_near_miss,
                "is_watched": is_watched,
                "changed": changed,
                "due": due,
            }
            family_descriptors.append(descriptor)
            if changed and is_watched:
                selected_family_slugs.add(family_slug)
                selected_reasons.setdefault(family_slug, set()).add("change")
            if due:
                selected_family_slugs.add(family_slug)
                selected_reasons.setdefault(family_slug, set()).add("due_interval")

        if (
            int(self.config.neg_risk_family_backstop_every_n_cycles) > 0
            and cycle_index > 0
            and cycle_index % int(self.config.neg_risk_family_backstop_every_n_cycles) == 0
        ):
            backstop_budget = max(0, int(self.config.neg_risk_family_backstop_budget))
            backstop_candidates = [descriptor for descriptor in family_descriptors if descriptor["family_slug"] not in selected_family_slugs]
            backstop_candidates.sort(
                key=lambda descriptor: (
                    0 if descriptor["is_productive"] else 1,
                    0 if descriptor["is_recent_qualified"] else 1,
                    0 if descriptor["is_recent_near_miss"] else 1,
                    descriptor["family_entry"].last_recomputed_at if descriptor["family_entry"] is not None and descriptor["family_entry"].last_recomputed_at is not None else datetime.min.replace(tzinfo=timezone.utc),
                    descriptor["family_slug"],
                )
            )
            if backstop_candidates:
                if self._family_backstop_cursor >= len(backstop_candidates):
                    self._family_backstop_cursor = 0
                rotated = backstop_candidates[self._family_backstop_cursor:] + backstop_candidates[:self._family_backstop_cursor]
                selected_backstop = rotated[:backstop_budget]
                if selected_backstop:
                    self._family_backstop_cursor = (self._family_backstop_cursor + len(selected_backstop)) % max(len(backstop_candidates), 1)
                for descriptor in selected_backstop:
                    selected_family_slugs.add(descriptor["family_slug"])
                    selected_reasons.setdefault(descriptor["family_slug"], set()).add("backstop")

        selected_groups: list[dict[str, Any]] = []
        selected_market_total = 0
        pinned_productive_families_evaluated = 0
        recent_near_miss_families_evaluated = 0
        families_recomputed_due_to_change = 0
        families_recomputed_due_to_due_interval = 0
        family_backstop_recompute_count = 0
        raw_candidates_by_family_per_cycle: dict[str, int] = {}

        for descriptor in family_descriptors:
            family_slug = descriptor["family_slug"]
            if family_slug not in selected_family_slugs:
                continue
            selected_groups.append(descriptor["event_group"])
            selected_market_total += len(descriptor["market_slugs"])
            reasons = selected_reasons.get(family_slug, set())
            if "change" in reasons:
                families_recomputed_due_to_change += 1
            if "due_interval" in reasons:
                families_recomputed_due_to_due_interval += 1
            if "backstop" in reasons:
                family_backstop_recompute_count += 1
            if descriptor["is_productive"]:
                pinned_productive_families_evaluated += 1
            if descriptor["is_recent_near_miss"]:
                recent_near_miss_families_evaluated += 1
            raw_candidates_by_family_per_cycle[family_slug] = 0

        family_count = len(selected_groups)
        metrics = {
            "families_considered": family_count,
            "families_recomputed_due_to_change": families_recomputed_due_to_change,
            "families_recomputed_due_to_due_interval": families_recomputed_due_to_due_interval,
            "pinned_productive_families_evaluated": pinned_productive_families_evaluated,
            "recent_near_miss_families_evaluated": recent_near_miss_families_evaluated,
            "family_backstop_recompute_count": family_backstop_recompute_count,
            "avg_markets_per_family_recompute": round(selected_market_total / max(family_count, 1), 6) if family_count else 0.0,
            "neg_risk_family_coverage_rate": round(family_count / max(len(event_groups), 1), 6) if event_groups else 0.0,
            "raw_candidates_by_family_per_cycle": raw_candidates_by_family_per_cycle,
        }
        return selected_groups, metrics

    def record_family_recompute(
        self,
        *,
        family_slug: str,
        observed_at: datetime,
        event_title: str | None = None,
        market_slugs: set[str] | None = None,
    ) -> None:
        if not family_slug:
            return
        family_entry = self._ensure_family_entry(family_slug, event_title=event_title)
        family_entry.active = True
        family_entry.last_recomputed_at = observed_at
        if market_slugs:
            family_entry.market_slugs.update(market_slugs)

    def evaluate_recompute(
        self,
        *,
        market_slug: str,
        snapshot: PairBookSnapshot,
        stream_event: bool,
    ) -> RecomputeDecision:
        entry = self.active_entry_for_slug(market_slug)
        if entry is None:
            return RecomputeDecision(triggered=False, reasons=["inactive"])

        previous = entry.last_snapshot
        reasons: list[str] = []
        if previous is None:
            reasons.append("first_refresh")
        if stream_event:
            reasons.append("websocket_update")

        if previous is not None:
            midpoint_delta = _abs_delta(snapshot.midpoint_sum, previous.midpoint_sum)
            spread_delta = _abs_delta(snapshot.max_spread, previous.max_spread)
            depth_delta_ratio = _relative_delta(snapshot.top_depth, previous.top_depth)
            inventory_delta = abs(snapshot.inventory_abs - previous.inventory_abs)
            if midpoint_delta is not None and midpoint_delta >= float(self.config.recompute_midpoint_delta_cents):
                reasons.append("midpoint_delta")
            if spread_delta is not None and spread_delta >= float(self.config.recompute_spread_delta_cents):
                reasons.append("spread_delta")
            if depth_delta_ratio is not None and depth_delta_ratio >= float(self.config.recompute_top_depth_delta_ratio):
                reasons.append("top_depth_delta")
            if inventory_delta >= float(self.config.recompute_inventory_delta_shares):
                reasons.append("inventory_delta")

        entry.last_snapshot = snapshot
        entry.last_refreshed_at = snapshot.ts
        decision = RecomputeDecision(triggered=bool(reasons), reasons=reasons or ["unchanged"])
        if decision.triggered:
            entry.last_evaluated_at = snapshot.ts
        return decision

    def _reassign_tiers(self, now: datetime) -> None:
        for entry in self.active_entries():
            entry.tier = self._tier_for_entry(entry, now)

    def _tier_for_entry(self, entry: UniverseEntry, now: datetime) -> MarketTier:
        hot_window = max(float(self.config.hot_refresh_interval_sec) * 10.0, float(self.config.discovery_refresh_interval_sec))
        warm_window = max(float(self.config.warm_refresh_interval_sec) * 10.0, float(self.config.discovery_refresh_interval_sec) * 2.0)
        if (
            _within_window(entry.last_qualified_at, now, hot_window)
            or _within_window(entry.last_near_miss_at, now, hot_window)
            or entry.productive_outcome_count > 0
            or entry.seeded_productive
        ):
            return MarketTier.HOT
        if _within_window(entry.last_raw_signal_at, now, warm_window):
            return MarketTier.WARM
        return MarketTier.COLD

    def _select_due_entries(self, now: datetime, tier: MarketTier, limit: int) -> list[UniverseEntry]:
        interval_map = {
            MarketTier.HOT: float(self.config.hot_refresh_interval_sec),
            MarketTier.WARM: float(self.config.warm_refresh_interval_sec),
            MarketTier.COLD: float(self.config.cold_refresh_interval_sec),
        }
        interval_sec = interval_map[tier]
        eligible = [
            entry
            for entry in self.active_entries()
            if entry.tier == tier and self._refresh_due(entry, now, interval_sec)
        ]
        eligible.sort(
            key=lambda entry: (
                -self._priority_score(entry),
                entry.last_refreshed_at or datetime.min.replace(tzinfo=timezone.utc),
                entry.market_slug,
            )
        )
        return eligible[: max(0, limit)]

    def _refresh_due(self, entry: UniverseEntry, now: datetime, interval_sec: float) -> bool:
        if entry.last_refreshed_at is None:
            return True
        if (
            entry.last_near_miss_at is not None
            and entry.last_evaluated_at is not None
            and (now - entry.last_evaluated_at).total_seconds() >= float(self.config.near_miss_retry_interval_sec)
        ):
            return True
        return (now - entry.last_refreshed_at).total_seconds() >= interval_sec

    def _select_backstop_entries(
        self,
        now: datetime,
        *,
        cycle_index: int,
        excluded_market_slugs: set[str],
    ) -> list[UniverseEntry]:
        budget = max(0, int(self.config.backstop_refresh_market_budget))
        if budget <= 0:
            return []
        force_productive = (
            int(self.config.force_refresh_productive_families_every_n_cycles) > 0
            and cycle_index > 0
            and cycle_index % int(self.config.force_refresh_productive_families_every_n_cycles) == 0
        )
        backstop_due = self.last_backstop_ts is None or (
            (now - self.last_backstop_ts).total_seconds() >= float(self.config.backstop_full_rescan_interval_sec)
        )
        if not backstop_due and not force_productive:
            return []

        productive_events = self.productive_event_slugs()
        candidates = [
            entry
            for entry in self.active_entries()
            if entry.market_slug not in excluded_market_slugs
        ]
        if not candidates:
            return []

        candidates.sort(
            key=lambda entry: (
                0 if force_productive and entry.event_slug in productive_events else 1,
                0 if entry.market_slug in self._pinned_productive_market_slugs else 1,
                entry.last_refreshed_at or datetime.min.replace(tzinfo=timezone.utc),
                entry.market_slug,
            )
        )
        if self._backstop_cursor >= len(candidates):
            self._backstop_cursor = 0
        rotated = candidates[self._backstop_cursor:] + candidates[:self._backstop_cursor]
        selected = rotated[:budget]
        if selected:
            self._backstop_cursor = (self._backstop_cursor + len(selected)) % max(len(candidates), 1)
            self.last_backstop_ts = now
        return selected

    def _priority_score(self, entry: UniverseEntry) -> float:
        return (
            (entry.productive_outcome_count * 5.0)
            + (entry.qualified_count * 3.0)
            + (entry.near_miss_count * 2.0)
            + float(entry.raw_signal_count)
            + min(entry.websocket_update_count, 10) * 0.25
        )

    def _ensure_family_entry(self, family_slug: str, *, event_title: str | None = None) -> FamilyEntry:
        entry = self.family_entries.get(family_slug)
        if entry is None:
            entry = FamilyEntry(family_slug=family_slug, event_title=event_title)
            self.family_entries[family_slug] = entry
        elif event_title and not entry.event_title:
            entry.event_title = event_title
        return entry

    def _family_entry_for_market_slugs(self, market_slugs: set[str]) -> FamilyEntry | None:
        for market_slug in market_slugs:
            entry = self.active_entry_for_slug(market_slug)
            if entry is not None and entry.event_slug:
                return self.family_entries.get(str(entry.event_slug))
        return None

    def _family_slug_for_group(self, event_group: dict[str, Any]) -> str:
        return str(event_group.get("event_slug") or event_group.get("event_id") or "").strip()

    def _family_refresh_due(self, family_entry: FamilyEntry | None, now: datetime) -> bool:
        if family_entry is None or family_entry.last_recomputed_at is None:
            return True
        return (now - family_entry.last_recomputed_at).total_seconds() >= float(self.config.neg_risk_family_due_refresh_interval_sec)


class HotMarketWebsocketClient:
    def __init__(
        self,
        *,
        url: str = MARKET_WS_URL,
        connect_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.url = url
        self.connect_factory = connect_factory or websockets.connect

    def collect_updates(
        self,
        *,
        token_ids: list[str],
        timeout_sec: float,
        stale_sec: float,
    ) -> dict[str, Any]:
        if not token_ids:
            return {
                "updated_tokens": [],
                "update_count": 0,
                "stale": False,
                "disconnected": False,
                "fallback_to_rest": False,
            }
        return asyncio.run(
            self._collect_updates(
                token_ids=list(dict.fromkeys(token_ids)),
                timeout_sec=max(0.0, float(timeout_sec)),
                stale_sec=max(0.1, float(stale_sec)),
            )
        )

    async def _collect_updates(
        self,
        *,
        token_ids: list[str],
        timeout_sec: float,
        stale_sec: float,
    ) -> dict[str, Any]:
        updated_tokens: set[str] = set()
        update_count = 0
        disconnected = False
        stale = False

        for attempt in range(2):
            try:
                async with self.connect_factory(
                    self.url,
                    ping_interval=10,
                    ping_timeout=10,
                    max_size=None,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "assets_ids": token_ids,
                                "type": "market",
                                "custom_feature_enabled": True,
                            }
                        )
                    )
                    deadline = time.monotonic() + timeout_sec
                    last_message_at = time.monotonic()
                    while time.monotonic() < deadline:
                        remaining = max(0.05, min(0.25, deadline - time.monotonic()))
                        try:
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                        except asyncio.TimeoutError:
                            if time.monotonic() - last_message_at >= stale_sec:
                                stale = True
                                break
                            continue
                        payload = json.loads(raw_message)
                        messages = payload if isinstance(payload, list) else [payload]
                        for message in messages:
                            if not isinstance(message, dict):
                                continue
                            token_id = str(message.get("asset_id") or "")
                            if token_id:
                                updated_tokens.add(token_id)
                                update_count += 1
                        last_message_at = time.monotonic()
                    break
            except Exception:
                disconnected = True
                if attempt >= 1:
                    break

        fallback_to_rest = stale or disconnected
        return {
            "updated_tokens": sorted(updated_tokens),
            "update_count": update_count,
            "stale": stale,
            "disconnected": disconnected,
            "fallback_to_rest": fallback_to_rest,
        }


def _within_window(ts: datetime | None, now: datetime, window_sec: float) -> bool:
    if ts is None:
        return False
    return (now - ts) <= timedelta(seconds=max(0.0, window_sec))


def _max_ts(lhs: datetime | None, rhs: datetime | None) -> datetime | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return rhs if rhs >= lhs else lhs


def _abs_delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return round(abs(current - previous), 6)


def _relative_delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    if abs(previous) <= 1e-9:
        return 1.0 if abs(current) > 1e-9 else 0.0
    return round(abs(current - previous) / abs(previous), 6)
