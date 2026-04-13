"""
auto_maker_loop — market_selector
polyarb_lab / research_lines / auto_maker_loop / modules

Ranks the three reward-eligible survivor markets and returns the best slug
to quote in each cycle of the automated maker loop.

Ranking formula:
    score = daily_rate_usdc / competitiveness_ref

Higher score = more reward per unit of competition.
Baseline ranking (from SURVIVOR_DATA, 2026-03-26):
    hungary : 150 / 46.9  = 3.20  ← winner (least competed, highest rate)
    rubio   : 70  / 104.1 = 0.67
    vance   : 100 / 576.9 = 0.17

At the start of each cycle the live reward config is fetched to refresh
competitiveness.  If the fetch fails, the hardcoded baseline is used.

Public interface:
    pick_best(survivor_data, live_configs=None) -> str
    rank_all(survivor_data, live_configs=None)  -> list[dict]
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def rank_all(
    survivor_data: dict,
    live_configs: Optional[dict] = None,
) -> list[dict]:
    """
    Rank all survivors by score = daily_rate_usdc / competitiveness_ref.

    Parameters
    ----------
    survivor_data : dict
        SURVIVOR_DATA from scoring_activation.py — keyed by full slug.
    live_configs : dict, optional
        Keyed by slug.  Each value is a RewardConfig object or dict with
        attributes/keys: daily_rate_usdc, competitiveness (float or None).
        When provided, live values override the hardcoded reference.

    Returns
    -------
    list of dicts, sorted best-first:
        slug, score, daily_rate_usdc, competitiveness, config_source
    """
    rows = []
    for slug, data in survivor_data.items():
        rate = data["daily_rate_usdc"]
        comp = data["competitiveness_ref"]
        source = "hardcoded"

        if live_configs and slug in live_configs:
            cfg = live_configs[slug]
            live_rate = _getv(cfg, "daily_rate_usdc")
            live_comp = _getv(cfg, "competitiveness")
            if live_rate and live_rate > 0:
                rate   = live_rate
                source = "live"
            if live_comp and live_comp > 0:
                comp   = live_comp
                source = "live"

        score = rate / comp if comp > 0 else 0.0
        rows.append({
            "slug":            slug,
            "score":           round(score, 4),
            "daily_rate_usdc": rate,
            "competitiveness": comp,
            "config_source":   source,
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def pick_best(
    survivor_data: dict,
    live_configs: Optional[dict] = None,
    skip_slugs: Optional[set] = None,
) -> Optional[str]:
    """
    Return the slug of the highest-scoring survivor.

    Parameters
    ----------
    survivor_data : dict
        SURVIVOR_DATA from scoring_activation.py.
    live_configs : dict, optional
        Live reward configs to override hardcoded competitiveness values.
    skip_slugs : set, optional
        Slugs to exclude (e.g. markets where inventory is currently zero).

    Returns
    -------
    str slug, or None if all markets are excluded.
    """
    ranked = rank_all(survivor_data, live_configs)
    if not ranked:
        return None

    for row in ranked:
        slug = row["slug"]
        if skip_slugs and slug in skip_slugs:
            logger.info(
                "market_selector: skip %s (in skip_slugs)", slug[:40]
            )
            continue
        logger.info(
            "market_selector: selected %s  score=%.4f  rate=$%.0f/day  comp=%.1f  src=%s",
            slug[:40], row["score"], row["daily_rate_usdc"],
            row["competitiveness"], row["config_source"],
        )
        return slug

    logger.warning("market_selector: all markets skipped — returning None")
    return None


def _getv(obj, key: str):
    """Get a value from a dict or object attribute. Returns None on failure."""
    try:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
    except Exception:
        return None
