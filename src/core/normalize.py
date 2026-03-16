from __future__ import annotations

from typing import Dict, Iterable, List
from src.core.models import MarketPair


def build_yes_no_pairs(markets: List[dict]) -> List[MarketPair]:
    """
    Build YES/NO token pairs from Gamma metadata.
    This function is intentionally defensive because market payloads differ.
    """
    pairs: List[MarketPair] = []
    for m in markets:
        slug = m.get("slug") or m.get("market_slug") or m.get("id")
        question = m.get("question") or m.get("title")
        outcomes = m.get("outcomes") or []
        token_ids = m.get("clobTokenIds") or m.get("clob_token_ids") or []

        # Gamma often stores outcomes and token ids as JSON strings
        if isinstance(outcomes, str):
            try:
                import json
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = []
        if isinstance(token_ids, str):
            try:
                import json
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = []

        if len(outcomes) != 2 or len(token_ids) != 2:
            continue

        norm = {str(out).strip().upper(): tid for out, tid in zip(outcomes, token_ids)}
        if "YES" in norm and "NO" in norm:
            pairs.append(MarketPair(
                market_slug=str(slug),
                yes_token_id=str(norm["YES"]),
                no_token_id=str(norm["NO"]),
                question=question,
            ))
    return pairs
