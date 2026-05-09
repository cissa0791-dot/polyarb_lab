from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from math import exp, isfinite
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "fill_probability_model.v1"
REPORT_TYPE = "fill_probability_model_report"
MODEL_VERSION = "fill_probability_model_v1.proxy"

UNCALIBRATED_STATUS = "MODEL_UNCALIBRATED_PROXY_ONLY"
CALIBRATED_STATUS = "MODEL_CALIBRATION_REPORT_READY"


def estimate_fill_probability(
    *,
    distance_to_best: float,
    queue_ahead_size: float = 0.0,
    same_price_depth: float = 0.0,
    recent_trade_rate: float = 0.0,
    cancel_velocity: float = 0.0,
    spread: float = 0.01,
    depth_imbalance: float = 0.0,
    quote_lifetime_seconds: float = 300.0,
    volatility: float = 0.0,
    time_to_resolution_hours: float | None = None,
    calibrated_history: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a deterministic v1 fill-probability report.

    Without historical calibration evidence this is explicitly a proxy, not a
    statistically proven model.
    """

    now = now or datetime.now(timezone.utc)
    features = {
        "distance_to_best": _finite(distance_to_best),
        "queue_ahead_size": _finite(queue_ahead_size),
        "same_price_depth": _finite(same_price_depth),
        "recent_trade_rate": _finite(recent_trade_rate),
        "cancel_velocity": _finite(cancel_velocity),
        "spread": max(0.0, _finite(spread)),
        "depth_imbalance": max(-1.0, min(1.0, _finite(depth_imbalance))),
        "quote_lifetime_seconds": max(1.0, _finite(quote_lifetime_seconds)),
        "volatility": max(0.0, _finite(volatility)),
        "time_to_resolution_hours": None if time_to_resolution_hours is None else max(0.0, _finite(time_to_resolution_hours)),
    }
    distance_penalty = features["distance_to_best"] / max(features["spread"], 0.01)
    queue_penalty = features["queue_ahead_size"] / max(features["same_price_depth"] + features["queue_ahead_size"], 1.0)
    activity = min(3.0, features["recent_trade_rate"] + features["cancel_velocity"])
    lifetime_boost = min(1.5, features["quote_lifetime_seconds"] / 300.0)
    imbalance_boost = max(0.0, abs(features["depth_imbalance"]) - 0.25)
    logit_300 = -1.2 - (1.6 * distance_penalty) - (1.1 * queue_penalty) + (0.8 * activity) + (0.7 * lifetime_boost) + imbalance_boost
    p_300 = _sigmoid(logit_300)
    p_30 = _clamp(1.0 - ((1.0 - p_300) ** 0.1))
    p_partial = _clamp(p_300 * min(0.85, 0.25 + queue_penalty + (features["same_price_depth"] / max(features["same_price_depth"] + 100.0, 1.0))))
    p_toxic = _clamp((features["volatility"] * 3.0) + max(0.0, abs(features["depth_imbalance"]) - 0.55) + (features["recent_trade_rate"] * 0.08))
    calibration = calibration_report(calibrated_history or [], now=now)
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "model_version": MODEL_VERSION,
        "model_input_hash": _hash(features),
        "features": features,
        "p_fill_30s": round(p_30, 6),
        "p_fill_300s": round(p_300, 6),
        "p_partial_fill": round(p_partial, 6),
        "p_toxic_fill": round(p_toxic, 6),
        "calibration_status": calibration["calibration_status"],
        "calibration": calibration,
        "statistically_proven": calibration["calibration_status"] != UNCALIBRATED_STATUS,
    }


def calibration_report(history: Iterable[dict[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    rows = [row for row in history if isinstance(row, dict)]
    buckets: dict[str, dict[str, Any]] = {}
    usable = 0
    for row in rows:
        predicted = _maybe_float(row.get("predicted_fill_probability") or row.get("p_fill_300s"))
        filled = row.get("filled")
        if predicted is None or filled is None:
            continue
        usable += 1
        bucket_floor = int(min(0.9, max(0.0, predicted)) * 10) / 10
        label = f"{bucket_floor:.1f}-{bucket_floor + 0.1:.1f}"
        bucket = buckets.setdefault(label, {"count": 0, "filled_count": 0, "predicted_sum": 0.0})
        bucket["count"] += 1
        bucket["filled_count"] += 1 if bool(filled) else 0
        bucket["predicted_sum"] += predicted
    for bucket in buckets.values():
        count = bucket["count"]
        bucket["avg_predicted"] = round(bucket["predicted_sum"] / count, 6)
        bucket["actual_fill_rate"] = round(bucket["filled_count"] / count, 6)
        del bucket["predicted_sum"]
    return {
        "generated_at_utc": now.isoformat(),
        "calibration_status": CALIBRATED_STATUS if usable >= 20 else UNCALIBRATED_STATUS,
        "usable_sample_count": usable,
        "buckets": buckets,
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite(value: Any) -> float:
    parsed = _maybe_float(value)
    return 0.0 if parsed is None else parsed


def _maybe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
