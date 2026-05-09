from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.build_fill_probability_model_report import main
from src.live.fill_probability_model import (
    UNCALIBRATED_STATUS,
    calibration_report,
    estimate_fill_probability,
)


NOW = datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc)


def test_stable_hash_and_output_for_identical_inputs() -> None:
    left = estimate_fill_probability(distance_to_best=0.02, spread=0.01, same_price_depth=100, now=NOW)
    right = estimate_fill_probability(distance_to_best=0.02, spread=0.01, same_price_depth=100, now=NOW)

    assert left["model_input_hash"] == right["model_input_hash"]
    assert left["p_fill_300s"] == right["p_fill_300s"]


def test_higher_distance_lowers_fill_probability() -> None:
    near = estimate_fill_probability(distance_to_best=0.0, spread=0.01, same_price_depth=100, now=NOW)
    far = estimate_fill_probability(distance_to_best=0.05, spread=0.01, same_price_depth=100, now=NOW)

    assert far["p_fill_300s"] < near["p_fill_300s"]


def test_toxic_volatility_increases_toxic_fill_probability() -> None:
    calm = estimate_fill_probability(distance_to_best=0.01, volatility=0.0, depth_imbalance=0.0, now=NOW)
    toxic = estimate_fill_probability(distance_to_best=0.01, volatility=0.2, depth_imbalance=0.9, now=NOW)

    assert toxic["p_toxic_fill"] > calm["p_toxic_fill"]


def test_uncalibrated_proxy_cannot_be_presented_as_proven() -> None:
    report = estimate_fill_probability(distance_to_best=0.01, now=NOW)

    assert report["calibration_status"] == UNCALIBRATED_STATUS
    assert report["statistically_proven"] is False


def test_calibration_report_checks_bucket_accuracy_when_history_exists() -> None:
    history = [{"predicted_fill_probability": 0.25, "filled": i % 4 == 0} for i in range(24)]
    report = calibration_report(history, now=NOW)

    assert report["calibration_status"] == "MODEL_CALIBRATION_REPORT_READY"
    assert report["usable_sample_count"] == 24
    assert "0.2-0.3" in report["buckets"]
    assert report["buckets"]["0.2-0.3"]["actual_fill_rate"] == 0.25


def test_cli_writes_model_report(tmp_path) -> None:
    out = tmp_path / "fill_model.json"

    rc = main(["--distance-to-best", "0.02", "--spread", "0.01", "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["model_version"] == "fill_probability_model_v1.proxy"
    assert payload["calibration_status"] == UNCALIBRATED_STATUS
