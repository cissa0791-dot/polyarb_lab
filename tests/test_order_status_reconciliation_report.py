from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts import build_order_status_reconciliation_report as report_script


class _FakeClient:
    def __init__(self, raw: dict):
        self.raw = raw

    def get_raw_order(self, order_id: str) -> dict:
        return dict(self.raw)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        probe_report=str(tmp_path / "missing_probe.json"),
        order_id="0xorder",
        host="https://clob.polymarket.com",
        out=str(tmp_path / "order_status.json"),
        pretty=False,
    )


def _patch_client(monkeypatch, raw: dict) -> None:
    monkeypatch.setattr(report_script, "load_live_credentials", lambda: object())
    monkeypatch.setattr(
        report_script.LiveWriteClient,
        "from_credentials",
        staticmethod(lambda *args, **kwargs: _FakeClient(raw)),
    )


def test_raw_order_missing_fill_fields_blocks(monkeypatch, tmp_path: Path) -> None:
    _patch_client(monkeypatch, {})

    report = report_script.build_report(_args(tmp_path))

    assert report["status"] == "ORDER_STATUS_RECONCILIATION_BLOCKED"
    assert report["blockers"] == [
        "RAW_ORDER_STATUS_MISSING",
        "SIZE_MATCHED_MISSING",
        "ORDER_SIZE_FIELDS_MISSING",
    ]
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False


def test_raw_order_with_status_and_size_fields_is_ready(monkeypatch, tmp_path: Path) -> None:
    _patch_client(
        monkeypatch,
        {
            "status": "CANCELED",
            "size_matched": "0",
            "size": "50",
            "price": "0.38",
            "side": "BUY",
            "market": "ivan-market",
        },
    )

    report = report_script.build_report(_args(tmp_path))

    assert report["status"] == "ORDER_STATUS_RECONCILIATION_READY"
    assert report["raw_order_status"] == "CANCELED"
    assert report["size_matched"] == 0.0
    assert report["original_size"] == 50.0
    assert report["raw_order_cancelled_zero_fill"] is True

