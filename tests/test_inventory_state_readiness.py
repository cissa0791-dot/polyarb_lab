from __future__ import annotations

from datetime import datetime, timezone

from src.live.inventory_state_readiness import build_inventory_state_report


NOW = datetime(2026, 5, 8, 2, 0, tzinfo=timezone.utc)


def test_flat_inventory_and_no_orders_is_clear() -> None:
    report = build_inventory_state_report(open_orders=[], positions=[], wallet_address="0xwallet", now=NOW)

    assert report["status"] == "INVENTORY_STATE_CLEAR"
    assert report["open_order_count"] == 0
    assert report["non_usdc_position_count"] == 0
    assert report["partial_fill_unresolved"] is False
    assert report["can_submit_order"] is False


def test_open_order_blocks_inventory_state() -> None:
    report = build_inventory_state_report(
        open_orders=[
            {
                "id": "order-1",
                "side": "BUY",
                "price": 0.37,
                "size": 50,
                "remaining_size": 50,
                "asset_id": "token",
            }
        ],
        positions=[],
        now=NOW,
    )

    assert report["status"] == "INVENTORY_STATE_BLOCKED"
    assert "OPEN_ORDERS_PRESENT" in report["blockers"]
    assert report["open_order_status"] == "OPEN_ORDER_PRESENT"


def test_non_dust_position_blocks_inventory_state() -> None:
    report = build_inventory_state_report(
        open_orders=[],
        positions=[{"marketSlug": "m", "asset": "YES", "size": 0.25}],
        now=NOW,
    )

    assert report["status"] == "INVENTORY_STATE_BLOCKED"
    assert "NON_USDC_POSITION_PRESENT" in report["blockers"]
    assert report["token_balance_shares"] == 0.25


def test_read_error_fails_closed_even_if_inventory_payload_is_empty() -> None:
    report = build_inventory_state_report(
        open_orders=[],
        positions=[],
        read_errors=["DATA_API_POSITIONS_READ_FAILED"],
        now=NOW,
    )

    assert report["status"] == "INVENTORY_STATE_BLOCKED"
    assert "INVENTORY_SOURCE_READ_FAILED" in report["blockers"]
