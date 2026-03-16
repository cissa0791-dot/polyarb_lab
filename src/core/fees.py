def total_buffer_cents(settings: dict) -> float:
    return float(settings.get("fee_buffer_cents", 0.0)) + float(settings.get("slippage_buffer_cents", 0.0))
