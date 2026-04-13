from __future__ import annotations

from src.config_runtime.models import RuntimeConfig


RUNTIME_PARAMETER_SET_OVERRIDES: dict[str, dict[str, dict[str, float | int | bool]]] = {
    "runtime_default": {},
    "strict": {
        "opportunity": {
            "min_edge_cents": 0.05,
            "min_net_profit_usd": 1.0,
            "max_spread_cents": 0.05,
            "min_depth_multiple": 4.0,
            "max_partial_fill_risk": 0.50,
            "max_non_atomic_risk": 0.50,
        }
    },
    "loose": {
        "opportunity": {
            "min_edge_cents": 0.02,
            "min_net_profit_usd": 0.25,
            "max_spread_cents": 0.10,
            "min_depth_multiple": 2.0,
            "max_partial_fill_risk": 0.75,
            "max_non_atomic_risk": 0.70,
        }
    },
    "fillability_probe": {
        "opportunity": {
            "min_edge_cents": 0.015,
            "min_net_profit_usd": 0.10,
            "max_spread_cents": 0.12,
            "min_depth_multiple": 1.5,
            "max_partial_fill_risk": 0.85,
            "max_non_atomic_risk": 0.80,
        }
    },
    "execution_microedge_probe": {
        "opportunity": {
            "fee_buffer_cents": 0.0,
            "slippage_buffer_cents": 0.0,
            "min_edge_cents": 0.001,
            "min_net_profit_usd": 0.05,
        }
    },
    "execution_tinybuffer_probe": {
        "opportunity": {
            "fee_buffer_cents": 0.001,
            "slippage_buffer_cents": 0.001,
            "min_edge_cents": 0.001,
            "min_net_profit_usd": 0.05,
        }
    },
}


def list_runtime_parameter_sets() -> list[str]:
    return sorted(RUNTIME_PARAMETER_SET_OVERRIDES)


def apply_runtime_parameter_set(config: RuntimeConfig, parameter_set_label: str) -> RuntimeConfig:
    if parameter_set_label not in RUNTIME_PARAMETER_SET_OVERRIDES:
        available = ", ".join(list_runtime_parameter_sets())
        raise ValueError(f"Unknown runtime parameter set '{parameter_set_label}'. Available: {available}")

    updated = config.model_copy(deep=True)
    overrides = RUNTIME_PARAMETER_SET_OVERRIDES[parameter_set_label]
    for section_name, section_overrides in overrides.items():
        section = getattr(updated, section_name)
        for field_name, value in section_overrides.items():
            setattr(section, field_name, value)
    return updated
