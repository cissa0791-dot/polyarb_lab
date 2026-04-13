from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.config_runtime.models import RuntimeConfig
from src.domain.models import RunSummary
from src.runtime.batch import BatchResearchRunner
from src.runtime.parameter_sets import apply_runtime_parameter_set


class RecordingRunner:
    def __init__(self):
        self.config = RuntimeConfig()
        self.applied_parameter_set_label: str | None = None
        self.calls = 0

    def apply_runtime_parameter_set(self, parameter_set_label: str) -> None:
        self.applied_parameter_set_label = parameter_set_label
        self.config = apply_runtime_parameter_set(self.config, parameter_set_label)

    def run_once(self, experiment_context=None):
        self.calls += 1
        now = datetime(2026, 3, 16, 0, self.calls, tzinfo=timezone.utc)
        return RunSummary(
            run_id=f"run-{self.calls}",
            started_ts=now,
            ended_ts=now + timedelta(seconds=1),
            metadata=experiment_context or {},
        )


class RuntimeParameterSetTests(unittest.TestCase):
    def test_execution_microedge_probe_applies_research_overrides_without_mutating_base(self) -> None:
        base = RuntimeConfig()

        updated = apply_runtime_parameter_set(base, "execution_microedge_probe")

        self.assertEqual(base.opportunity.fee_buffer_cents, 0.01)
        self.assertEqual(base.opportunity.slippage_buffer_cents, 0.01)
        self.assertEqual(base.opportunity.min_edge_cents, 0.03)
        self.assertEqual(base.opportunity.min_net_profit_usd, 0.50)
        self.assertEqual(updated.opportunity.fee_buffer_cents, 0.0)
        self.assertEqual(updated.opportunity.slippage_buffer_cents, 0.0)
        self.assertEqual(updated.opportunity.min_edge_cents, 0.001)
        self.assertEqual(updated.opportunity.min_net_profit_usd, 0.05)

    def test_execution_tinybuffer_probe_applies_small_nonzero_buffers(self) -> None:
        updated = apply_runtime_parameter_set(RuntimeConfig(), "execution_tinybuffer_probe")

        self.assertEqual(updated.opportunity.fee_buffer_cents, 0.001)
        self.assertEqual(updated.opportunity.slippage_buffer_cents, 0.001)
        self.assertEqual(updated.opportunity.min_edge_cents, 0.001)
        self.assertEqual(updated.opportunity.min_net_profit_usd, 0.05)

    def test_batch_runner_applies_parameter_set_to_runtime_runner_before_run(self) -> None:
        batch = BatchResearchRunner(runner_factory=RecordingRunner)

        result = batch.run_batch(
            cycles=1,
            experiment_label="microedge-probe",
            parameter_set_label="execution_microedge_probe",
            market_limit=11,
        )

        self.assertEqual(result.parameter_set_label, "execution_microedge_probe")
        self.assertEqual(result.metadata["market_limit"], 11)
        self.assertEqual(result.aggregate_summary.metadata["parameter_set_label"], "execution_microedge_probe")
        self.assertEqual(result.per_run_summaries[0].metadata["parameter_set_label"], "execution_microedge_probe")

    def test_unknown_parameter_set_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_runtime_parameter_set(RuntimeConfig(), "not-a-real-parameter-set")


if __name__ == "__main__":
    unittest.main()
