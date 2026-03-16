from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from src.domain.models import RunSummary
from src.reporting.models import BatchExperimentSummary, ResearchCampaignExecutionSummary
from src.reporting.summary import aggregate_run_summaries
from src.runtime.campaigns import ResearchCampaignManifest
from src.runtime.runner import ResearchRunner


class BatchResearchRunner:
    def __init__(self, runner_factory: Callable[[], ResearchRunner] | None = None):
        self.runner_factory = runner_factory or ResearchRunner

    def run_batch(
        self,
        cycles: int,
        sleep_sec: float = 0.0,
        experiment_label: str | None = None,
        parameter_set_label: str | None = None,
        market_limit: int | None = None,
        extra_context: dict[str, object] | None = None,
    ) -> BatchExperimentSummary:
        if cycles <= 0:
            raise ValueError("cycles must be positive")

        runner = self.runner_factory()
        if market_limit is not None:
            runner.config.market_data.market_limit = market_limit

        experiment_id = str(uuid4())
        started_ts = datetime.now(timezone.utc)
        run_summaries: list[RunSummary] = []

        for index in range(cycles):
            merged_context = {
                "experiment_id": experiment_id,
                "experiment_label": experiment_label or f"batch-{started_ts.strftime('%Y%m%d%H%M%S')}",
                "parameter_set_label": parameter_set_label or "runtime_default",
                "batch_cycle_index": index,
                "batch_cycles_requested": cycles,
            }
            if extra_context:
                merged_context.update({key: value for key, value in extra_context.items() if value is not None})
            summary = runner.run_once(
                experiment_context=merged_context
            )
            run_summaries.append(summary)
            if sleep_sec > 0 and index < cycles - 1:
                time.sleep(sleep_sec)

        ended_ts = datetime.now(timezone.utc)
        aggregate = aggregate_run_summaries(
            run_id=f"batch:{experiment_id}",
            started_ts=started_ts,
            ended_ts=ended_ts,
            summaries=run_summaries,
        )
        aggregate.metadata.update(
            {
                "experiment_id": experiment_id,
                "experiment_label": experiment_label or f"batch-{started_ts.strftime('%Y%m%d%H%M%S')}",
                "parameter_set_label": parameter_set_label or "runtime_default",
                "batch_cycles_requested": cycles,
                "batch_cycles_completed": len(run_summaries),
                **({key: value for key, value in (extra_context or {}).items() if value is not None}),
            }
        )
        return BatchExperimentSummary(
            experiment_id=experiment_id,
            experiment_label=experiment_label or f"batch-{started_ts.strftime('%Y%m%d%H%M%S')}",
            parameter_set_label=parameter_set_label or "runtime_default",
            campaign_id=str((extra_context or {}).get("campaign_id")) if (extra_context or {}).get("campaign_id") is not None else None,
            campaign_label=str((extra_context or {}).get("campaign_label")) if (extra_context or {}).get("campaign_label") is not None else None,
            started_ts=started_ts,
            ended_ts=ended_ts,
            cycles_requested=cycles,
            cycles_completed=len(run_summaries),
            run_ids=[summary.run_id for summary in run_summaries],
            aggregate_summary=aggregate,
            per_run_summaries=run_summaries,
            metadata={
                "market_limit": runner.config.market_data.market_limit,
                **({key: value for key, value in (extra_context or {}).items() if value is not None}),
            },
        )

    def run_campaign(self, manifest: ResearchCampaignManifest) -> ResearchCampaignExecutionSummary:
        campaign_id = manifest.resolved_campaign_id()
        parameter_sets = manifest.resolved_parameter_sets()
        started_ts = datetime.now(timezone.utc)
        batch_summaries: list[BatchExperimentSummary] = []
        all_runs: list[RunSummary] = []

        for parameter_set_label in parameter_sets:
            batch_summary = self.run_batch(
                cycles=manifest.cycles,
                sleep_sec=manifest.sleep_sec,
                experiment_label=manifest.resolved_experiment_label(parameter_set_label),
                parameter_set_label=parameter_set_label,
                market_limit=manifest.market_limit,
                extra_context={
                    "campaign_id": campaign_id,
                    "campaign_label": manifest.campaign_label,
                    "campaign_purpose": manifest.purpose,
                    "campaign_notes": manifest.notes,
                    "campaign_target_strategy_families": manifest.target_strategy_families,
                    "campaign_target_parameter_sets": parameter_sets,
                    "campaign_run_cadence_note": manifest.run_cadence_note,
                    **manifest.metadata,
                },
            )
            batch_summaries.append(batch_summary)
            all_runs.extend(batch_summary.per_run_summaries)

        ended_ts = datetime.now(timezone.utc)
        aggregate = aggregate_run_summaries(
            run_id=f"campaign:{campaign_id}",
            started_ts=started_ts,
            ended_ts=ended_ts,
            summaries=all_runs,
        )
        aggregate.metadata.update(
            {
                "campaign_id": campaign_id,
                "campaign_label": manifest.campaign_label,
                "campaign_purpose": manifest.purpose,
                "campaign_notes": manifest.notes,
                "campaign_target_strategy_families": manifest.target_strategy_families,
                "campaign_target_parameter_sets": parameter_sets,
                "campaign_run_cadence_note": manifest.run_cadence_note,
                "campaign_cycles_requested_per_parameter_set": manifest.cycles,
                "campaign_cycles_completed": sum(summary.cycles_completed for summary in batch_summaries),
                **manifest.metadata,
            }
        )
        return ResearchCampaignExecutionSummary(
            campaign_id=campaign_id,
            campaign_label=manifest.campaign_label,
            purpose=manifest.purpose,
            notes=manifest.notes,
            target_strategy_families=list(manifest.target_strategy_families),
            target_parameter_sets=list(parameter_sets),
            started_ts=started_ts,
            ended_ts=ended_ts,
            cycles_requested_per_parameter_set=manifest.cycles,
            cycles_completed=sum(summary.cycles_completed for summary in batch_summaries),
            run_ids=[run.run_id for run in all_runs],
            experiment_ids=[summary.experiment_id for summary in batch_summaries if summary.experiment_id],
            batch_summaries=batch_summaries,
            aggregate_summary=aggregate,
            metadata={
                "market_limit": manifest.market_limit,
                "sleep_sec": manifest.sleep_sec,
                "campaign_run_cadence_note": manifest.run_cadence_note,
                **manifest.metadata,
            },
        )
