from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ResearchCampaignManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    campaign_id: str | None = None
    campaign_label: str = Field(min_length=1)
    purpose: str | None = None
    notes: str | None = None
    target_strategy_families: list[str] = Field(default_factory=list)
    target_parameter_sets: list[str] = Field(default_factory=list)
    cycles: int = Field(default=1, ge=1)
    sleep_sec: float = Field(default=0.0, ge=0.0)
    market_limit: int | None = Field(default=None, ge=1)
    run_cadence_note: str | None = None
    experiment_label_prefix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolved_campaign_id(self) -> str:
        return self.campaign_id or str(uuid4())

    def resolved_parameter_sets(self) -> list[str]:
        if self.target_parameter_sets:
            return [str(item) for item in self.target_parameter_sets]
        return ["runtime_default"]

    def resolved_experiment_label(self, parameter_set_label: str) -> str:
        prefix = self.experiment_label_prefix or self.campaign_label
        return f"{prefix}:{parameter_set_label}"


class ResearchCampaignPreset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    preset_name: str
    description: str
    purpose: str | None = None
    notes: str | None = None
    target_strategy_families: list[str] = Field(default_factory=list)
    target_parameter_sets: list[str] = Field(default_factory=list)
    cycles: int = Field(default=1, ge=1)
    sleep_sec: float = Field(default=0.0, ge=0.0)
    market_limit: int | None = Field(default=None, ge=1)
    run_cadence_note: str | None = None
    experiment_label_prefix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


CAMPAIGN_PRESETS: dict[str, ResearchCampaignPreset] = {
    "single_market_focus": ResearchCampaignPreset(
        preset_name="single_market_focus",
        description="Collect concentrated evidence for single-market opportunities.",
        purpose="Collect repeatable single-market evidence.",
        notes="Use when the next collection budget should focus on paper-eligible single-market candidates.",
        target_strategy_families=["single_market_mispricing"],
        target_parameter_sets=["runtime_default"],
        cycles=2,
        market_limit=25,
        run_cadence_note="manual-burst-single-market",
        experiment_label_prefix="single-market-focus",
        metadata={"campaign_preset": "single_market_focus"},
    ),
    "single_market_broad_500": ResearchCampaignPreset(
        preset_name="single_market_broad_500",
        description="Collect broad single-market evidence across 500 active markets.",
        purpose="Increase single-market coverage breadth with runtime-default thresholds.",
        notes="Use for broader overnight evidence collection without changing downstream gating.",
        target_strategy_families=["single_market_mispricing"],
        target_parameter_sets=["runtime_default"],
        cycles=1,
        market_limit=500,
        run_cadence_note="manual-broad-single-market",
        experiment_label_prefix="single-market-broad-500",
        metadata={"campaign_preset": "single_market_broad_500"},
    ),
    "single_market_broad_1000": ResearchCampaignPreset(
        preset_name="single_market_broad_1000",
        description="Collect broad single-market evidence across 1000 active markets.",
        purpose="Maximize single-market coverage breadth with runtime-default thresholds.",
        notes="Use when 500-market coverage still produces no raw candidates and broader scope is needed.",
        target_strategy_families=["single_market_mispricing"],
        target_parameter_sets=["runtime_default"],
        cycles=1,
        market_limit=1000,
        run_cadence_note="manual-broad-single-market",
        experiment_label_prefix="single-market-broad-1000",
        metadata={"campaign_preset": "single_market_broad_1000"},
    ),
    "single_market_repeated_500": ResearchCampaignPreset(
        preset_name="single_market_repeated_500",
        description="Collect repeated single-market evidence across 500 active markets.",
        purpose="Accumulate repeated single-market samples at broader scope with runtime-default thresholds.",
        notes="Use for overnight repeated collection while preserving current thresholds and gating.",
        target_strategy_families=["single_market_mispricing"],
        target_parameter_sets=["runtime_default"],
        cycles=3,
        market_limit=500,
        run_cadence_note="manual-repeated-single-market",
        experiment_label_prefix="single-market-repeated-500",
        metadata={"campaign_preset": "single_market_repeated_500"},
    ),
    "cross_market_focus": ResearchCampaignPreset(
        preset_name="cross_market_focus",
        description="Collect research-only cross-market evidence and near-misses.",
        purpose="Collect cross-market qualification evidence.",
        notes="Useful when cross-market families need more qualified and rejection data.",
        target_strategy_families=["cross_market_constraint"],
        target_parameter_sets=["runtime_default"],
        cycles=2,
        market_limit=25,
        run_cadence_note="manual-burst-cross-market",
        experiment_label_prefix="cross-market-focus",
        metadata={"campaign_preset": "cross_market_focus"},
    ),
    "broad_evidence_collection": ResearchCampaignPreset(
        preset_name="broad_evidence_collection",
        description="Collect broad multi-family evidence across the current research universe.",
        purpose="Increase broad evidence coverage across active strategy families.",
        notes="Best for general sample accumulation when no single family is clearly preferred yet.",
        target_strategy_families=[
            "single_market_mispricing",
            "cross_market_constraint",
            "rebalancing",
            "external_belief",
        ],
        target_parameter_sets=["runtime_default"],
        cycles=2,
        market_limit=50,
        run_cadence_note="manual-broad-coverage",
        experiment_label_prefix="broad-evidence",
        metadata={"campaign_preset": "broad_evidence_collection"},
    ),
    "fillability_focused": ResearchCampaignPreset(
        preset_name="fillability_focused",
        description="Collect evidence that emphasizes fillability and shadow execution diagnostics.",
        purpose="Improve fillability and shadow-execution sample quality.",
        notes="Prefer when readiness is bottlenecked by shadow viability or execution gap evidence.",
        target_strategy_families=["single_market_mispricing", "cross_market_constraint"],
        target_parameter_sets=["runtime_default", "fillability_probe"],
        cycles=2,
        market_limit=20,
        run_cadence_note="manual-fillability-probe",
        experiment_label_prefix="fillability-focus",
        metadata={"campaign_preset": "fillability_focused"},
    ),
    "promotion_gap_collection": ResearchCampaignPreset(
        preset_name="promotion_gap_collection",
        description="Target families that look promising but still miss promotion evidence gates.",
        purpose="Close promotion evidence gaps for promising families.",
        notes="Use after reviewing promotion blockers and evidence target trackers.",
        target_strategy_families=["single_market_mispricing"],
        target_parameter_sets=["runtime_default", "strict"],
        cycles=2,
        market_limit=20,
        run_cadence_note="manual-promotion-gap",
        experiment_label_prefix="promotion-gap",
        metadata={"campaign_preset": "promotion_gap_collection"},
    ),
    "diversification_collection": ResearchCampaignPreset(
        preset_name="diversification_collection",
        description="Broaden parameter-set and time-window coverage for already-active families.",
        purpose="Reduce concentration risk in evidence collection.",
        notes="Use when coverage and stability diagnostics call for broader collection slices.",
        target_strategy_families=["single_market_mispricing", "cross_market_constraint"],
        target_parameter_sets=["runtime_default", "strict", "loose"],
        cycles=1,
        market_limit=15,
        run_cadence_note="manual-diversification",
        experiment_label_prefix="diversification",
        metadata={"campaign_preset": "diversification_collection"},
    ),
    "maker_rewarded_event_mm_v1": ResearchCampaignPreset(
        preset_name="maker_rewarded_event_mm_v1",
        description="Collect maker-MM evidence for the reward-eligible event cohort.",
        purpose="Accumulate paper-execution evidence for the MAKER_REWARDED_EVENT_MM_V1 strategy family.",
        notes=(
            "Uses experiment-context pass-through for all maker-MM controls. "
            "Does NOT use a parameter set override — maker-MM config is path-local "
            "and must not contaminate shared arb opportunity filtering. "
            "To extend the cohort, pass maker_mm_cohort via extra_context or update the class constant."
        ),
        target_strategy_families=["maker_rewarded_event_mm_v1"],
        target_parameter_sets=["runtime_default"],
        cycles=1,
        market_limit=200,
        run_cadence_note="manual-maker-mm-v1",
        experiment_label_prefix="maker-mm-v1",
        metadata={
            "campaign_preset": "maker_rewarded_event_mm_v1",
            "maker_mm_cohort": [
                "next-prime-minister-of-hungary",
                "netanyahu-out-before-2027",
                "balance-of-power-2026-midterms",
                "next-james-bond-actor-635",
            ],
            "maker_mm_min_edge": 0.005,
            "maker_mm_g6_margin": 1.25,
        },
    ),
}


def load_campaign_manifest(path: str | Path) -> ResearchCampaignManifest:
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    return ResearchCampaignManifest.model_validate(payload)


def list_campaign_presets() -> list[ResearchCampaignPreset]:
    return [CAMPAIGN_PRESETS[name] for name in sorted(CAMPAIGN_PRESETS)]


def build_campaign_manifest_from_preset(
    preset_name: str,
    *,
    campaign_label: str | None = None,
    cycles: int | None = None,
    market_limit: int | None = None,
    sleep_sec: float | None = None,
    target_parameter_sets: list[str] | None = None,
    target_strategy_families: list[str] | None = None,
    purpose: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResearchCampaignManifest:
    if preset_name not in CAMPAIGN_PRESETS:
        available = ", ".join(sorted(CAMPAIGN_PRESETS))
        raise ValueError(f"Unknown campaign preset '{preset_name}'. Available presets: {available}")
    preset = CAMPAIGN_PRESETS[preset_name]
    merged_metadata = dict(preset.metadata)
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata.setdefault("campaign_preset", preset_name)
    return ResearchCampaignManifest(
        campaign_label=campaign_label or preset.preset_name,
        purpose=purpose if purpose is not None else preset.purpose,
        notes=notes if notes is not None else preset.notes,
        target_strategy_families=list(target_strategy_families) if target_strategy_families is not None else list(preset.target_strategy_families),
        target_parameter_sets=list(target_parameter_sets) if target_parameter_sets is not None else list(preset.target_parameter_sets),
        cycles=cycles if cycles is not None else preset.cycles,
        sleep_sec=sleep_sec if sleep_sec is not None else preset.sleep_sec,
        market_limit=market_limit if market_limit is not None else preset.market_limit,
        run_cadence_note=preset.run_cadence_note,
        experiment_label_prefix=preset.experiment_label_prefix or campaign_label or preset.preset_name,
        metadata=merged_metadata,
    )


def save_campaign_manifest(manifest: ResearchCampaignManifest, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json", exclude_none=True)
    if manifest_path.suffix.lower() == ".json":
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path
