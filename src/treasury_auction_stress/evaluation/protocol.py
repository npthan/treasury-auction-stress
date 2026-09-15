"""Loads `configs/phase_6_evaluation.yml`, the frozen Phase 6 protocol,
so every downstream module reads its decisions from one file instead
of a second, hand-copied set of constants. See that file's own header
comment for why it was frozen before any out-of-sample score existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from treasury_auction_stress.features.feature_manifest import (
    FeatureEntry,
    literal_predictor_names,
)
from treasury_auction_stress.features.feature_matrix import INCREMENTAL_SOURCE_FAMILY

DEFAULT_PROTOCOL_PATH = Path("configs/phase_6_evaluation.yml")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    family: str  # "baseline" | "linear"
    description: str
    estimator: str | None = None
    predictor_tier: str | None = None
    tuned: bool = False
    is_strong_baseline: bool = False
    sensitivity_only: bool = False
    recent_history_n_lookback: int | None = None


@dataclass(frozen=True)
class Protocol:
    raw: dict[str, Any]
    primary_target: str
    secondary_targets: tuple[str, ...]
    clipping: dict[str, tuple[float | None, float | None]]
    cutoff_views: tuple[str, ...]
    complete_test_years: tuple[int, ...]
    provisional_test_year: int
    initial_training_start: str
    initial_training_end: str
    models: tuple[ModelSpec, ...]
    structural_only_columns: tuple[str, ...]

    @property
    def all_targets(self) -> tuple[str, ...]:
        return (self.primary_target, *self.secondary_targets)

    @property
    def strong_baseline_model_id(self) -> str:
        for m in self.models:
            if m.is_strong_baseline:
                return m.id
        raise ValueError("protocol: no model has is_strong_baseline: true")


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> Protocol:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    clipping = {
        name: (spec.get("lower"), spec.get("upper")) for name, spec in raw["targets"]["clipping"].items()
    }

    models = []
    for m in raw["models"]:
        recent_history = m.get("recent_history")
        models.append(
            ModelSpec(
                id=m["id"],
                label=m["label"],
                family=m["family"],
                description=m.get("description", ""),
                estimator=m.get("estimator"),
                predictor_tier=m.get("predictor_tier"),
                tuned=bool(m.get("tuned", False)),
                is_strong_baseline=bool(m.get("is_strong_baseline", False)),
                sensitivity_only=bool(m.get("sensitivity_only", False)),
                recent_history_n_lookback=(recent_history or {}).get("n_lookback"),
            )
        )

    fold_schedule = raw["fold_schedule"]
    return Protocol(
        raw=raw,
        primary_target=raw["targets"]["primary"],
        secondary_targets=tuple(raw["targets"]["secondary"]),
        clipping=clipping,
        cutoff_views=tuple(raw["cutoff_views"]),
        complete_test_years=tuple(fold_schedule["complete_test_years"]),
        provisional_test_year=fold_schedule["provisional_test_year"],
        initial_training_start=fold_schedule["initial_training_span"]["start"],
        initial_training_end=fold_schedule["initial_training_span"]["end"],
        models=tuple(models),
        structural_only_columns=tuple(raw["predictor_tiers"]["structural_only"]["columns"]),
    )


def resolve_tier_feature_columns(
    protocol: Protocol, tier: str, entries: list[FeatureEntry]
) -> list[str]:
    """The literal predictor column names for one predictor tier.
    `entries` is the parsed `configs/phase_5_features.yml` contract
    (`treasury_auction_stress.features.feature_manifest.parse_entries`)
    -- `core`/`extended_sensitivity` are read directly from it so this
    project never maintains a second, hand-copied predictor list that
    could silently drift from the Phase 5 contract.
    """
    predictor_names = literal_predictor_names(entries, exclude_source_families=frozenset({INCREMENTAL_SOURCE_FAMILY}))
    if tier == "structural_only":
        return list(protocol.structural_only_columns)
    if tier == "core":
        return sorted(name for name, t in predictor_names.items() if t == "core")
    if tier == "extended_sensitivity":
        return sorted(predictor_names.keys())
    raise ValueError(f"unknown predictor tier {tier!r}")


def clip_series(values, target_name: str, protocol: Protocol):
    """Apply the pre-declared clipping policy for `target_name`. Never
    called conditionally on how the unclipped values look -- the
    bounds come only from the frozen protocol.
    """
    lower, upper = protocol.clipping[target_name]
    return values.clip(lower=lower, upper=upper)
