"""Loads the small set of structural constants from
`configs/phase_7_protocol.yml` that the Phase 7 evaluation driver reads
at runtime (fold schedule, primary target, quantile levels) -- exactly
the same "loop-over" structural elements
`treasury_auction_stress.evaluation.protocol.load_protocol` reads from
`configs/phase_6_evaluation.yml`. Every other Phase 7 decision (model
hyperparameter grids, the stress percentile, min-count thresholds) is a
literal constant in its own implementation module
(`models.gbm`, `models.shrinkage`, `evaluation.stress_event`, ...) --
this mirrors Phase 6's own convention exactly:
`models.linear_models.RIDGE_ALPHA_GRID` is a hardcoded constant, not
re-parsed from YAML on every call either. The YAML is the frozen,
audited DECISION record; the code constants are the single
IMPLEMENTATION of that decision -- both must agree, and this project's
existing tests (e.g. `tests/test_phase6_protocol.py`) are the
established pattern for catching drift, not a runtime YAML re-read of
every constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROTOCOL_PATH = Path("configs/phase_7_protocol.yml")


@dataclass(frozen=True)
class Protocol7:
    raw: dict[str, Any]
    primary_target: str
    complete_test_years: tuple[int, ...]
    provisional_test_year: int
    initial_training_start: str
    initial_training_end: str
    quantile_levels: tuple[float, ...]


def load_protocol7(path: Path = DEFAULT_PROTOCOL_PATH) -> Protocol7:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    fold_schedule_reused = raw["fold_schedule"]
    return Protocol7(
        raw=raw,
        primary_target=raw["targets"]["primary"],
        complete_test_years=tuple(fold_schedule_reused["complete_test_years"]),
        provisional_test_year=fold_schedule_reused["provisional_test_year"],
        initial_training_start=fold_schedule_reused["initial_training_span"]["start"],
        initial_training_end=fold_schedule_reused["initial_training_span"]["end"],
        quantile_levels=tuple(raw["probabilistic"]["quantile_levels"]),
    )
