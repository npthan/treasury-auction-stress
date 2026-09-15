"""Phase 8 CLI: interpretation/diagnostics computed from the ACCEPTED
Phase 7 out-of-sample prediction tables, and the single JSON aggregate
the local dashboard reads.

    uv run python -m treasury_auction_stress.evaluation.phase8_cli

This module does not fit, tune, or re-score any model, and does not
touch Phase 5/6/7 inputs beyond reading their already-published output
tables. Nothing here can change a Phase 1-7 target definition, feature
eligibility, fold construction, model setting, threshold, prediction,
or accepted metric.

## Fail before write

Identical two-phase discipline to `phase6_cli`/`phase7_cli`: Phase 1
(`_generate`) computes every artifact -- including running the Phase 8
test suite as a mandatory gate -- entirely in memory; any exception
blocks every write. Phase 2 (`run`) writes only if Phase 1 raised
nothing, and each individual file write is atomic (temp file +
rename).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from treasury_auction_stress.evaluation.protocol7 import load_protocol7
from treasury_auction_stress.evaluation.reporting8 import (
    build_dashboard_data,
    build_phase8_tables,
    render_phase8_interpretation_report,
)
from treasury_auction_stress.features.feature_matrix import compute_content_digest

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_REPORTS_DIR = Path("artifacts")
DEFAULT_DASHBOARD_DATA_DIR = Path("dashboard/data")

REQUIRED_PHASE7_FILES: tuple[str, ...] = (
    "phase_7_point_predictions.parquet",
    "phase_7_probabilistic_predictions.parquet",
    "phase_7_classifier_predictions.parquet",
)

PHASE7_REGENERATION_COMMAND = "uv run python -m treasury_auction_stress.evaluation.phase7_cli"

REPORT_FILE = "phase_8_interpretation.md"
DASHBOARD_DATA_FILE = "phase8_dashboard_data.json"

POINT_MODEL_LABELS = {
    "baseline_recent_history_frozen": "Recent-history (frozen)",
    "baseline_recent_history_adaptive": "Recent-history (adaptive)",
    "challenger_gbm_core": "GBM challenger (core tier)",
    "challenger_shrinkage_tenor_reopening": "Shrinkage (tenor x reopening)",
}
PROBABILISTIC_MODEL_LABELS = {
    "quantile_recent_history_residual": "Residual quantiles (recent-history)",
    "quantile_gbm_core": "Quantile GBM (core tier)",
}

# Deliberately EXCLUDES tests/test_phase8_cli.py itself -- identical
# reasoning to phase6_cli/phase7_cli's own gate-list exclusion: that
# file's own tests call `phase8_cli.run(...)` for real, which would
# re-trigger this exact gate, causing unbounded recursive subprocess
# spawning.
DEFAULT_PHASE8_TEST_FILES: tuple[str, ...] = ("tests/test_phase8_interpretation.py",)


class Phase8GenerationError(RuntimeError):
    """Raised anywhere in `_generate` -- guarantees `run` writes nothing."""


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _require_phase7_files(processed_dir: Path) -> None:
    missing = [name for name in REQUIRED_PHASE7_FILES if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Phase 8: missing required Phase 7 artifact(s) in {processed_dir}: {missing}. "
            f"Regenerate Phase 7 first with:\n    {PHASE7_REGENERATION_COMMAND}\n"
            "Phase 8 never rebuilds these differently or re-scores any model itself."
        )


def _load_phase7_outputs(processed_dir: Path) -> dict[str, pd.DataFrame]:
    _require_phase7_files(processed_dir)
    return {
        "point_predictions": pd.read_parquet(processed_dir / "phase_7_point_predictions.parquet"),
        "probabilistic_predictions": pd.read_parquet(processed_dir / "phase_7_probabilistic_predictions.parquet"),
        "classifier_predictions": pd.read_parquet(processed_dir / "phase_7_classifier_predictions.parquet"),
    }


def _run_gate_tests(test_files: tuple[str, ...] = DEFAULT_PHASE8_TEST_FILES) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_files], capture_output=True, text=True, cwd=Path.cwd(), check=False
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise Phase8GenerationError(
            f"Phase 8 gate test suite failed (exit code {result.returncode}) -- no output written:\n{output}"
        )
    return re.sub(r" in \d+\.\d+s( \(\d+:\d+:\d+\))?", "", output)


def _generate(*, processed_dir: Path, test_files: tuple[str, ...] = DEFAULT_PHASE8_TEST_FILES) -> dict:
    protocol7 = load_protocol7()

    print("Loading accepted Phase 7 outputs...")
    inputs = _load_phase7_outputs(processed_dir)

    point_digest = compute_content_digest(inputs["point_predictions"])
    prob_digest = compute_content_digest(inputs["probabilistic_predictions"])
    print(f"  point predictions content digest: {point_digest}")
    print(f"  probabilistic predictions content digest: {prob_digest}")

    print("Computing Phase 8 interpretation tables (no re-fitting, no re-scoring)...")
    tables = build_phase8_tables(
        point_predictions=inputs["point_predictions"],
        probabilistic_predictions=inputs["probabilistic_predictions"],
        classifier_predictions=inputs["classifier_predictions"],
    )

    print("Running the Phase 8 gate test suite (mandatory; blocks all writes on failure)...")
    pytest_summary = _run_gate_tests(test_files)
    print(pytest_summary)

    print("Rendering the interpretation report and dashboard data (in memory)...")
    report = render_phase8_interpretation_report(
        tables=tables,
        point_digest=point_digest,
        probabilistic_digest=prob_digest,
        classifier_row_count=len(inputs["classifier_predictions"]),
    )
    dashboard_data = build_dashboard_data(
        tables=tables,
        point_digest=point_digest,
        probabilistic_digest=prob_digest,
        point_model_labels=POINT_MODEL_LABELS,
        probabilistic_model_labels=PROBABILISTIC_MODEL_LABELS,
        complete_test_years=tuple(protocol7.complete_test_years),
        provisional_year=protocol7.provisional_test_year,
    )
    dashboard_json = json.dumps(dashboard_data, indent=2, sort_keys=True) + "\n"

    return {
        "report": report,
        "dashboard_json": dashboard_json,
        "point_digest": point_digest,
        "prob_digest": prob_digest,
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, type=Path)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR, type=Path)
    parser.add_argument("--dashboard-data-dir", default=DEFAULT_DASHBOARD_DATA_DIR, type=Path)
    args = parser.parse_args(argv)

    try:
        artifacts = _generate(processed_dir=args.processed_dir)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any failure here must block all writes
        print(f"PHASE 8 GENERATION FAILED -- no output written: {exc}", file=sys.stderr)
        return 1

    print("Writing report...")
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(artifacts["report"], args.reports_dir / REPORT_FILE)

    print("Writing dashboard data...")
    args.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(artifacts["dashboard_json"], args.dashboard_data_dir / DASHBOARD_DATA_FILE)

    print("Done.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
