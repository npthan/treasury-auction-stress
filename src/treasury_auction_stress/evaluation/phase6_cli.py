"""Phase 6 CLI: run the full evaluation from the frozen Phase 5
artifacts, write the out-of-sample prediction table, fold manifest, and
metrics tables to `data/processed/` (gitignored), regenerate the three
Phase 6 reports and two figures, and print a summary.

    uv run python -m treasury_auction_stress.evaluation.phase6_cli

## Fail before write (acceptance-review discipline, mirrored from Phase 5)

`run` has two strictly ordered phases. Phase 1 (`_generate`) computes
every artifact -- including actually running the Phase 6 leakage-
relevant test suite as a gate -- entirely in memory; ANY exception
anywhere in this phase (a validation failure, a gate test failure, a
nonzero test-runner exit code) is caught once, at the top of `run`,
prints an error, and returns exit code 1 WITHOUT WRITING ANYTHING.
Phase 2 (writing) only ever runs if Phase 1 completed with no
exception, and every write is atomic (`_atomic_write_parquet`/
`_atomic_write_text`, and `phase6_plots`'s own atomic figure write) --
a failing run can never leave a partially-written or corrupted "final"
file, and a pre-existing valid file is never touched until its
replacement is fully written. See `tests/test_phase6_cli.py` for the
regression tests proving this directly (a deliberately-failing gate
leaves previously-published outputs byte-for-byte unchanged, and a
failing structural step writes nothing to a fresh directory).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from treasury_auction_stress.evaluation.data_loading import (
    DEFAULT_PROCESSED_DIR,
    load_phase6_inputs,
    validate_phase6_inputs,
)
from treasury_auction_stress.evaluation.dealer_absorption_audit import (
    summarize_regime_feature_risk,
)
from treasury_auction_stress.evaluation.fold_manifest import build_fold_manifest
from treasury_auction_stress.evaluation.metrics import (
    build_breakdown_metrics_table,
    build_pooled_metrics_table,
    build_provisional_only_metrics_table,
)
from treasury_auction_stress.evaluation.protocol import load_protocol
from treasury_auction_stress.evaluation.reporting import (
    render_baseline_results_report,
    render_evaluation_protocol_report,
    render_leakage_audit_report,
)
from treasury_auction_stress.evaluation.run_evaluation import (
    build_modeling_frame,
    run_full_evaluation,
)
from treasury_auction_stress.features.feature_manifest import (
    load_contract,
    parse_entries,
)
from treasury_auction_stress.features.feature_matrix import compute_content_digest
from treasury_auction_stress.visualization.phase6_plots import (
    plot_actual_vs_forecast_over_time,
    plot_annual_mae_by_method,
    write_figure_atomically,
)

DEFAULT_REPORTS_DIR = Path("artifacts")

PROCESSED_OUTPUT_FILES: tuple[str, ...] = (
    "phase_6_oos_predictions.parquet",
    "phase_6_fold_manifest.parquet",
    "phase_6_metrics_pooled.parquet",
    "phase_6_metrics_provisional.parquet",
    "phase_6_metrics_by_year.parquet",
    "phase_6_metrics_by_tenor.parquet",
    "phase_6_metrics_by_reopening.parquet",
)
REPORT_FILES: tuple[str, ...] = (
    "phase_6_evaluation_protocol.md",
    "phase_6_baseline_results.md",
    "phase_6_leakage_audit.md",
)
FIGURE_FILES: tuple[str, ...] = (
    "phase6_annual_mae_by_method.png",
    "phase6_actual_vs_forecast.png",
)

HEADLINE_MODEL_IDS = [
    "baseline_global_mean",
    "baseline_tenor_mean",
    "baseline_tenor_reopening_mean",
    "baseline_recent_history",
    "structural_ridge",
    "core_ridge",
    "core_elastic_net",
]
ALL_MODEL_IDS = [*HEADLINE_MODEL_IDS, "extended_ridge_sensitivity"]

#
# Deliberately EXCLUDES tests/test_phase6_cli.py: several of that
# file's own tests call `phase6_cli.run(...)` for real, which would
# invoke this exact gate again, which would re-run test_phase6_cli.py
# again -- unbounded recursive subprocess spawning. This was a real
# bug introduced and caught during the Phase 6 acceptance review (the
# CLI hung and never returned); test_phase6_cli.py is still run by the
# ordinary `uv run pytest` suite and by the acceptance review's own
# manual verification commands, just never as this in-process gate.
DEFAULT_PHASE6_TEST_FILES: tuple[str, ...] = (
    "tests/test_phase6_timing.py",
    "tests/test_phase6_baselines.py",
    "tests/test_phase6_linear_models.py",
    "tests/test_phase6_protocol.py",
    "tests/test_phase6_metrics.py",
    "tests/test_phase6_data_loading.py",
    "tests/test_phase6_evaluation.py",
    "tests/test_phase6_dealer_absorption_audit.py",
    "tests/test_phase6_fold_manifest.py",
)


class Phase6GenerationError(RuntimeError):
    """Raised anywhere in `_generate` -- guarantees `run` writes nothing."""


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _run_gate_tests(test_files: tuple[str, ...] = DEFAULT_PHASE6_TEST_FILES) -> str:
    """Actually run the Phase 6 leakage-relevant test files as a
    mandatory gate. Raises `Phase6GenerationError` (never returns) if
    the test runner exits nonzero -- `docs/project_rules.md`'s "no fabricated
    results" rule means this report can only ever reflect a real,
    passing run; it must never embed a failing run's output while
    still proceeding to publish artifacts.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_files], capture_output=True, text=True, cwd=Path.cwd(), check=False
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise Phase6GenerationError(
            f"Phase 6 gate test suite failed (exit code {result.returncode}) -- no output written:\n{output}"
        )
    # Strip wall-clock timing (e.g. "in 33.29s") so this report's content
    # digest depends only on the real pass/fail outcome, not how long the
    # run happened to take -- required for the Section 14 reproducibility
    # check (run twice, expect identical digests).
    return re.sub(r" in \d+\.\d+s( \(\d+:\d+:\d+\))?", "", output)


def _generate(*, processed_dir: Path, test_files: tuple[str, ...] = DEFAULT_PHASE6_TEST_FILES) -> dict:
    """Phase 1: compute every artifact and run every gate, entirely in
    memory -- no filesystem write of any Phase 6 output happens in this
    function. Raises on the first failure found.
    """
    protocol = load_protocol()
    contract = load_contract()
    entries = parse_entries(contract)

    print("Loading and validating Phase 5 artifacts...")
    inputs = load_phase6_inputs(processed_dir)
    validation_report = validate_phase6_inputs(inputs)
    print(f"  validated {validation_report['n_rows']} rows, 0 timing violations")

    ann_frame = build_modeling_frame(inputs.announcement_matrix, inputs.target_table)
    pre_frame = build_modeling_frame(inputs.pre_auction_matrix, inputs.target_table)

    print("Auditing Dealer Absorption Surprise regime-feature reuse risk...")
    regime_risk = summarize_regime_feature_risk(ann_frame)
    print(f"  {regime_risk['n_total_violations']} real availability violations found across both cutoffs, full 8-window (see protocol report)")

    print("Running the full evaluation (this takes ~30s)...")
    predictions = run_full_evaluation(
        announcement_frame=ann_frame, pre_auction_frame=pre_frame, entries=entries, protocol=protocol
    )
    print(f"  {len(predictions)} out-of-sample predictions produced")

    print("Building the fold manifest...")
    all_fold_years = [(y, False) for y in protocol.complete_test_years] + [(protocol.provisional_test_year, True)]
    fold_manifest = build_fold_manifest(
        frame_by_cutoff={"announcement": ann_frame, "pre_auction": pre_frame},
        fold_years_with_provisional_flag=all_fold_years,
    )

    print("Computing metrics tables...")
    pooled = build_pooled_metrics_table(
        predictions, target_names=protocol.all_targets, model_ids=ALL_MODEL_IDS, cutoff_views=protocol.cutoff_views
    )
    provisional_only = build_provisional_only_metrics_table(
        predictions, target_names=protocol.all_targets, model_ids=ALL_MODEL_IDS, cutoff_views=protocol.cutoff_views
    )
    by_year = build_breakdown_metrics_table(
        predictions,
        target_names=(protocol.primary_target,),
        model_ids=HEADLINE_MODEL_IDS,
        cutoff_views=protocol.cutoff_views,
        group_col="test_year",
        complete_years_only=False,
    )
    by_tenor = build_breakdown_metrics_table(
        predictions,
        target_names=(protocol.primary_target,),
        model_ids=HEADLINE_MODEL_IDS,
        cutoff_views=protocol.cutoff_views,
        group_col="tenor",
        complete_years_only=True,
    )
    by_reopening = build_breakdown_metrics_table(
        predictions,
        target_names=(protocol.primary_target,),
        model_ids=HEADLINE_MODEL_IDS,
        cutoff_views=protocol.cutoff_views,
        group_col="is_reopening",
        complete_years_only=True,
    )

    predictions_digest = compute_content_digest(predictions)

    print("Generating figures (in memory)...")
    fig1_bytes = plot_annual_mae_by_method(
        predictions,
        target_name=protocol.primary_target,
        cutoff_view="announcement",
        model_ids=HEADLINE_MODEL_IDS,
        model_labels={m.id: m.label for m in protocol.models},
    )
    fig2_bytes = plot_actual_vs_forecast_over_time(
        predictions,
        target_name=protocol.primary_target,
        cutoff_view="announcement",
        model_id="core_ridge",
        model_label="Ridge, core tier",
    )

    print("Running the Phase 6 gate test suite (mandatory; blocks all writes on failure)...")
    pytest_summary = _run_gate_tests(test_files)

    print("Rendering reports (in memory)...")
    protocol_report = render_evaluation_protocol_report(
        protocol=protocol,
        fold_manifest=fold_manifest,
        regime_risk=regime_risk,
        validation_report=validation_report,
        input_fingerprints=inputs.fingerprints,
    )
    baseline_report = render_baseline_results_report(
        predictions=predictions,
        pooled=pooled,
        provisional_only=provisional_only,
        by_year=by_year,
        by_tenor=by_tenor,
        by_reopening=by_reopening,
        protocol=protocol,
        headline_model_ids=HEADLINE_MODEL_IDS,
        predictions_digest=predictions_digest,
    )
    leakage_report = render_leakage_audit_report(
        validation_report=validation_report,
        regime_risk=regime_risk,
        pytest_summary=pytest_summary,
        input_fingerprints=inputs.fingerprints,
    )

    return {
        "predictions": predictions,
        "fold_manifest": fold_manifest,
        "pooled": pooled,
        "provisional_only": provisional_only,
        "by_year": by_year,
        "by_tenor": by_tenor,
        "by_reopening": by_reopening,
        "predictions_digest": predictions_digest,
        "fig1_bytes": fig1_bytes,
        "fig2_bytes": fig2_bytes,
        "protocol_report": protocol_report,
        "baseline_report": baseline_report,
        "leakage_report": leakage_report,
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, type=Path)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR, type=Path)
    args = parser.parse_args(argv)
    processed_dir: Path = args.processed_dir
    reports_dir: Path = args.reports_dir
    figures_dir = reports_dir / "figures"

    try:
        artifacts = _generate(processed_dir=processed_dir)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any failure here must block all writes
        print(f"PHASE 6 GENERATION FAILED -- no output written: {exc}", file=sys.stderr)
        return 1

    print("Writing processed outputs...")
    processed_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(artifacts["predictions"], processed_dir / "phase_6_oos_predictions.parquet")
    _atomic_write_parquet(artifacts["fold_manifest"], processed_dir / "phase_6_fold_manifest.parquet")
    _atomic_write_parquet(artifacts["pooled"], processed_dir / "phase_6_metrics_pooled.parquet")
    _atomic_write_parquet(artifacts["provisional_only"], processed_dir / "phase_6_metrics_provisional.parquet")
    _atomic_write_parquet(artifacts["by_year"], processed_dir / "phase_6_metrics_by_year.parquet")
    _atomic_write_parquet(artifacts["by_tenor"], processed_dir / "phase_6_metrics_by_tenor.parquet")
    _atomic_write_parquet(artifacts["by_reopening"], processed_dir / "phase_6_metrics_by_reopening.parquet")
    print(f"  predictions content digest: {artifacts['predictions_digest']}")

    print("Writing figures...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for filename, key in (
        ("phase6_annual_mae_by_method.png", "fig1_bytes"),
        ("phase6_actual_vs_forecast.png", "fig2_bytes"),
    ):
        path = write_figure_atomically(artifacts[key], figures_dir / filename)
        print(f"  wrote {path}")

    print("Writing reports...")
    reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(artifacts["protocol_report"], reports_dir / "phase_6_evaluation_protocol.md")
    _atomic_write_text(artifacts["baseline_report"], reports_dir / "phase_6_baseline_results.md")
    _atomic_write_text(artifacts["leakage_report"], reports_dir / "phase_6_leakage_audit.md")

    print("Done.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
