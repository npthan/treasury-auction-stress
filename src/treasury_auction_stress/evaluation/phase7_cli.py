"""Phase 7 CLI: run the full Phase 7 evaluation (adaptive recent-history
benchmark, GBM challenger, shrinkage challenger, probabilistic
quantile forecasts, and -- gated -- the stress-event classifier),
write out-of-sample tables to `data/processed/` (gitignored), and
regenerate the Phase 7 reports and figures.

    uv run python -m treasury_auction_stress.evaluation.phase7_cli

## Fail before write

Identical two-phase discipline to `treasury_auction_stress.evaluation.
phase6_cli`: Phase 1 (`_generate`) computes every artifact -- including
running the Phase 7 leakage-relevant test suite as a mandatory gate --
entirely in memory; any exception blocks every write. Phase 2 (writing)
only runs if Phase 1 raised nothing, and EACH INDIVIDUAL file write is
atomic (temp file + rename, via `_atomic_write_parquet`/
`_atomic_write_text`/`write_figure_atomically`) -- a single file can
never be observed half-written.

**This is per-file atomicity, not whole-run atomicity.** Phase 2 writes
roughly a dozen files (8 processed tables, 4 reports, 2 figures) in
sequence, one `rename()` at a time. If the process is killed or the
disk fails partway through this sequence (after some files' renames
have already committed but before the rest), the files already renamed
DO reflect this run's new content, and any file not yet reached retains
its PREVIOUS content -- there is no cross-file transaction, and no
attempt to roll the already-renamed files back. A reader could
therefore observe a MIXED set (some files from this run, some from an
earlier one) if it inspects `data/processed/`/`artifacts/` in the middle
of such a failure -- this is a real, disclosed limitation, not a claim
this CLI does not make. A single run's *own* internal consistency
(predictions vs. the reports and figures rendered from those SAME in-memory
predictions) is unaffected by this, because all of it is held in memory
and derived from one `_generate()` call before any file is written.
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
from treasury_auction_stress.evaluation.protocol7 import load_protocol7
from treasury_auction_stress.evaluation.reporting7 import (
    apply_decision_rule,
    build_paired_vs_baseline_table,
    build_point_by_tenor_table,
    build_point_by_year_table,
    build_point_pooled_table,
    build_point_provisional_table,
    build_probabilistic_by_tenor_table,
    build_probabilistic_by_year_table,
    build_probabilistic_pooled_table,
    render_phase7_leakage_audit_report,
    render_point_results_report,
    render_probabilistic_results_report,
    render_stress_gate_report,
)
from treasury_auction_stress.evaluation.run_evaluation_phase7 import (
    POINT_MODEL_IDS,
    PROBABILISTIC_MODEL_IDS,
    build_modeling_frame,
    run_full_phase7_evaluation,
)
from treasury_auction_stress.features.feature_manifest import (
    load_contract,
    parse_entries,
)
from treasury_auction_stress.features.feature_matrix import compute_content_digest
from treasury_auction_stress.visualization.phase7_plots import (
    plot_annual_mae_by_method,
    plot_coverage_by_year,
    write_figure_atomically,
)

DEFAULT_REPORTS_DIR = Path("artifacts")

PROCESSED_OUTPUT_FILES: tuple[str, ...] = (
    "phase_7_point_predictions.parquet",
    "phase_7_probabilistic_predictions.parquet",
    "phase_7_classifier_predictions.parquet",
    "phase_7_point_metrics_pooled.parquet",
    "phase_7_point_metrics_provisional.parquet",
    "phase_7_point_metrics_by_year.parquet",
    "phase_7_point_metrics_by_tenor.parquet",
    "phase_7_probabilistic_metrics_pooled.parquet",
)
REPORT_FILES: tuple[str, ...] = (
    "phase_7_point_forecast_results.md",
    "phase_7_probabilistic_results.md",
    "phase_7_stress_event_gate.md",
    "phase_7_leakage_audit.md",
)
FIGURE_FILES: tuple[str, ...] = (
    "phase7_annual_mae_by_method.png",
    "phase7_coverage_by_year.png",
)

# POINT_MODEL_IDS / PROBABILISTIC_MODEL_IDS are imported from
# run_evaluation_phase7 (the single source of truth) rather than
# re-declared here -- an acceptance-review fix for three independently
# hand-copied duplicates of the same lists that could otherwise drift.
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

# Deliberately EXCLUDES tests/test_phase7_cli.py itself -- see
# phase6_cli.py's identical comment: several of that file's own tests
# call `phase7_cli.run(...)` for real, which would re-trigger this
# exact gate, causing unbounded recursive subprocess spawning.
DEFAULT_PHASE7_TEST_FILES: tuple[str, ...] = (
    "tests/test_phase7_safe_as_of_lookback.py",
    "tests/test_phase7_adaptive_baseline.py",
    "tests/test_phase7_walk_forward.py",
    "tests/test_phase7_gbm.py",
    "tests/test_phase7_shrinkage.py",
    "tests/test_phase7_probabilistic_metrics.py",
    "tests/test_phase7_quantile_residual.py",
    "tests/test_phase7_logistic_classifier.py",
    "tests/test_phase7_stress_event.py",
    "tests/test_phase7_protocol.py",
    "tests/test_phase7_evaluation.py",
    "tests/test_phase7_leakage_adversarial.py",
    "tests/test_phase7_reporting.py",
    "tests/test_phase7_plots.py",
)


class Phase7GenerationError(RuntimeError):
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


def _run_gate_tests(test_files: tuple[str, ...] = DEFAULT_PHASE7_TEST_FILES) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_files], capture_output=True, text=True, cwd=Path.cwd(), check=False
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise Phase7GenerationError(
            f"Phase 7 gate test suite failed (exit code {result.returncode}) -- no output written:\n{output}"
        )
    return re.sub(r" in \d+\.\d+s( \(\d+:\d+:\d+\))?", "", output)


def _generate(*, processed_dir: Path, test_files: tuple[str, ...] = DEFAULT_PHASE7_TEST_FILES) -> dict:
    contract = load_contract()
    entries = parse_entries(contract)
    protocol7 = load_protocol7()

    print("Loading and validating Phase 5/6 artifacts...")
    inputs = load_phase6_inputs(processed_dir)
    validate_phase6_inputs(inputs)

    ann_frame = build_modeling_frame(inputs.announcement_matrix, inputs.target_table)
    pre_frame = build_modeling_frame(inputs.pre_auction_matrix, inputs.target_table)

    print("Running the full Phase 7 evaluation (point, probabilistic, stress-event gate)...")
    results = run_full_phase7_evaluation(
        announcement_frame=ann_frame, pre_auction_frame=pre_frame, entries=entries, protocol7=protocol7
    )
    print(f"  {len(results.point_predictions)} point predictions, {len(results.probabilistic_predictions)} probabilistic predictions")
    for cutoff_view, decision in results.gate_decisions.items():
        print(f"  stress-event gate ({cutoff_view}): {'PASSES' if decision.passes else 'FAILS'} ({decision.pooled_positive_count} pooled positives)")

    print("Computing point-forecast metrics tables...")
    point_pooled = build_point_pooled_table(results.point_predictions)
    point_provisional = build_point_provisional_table(results.point_predictions)
    point_by_year = build_point_by_year_table(results.point_predictions)
    point_by_tenor = build_point_by_tenor_table(results.point_predictions)
    challengers = ("challenger_gbm_core", "challenger_shrinkage_tenor_reopening")
    paired_vs_frozen = build_paired_vs_baseline_table(
        results.point_predictions, model_ids=challengers, benchmark_model_id="baseline_recent_history_frozen"
    )
    paired_vs_adaptive = build_paired_vs_baseline_table(
        results.point_predictions, model_ids=challengers, benchmark_model_id="baseline_recent_history_adaptive"
    )
    paired_adaptive_vs_frozen = build_paired_vs_baseline_table(
        results.point_predictions,
        model_ids=("baseline_recent_history_adaptive",),
        benchmark_model_id="baseline_recent_history_frozen",
    )
    decision_rule_verdicts = apply_decision_rule(paired_vs_frozen, paired_vs_adaptive)

    print("Computing probabilistic metrics tables...")
    prob_pooled = build_probabilistic_pooled_table(results.probabilistic_predictions)
    prob_by_year = build_probabilistic_by_year_table(results.probabilistic_predictions)
    prob_by_tenor = build_probabilistic_by_tenor_table(results.probabilistic_predictions)

    point_digest = compute_content_digest(results.point_predictions)
    prob_digest = compute_content_digest(results.probabilistic_predictions)

    print("Generating figures (in memory)...")
    fig1_bytes = plot_annual_mae_by_method(
        results.point_predictions,
        target_name="primary_dealer_share",
        cutoff_view="announcement",
        model_ids=POINT_MODEL_IDS,
        model_labels=POINT_MODEL_LABELS,
    )
    fig2_bytes = plot_coverage_by_year(
        results.probabilistic_predictions,
        model_ids=PROBABILISTIC_MODEL_IDS,
        model_labels=PROBABILISTIC_MODEL_LABELS,
        cutoff_view="announcement",
        interval_label="80%",
    )

    print("Running the Phase 7 gate test suite (mandatory; blocks all writes on failure)...")
    pytest_summary = _run_gate_tests(test_files)

    print("Rendering reports (in memory)...")
    point_report = render_point_results_report(
        predictions=results.point_predictions,
        pooled=point_pooled,
        provisional=point_provisional,
        by_year=point_by_year,
        by_tenor=point_by_tenor,
        paired_vs_frozen=paired_vs_frozen,
        paired_vs_adaptive=paired_vs_adaptive,
        paired_adaptive_vs_frozen=paired_adaptive_vs_frozen,
        decision_rule_verdicts=decision_rule_verdicts,
        predictions_digest=point_digest,
    )
    probabilistic_report = render_probabilistic_results_report(
        pooled=prob_pooled, by_year=prob_by_year, by_tenor=prob_by_tenor, predictions_digest=prob_digest
    )
    stress_report = render_stress_gate_report(
        gate_decisions=results.gate_decisions,
        classifier_predictions=results.classifier_predictions,
        years_excluded=results.years_excluded_from_classifier,
    )
    leakage_report = render_phase7_leakage_audit_report(
        pytest_summary=pytest_summary, gate_decisions=results.gate_decisions
    )

    return {
        "point_predictions": results.point_predictions,
        "probabilistic_predictions": results.probabilistic_predictions,
        "classifier_predictions": results.classifier_predictions,
        "point_pooled": point_pooled,
        "point_provisional": point_provisional,
        "point_by_year": point_by_year,
        "point_by_tenor": point_by_tenor,
        "prob_pooled": prob_pooled,
        "point_digest": point_digest,
        "prob_digest": prob_digest,
        "fig1_bytes": fig1_bytes,
        "fig2_bytes": fig2_bytes,
        "point_report": point_report,
        "probabilistic_report": probabilistic_report,
        "stress_report": stress_report,
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
        print(f"PHASE 7 GENERATION FAILED -- no output written: {exc}", file=sys.stderr)
        return 1

    print("Writing processed outputs...")
    processed_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(artifacts["point_predictions"], processed_dir / "phase_7_point_predictions.parquet")
    _atomic_write_parquet(artifacts["probabilistic_predictions"], processed_dir / "phase_7_probabilistic_predictions.parquet")
    _atomic_write_parquet(artifacts["classifier_predictions"], processed_dir / "phase_7_classifier_predictions.parquet")
    _atomic_write_parquet(artifacts["point_pooled"], processed_dir / "phase_7_point_metrics_pooled.parquet")
    _atomic_write_parquet(artifacts["point_provisional"], processed_dir / "phase_7_point_metrics_provisional.parquet")
    _atomic_write_parquet(artifacts["point_by_year"], processed_dir / "phase_7_point_metrics_by_year.parquet")
    _atomic_write_parquet(artifacts["point_by_tenor"], processed_dir / "phase_7_point_metrics_by_tenor.parquet")
    _atomic_write_parquet(artifacts["prob_pooled"], processed_dir / "phase_7_probabilistic_metrics_pooled.parquet")
    print(f"  point predictions content digest: {artifacts['point_digest']}")
    print(f"  probabilistic predictions content digest: {artifacts['prob_digest']}")

    print("Writing figures...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for filename, key in (
        ("phase7_annual_mae_by_method.png", "fig1_bytes"),
        ("phase7_coverage_by_year.png", "fig2_bytes"),
    ):
        path = write_figure_atomically(artifacts[key], figures_dir / filename)
        print(f"  wrote {path}")

    print("Writing reports...")
    reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(artifacts["point_report"], reports_dir / "phase_7_point_forecast_results.md")
    _atomic_write_text(artifacts["probabilistic_report"], reports_dir / "phase_7_probabilistic_results.md")
    _atomic_write_text(artifacts["stress_report"], reports_dir / "phase_7_stress_event_gate.md")
    _atomic_write_text(artifacts["leakage_report"], reports_dir / "phase_7_leakage_audit.md")

    print("Done.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
