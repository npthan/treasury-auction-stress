"""Phase 7: smoke/correctness tests for
`treasury_auction_stress.evaluation.reporting7` -- every render function
must run end-to-end on a real (small) Phase 7 result set and produce
non-empty, well-formed markdown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from treasury_auction_stress.evaluation.data_loading import (
    load_phase6_inputs,
    validate_phase6_inputs,
)
from treasury_auction_stress.evaluation.protocol7 import Protocol7, load_protocol7
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
    render_point_results_report,
    render_probabilistic_results_report,
    render_stress_gate_report,
)
from treasury_auction_stress.evaluation.run_evaluation_phase7 import (
    build_modeling_frame,
    run_full_phase7_evaluation,
)
from treasury_auction_stress.features.feature_manifest import (
    DEFAULT_CONTRACT_PATH,
    load_contract,
    parse_entries,
)

REQUIRED_FILES = (
    "data/processed/feature_matrix_announcement.parquet",
    "data/processed/feature_matrix_pre_auction.parquet",
    "data/processed/auction_targets.parquet",
    "data/processed/feature_audit_announcement.parquet",
    "data/processed/feature_audit_pre_auction.parquet",
    "data/processed/pre_auction_information_updates.parquet",
)


@pytest.fixture(scope="module")
def small_results():
    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        pytest.skip(f"Phase 5 processed artifacts not present in this environment: {missing}")

    inputs = load_phase6_inputs()
    validate_phase6_inputs(inputs)
    entries = parse_entries(load_contract(DEFAULT_CONTRACT_PATH))
    full = load_protocol7()
    small = Protocol7(
        raw=full.raw,
        primary_target=full.primary_target,
        complete_test_years=(2015, 2016, 2017),
        provisional_test_year=full.provisional_test_year,
        initial_training_start=full.initial_training_start,
        initial_training_end=full.initial_training_end,
        quantile_levels=full.quantile_levels,
    )
    ann_frame = build_modeling_frame(inputs.announcement_matrix, inputs.target_table)
    pre_frame = build_modeling_frame(inputs.pre_auction_matrix, inputs.target_table)
    return run_full_phase7_evaluation(announcement_frame=ann_frame, pre_auction_frame=pre_frame, entries=entries, protocol7=small)


def test_point_report_renders_with_all_sections(small_results):
    preds = small_results.point_predictions
    pooled = build_point_pooled_table(preds)
    provisional = build_point_provisional_table(preds)
    by_year = build_point_by_year_table(preds)
    by_tenor = build_point_by_tenor_table(preds)
    paired_vs_frozen = build_paired_vs_baseline_table(preds, model_ids=("challenger_gbm_core", "challenger_shrinkage_tenor_reopening"), benchmark_model_id="baseline_recent_history_frozen")
    paired_vs_adaptive = build_paired_vs_baseline_table(preds, model_ids=("challenger_gbm_core", "challenger_shrinkage_tenor_reopening"), benchmark_model_id="baseline_recent_history_adaptive")
    paired_adaptive_vs_frozen = build_paired_vs_baseline_table(preds, model_ids=("baseline_recent_history_adaptive",), benchmark_model_id="baseline_recent_history_frozen")
    verdicts = apply_decision_rule(paired_vs_frozen, paired_vs_adaptive)

    report = render_point_results_report(
        predictions=preds,
        pooled=pooled,
        provisional=provisional,
        by_year=by_year,
        by_tenor=by_tenor,
        paired_vs_frozen=paired_vs_frozen,
        paired_vs_adaptive=paired_vs_adaptive,
        paired_adaptive_vs_frozen=paired_adaptive_vs_frozen,
        decision_rule_verdicts=verdicts,
        predictions_digest="testdigest",
    )
    assert "Phase 7 Point-Forecast Results" in report
    assert "challenger_gbm_core" in report
    assert "Decision-rule verdicts" in report
    assert len(report) > 500


def test_probabilistic_report_renders(small_results):
    preds = small_results.probabilistic_predictions
    pooled = build_probabilistic_pooled_table(preds)
    by_year = build_probabilistic_by_year_table(preds)
    by_tenor = build_probabilistic_by_tenor_table(preds)
    report = render_probabilistic_results_report(pooled=pooled, by_year=by_year, by_tenor=by_tenor, predictions_digest="testdigest")
    assert "Probabilistic Forecast Results" in report
    assert "coverage" in report.lower()


def test_stress_gate_report_renders(small_results):
    report = render_stress_gate_report(
        gate_decisions=small_results.gate_decisions,
        classifier_predictions=small_results.classifier_predictions,
        years_excluded=small_results.years_excluded_from_classifier,
    )
    assert "Stress-Event Gate" in report
    assert "operational" in report.lower() or "OPERATIONAL" in report


def test_build_coverage_trend_summary_detects_under_to_over_coverage_drift():
    import pandas as pd

    from treasury_auction_stress.evaluation.reporting7 import (
        build_coverage_trend_summary,
    )

    rows = []
    coverages = [0.70, 0.75, 0.72, 0.88, 0.90, 0.85, 0.95, 0.98, 1.00]
    for i, cov in enumerate(coverages):
        rows.append(
            {"cutoff_view": "announcement", "model_id": "quantile_recent_history_residual", "test_year": 2015 + i, "coverage_90%": cov}
        )
    by_year = pd.DataFrame(rows)
    trend = build_coverage_trend_summary(by_year, n_years_each_end=3)
    assert len(trend) == 1
    row = trend.iloc[0]
    assert row["early_years"] == "2015-2017"
    assert row["late_years"] == "2021-2023"
    assert row["early_mean_coverage"] == pytest.approx((0.70 + 0.75 + 0.72) / 3)
    assert row["late_mean_coverage"] == pytest.approx((0.95 + 0.98 + 1.00) / 3)
    assert row["drift"] > 0


def test_build_coverage_trend_summary_skips_groups_with_too_few_years():
    import pandas as pd

    from treasury_auction_stress.evaluation.reporting7 import (
        build_coverage_trend_summary,
    )

    by_year = pd.DataFrame(
        [{"cutoff_view": "announcement", "model_id": "m1", "test_year": 2015, "coverage_90%": 0.9}]
    )
    trend = build_coverage_trend_summary(by_year, n_years_each_end=3)
    assert trend.empty
