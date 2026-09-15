"""Tests for `treasury_auction_stress.evaluation.phase8_interpretation`
and `reporting8`. Unit tests use small, hand-constructed frames (no
network, no real data dependency); a final class of tests pins the
Phase 8 report/dashboard's headline numbers against the real, accepted
Phase 7 output tables if present in this environment (skipped
otherwise, matching the project's existing convention for real-data
regression tests, e.g.
`tests/test_phase6_dealer_absorption_audit.py::test_real_data_full_window_both_cutoffs_matches_known_counts`).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.evaluation.phase8_interpretation import (
    ADAPTIVE_MODEL_ID,
    FROZEN_MODEL_ID,
    build_error_distribution_table,
    build_paired_comparison_by_group,
    build_stress_metrics_by_year_table,
    compute_quantile_clip_fraction,
    select_case_studies,
    year_block_bootstrap_ci,
)
from treasury_auction_stress.evaluation.reporting7 import (
    build_paired_vs_baseline_table,
    build_point_by_year_table,
    build_point_pooled_table,
    build_probabilistic_pooled_table,
)
from treasury_auction_stress.evaluation.reporting8 import (
    build_dashboard_data,
    build_phase8_tables,
    describe_broad_or_concentrated,
)
from treasury_auction_stress.evaluation.stress_event import compute_classifier_metrics

PROCESSED_DIR = Path("data/processed")


def _has_real_phase7_outputs() -> bool:
    return all(
        (PROCESSED_DIR / name).exists()
        for name in (
            "phase_7_point_predictions.parquet",
            "phase_7_probabilistic_predictions.parquet",
            "phase_7_classifier_predictions.parquet",
        )
    )


# ============================================================
# Synthetic fixtures
# ============================================================


def _point_row(key, cutoff_view, model_id, test_year, tenor, is_reopening, actual, forecast, is_provisional=False):
    return {
        "auction_key": key,
        "cutoff_view": cutoff_view,
        "target_name": "primary_dealer_share",
        "model_id": model_id,
        "test_year": test_year,
        "is_provisional": is_provisional,
        "tenor": tenor,
        "is_reopening": is_reopening,
        "actual": actual,
        "forecast": forecast,
    }


@pytest.fixture
def synthetic_point_predictions() -> pd.DataFrame:
    rows = [
        # 2015, 2-Year, not reopening: adaptive improves a lot
        _point_row("a1", "announcement", FROZEN_MODEL_ID, 2015, "2-Year", False, 0.30, 0.50),
        _point_row("a1", "announcement", ADAPTIVE_MODEL_ID, 2015, "2-Year", False, 0.30, 0.32),
        # 2015, 5-Year, reopening: adaptive worse
        _point_row("a2", "announcement", FROZEN_MODEL_ID, 2015, "5-Year", True, 0.40, 0.41),
        _point_row("a2", "announcement", ADAPTIVE_MODEL_ID, 2015, "5-Year", True, 0.40, 0.55),
        # 2016, 2-Year, not reopening: adaptive mildly better (typical case)
        _point_row("a3", "announcement", FROZEN_MODEL_ID, 2016, "2-Year", False, 0.20, 0.25),
        _point_row("a3", "announcement", ADAPTIVE_MODEL_ID, 2016, "2-Year", False, 0.20, 0.22),
        # 2016, 5-Year, reopening: adaptive mildly better
        _point_row("a4", "announcement", FROZEN_MODEL_ID, 2016, "5-Year", True, 0.35, 0.40),
        _point_row("a4", "announcement", ADAPTIVE_MODEL_ID, 2016, "5-Year", True, 0.35, 0.37),
        # provisional year, excluded from complete-year comparisons by default
        _point_row("a5", "announcement", FROZEN_MODEL_ID, 2026, "2-Year", False, 0.25, 0.30, is_provisional=True),
        _point_row("a5", "announcement", ADAPTIVE_MODEL_ID, 2026, "2-Year", False, 0.25, 0.26, is_provisional=True),
    ]
    # Mirror everything for the pre_auction cutoff view too (identical values is fine for these tests).
    pre_rows = [dict(r, cutoff_view="pre_auction") for r in rows]
    return pd.DataFrame(rows + pre_rows)


@pytest.fixture
def synthetic_probabilistic_predictions() -> pd.DataFrame:
    rows = [
        {
            "auction_key": "a1",
            "cutoff_view": "announcement",
            "model_id": "quantile_recent_history_residual",
            "test_year": 2015,
            "is_provisional": False,
            "tenor": "2-Year",
            "actual": 0.30,
            "q0.05": 0.0,
            "q0.10": 0.05,
            "q0.25": 0.15,
            "q0.50": 0.30,
            "q0.75": 0.45,
            "q0.90": 0.55,
            "q0.95": 0.60,
        },
        {
            "auction_key": "a2",
            "cutoff_view": "announcement",
            "model_id": "quantile_recent_history_residual",
            "test_year": 2015,
            "is_provisional": False,
            "tenor": "5-Year",
            "actual": 0.40,
            "q0.05": 0.10,
            "q0.10": 0.20,
            "q0.25": 0.30,
            "q0.50": 0.40,
            "q0.75": 0.50,
            "q0.90": 0.60,
            "q0.95": 1.0,
        },
    ]
    return pd.DataFrame(rows)


# ============================================================
# build_paired_comparison_by_group
# ============================================================


def test_build_paired_comparison_by_group_by_tenor(synthetic_point_predictions):
    table = build_paired_comparison_by_group(synthetic_point_predictions, cutoff_view="announcement", group_col="tenor")
    table = table.set_index("tenor")
    # 2-Year: a1 (frozen ae=0.20*100=20, adaptive ae=0.02*100=2, diff=-18) and a3 (frozen ae=5, adaptive ae=2, diff=-3)
    assert table.loc["2-Year", "n"] == 2
    assert table.loc["2-Year", "mean_paired_diff_model_minus_benchmark"] == pytest.approx((-18 + -3) / 2)
    assert table.loc["2-Year", "n_model_better"] == 2
    # 5-Year: a2 (frozen ae=1, adaptive ae=15, diff=+14) and a4 (frozen ae=5, adaptive ae=2, diff=-3)
    assert table.loc["5-Year", "n"] == 2
    assert table.loc["5-Year", "n_benchmark_better"] == 1
    assert table.loc["5-Year", "n_model_better"] == 1


def test_build_paired_comparison_by_group_by_reopening(synthetic_point_predictions):
    table = build_paired_comparison_by_group(synthetic_point_predictions, cutoff_view="announcement", group_col="is_reopening")
    table = table.set_index("is_reopening")
    assert set(table.index) == {False, True}
    assert table.loc[False, "n"] == 2
    assert table.loc[True, "n"] == 2


def test_build_paired_comparison_by_group_excludes_provisional_by_default(synthetic_point_predictions):
    table = build_paired_comparison_by_group(synthetic_point_predictions, cutoff_view="announcement", group_col="tenor")
    assert table["n"].sum() == 4  # a1, a2, a3, a4 only -- a5 (2026, provisional) excluded


def test_build_paired_comparison_by_group_raises_on_mismatched_actuals(synthetic_point_predictions):
    poisoned = synthetic_point_predictions.copy()
    mask = (poisoned["auction_key"] == "a1") & (poisoned["model_id"] == ADAPTIVE_MODEL_ID) & (poisoned["cutoff_view"] == "announcement")
    poisoned.loc[mask, "actual"] = 0.99
    with pytest.raises(AssertionError, match="actual values disagree"):
        build_paired_comparison_by_group(poisoned, cutoff_view="announcement", group_col="tenor")


# ============================================================
# build_error_distribution_table
# ============================================================


def test_build_error_distribution_table_signed_bias(synthetic_point_predictions):
    table = build_error_distribution_table(
        synthetic_point_predictions, cutoff_view="announcement", model_ids=(FROZEN_MODEL_ID, ADAPTIVE_MODEL_ID)
    )
    table = table.set_index("model_id")
    # Frozen: errors (forecast-actual)*100 = [20, 1, 5, 5] -> all positive bias
    assert table.loc[FROZEN_MODEL_ID, "mean_signed_error_bias"] == pytest.approx(np.mean([20, 1, 5, 5]))
    assert table.loc[FROZEN_MODEL_ID, "n"] == 4


def test_build_error_distribution_table_by_group(synthetic_point_predictions):
    table = build_error_distribution_table(
        synthetic_point_predictions,
        cutoff_view="announcement",
        model_ids=(FROZEN_MODEL_ID,),
        group_col="tenor",
    )
    assert set(table["tenor"]) == {"2-Year", "5-Year"}
    assert (table["n"] == 2).all()


# ============================================================
# select_case_studies
# ============================================================


def test_select_case_studies_picks_expected_extremes(synthetic_point_predictions):
    cases = select_case_studies(synthetic_point_predictions, cutoff_view="announcement")
    by_rule = cases.set_index("case_selection_rule")
    # a1: frozen ae=20, adaptive ae=2, improvement=+18 -- the largest improvement
    assert by_rule.loc["largest_adaptive_improvement", "auction_key"] == "a1"
    # a2: frozen ae=1, adaptive ae=15, improvement=-14 -- the largest deterioration
    assert by_rule.loc["largest_adaptive_deterioration", "auction_key"] == "a2"
    # known-at-cutoff vs. learned-after-settlement columns both present and distinct
    row = by_rule.loc["largest_adaptive_improvement"]
    assert row["known_at_cutoff__frozen_forecast_pct"] == pytest.approx(50.0)
    assert row["known_at_cutoff__adaptive_forecast_pct"] == pytest.approx(32.0)
    assert row["learned_after_settlement__actual_pct"] == pytest.approx(30.0)


def test_select_case_studies_is_deterministic(synthetic_point_predictions):
    first = select_case_studies(synthetic_point_predictions, cutoff_view="announcement")
    second = select_case_studies(synthetic_point_predictions, cutoff_view="announcement")
    pd.testing.assert_frame_equal(first, second)


def test_select_case_studies_deterministic_under_row_shuffle(synthetic_point_predictions):
    shuffled = synthetic_point_predictions.sample(frac=1.0, random_state=7).reset_index(drop=True)
    original = select_case_studies(synthetic_point_predictions, cutoff_view="announcement")
    reshuffled = select_case_studies(shuffled, cutoff_view="announcement")
    pd.testing.assert_frame_equal(original, reshuffled)


def test_select_case_studies_empty_for_unknown_cutoff(synthetic_point_predictions):
    result = select_case_studies(synthetic_point_predictions, cutoff_view="nonexistent")
    assert result.empty


# ============================================================
# compute_quantile_clip_fraction
# ============================================================


def test_compute_quantile_clip_fraction_pooled(synthetic_probabilistic_predictions):
    result = compute_quantile_clip_fraction(
        synthetic_probabilistic_predictions,
        cutoff_view="announcement",
        model_id="quantile_recent_history_residual",
        quantile_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    )
    assert result.iloc[0]["n"] == 2
    # a1 touches 0.0 at q0.05; a2 touches 1.0 at q0.95 -- both rows clipped
    assert result.iloc[0]["n_rows_touching_clip_bound"] == 2
    assert result.iloc[0]["fraction_rows_touching_clip_bound"] == pytest.approx(1.0)


def test_compute_quantile_clip_fraction_by_group(synthetic_probabilistic_predictions):
    result = compute_quantile_clip_fraction(
        synthetic_probabilistic_predictions,
        cutoff_view="announcement",
        model_id="quantile_recent_history_residual",
        quantile_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        group_col="tenor",
    )
    result = result.set_index("tenor")
    assert result.loc["2-Year", "n_rows_touching_clip_bound"] == 1
    assert result.loc["5-Year", "n_rows_touching_clip_bound"] == 1


def test_compute_quantile_clip_fraction_empty_for_missing_model():
    empty = pd.DataFrame(columns=["auction_key", "cutoff_view", "model_id", "is_provisional", "q0.05"])
    result = compute_quantile_clip_fraction(
        empty, cutoff_view="announcement", model_id="nope", quantile_levels=(0.05,)
    )
    assert result.empty


# ============================================================
# year_block_bootstrap_ci
# ============================================================


def test_year_block_bootstrap_ci_is_deterministic():
    values = pd.Series([-2.0, -1.0, 0.5, -3.0, 1.0, -0.5, -2.5, 0.2, -1.2, -0.8, -1.5], index=range(2015, 2026))
    first = year_block_bootstrap_ci(values, n_boot=2000, seed=42)
    second = year_block_bootstrap_ci(values, n_boot=2000, seed=42)
    assert first == second
    assert first["n_years"] == 11
    assert first["ci_low"] <= first["point_estimate"] <= first["ci_high"]


def test_year_block_bootstrap_ci_different_seed_can_differ():
    values = pd.Series([-2.0, -1.0, 0.5, -3.0, 1.0, -0.5, -2.5, 0.2, -1.2, -0.8, -1.5], index=range(2015, 2026))
    a = year_block_bootstrap_ci(values, n_boot=2000, seed=1)
    b = year_block_bootstrap_ci(values, n_boot=2000, seed=2)
    assert a["point_estimate"] == b["point_estimate"]  # point estimate does not depend on resampling
    # CI bounds need not be identical across seeds (though they may coincidentally be close)


def test_year_block_bootstrap_ci_too_few_years():
    result = year_block_bootstrap_ci(pd.Series([1.0]), n_boot=100, seed=42)
    assert result["n_years"] == 1
    assert result["ci_low"] is None
    assert result["ci_high"] is None


def test_year_block_bootstrap_ci_empty():
    result = year_block_bootstrap_ci(pd.Series([], dtype=float), n_boot=100, seed=42)
    assert result["n_years"] == 0
    assert result["point_estimate"] is None


# ============================================================
# build_stress_metrics_by_year_table
# ============================================================


def test_build_stress_metrics_by_year_table_synthetic():
    df = pd.DataFrame(
        {
            "auction_key": [f"k{i}" for i in range(6)],
            "cutoff_view": ["announcement"] * 6,
            "test_year": [2020, 2020, 2020, 2021, 2021, 2021],
            "is_provisional": [False] * 6,
            "predicted_proba": [0.1, 0.8, 0.3, 0.2, 0.9, 0.05],
            "predicted_label": [False, True, False, False, True, False],
            "actual_label": [False, True, False, False, True, False],
            "train_prevalence": [0.1] * 6,
        }
    )
    table = build_stress_metrics_by_year_table(df, cutoff_view="announcement")
    assert set(table["test_year"]) == {2020, 2021}
    assert (table["n"] == 3).all()


def test_build_stress_metrics_by_year_table_empty():
    empty = pd.DataFrame()
    assert build_stress_metrics_by_year_table(empty, cutoff_view="announcement").empty


# ============================================================
# build_phase8_tables / build_dashboard_data end-to-end on synthetic data
# ============================================================


def test_build_phase8_tables_end_to_end_smoke(synthetic_point_predictions, synthetic_probabilistic_predictions):
    classifier_predictions = pd.DataFrame(
        {
            "auction_key": ["a1", "a2"],
            "cutoff_view": ["announcement", "announcement"],
            "test_year": [2015, 2015],
            "is_provisional": [False, False],
            "predicted_proba": [0.2, 0.6],
            "predicted_label": [False, True],
            "actual_label": [False, True],
            "train_prevalence": [0.1, 0.1],
        }
    )
    tables = build_phase8_tables(
        point_predictions=synthetic_point_predictions,
        probabilistic_predictions=synthetic_probabilistic_predictions,
        classifier_predictions=classifier_predictions,
    )
    assert "paired_by_tenor_announcement" in tables
    assert "case_studies_announcement" in tables
    assert "adaptive_vs_frozen_bootstrap" in tables

    dashboard_data = build_dashboard_data(
        tables=tables,
        point_digest="deadbeef",
        probabilistic_digest="cafef00d",
        point_model_labels={FROZEN_MODEL_ID: "Frozen", ADAPTIVE_MODEL_ID: "Adaptive"},
        probabilistic_model_labels={"quantile_recent_history_residual": "Residual quantiles"},
        complete_test_years=(2015, 2016),
        provisional_year=2026,
    )
    assert dashboard_data["meta"]["cutoff_views"] == ["announcement", "pre_auction"]
    assert dashboard_data["warnings"]["classifier_poorly_calibrated"]
    # JSON-serializable end to end (no NaN/NaT/numpy scalar leaks)
    import json

    json.dumps(dashboard_data)


# ============================================================
# describe_broad_or_concentrated -- the by-year win/loss narrative must
# be DERIVED from the paired-by-year table, never hand-typed (a
# hand-typed version once claimed the frozen benchmark won 2025 when
# the table showed a 48-29 adaptive win that year).
# ============================================================


def _expand_year_ranges(text: str) -> set[int]:
    """`describe_broad_or_concentrated` compresses consecutive years into
    `"2018-2023"` for readability -- a naive `\\d{4}` scan would miss
    every year strictly between the two endpoints of a range, so this
    expands each `YYYY` or `YYYY-YYYY` token into the full set of years
    it names before a test compares it against the table.
    """
    years: set[int] = set()
    for start, end in re.findall(r"(\d{4})(?:-(\d{4}))?", text):
        years.update(range(int(start), int(end) + 1) if end else {int(start)})
    return years


def _paired_year_row(cutoff_view, test_year, n_model_better, n_benchmark_better, n_tied=0, is_provisional=False):
    return {
        "cutoff_view": cutoff_view,
        "model_id": ADAPTIVE_MODEL_ID,
        "benchmark_model_id": FROZEN_MODEL_ID,
        "test_year": test_year,
        "is_provisional": is_provisional,
        "n": n_model_better + n_benchmark_better + n_tied,
        "unit": "percentage points",
        "mae_model": 1.0,
        "mae_benchmark": 2.0,
        "mean_paired_diff_model_minus_benchmark": -1.0,
        "n_model_better": n_model_better,
        "n_benchmark_better": n_benchmark_better,
        "n_tied": n_tied,
    }


def test_describe_broad_or_concentrated_names_the_correct_years():
    # Adaptive wins 2015-2016 and 2018; frozen wins 2017. Identical at
    # both cutoffs. A provisional 2019 row (frozen "wins" 40-10) must be
    # excluded entirely -- if the narrative counted it, 2019 would
    # wrongly appear in the frozen clause.
    rows = []
    for cv in ("announcement", "pre_auction"):
        rows += [
            _paired_year_row(cv, 2015, n_model_better=45, n_benchmark_better=21),
            _paired_year_row(cv, 2016, n_model_better=38, n_benchmark_better=28),
            _paired_year_row(cv, 2017, n_model_better=32, n_benchmark_better=34),
            _paired_year_row(cv, 2018, n_model_better=43, n_benchmark_better=23),
            _paired_year_row(cv, 2019, n_model_better=10, n_benchmark_better=40, is_provisional=True),
        ]
    paired_by_year = pd.DataFrame(rows)

    sentence = describe_broad_or_concentrated(paired_by_year)

    assert "2015-2016" in sentence or ("2015" in sentence and "2016" in sentence)
    assert "2018" in sentence
    before, _, after = sentence.partition("frozen benchmark wins more auctions in")
    adaptive_years = _expand_year_ranges(before)
    frozen_years = _expand_year_ranges(after)
    assert frozen_years == {2017}
    assert adaptive_years == {2015, 2016, 2018}
    assert 2019 not in adaptive_years and 2019 not in frozen_years  # provisional, must be excluded


def test_describe_broad_or_concentrated_disjoint_year_sets():
    # No year the table says the model won may also appear in the
    # sentence's frozen-won clause, and vice versa -- the exact
    # invariant a hand-typed narrative can silently violate.
    rows = []
    for cv in ("announcement", "pre_auction"):
        rows += [
            _paired_year_row(cv, 2021, n_model_better=48, n_benchmark_better=29),
            _paired_year_row(cv, 2024, n_model_better=34, n_benchmark_better=43),
            _paired_year_row(cv, 2025, n_model_better=48, n_benchmark_better=29),
        ]
    paired_by_year = pd.DataFrame(rows)

    sentence = describe_broad_or_concentrated(paired_by_year)
    before, _, after = sentence.partition("frozen benchmark wins more auctions in")
    adaptive_years = _expand_year_ranges(before)
    frozen_years = _expand_year_ranges(after)
    assert adaptive_years.isdisjoint(frozen_years)
    assert frozen_years == {2024}
    assert 2025 in adaptive_years  # the specific case this regression test exists to guard


# ============================================================
# Real-data regression pins (skipped if this environment lacks the
# gitignored data/processed Phase 7 outputs)
# ============================================================


@pytest.mark.skipif(not _has_real_phase7_outputs(), reason="Phase 7 processed outputs not present in this environment")
class TestHeadlineNumbersPinnedAgainstAcceptedPhase7Outputs:
    """Pins the exact headline numbers this project's Phase 8 report and
    dashboard state publicly. If a future change to Phase 7 (or an
    accidental regeneration with different code) moves any of these,
    this test fails loudly rather than letting the report/dashboard
    silently drift from the underlying accepted artifacts."""

    @staticmethod
    @pytest.fixture(scope="class")
    def point_predictions():
        return pd.read_parquet(PROCESSED_DIR / "phase_7_point_predictions.parquet")

    @staticmethod
    @pytest.fixture(scope="class")
    def probabilistic_predictions():
        return pd.read_parquet(PROCESSED_DIR / "phase_7_probabilistic_predictions.parquet")

    @staticmethod
    @pytest.fixture(scope="class")
    def classifier_predictions():
        return pd.read_parquet(PROCESSED_DIR / "phase_7_classifier_predictions.parquet")

    def test_frozen_vs_adaptive_pooled_mae(self, point_predictions):
        pooled = build_point_pooled_table(point_predictions).set_index(["cutoff_view", "model_id"])
        assert pooled.loc[("announcement", FROZEN_MODEL_ID), "mae"] == pytest.approx(5.104, abs=1e-3)
        assert pooled.loc[("announcement", ADAPTIVE_MODEL_ID), "mae"] == pytest.approx(4.216, abs=1e-3)
        assert pooled.loc[("pre_auction", FROZEN_MODEL_ID), "mae"] == pytest.approx(5.104, abs=1e-3)
        assert pooled.loc[("pre_auction", ADAPTIVE_MODEL_ID), "mae"] == pytest.approx(4.217, abs=1e-3)

    def test_challengers_do_not_beat_recent_history(self, point_predictions):
        pooled = build_point_pooled_table(point_predictions).set_index(["cutoff_view", "model_id"])
        adaptive_mae = pooled.loc[("announcement", ADAPTIVE_MODEL_ID), "mae"]
        assert pooled.loc[("announcement", "challenger_gbm_core"), "mae"] > adaptive_mae
        assert pooled.loc[("announcement", "challenger_shrinkage_tenor_reopening"), "mae"] > adaptive_mae

    def test_stress_classifier_worse_than_no_skill(self, classifier_predictions):
        for cutoff_view, expected_brier in (("announcement", 0.13528), ("pre_auction", 0.11870)):
            subset = classifier_predictions.loc[
                (classifier_predictions["cutoff_view"] == cutoff_view) & (~classifier_predictions["is_provisional"])
            ]
            metrics = compute_classifier_metrics(subset)
            assert metrics["n_positive"] == 53
            assert metrics["n"] == 860
            assert metrics["brier_score"] == pytest.approx(expected_brier, abs=5e-5)
            assert metrics["no_skill_brier_score"] == pytest.approx(0.05937, abs=5e-5)
            assert metrics["worse_than_no_skill"] is True

    def test_probabilistic_90_coverage_and_width(self, probabilistic_predictions):
        pooled = build_probabilistic_pooled_table(probabilistic_predictions).set_index(
            ["cutoff_view", "model_id", "scope"]
        )
        row = pooled.loc[("announcement", "quantile_recent_history_residual", "complete_years")]
        assert row["coverage_90%"] == pytest.approx(0.922, abs=1e-3)
        assert row["width_90%"] == pytest.approx(23.048, abs=1e-2)

    def test_adaptive_vs_frozen_paired_by_year_matches_accepted_report(self, point_predictions):
        paired = build_paired_vs_baseline_table(
            point_predictions, model_ids=(ADAPTIVE_MODEL_ID,), benchmark_model_id=FROZEN_MODEL_ID
        )
        row2015 = paired.loc[(paired["cutoff_view"] == "announcement") & (paired["test_year"] == 2015)].iloc[0]
        assert row2015["mae_model"] == pytest.approx(5.275, abs=1e-3)
        assert row2015["mae_benchmark"] == pytest.approx(7.336, abs=1e-3)

    def test_2020_regime_shift_year_matches_research_report(self, point_predictions):
        by_year = build_point_by_year_table(point_predictions)
        row = by_year.loc[
            (by_year["cutoff_view"] == "announcement")
            & (by_year["test_year"] == 2020)
            & (by_year["model_id"] == ADAPTIVE_MODEL_ID)
        ].iloc[0]
        assert row["mae"] == pytest.approx(3.784, abs=1e-3)
        frozen_row = by_year.loc[
            (by_year["cutoff_view"] == "announcement")
            & (by_year["test_year"] == 2020)
            & (by_year["model_id"] == FROZEN_MODEL_ID)
        ].iloc[0]
        assert frozen_row["mae"] == pytest.approx(5.086, abs=1e-3)

    def test_adaptive_beats_frozen_in_9_of_11_complete_years(self, point_predictions):
        paired = build_paired_vs_baseline_table(
            point_predictions, model_ids=(ADAPTIVE_MODEL_ID,), benchmark_model_id=FROZEN_MODEL_ID
        )
        complete = paired.loc[(paired["cutoff_view"] == "announcement") & (~paired["is_provisional"])]
        adaptive_wins = complete.loc[complete["mae_model"] < complete["mae_benchmark"], "test_year"].tolist()
        frozen_wins = complete.loc[complete["mae_model"] > complete["mae_benchmark"], "test_year"].tolist()
        assert len(complete) == 11
        assert len(adaptive_wins) == 9
        assert sorted(frozen_wins) == [2017, 2024]

    def test_narrative_matches_paired_by_year_table_on_real_data(self, point_predictions):
        """End-to-end guard against the exact defect found in acceptance
        review: `artifacts/phase_8_interpretation.md` once stated the
        frozen benchmark won 2025 in prose while its own by-year table
        showed a 48-29 ADAPTIVE win. Recomputes win/loss years directly
        from the real accepted table (independently of
        `describe_broad_or_concentrated`'s own internals) and asserts
        the rendered sentence names every year on the correct side and
        never on the wrong one.
        """
        paired_by_year = build_paired_vs_baseline_table(
            point_predictions, model_ids=(ADAPTIVE_MODEL_ID,), benchmark_model_id=FROZEN_MODEL_ID
        )
        complete = paired_by_year.loc[
            (paired_by_year["cutoff_view"] == "announcement") & (~paired_by_year["is_provisional"])
        ]
        expected_adaptive_wins = {
            int(y) for y in complete.loc[complete["n_model_better"] > complete["n_benchmark_better"], "test_year"]
        }
        expected_frozen_wins = {
            int(y) for y in complete.loc[complete["n_benchmark_better"] > complete["n_model_better"], "test_year"]
        }
        assert expected_adaptive_wins.isdisjoint(expected_frozen_wins)

        sentence = describe_broad_or_concentrated(paired_by_year)
        before, _, after = sentence.partition("frozen benchmark wins more auctions in")
        adaptive_years_in_text = _expand_year_ranges(before)
        frozen_years_in_text = _expand_year_ranges(after)

        assert expected_frozen_wins <= frozen_years_in_text
        assert expected_frozen_wins.isdisjoint(adaptive_years_in_text)
        assert expected_adaptive_wins <= adaptive_years_in_text
        assert expected_adaptive_wins.isdisjoint(frozen_years_in_text)
        # the specific 2025 mislabeling this test exists to catch
        assert 2025 in expected_adaptive_wins
        assert 2025 not in frozen_years_in_text

    def test_paired_by_tenor_extremes_match_research_report(self, point_predictions):
        table = build_paired_comparison_by_group(point_predictions, cutoff_view="announcement", group_col="tenor")
        table = table.set_index("tenor")
        assert table["mean_paired_diff_model_minus_benchmark"].idxmin() == "20-Year"
        assert table.loc["20-Year", "mean_paired_diff_model_minus_benchmark"] == pytest.approx(-1.884, abs=1e-3)
        assert table["mean_paired_diff_model_minus_benchmark"].abs().idxmin() == "7-Year"
        assert table.loc["7-Year", "mean_paired_diff_model_minus_benchmark"] == pytest.approx(-0.150, abs=1e-3)

    def test_paired_by_reopening_matches_research_report(self, point_predictions):
        table = build_paired_comparison_by_group(point_predictions, cutoff_view="announcement", group_col="is_reopening")
        table = table.set_index("is_reopening")
        assert table.loc[False, "mean_paired_diff_model_minus_benchmark"] == pytest.approx(-0.884, abs=1e-2)
        assert table.loc[True, "mean_paired_diff_model_minus_benchmark"] == pytest.approx(-0.899, abs=1e-2)

    def test_case_studies_match_research_report(self, point_predictions):
        cases = select_case_studies(point_predictions, cutoff_view="announcement").set_index("case_selection_rule")
        improved = cases.loc["largest_adaptive_improvement"]
        assert improved["auction_key"] == "912810SR0_2020-07-22"
        assert improved["tenor"] == "20-Year"
        assert improved["known_at_cutoff__frozen_forecast_pct"] == pytest.approx(38.248, abs=1e-2)
        assert improved["known_at_cutoff__adaptive_forecast_pct"] == pytest.approx(23.237, abs=1e-2)
        assert improved["learned_after_settlement__actual_pct"] == pytest.approx(21.228, abs=1e-2)

        deteriorated = cases.loc["largest_adaptive_deterioration"]
        assert deteriorated["auction_key"] == "9128283N8_2017-12-26"
        assert deteriorated["tenor"] == "2-Year"
        assert bool(deteriorated["is_reopening"]) is False

    def test_clip_fraction_matches_research_report(self, probabilistic_predictions):
        from treasury_auction_stress.evaluation.phase8_interpretation import (
            compute_quantile_clip_fraction as _clip,
        )

        pooled = _clip(
            probabilistic_predictions,
            cutoff_view="announcement",
            model_id="quantile_recent_history_residual",
            quantile_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        )
        assert pooled.iloc[0]["fraction_rows_touching_clip_bound"] == pytest.approx(0.085, abs=1e-3)

        by_tenor = _clip(
            probabilistic_predictions,
            cutoff_view="announcement",
            model_id="quantile_recent_history_residual",
            quantile_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
            group_col="tenor",
        ).set_index("tenor")
        assert by_tenor.loc["20-Year", "fraction_rows_touching_clip_bound"] == pytest.approx(0.529, abs=1e-3)

        by_year = _clip(
            probabilistic_predictions,
            cutoff_view="announcement",
            model_id="quantile_recent_history_residual",
            quantile_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
            group_col="test_year",
        ).set_index("test_year")
        assert by_year.loc[2025, "fraction_rows_touching_clip_bound"] == pytest.approx(0.440, abs=1e-3)
