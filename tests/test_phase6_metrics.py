from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import r2_score

from treasury_auction_stress.evaluation.metrics import (
    build_paired_model_vs_benchmark_by_year,
    compute_error_metrics,
    n_excluded,
    r2_vs_fold_own_mean,
    r2_vs_strong_baseline,
    summarize_pooled,
    unit_label,
)


def test_unit_label():
    assert unit_label("primary_dealer_share") == "percentage points"
    assert unit_label("direct_bidder_share") == "percentage points"
    assert unit_label("bid_to_cover_ratio") == "ratio units"


def test_compute_error_metrics_share_target_scales_to_percentage_points():
    df = pd.DataFrame({"actual": [0.30, 0.40], "forecast": [0.32, 0.35]})
    metrics = compute_error_metrics(df, target_name="primary_dealer_share")
    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(3.5)  # (|0.02|+|0.05|)/2 * 100
    assert metrics["bias"] == pytest.approx((0.02 - 0.05) / 2 * 100)
    assert metrics["unit"] == "percentage points"


def test_compute_error_metrics_ratio_target_stays_in_native_units():
    df = pd.DataFrame({"actual": [2.5, 3.0], "forecast": [2.7, 2.9]})
    metrics = compute_error_metrics(df, target_name="bid_to_cover_ratio")
    assert metrics["mae"] == pytest.approx((0.2 + 0.1) / 2)
    assert metrics["unit"] == "ratio units"


def test_compute_error_metrics_empty_returns_none_not_crash():
    metrics = compute_error_metrics(pd.DataFrame({"actual": [], "forecast": []}), target_name="primary_dealer_share")
    assert metrics["n"] == 0
    assert metrics["mae"] is None


def test_r2_vs_fold_own_mean_matches_sklearn():
    df = pd.DataFrame({"actual": [0.1, 0.2, 0.3, 0.4], "forecast": [0.15, 0.18, 0.33, 0.36]})
    expected = r2_score(df["actual"], df["forecast"])
    assert r2_vs_fold_own_mean(df) == pytest.approx(expected)


def test_r2_vs_strong_baseline_correctness():
    model_df = pd.DataFrame(
        {"auction_key": ["a", "b", "c"], "actual": [0.3, 0.4, 0.5], "forecast": [0.31, 0.39, 0.52]}
    )
    baseline_df = pd.DataFrame(
        {"auction_key": ["a", "b", "c"], "actual": [0.3, 0.4, 0.5], "forecast": [0.35, 0.45, 0.55]}
    )
    sse_model = ((model_df["forecast"] - model_df["actual"]) ** 2).sum()
    sse_baseline = ((baseline_df["forecast"] - baseline_df["actual"]) ** 2).sum()
    expected = 1.0 - sse_model / sse_baseline
    result = r2_vs_strong_baseline(model_df, baseline_df)
    assert result == pytest.approx(expected)
    assert result > 0  # model beats the baseline here


def test_r2_vs_strong_baseline_raises_on_mismatched_keys():
    model_df = pd.DataFrame({"auction_key": ["a", "b"], "actual": [0.3, 0.4], "forecast": [0.3, 0.4]})
    baseline_df = pd.DataFrame({"auction_key": ["a", "c"], "actual": [0.3, 0.5], "forecast": [0.3, 0.5]})
    with pytest.raises(AssertionError):
        r2_vs_strong_baseline(model_df, baseline_df)


def _predictions_fixture() -> pd.DataFrame:
    rows = []
    for year, is_prov in ((2015, False), (2016, False), (2026, True)):
        for i in range(5):
            rows.append(
                {
                    "auction_key": f"K{year}_{i}",
                    "cutoff_view": "announcement",
                    "target_name": "primary_dealer_share",
                    "model_id": "core_ridge",
                    "test_year": year,
                    "is_provisional": is_prov,
                    "actual": 0.3 + 0.01 * i,
                    "forecast": 0.3 + 0.01 * i + 0.02,
                }
            )
            rows.append(
                {
                    "auction_key": f"K{year}_{i}",
                    "cutoff_view": "announcement",
                    "target_name": "primary_dealer_share",
                    "model_id": "baseline_tenor_reopening_mean",
                    "test_year": year,
                    "is_provisional": is_prov,
                    "actual": 0.3 + 0.01 * i,
                    "forecast": 0.3 + 0.01 * i + 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_pooled_excludes_provisional_by_default():
    preds = _predictions_fixture()
    pooled = summarize_pooled(
        preds, target_name="primary_dealer_share", cutoff_view="announcement", model_id="core_ridge", include_provisional=False
    )
    assert pooled["n"] == 10  # 2015+2016, 5 rows each -- 2026 excluded


def test_summarize_pooled_can_include_provisional():
    preds = _predictions_fixture()
    pooled = summarize_pooled(
        preds, target_name="primary_dealer_share", cutoff_view="announcement", model_id="core_ridge", include_provisional=True
    )
    assert pooled["n"] == 15


def test_summarize_pooled_r2_vs_strong_baseline_is_none_for_the_baseline_itself():
    preds = _predictions_fixture()
    pooled = summarize_pooled(
        preds,
        target_name="primary_dealer_share",
        cutoff_view="announcement",
        model_id="baseline_tenor_reopening_mean",
        include_provisional=False,
    )
    assert pooled["r2_vs_strong_baseline"] is None


def test_n_excluded():
    attempted = pd.DataFrame({"a": range(10)})
    scored = pd.DataFrame({"a": range(8)})
    result = n_excluded(attempted, scored)
    assert result == {"n_attempted": 10, "n_scored": 8, "n_excluded": 2, "reason": "missing actual target value"}


def _paired_fixture(years=(2015, 2016), n_per_year=4, provisional_year=2026) -> pd.DataFrame:
    rows = []
    for year in (*years, provisional_year):
        for i in range(n_per_year):
            actual = 0.3 + 0.02 * i
            rows.append(
                {
                    "auction_key": f"K{year}_{i}",
                    "cutoff_view": "announcement",
                    "target_name": "primary_dealer_share",
                    "model_id": "core_ridge",
                    "test_year": year,
                    "is_provisional": year == provisional_year,
                    "actual": actual,
                    "forecast": actual + 0.01,  # model AE = 0.01 * 100 = 1.0pp
                }
            )
            rows.append(
                {
                    "auction_key": f"K{year}_{i}",
                    "cutoff_view": "announcement",
                    "target_name": "primary_dealer_share",
                    "model_id": "baseline_recent_history",
                    "test_year": year,
                    "is_provisional": year == provisional_year,
                    "actual": actual,
                    "forecast": actual + 0.03,  # benchmark AE = 0.03 * 100 = 3.0pp -- model wins
                }
            )
    return pd.DataFrame(rows)


def test_build_paired_model_vs_benchmark_by_year_correctness_and_units():
    preds = _paired_fixture()
    result = build_paired_model_vs_benchmark_by_year(
        preds,
        target_name="primary_dealer_share",
        cutoff_view="announcement",
        model_ids=["core_ridge"],
        benchmark_model_id="baseline_recent_history",
    )
    complete = result.loc[~result["is_provisional"]]
    assert set(complete["test_year"]) == {2015, 2016}
    assert (complete["n"] == 4).all()
    assert complete["mae_model"].round(6).eq(1.0).all()
    assert complete["mae_benchmark"].round(6).eq(3.0).all()
    assert complete["mean_paired_diff_model_minus_benchmark"].round(6).eq(-2.0).all()
    assert (complete["n_model_better"] == 4).all()
    assert (complete["n_benchmark_better"] == 0).all()
    assert (complete["unit"] == "percentage points").all()

    provisional = result.loc[result["is_provisional"]]
    assert set(provisional["test_year"]) == {2026}
    assert len(provisional) == 1  # kept as its own row, never merged with complete years


def test_build_paired_model_vs_benchmark_by_year_raises_on_key_set_mismatch():
    preds = _paired_fixture()
    # Drop one benchmark row for 2015 so the key sets no longer match.
    drop_mask = (
        (preds["model_id"] == "baseline_recent_history") & (preds["test_year"] == 2015) & (preds["auction_key"] == "K2015_0")
    )
    corrupted = preds.loc[~drop_mask]
    with pytest.raises(AssertionError, match="key sets disagree"):
        build_paired_model_vs_benchmark_by_year(
            corrupted,
            target_name="primary_dealer_share",
            cutoff_view="announcement",
            model_ids=["core_ridge"],
            benchmark_model_id="baseline_recent_history",
        )


def test_build_paired_model_vs_benchmark_by_year_raises_on_actual_value_mismatch():
    preds = _paired_fixture()
    mismatch_mask = (
        (preds["model_id"] == "baseline_recent_history") & (preds["test_year"] == 2015) & (preds["auction_key"] == "K2015_0")
    )
    corrupted = preds.copy()
    corrupted.loc[mismatch_mask, "actual"] = 999.0  # same auction/target must never disagree on 'actual'
    with pytest.raises(AssertionError, match="actual values disagree"):
        build_paired_model_vs_benchmark_by_year(
            corrupted,
            target_name="primary_dealer_share",
            cutoff_view="announcement",
            model_ids=["core_ridge"],
            benchmark_model_id="baseline_recent_history",
        )


def test_build_paired_model_vs_benchmark_by_year_raises_on_non_finite_prediction():
    preds = _paired_fixture()
    nan_mask = (preds["model_id"] == "core_ridge") & (preds["test_year"] == 2015) & (preds["auction_key"] == "K2015_0")
    corrupted = preds.copy()
    corrupted.loc[nan_mask, "forecast"] = float("inf")
    with pytest.raises(AssertionError, match="non-finite"):
        build_paired_model_vs_benchmark_by_year(
            corrupted,
            target_name="primary_dealer_share",
            cutoff_view="announcement",
            model_ids=["core_ridge"],
            benchmark_model_id="baseline_recent_history",
        )
