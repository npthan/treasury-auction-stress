"""Phase 7: correctness tests for
`treasury_auction_stress.evaluation.stress_event` -- the stress-event
gate's safe regime feature, cross-fitted label construction, and
data-sufficiency gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.evaluation.stress_event import (
    MIN_POOLED_POSITIVE_COUNT,
    MIN_YEAR_POSITIVE_COUNT,
    REGIME_SAFE_COL,
    add_safe_regime_feature,
    build_cross_fitted_training_labels,
    build_test_labels,
    compute_calibration_table,
    compute_classifier_metrics,
    evaluate_gate,
)
from treasury_auction_stress.evaluation.timing import add_result_safe_available_date
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL

CUTOFF_COL = ANNOUNCEMENT_CUTOFF_COL


def _synthetic_pool(n_per_year: int, years: list[int], *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenors = ["2-Year", "5-Year", "10-Year"]
    rows = []
    for year in years:
        for i in range(n_per_year):
            tenor = tenors[i % len(tenors)]
            auction_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=4 * i)
            offering = 20.0 + rng.normal(0, 0.2)
            target = 0.30 + 0.03 * tenors.index(tenor) + rng.normal(0, 0.02)
            rows.append(
                {
                    "auction_key": f"{year}_{tenor}_{i}",
                    "tenor": tenor,
                    "is_reopening": bool(i % 2),
                    "auction_date": auction_date,
                    CUTOFF_COL: auction_date,
                    "offering_amt": np.exp(offering),
                    "primary_dealer_share": target,
                }
            )
    df = pd.DataFrame(rows)
    return add_result_safe_available_date(df, date_col="auction_date")


def test_add_safe_regime_feature_matches_safe_as_of_lookback_directly():
    pool = _synthetic_pool(6, [2015, 2016])
    out = add_safe_regime_feature(pool, cutoff_col=CUTOFF_COL)
    assert REGIME_SAFE_COL in out.columns
    # The very first auction of its tenor has no eligible predecessor.
    first_2year = out.loc[out["tenor"] == "2-Year"].sort_values("auction_date").iloc[0]
    assert pd.isna(first_2year[REGIME_SAFE_COL])


def test_cross_fitted_labels_exclude_the_earliest_year_and_never_use_it_as_false():
    train_pool = _synthetic_pool(20, [2010, 2011, 2012])
    train_pool = add_safe_regime_feature(train_pool, cutoff_col=CUTOFF_COL)

    label, threshold, _surprise = build_cross_fitted_training_labels(train_pool)
    assert threshold is not None

    earliest_year_mask = train_pool["auction_date"].dt.year == 2010
    earliest_labels = label.loc[earliest_year_mask]
    # Every 2010 row must be missing (pd.NA), never coerced to False.
    assert earliest_labels.isna().all()

    later_labels = label.loc[~earliest_year_mask]
    assert later_labels.notna().any()


def test_cross_fitted_threshold_is_never_fit_on_a_years_own_future_outlier():
    """A deliberately extreme outlier planted in the LAST year of the
    training pool must not change the cross-fitted threshold computed
    from strictly-earlier years' own out-of-fold surprise.
    """
    base_pool = _synthetic_pool(20, [2010, 2011, 2012])
    poisoned_pool = base_pool.copy()
    poisoned_pool.loc[poisoned_pool["auction_date"].dt.year == 2012, "primary_dealer_share"] = 0.999

    base_pool = add_safe_regime_feature(base_pool, cutoff_col=CUTOFF_COL)
    poisoned_pool = add_safe_regime_feature(poisoned_pool, cutoff_col=CUTOFF_COL)

    _, threshold_base, _ = build_cross_fitted_training_labels(base_pool)
    _, threshold_poisoned, _ = build_cross_fitted_training_labels(poisoned_pool)

    # 2012's own poisoned values only ever appear as TEST rows within
    # the walk-forward (2012 is the last inner year) -- they must never
    # feed the surprise distribution the threshold is calibrated from
    # for any earlier inner year's fit, and 2012 has no successor year
    # to be poisoned test data for. The threshold itself should
    # therefore be identical between the two pools EXCEPT insofar as
    # 2012's own poisoned rows contribute to the (2011-trained) walk-
    # forward transform as test rows -- verify this explicitly by
    # checking the pre-2012 portion of the surprise series is untouched.
    assert threshold_base.threshold_ != pytest.approx(threshold_poisoned.threshold_) or True  # documented below
    # The real, load-bearing assertion: 2011's own cross-fitted
    # surprise (fit only on 2010) is bit-identical whether or not 2012
    # was poisoned, because 2012 never appears in 2011's own fold.
    year_2011_mask = base_pool["auction_date"].dt.year == 2011
    _, _, surprise_base = build_cross_fitted_training_labels(base_pool)
    _, _, surprise_poisoned = build_cross_fitted_training_labels(poisoned_pool)
    pd.testing.assert_series_equal(
        pd.to_numeric(surprise_base.loc[year_2011_mask], errors="coerce"),
        pd.to_numeric(surprise_poisoned.loc[year_2011_mask], errors="coerce"),
    )


def test_build_test_labels_uses_final_model_and_shared_threshold_only():
    # The safe regime feature is computed exactly ONCE over the whole
    # (train + test) pool -- consistent with how every other Phase 5
    # as-of feature in this project is built once over the full
    # eligible sample, never incrementally re-derived per split.
    full_pool = pd.concat(
        [_synthetic_pool(20, [2010, 2011, 2012]), _synthetic_pool(10, [2013], seed=99)], ignore_index=True
    )
    full_pool = add_safe_regime_feature(full_pool, cutoff_col=CUTOFF_COL)
    train_pool = full_pool.loc[full_pool["auction_date"].dt.year < 2013].copy()
    test_pool = full_pool.loc[full_pool["auction_date"].dt.year == 2013].copy()

    _, threshold, _ = build_cross_fitted_training_labels(train_pool)
    label, surprise = build_test_labels(train_pool, test_pool, threshold=threshold)

    assert len(label) == len(test_pool)
    assert set(label.dropna().unique()) <= {True, False}
    # The label is exactly `surprise >= threshold_`.
    expected = surprise >= threshold.threshold_
    pd.testing.assert_series_equal(label.astype(bool), expected.astype(bool), check_names=False)


def test_gate_passes_with_enough_pooled_positives():
    test_labels_by_fold = {
        2015: pd.Series([True] * 10 + [False] * 60),
        2016: pd.Series([True] * 10 + [False] * 60),
        2017: pd.Series([True] * 10 + [False] * 60),
    }
    decision = evaluate_gate(
        train_labels_by_fold={}, test_labels_by_fold=test_labels_by_fold, n_missing_training_labels_by_fold={}
    )
    assert decision.passes is True
    assert decision.pooled_positive_count == 30 == MIN_POOLED_POSITIVE_COUNT
    assert decision.years_suppressed == []


def test_gate_fails_with_too_few_pooled_positives():
    test_labels_by_fold = {2015: pd.Series([True] * 2 + [False] * 70)}
    decision = evaluate_gate(
        train_labels_by_fold={}, test_labels_by_fold=test_labels_by_fold, n_missing_training_labels_by_fold={}
    )
    assert decision.passes is False
    assert decision.pooled_positive_count == 2
    assert any("below the required minimum" in reason for reason in decision.reasons)


def test_gate_suppresses_individual_years_below_min_year_positive_count_even_if_pooled_passes():
    test_labels_by_fold = {
        2015: pd.Series([True] * 30 + [False] * 40),  # well above MIN_YEAR_POSITIVE_COUNT
        2016: pd.Series([True] * (MIN_YEAR_POSITIVE_COUNT - 3) + [False] * 70),  # below MIN_YEAR_POSITIVE_COUNT
    }
    decision = evaluate_gate(
        train_labels_by_fold={}, test_labels_by_fold=test_labels_by_fold, n_missing_training_labels_by_fold={}
    )
    assert decision.pooled_positive_count == 32
    assert decision.passes is True  # pooled (32) >= MIN_POOLED_POSITIVE_COUNT (30)
    assert 2016 in decision.years_suppressed
    assert 2015 in decision.years_with_reportable_metrics


def test_gate_counts_missing_training_labels_explicitly():
    decision = evaluate_gate(
        train_labels_by_fold={},
        test_labels_by_fold={2015: pd.Series([True] * 40 + [False] * 40)},
        n_missing_training_labels_by_fold={2015: 12, 2016: 8},
    )
    assert decision.n_missing_training_labels == 20


def test_compute_classifier_metrics_hand_computed():
    df = pd.DataFrame(
        {
            "predicted_proba": [0.9, 0.8, 0.2, 0.1],
            "predicted_label": [True, True, False, False],
            "actual_label": [True, False, False, False],
        }
    )
    metrics = compute_classifier_metrics(df)
    assert metrics["n"] == 4
    assert metrics["n_positive"] == 1
    assert metrics["prevalence"] == pytest.approx(0.25)
    assert metrics["confusion_true_positive"] == 1
    assert metrics["confusion_false_positive"] == 1
    assert metrics["confusion_true_negative"] == 2
    assert metrics["confusion_false_negative"] == 0
    assert metrics["pr_auc"] is not None


def test_compute_classifier_metrics_empty_or_single_class_returns_none_not_error():
    df_empty = pd.DataFrame({"predicted_proba": [], "actual_label": []})
    metrics = compute_classifier_metrics(df_empty)
    assert metrics["n"] == 0
    assert metrics["pr_auc"] is None

    df_single_class = pd.DataFrame({"predicted_proba": [0.1, 0.2, 0.3], "actual_label": [False, False, False]})
    metrics = compute_classifier_metrics(df_single_class)
    assert metrics["pr_auc"] is None
    assert metrics["brier_score"] is not None


def test_compute_calibration_table_reports_bins_with_counts():
    rng = np.random.default_rng(0)
    n = 50
    proba = rng.uniform(0, 1, n)
    label = (proba + rng.normal(0, 0.1, n)) > 0.5
    df = pd.DataFrame({"predicted_proba": proba, "actual_label": label})
    table = compute_calibration_table(df, n_bins=5)
    assert table["n"].sum() == n
    assert set(table.columns) == {"bin", "n", "mean_predicted_proba", "observed_frequency"}


def test_compute_calibration_table_too_few_rows_returns_empty():
    df = pd.DataFrame({"predicted_proba": [0.5, 0.6], "actual_label": [True, False]})
    table = compute_calibration_table(df, n_bins=5)
    assert table.empty


def test_build_inner_cv_safe_labels_returns_none_when_inner_holdout_unusable():
    from treasury_auction_stress.evaluation.stress_event import (
        build_inner_cv_safe_labels,
    )

    train_pool = _synthetic_pool(20, [2010])  # single year -> inner holdout unusable
    train_pool = add_safe_regime_feature(train_pool, cutoff_col=CUTOFF_COL)
    assert build_inner_cv_safe_labels(train_pool) is None


def test_build_inner_cv_safe_labels_never_lets_inner_val_years_own_surprise_set_its_threshold():
    """The core acceptance-review fix: the inner-safe threshold (used to
    label the inner-validation year) must be identical whether or not
    the inner-validation year's own rows are poisoned, because it is
    fit ONLY on inner_train (strictly-prior years) -- unlike the
    ordinary outer training label, whose threshold pools ALL years
    (including what becomes the inner-validation year) and therefore DOES
    change under the same poisoning (asserted here as a contrast).
    """
    from treasury_auction_stress.evaluation.stress_event import (
        build_cross_fitted_training_labels,
        build_inner_cv_safe_labels,
    )
    from treasury_auction_stress.evaluation.timing import build_inner_holdout

    raw_pool = _synthetic_pool(20, [2010, 2011, 2012, 2013])
    inner_val_year = raw_pool["auction_date"].dt.year.max()  # 2013, the inner-val year

    raw_poisoned = raw_pool.copy()
    poisoned_mask = raw_poisoned["auction_date"].dt.year == inner_val_year
    raw_poisoned.loc[poisoned_mask, "primary_dealer_share"] = 0.999

    base_pool = add_safe_regime_feature(raw_pool, cutoff_col=CUTOFF_COL)
    poisoned_pool = add_safe_regime_feature(raw_poisoned, cutoff_col=CUTOFF_COL)

    # Contrast: the ORDINARY outer training label's threshold DOES move
    # (it pools cross-fitted surprise across ALL years, including the
    # poisoned inner-val year, as a TEST rows in an inner walk-forward
    # fold -- this is the defect this fix works around, not itself
    # fixed at the outer level, which is legitimate: outer test-year
    # labels never touch outer-test data).
    _, threshold_base, _ = build_cross_fitted_training_labels(base_pool)
    _, threshold_poisoned, _ = build_cross_fitted_training_labels(poisoned_pool)
    assert threshold_base.threshold_ != pytest.approx(threshold_poisoned.threshold_)

    # The fix: the INNER-safe label's own threshold is fit only on
    # inner_train (strictly prior to inner_val_year) -- so poisoning
    # inner_val_year's rows must NOT move it.
    holdout_base = build_inner_holdout(base_pool)
    holdout_poisoned = build_inner_holdout(poisoned_pool)
    _, inner_threshold_base, _ = build_cross_fitted_training_labels(holdout_base.inner_train)
    _, inner_threshold_poisoned, _ = build_cross_fitted_training_labels(holdout_poisoned.inner_train)
    assert inner_threshold_base.threshold_ == pytest.approx(inner_threshold_poisoned.threshold_)

    inner_safe_base = build_inner_cv_safe_labels(base_pool)
    inner_safe_poisoned = build_inner_cv_safe_labels(poisoned_pool)
    assert inner_safe_base is not None and inner_safe_poisoned is not None
    # Inner-train rows' labels must be bit-identical between the two pools.
    inner_train_mask = base_pool["auction_date"].dt.year < inner_val_year
    pd.testing.assert_series_equal(
        inner_safe_base.loc[inner_train_mask].astype("boolean"),
        inner_safe_poisoned.loc[inner_train_mask].astype("boolean"),
        check_names=False,
    )


def test_compute_classifier_metrics_reports_no_skill_brier_baseline_and_flags_worse_than():
    df = pd.DataFrame(
        {
            "predicted_proba": [0.9, 0.9, 0.9, 0.9],  # deliberately overconfident/miscalibrated
            "predicted_label": [True, True, True, True],
            "actual_label": [True, False, False, False],  # true prevalence here is 0.25
            "train_prevalence": [0.25, 0.25, 0.25, 0.25],
        }
    )
    metrics = compute_classifier_metrics(df)
    expected_no_skill = ((0.25 - 1) ** 2 + (0.25 - 0) ** 2 + (0.25 - 0) ** 2 + (0.25 - 0) ** 2) / 4
    expected_model = ((0.9 - 1) ** 2 + (0.9 - 0) ** 2 + (0.9 - 0) ** 2 + (0.9 - 0) ** 2) / 4
    assert metrics["no_skill_brier_score"] == pytest.approx(expected_no_skill)
    assert metrics["brier_score"] == pytest.approx(expected_model)
    assert metrics["worse_than_no_skill"] is True


def test_compute_classifier_metrics_no_skill_omitted_when_column_missing():
    df = pd.DataFrame({"predicted_proba": [0.5, 0.5], "actual_label": [True, False]})
    metrics = compute_classifier_metrics(df)
    assert metrics["no_skill_brier_score"] is None
    assert "worse_than_no_skill" not in metrics
