"""Phase 7: correctness tests for `treasury_auction_stress.models.gbm`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.evaluation.probabilistic_metrics import (
    quantile_column_name,
)
from treasury_auction_stress.evaluation.timing import add_result_safe_available_date
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL
from treasury_auction_stress.models.gbm import (
    FALLBACK_GBM_HYPERPARAMS,
    fit_predict_gbm,
    fit_predict_gbm_quantiles,
)

FEATURE_COLS = ["tenor", "log_offering_amt", "days_announcement_to_auction", "mostly_missing"]
TARGET_COL = "primary_dealer_share"


def _synthetic_pool(n_per_year: int, years: list[int], *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    tenors = ["2-Year", "5-Year", "10-Year"]
    for year in years:
        for i in range(n_per_year):
            tenor = tenors[i % len(tenors)]
            auction_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=3 * i)
            log_offering = 17.0 + rng.normal(0, 0.1)
            days_to_auction = 5 + (i % 4)
            target = 0.30 + 0.02 * tenors.index(tenor) + rng.normal(0, 0.01)
            rows.append(
                {
                    "auction_key": f"{year}_{i}",
                    "tenor": tenor,
                    "auction_date": auction_date,
                    ANNOUNCEMENT_CUTOFF_COL: auction_date,
                    "log_offering_amt": log_offering,
                    "days_announcement_to_auction": days_to_auction,
                    "mostly_missing": np.nan,
                    TARGET_COL: target,
                }
            )
    df = pd.DataFrame(rows)
    return add_result_safe_available_date(df, date_col="auction_date")


def test_fit_predict_gbm_uses_inner_cv_when_history_is_sufficient():
    train_df = _synthetic_pool(60, [2010, 2011])
    test_df = _synthetic_pool(10, [2012], seed=1)

    result = fit_predict_gbm(train_df, test_df, feature_cols=FEATURE_COLS, target_col=TARGET_COL)

    assert result.used_inner_cv is True
    assert "mostly_missing" in result.dropped_all_missing_columns
    assert "mostly_missing" not in result.kept_feature_columns
    assert np.isfinite(result.predictions.to_numpy()).all()
    assert list(result.predictions.index) == list(test_df.index)


def test_fit_predict_gbm_falls_back_with_insufficient_history():
    train_df = _synthetic_pool(20, [2010])
    test_df = _synthetic_pool(5, [2011], seed=2)

    result = fit_predict_gbm(train_df, test_df, feature_cols=FEATURE_COLS, target_col=TARGET_COL)

    assert result.used_inner_cv is False
    assert result.selected_hyperparams == FALLBACK_GBM_HYPERPARAMS
    assert np.isfinite(result.predictions.to_numpy()).all()


def test_fit_predict_gbm_handles_unseen_category_at_test_time():
    train_df = _synthetic_pool(30, [2010, 2011])
    train_df = train_df.loc[train_df["tenor"] != "10-Year"].copy()  # 10-Year never seen in training
    test_df = _synthetic_pool(6, [2012], seed=3)
    assert (test_df["tenor"] == "10-Year").any()

    result = fit_predict_gbm(train_df, test_df, feature_cols=FEATURE_COLS, target_col=TARGET_COL)
    assert np.isfinite(result.predictions.to_numpy()).all()


def test_quantile_gbm_predictions_are_never_crossed():
    train_df = _synthetic_pool(60, [2010, 2011])
    test_df = _synthetic_pool(15, [2012], seed=4)
    quantile_levels = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    point_result = fit_predict_gbm(train_df, test_df, feature_cols=FEATURE_COLS, target_col=TARGET_COL)
    quantile_df = fit_predict_gbm_quantiles(
        train_df,
        test_df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        quantile_levels=quantile_levels,
        tree_hyperparams=point_result.selected_hyperparams,
    )

    values = quantile_df[[quantile_column_name(q) for q in quantile_levels]].to_numpy()
    assert (np.diff(values, axis=1) >= -1e-12).all()  # non-decreasing across each row
    assert list(quantile_df.index) == list(test_df.index)


def test_fit_predict_gbm_raises_when_all_columns_are_missing():
    train_df = _synthetic_pool(20, [2010])
    with pytest.raises(ValueError):
        fit_predict_gbm(train_df, train_df, feature_cols=["mostly_missing"], target_col=TARGET_COL)
