"""Phase 7: correctness tests for
`treasury_auction_stress.models.quantile_residual.ResidualQuantileCalibrator`.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.evaluation.probabilistic_metrics import (
    quantile_column_name,
)
from treasury_auction_stress.evaluation.timing import add_result_safe_available_date
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL
from treasury_auction_stress.models.baselines import RecentHistoryBaseline
from treasury_auction_stress.models.quantile_residual import ResidualQuantileCalibrator

QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _pool(n_per_year: int, years: list[int], tenor: str = "2-Year", *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for i in range(n_per_year):
            auction_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=5 * i)
            rows.append(
                {
                    "auction_key": f"{year}_{i}",
                    "tenor": tenor,
                    "auction_date": auction_date,
                    ANNOUNCEMENT_CUTOFF_COL: auction_date,
                    "primary_dealer_share": 0.30 + rng.normal(0, 0.05),
                }
            )
    df = pd.DataFrame(rows)
    return add_result_safe_available_date(df, date_col="auction_date")


def _point_predict_fn(model, df):
    return model.predict(df)


def test_fit_produces_pooled_quantiles_from_walk_forward_residuals():
    train_pool = _pool(20, [2010, 2011, 2012])
    calibrator = ResidualQuantileCalibrator(quantile_levels=QUANTILE_LEVELS).fit(
        train_pool,
        point_model_factory=lambda: RecentHistoryBaseline(n_lookback=8),
        point_predict_fn=_point_predict_fn,
    )
    assert calibrator.n_pooled_residuals_ > 0
    # Quantiles must be non-decreasing in level.
    values = [calibrator.pooled_quantiles_[level] for level in QUANTILE_LEVELS]
    assert all(a <= b + 1e-12 for a, b in pairwise(values))


def test_sparse_tenor_falls_back_to_pooled_quantiles():
    common = _pool(20, [2010, 2011, 2012], tenor="2-Year")
    sparse = _pool(2, [2012], tenor="20-Year", seed=99)
    train_pool = pd.concat([common, sparse], ignore_index=True)

    calibrator = ResidualQuantileCalibrator(quantile_levels=QUANTILE_LEVELS, min_residuals_per_tenor=30).fit(
        train_pool,
        point_model_factory=lambda: RecentHistoryBaseline(n_lookback=8),
        point_predict_fn=_point_predict_fn,
    )
    assert "20-Year" not in calibrator.tenor_quantiles_
    assert "2-Year" in calibrator.tenor_quantiles_

    test_df = pd.DataFrame(
        [
            {"auction_key": "t1", "tenor": "20-Year", "auction_date": pd.Timestamp("2013-01-01")},
            {"auction_key": "t2", "tenor": "2-Year", "auction_date": pd.Timestamp("2013-01-01")},
        ]
    )
    point_forecast = pd.Series([0.5, 0.5], index=test_df.index)
    quantiles = calibrator.predict_quantiles(test_df, point_forecast)

    median_col = quantile_column_name(0.50)
    sparse_row = quantiles.iloc[0][median_col]
    expected_pooled = 0.5 + calibrator.pooled_quantiles_[0.50]
    assert sparse_row == pytest.approx(expected_pooled)


def test_predict_quantiles_offsets_are_centered_on_the_supplied_point_forecast_and_sorted():
    train_pool = _pool(20, [2010, 2011, 2012])
    calibrator = ResidualQuantileCalibrator(quantile_levels=QUANTILE_LEVELS).fit(
        train_pool,
        point_model_factory=lambda: RecentHistoryBaseline(n_lookback=8),
        point_predict_fn=_point_predict_fn,
    )
    test_df = pd.DataFrame(
        [{"auction_key": "t1", "tenor": "2-Year", "auction_date": pd.Timestamp("2013-01-01")}]
    )
    point_forecast = pd.Series([0.42], index=test_df.index)
    quantiles = calibrator.predict_quantiles(test_df, point_forecast)

    values = quantiles.iloc[0][[quantile_column_name(level) for level in QUANTILE_LEVELS]].to_numpy()
    assert (np.diff(values) >= -1e-12).all()
    median = quantiles.iloc[0][quantile_column_name(0.50)]
    assert median == pytest.approx(0.42 + calibrator.tenor_quantiles_["2-Year"][0.50])


def test_single_year_training_pool_yields_zero_offsets_not_an_error():
    train_pool = _pool(20, [2010])
    calibrator = ResidualQuantileCalibrator(quantile_levels=QUANTILE_LEVELS).fit(
        train_pool,
        point_model_factory=lambda: RecentHistoryBaseline(n_lookback=8),
        point_predict_fn=_point_predict_fn,
    )
    assert calibrator.n_pooled_residuals_ == 0
    assert all(v == 0.0 for v in calibrator.pooled_quantiles_.values())
