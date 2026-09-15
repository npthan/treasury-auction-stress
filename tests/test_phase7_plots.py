"""Phase 7: smoke tests for `treasury_auction_stress.visualization.phase7_plots`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.visualization.phase7_plots import (
    plot_annual_mae_by_method,
    plot_coverage_by_year,
)


def _point_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in (2015, 2016, 2017):
        for model_id in ("baseline_recent_history_frozen", "challenger_gbm_core"):
            for i in range(20):
                actual = 0.30 + rng.normal(0, 0.05)
                rows.append(
                    {
                        "target_name": "primary_dealer_share",
                        "cutoff_view": "announcement",
                        "model_id": model_id,
                        "test_year": year,
                        "is_provisional": False,
                        "actual": actual,
                        "forecast": actual + rng.normal(0, 0.02),
                    }
                )
    return pd.DataFrame(rows)


def _probabilistic_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for year in (2015, 2016, 2017):
        for model_id in ("quantile_recent_history_residual", "quantile_gbm_core"):
            for i in range(20):
                actual = 0.30 + rng.normal(0, 0.05)
                rows.append(
                    {
                        "cutoff_view": "announcement",
                        "model_id": model_id,
                        "test_year": year,
                        "is_provisional": False,
                        "actual": actual,
                        "q0.10": actual - 0.10,
                        "q0.90": actual + 0.10,
                        "q0.05": actual - 0.15,
                        "q0.95": actual + 0.15,
                        "q0.25": actual - 0.05,
                        "q0.75": actual + 0.05,
                        "q0.50": actual,
                    }
                )
    return pd.DataFrame(rows)


def test_plot_annual_mae_by_method_returns_nonempty_png():
    png = plot_annual_mae_by_method(
        _point_predictions(),
        target_name="primary_dealer_share",
        cutoff_view="announcement",
        model_ids=["baseline_recent_history_frozen", "challenger_gbm_core"],
        model_labels={"baseline_recent_history_frozen": "Frozen", "challenger_gbm_core": "GBM"},
    )
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000


def test_plot_coverage_by_year_returns_nonempty_png():
    png = plot_coverage_by_year(
        _probabilistic_predictions(),
        model_ids=["quantile_recent_history_residual", "quantile_gbm_core"],
        model_labels={"quantile_recent_history_residual": "Residual", "quantile_gbm_core": "GBM quantile"},
        cutoff_view="announcement",
        interval_label="80%",
    )
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000
