"""Phase 7 diagnostic figures. Point-forecast figures reuse Phase 6's
own `plot_annual_mae_by_method`/`plot_actual_vs_forecast_over_time`
directly (Phase 7's point-prediction table shares the exact same
column names) rather than re-deriving the same plotting logic. Only
the probabilistic coverage figure is new.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from treasury_auction_stress.evaluation.probabilistic_metrics import NOMINAL_INTERVALS
from treasury_auction_stress.visualization.phase6_plots import (
    _finish,
    plot_actual_vs_forecast_over_time,
    plot_annual_mae_by_method,
    write_figure_atomically,
)

__all__ = [
    "plot_actual_vs_forecast_over_time",
    "plot_annual_mae_by_method",
    "plot_coverage_by_year",
    "write_figure_atomically",
]


def plot_coverage_by_year(
    predictions: pd.DataFrame, *, model_ids: list[str], model_labels: dict[str, str], cutoff_view: str, interval_label: str = "80%"
) -> bytes:
    """Empirical coverage of one nominal interval, by test year, for
    each probabilistic model -- a horizontal reference line marks the
    nominal (target) coverage level so under/over-coverage is visible
    directly.
    """
    nominal_fraction = {"50%": 0.50, "80%": 0.80, "90%": 0.90}[interval_label]
    low_q, high_q = next((lo, hi) for label, lo, hi in NOMINAL_INTERVALS if label == interval_label)
    low_col, high_col = f"q{low_q:.2f}", f"q{high_q:.2f}"

    subset = predictions.loc[(predictions["cutoff_view"] == cutoff_view) & (predictions["model_id"].isin(model_ids)) & (~predictions["is_provisional"])]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model_id in model_ids:
        model_subset = subset.loc[subset["model_id"] == model_id]
        covered = (model_subset["actual"] >= model_subset[low_col]) & (model_subset["actual"] <= model_subset[high_col])
        by_year = covered.groupby(model_subset["test_year"]).mean()
        ax.plot(by_year.index, by_year.to_numpy(), marker="o", label=model_labels.get(model_id, model_id), linewidth=1.5)

    ax.axhline(nominal_fraction, color="black", linestyle="--", linewidth=1, label=f"Nominal {interval_label}")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Empirical {interval_label} interval coverage by year -- primary_dealer_share ({cutoff_view} cutoff)")
    ax.set_xlabel("Test year (complete years only)")
    ax.set_ylabel("Empirical coverage")
    ax.legend(fontsize=8)
    caption = (
        f"Out-of-sample only, complete test years. Nominal interval: {interval_label} "
        f"({low_col}-{high_col}). Cutoff view: {cutoff_view}. A well-calibrated method's line "
        "should sit near the dashed nominal-coverage reference line; no claim is made that "
        "nominal coverage holds under a genuine regime shift."
    )
    return _finish(fig, caption)
