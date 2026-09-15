"""Phase 6: honest out-of-sample metrics, computed only from the
long-format out-of-sample prediction table (`build_predictions_table`
in `treasury_auction_stress.evaluation.run_evaluation`) -- never from
an in-sample fitted value, and never from a hand-typed number.

Share targets (`primary_dealer_share`, `direct_bidder_share`,
`indirect_bidder_share`) are reported in PERCENTAGE POINTS (raw share
x 100); `bid_to_cover_ratio` is reported in its own native ratio units
-- the two must never be mixed in one table (`docs/project_rules.md`'s "Secondary
targets ... need target-appropriate units").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

PERCENTAGE_POINT_TARGETS: frozenset[str] = frozenset(
    {"primary_dealer_share", "direct_bidder_share", "indirect_bidder_share"}
)

STRONG_BASELINE_MODEL_ID = "baseline_tenor_reopening_mean"


def unit_label(target_name: str) -> str:
    return "percentage points" if target_name in PERCENTAGE_POINT_TARGETS else "ratio units"


def _scaled_error(actual: pd.Series, forecast: pd.Series, target_name: str) -> pd.Series:
    scale = 100.0 if target_name in PERCENTAGE_POINT_TARGETS else 1.0
    return (forecast - actual) * scale


def compute_error_metrics(df: pd.DataFrame, *, target_name: str) -> dict:
    """`df` must have `actual` and `forecast` columns, already
    restricted to exactly the rows to score. Returns `n=0` metrics
    (all `None`) for an empty `df` -- never raises or divides by zero.
    """
    if df.empty:
        return {"n": 0, "mae": None, "median_ae": None, "rmse": None, "bias": None, "unit": unit_label(target_name)}

    error = _scaled_error(df["actual"], df["forecast"], target_name)
    abs_error = error.abs()
    return {
        "n": len(df),
        "mae": float(abs_error.mean()),
        "median_ae": float(abs_error.median()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "unit": unit_label(target_name),
    }


def r2_vs_fold_own_mean(df: pd.DataFrame) -> float | None:
    """Ordinary `sklearn.metrics.r2_score`: benchmarks against THIS
    group's own actual mean, not a pre-specified baseline model. Kept
    clearly distinct from `r2_vs_strong_baseline` below -- conflating
    the two is exactly the mistake `docs/project_rules.md` warns against.
    """
    if len(df) < 2:
        return None
    return float(r2_score(df["actual"], df["forecast"]))


def r2_vs_strong_baseline(
    model_df: pd.DataFrame, baseline_df: pd.DataFrame, *, key_cols: tuple[str, ...] = ("auction_key",)
) -> float | None:
    """Benchmark-relative out-of-sample R2: `1 - SSE_model / SSE_baseline`,
    computed on the SAME evaluated rows for both (an inner join on
    `key_cols`, never assumed to already be aligned). `baseline_df`
    should be the strong baseline's (`STRONG_BASELINE_MODEL_ID`) own
    predictions for the identical target/cutoff/fold selection.
    """
    merged = model_df.merge(baseline_df, on=list(key_cols), suffixes=("_model", "_baseline"), how="inner")
    if len(merged) != len(model_df) or len(merged) != len(baseline_df):
        raise AssertionError(
            "r2_vs_strong_baseline: model_df and baseline_df do not share an identical key set "
            f"({len(model_df)} model rows, {len(baseline_df)} baseline rows, {len(merged)} matched)"
        )
    if merged.empty:
        return None
    sse_model = float(((merged["forecast_model"] - merged["actual_model"]) ** 2).sum())
    sse_baseline = float(((merged["forecast_baseline"] - merged["actual_baseline"]) ** 2).sum())
    if sse_baseline == 0.0:
        return None
    return 1.0 - sse_model / sse_baseline


def _select(df: pd.DataFrame, **filters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        mask &= df[col] == value
    return df.loc[mask]


def summarize_pooled(
    predictions: pd.DataFrame, *, target_name: str, cutoff_view: str, model_id: str, include_provisional: bool
) -> dict:
    """Pooled metrics across complete test-year folds (optionally
    including the provisional partial year -- always reported
    separately elsewhere too, never silently merged into a "complete-
    year" headline without being asked for).
    """
    subset = _select(predictions, target_name=target_name, cutoff_view=cutoff_view, model_id=model_id)
    if not include_provisional:
        subset = subset.loc[~subset["is_provisional"]]
    metrics = compute_error_metrics(subset, target_name=target_name)
    metrics["r2_vs_fold_own_mean"] = r2_vs_fold_own_mean(subset)

    baseline_subset = _select(
        predictions, target_name=target_name, cutoff_view=cutoff_view, model_id=STRONG_BASELINE_MODEL_ID
    )
    if not include_provisional:
        baseline_subset = baseline_subset.loc[~baseline_subset["is_provisional"]]
    metrics["r2_vs_strong_baseline"] = (
        None
        if model_id == STRONG_BASELINE_MODEL_ID or subset.empty
        else r2_vs_strong_baseline(
            subset[["auction_key", "actual", "forecast"]], baseline_subset[["auction_key", "actual", "forecast"]]
        )
    )
    return metrics


def summarize_by(
    predictions: pd.DataFrame, *, target_name: str, cutoff_view: str, model_id: str, group_col: str
) -> pd.DataFrame:
    """One row per distinct value of `group_col` (e.g. `test_year`,
    `tenor`, `is_reopening`), with the group's own row count and error
    metrics. The provisional partial year is included in a `by_test_year`
    breakdown as its own row (already labeled `is_provisional`), never
    pooled with complete years there either.
    """
    subset = _select(predictions, target_name=target_name, cutoff_view=cutoff_view, model_id=model_id)
    rows = []
    for value, group in subset.groupby(group_col):
        metrics = compute_error_metrics(group, target_name=target_name)
        metrics["r2_vs_fold_own_mean"] = r2_vs_fold_own_mean(group)
        rows.append({group_col: value, **metrics})
    return pd.DataFrame(rows)


def build_pooled_metrics_table(predictions: pd.DataFrame, *, target_names: tuple[str, ...], model_ids: tuple[str, ...], cutoff_views: tuple[str, ...]) -> pd.DataFrame:
    """One row per (target, cutoff_view, model_id, include_provisional)
    -- the full pooled-metrics cross product, computed once so every
    report table is a `.loc`/`.query` on this, never a freshly
    recomputed number.
    """
    rows = []
    for target_name in target_names:
        for cutoff_view in cutoff_views:
            for model_id in model_ids:
                for include_provisional, label in ((False, "complete_years"), (True, "complete_years_plus_provisional")):
                    metrics = summarize_pooled(
                        predictions,
                        target_name=target_name,
                        cutoff_view=cutoff_view,
                        model_id=model_id,
                        include_provisional=include_provisional,
                    )
                    rows.append({"target_name": target_name, "cutoff_view": cutoff_view, "model_id": model_id, "scope": label, **metrics})
    return pd.DataFrame(rows)


def build_provisional_only_metrics_table(predictions: pd.DataFrame, *, target_names: tuple[str, ...], model_ids: tuple[str, ...], cutoff_views: tuple[str, ...]) -> pd.DataFrame:
    """The 2026 provisional fold's OWN metrics, reported separately --
    never pooled silently into a "complete-year" headline number.
    """
    rows = []
    for target_name in target_names:
        for cutoff_view in cutoff_views:
            for model_id in model_ids:
                subset = _select(
                    predictions, target_name=target_name, cutoff_view=cutoff_view, model_id=model_id
                )
                subset = subset.loc[subset["is_provisional"]]
                metrics = compute_error_metrics(subset, target_name=target_name)
                metrics["r2_vs_fold_own_mean"] = r2_vs_fold_own_mean(subset)
                rows.append({"target_name": target_name, "cutoff_view": cutoff_view, "model_id": model_id, **metrics})
    return pd.DataFrame(rows)


def build_breakdown_metrics_table(
    predictions: pd.DataFrame,
    *,
    target_names: tuple[str, ...],
    model_ids: tuple[str, ...],
    cutoff_views: tuple[str, ...],
    group_col: str,
    complete_years_only: bool,
) -> pd.DataFrame:
    """`group_col` in {"test_year", "tenor", "is_reopening"} -- one row
    per (target, cutoff_view, model_id, group value).
    """
    base = predictions.loc[~predictions["is_provisional"]] if complete_years_only else predictions
    rows = []
    for target_name in target_names:
        for cutoff_view in cutoff_views:
            for model_id in model_ids:
                table = summarize_by(
                    base, target_name=target_name, cutoff_view=cutoff_view, model_id=model_id, group_col=group_col
                )
                table.insert(0, "model_id", model_id)
                table.insert(0, "cutoff_view", cutoff_view)
                table.insert(0, "target_name", target_name)
                rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_paired_model_vs_benchmark_by_year(
    predictions: pd.DataFrame,
    *,
    target_name: str,
    cutoff_view: str,
    model_ids: tuple[str, ...],
    benchmark_model_id: str,
) -> pd.DataFrame:
    """Acceptance-review addition: paired absolute-error comparison of
    every model in `model_ids` against `benchmark_model_id` (intended
    for `baseline_recent_history`, the empirically much stronger
    baseline -- NOT the same thing as `STRONG_BASELINE_MODEL_ID`, which
    remains `baseline_tenor_reopening_mean` for `r2_vs_strong_baseline`
    per the pre-declared protocol). One row per (model, test_year),
    with the provisional year kept as its OWN row -- never pooled with
    complete years.

    Every row is independently verified before being used, never
    assumed: matching auction-key sets between the model and benchmark
    slices (raises `AssertionError` otherwise), identical `actual`
    values between them for the same auction (a sanity check that both
    slices really do describe the same evaluated auctions under the
    same target), and finite (non-NaN, non-infinite) predictions on
    both sides.
    """
    subset = predictions.loc[(predictions["target_name"] == target_name) & (predictions["cutoff_view"] == cutoff_view)]
    benchmark_all = subset.loc[subset["model_id"] == benchmark_model_id]
    scale = 100.0 if target_name in PERCENTAGE_POINT_TARGETS else 1.0

    rows = []
    for model_id in model_ids:
        model_all = subset.loc[subset["model_id"] == model_id]
        for test_year, model_year in model_all.groupby("test_year"):
            benchmark_year = benchmark_all.loc[benchmark_all["test_year"] == test_year]
            merged = model_year.merge(
                benchmark_year[["auction_key", "actual", "forecast"]],
                on="auction_key",
                suffixes=("_model", "_benchmark"),
                how="inner",
            )
            if len(merged) != len(model_year) or len(merged) != len(benchmark_year):
                raise AssertionError(
                    f"build_paired_model_vs_benchmark_by_year: {model_id} vs {benchmark_model_id}, "
                    f"year {test_year}: key sets disagree ({len(model_year)} model, {len(benchmark_year)} "
                    f"benchmark, {len(merged)} matched)"
                )
            if not np.array_equal(merged["actual_model"].to_numpy(), merged["actual_benchmark"].to_numpy()):
                raise AssertionError(
                    f"build_paired_model_vs_benchmark_by_year: {model_id} vs {benchmark_model_id}, "
                    f"year {test_year}: actual values disagree between model and benchmark rows for "
                    "the same auction/target -- these should be identical by construction"
                )
            model_forecast = merged["forecast_model"].to_numpy()
            benchmark_forecast = merged["forecast_benchmark"].to_numpy()
            if not (np.isfinite(model_forecast).all() and np.isfinite(benchmark_forecast).all()):
                raise AssertionError(
                    f"build_paired_model_vs_benchmark_by_year: {model_id} vs {benchmark_model_id}, "
                    f"year {test_year}: non-finite prediction found"
                )

            ae_model = np.abs(model_forecast - merged["actual_model"].to_numpy()) * scale
            ae_benchmark = np.abs(benchmark_forecast - merged["actual_benchmark"].to_numpy()) * scale
            diff = ae_model - ae_benchmark
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_model_id": benchmark_model_id,
                    "test_year": test_year,
                    "is_provisional": bool(model_year["is_provisional"].iloc[0]),
                    "n": len(merged),
                    "unit": unit_label(target_name),
                    "mae_model": float(ae_model.mean()),
                    "mae_benchmark": float(ae_benchmark.mean()),
                    "mean_paired_diff_model_minus_benchmark": float(diff.mean()),
                    "n_model_better": int((diff < 0).sum()),
                    "n_benchmark_better": int((diff > 0).sum()),
                    "n_tied": int((diff == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def n_excluded(predictions_attempted: pd.DataFrame, predictions_scored: pd.DataFrame) -> dict:
    """How many rows were attempted vs. actually scored, and why --
    `docs/project_rules.md` requires this be reported explicitly, never silently
    varying the evaluated subset per model.
    """
    excluded = len(predictions_attempted) - len(predictions_scored)
    return {
        "n_attempted": len(predictions_attempted),
        "n_scored": len(predictions_scored),
        "n_excluded": excluded,
        "reason": "missing actual target value" if excluded else "n/a",
    }
