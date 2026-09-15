"""Phase 8: interpretation and diagnostics, computed strictly from the
already-ACCEPTED Phase 7 out-of-sample prediction tables
(`data/processed/phase_7_point_predictions.parquet`,
`phase_7_probabilistic_predictions.parquet`,
`phase_7_classifier_predictions.parquet`). This module changes nothing
about Phase 1-7: no target definition, feature eligibility, fold
construction, model setting, threshold, prediction, or accepted metric
is recomputed differently here -- every function below only reads
already-produced `actual`/`forecast`/`predicted_proba` columns and
summarizes them a different way (paired by a new grouping column,
error-distribution statistics, a transparently-defined case-selection
rule, clip-boundary fractions, or a descriptive year-block bootstrap).

No new model, feature, or backtest is added anywhere in this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.evaluation.metrics import (
    PERCENTAGE_POINT_TARGETS,
    unit_label,
)
from treasury_auction_stress.evaluation.probabilistic_metrics import (
    quantile_column_name,
)

FROZEN_MODEL_ID = "baseline_recent_history_frozen"
ADAPTIVE_MODEL_ID = "baseline_recent_history_adaptive"


def _scale_for(target_name: str) -> float:
    return 100.0 if target_name in PERCENTAGE_POINT_TARGETS else 1.0


def build_paired_comparison_by_group(
    predictions: pd.DataFrame,
    *,
    cutoff_view: str,
    group_col: str,
    model_id: str = ADAPTIVE_MODEL_ID,
    benchmark_model_id: str = FROZEN_MODEL_ID,
    complete_years_only: bool = True,
) -> pd.DataFrame:
    """Paired absolute-error comparison of `model_id` vs.
    `benchmark_model_id`, broken out by an arbitrary `group_col` (e.g.
    `"tenor"` or `"is_reopening"`) instead of `test_year` -- generalizes
    `evaluation.metrics.build_paired_model_vs_benchmark_by_year`'s exact
    integrity checks (matched auction-key sets, identical `actual`
    values, finite predictions on both sides, all enforced by raising
    rather than assumed) to a grouping column that function does not
    support directly.
    """
    subset = predictions.loc[predictions["cutoff_view"] == cutoff_view]
    if complete_years_only:
        subset = subset.loc[~subset["is_provisional"]]
    if subset.empty:
        return pd.DataFrame()
    target_name = subset["target_name"].iloc[0]
    scale = _scale_for(target_name)

    model_all = subset.loc[subset["model_id"] == model_id]
    benchmark_all = subset.loc[subset["model_id"] == benchmark_model_id]

    rows = []
    for group_value, model_group in model_all.groupby(group_col):
        benchmark_group = benchmark_all.loc[benchmark_all[group_col] == group_value]
        merged = model_group.merge(
            benchmark_group[["auction_key", "actual", "forecast"]],
            on="auction_key",
            suffixes=("_model", "_benchmark"),
            how="inner",
        )
        if len(merged) != len(model_group) or len(merged) != len(benchmark_group):
            raise AssertionError(
                f"build_paired_comparison_by_group: {group_col}={group_value}: key sets disagree "
                f"({len(model_group)} model, {len(benchmark_group)} benchmark, {len(merged)} matched)"
            )
        if not np.array_equal(merged["actual_model"].to_numpy(), merged["actual_benchmark"].to_numpy()):
            raise AssertionError(
                f"build_paired_comparison_by_group: {group_col}={group_value}: actual values disagree "
                "between model and benchmark rows for the same auction -- these should be identical by construction"
            )
        model_forecast = merged["forecast_model"].to_numpy()
        benchmark_forecast = merged["forecast_benchmark"].to_numpy()
        if not (np.isfinite(model_forecast).all() and np.isfinite(benchmark_forecast).all()):
            raise AssertionError(f"build_paired_comparison_by_group: {group_col}={group_value}: non-finite prediction found")

        ae_model = np.abs(model_forecast - merged["actual_model"].to_numpy()) * scale
        ae_benchmark = np.abs(benchmark_forecast - merged["actual_benchmark"].to_numpy()) * scale
        diff = ae_model - ae_benchmark
        rows.append(
            {
                group_col: group_value,
                "model_id": model_id,
                "benchmark_model_id": benchmark_model_id,
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


def build_error_distribution_table(
    predictions: pd.DataFrame,
    *,
    cutoff_view: str,
    model_ids: tuple[str, ...],
    group_col: str | None = None,
    complete_years_only: bool = True,
) -> pd.DataFrame:
    """Signed-error (`forecast - actual`, scaled to the target's own
    unit) distribution: mean (bias), median, standard deviation, and
    skew, per model and optionally broken out by `group_col`. Every row
    carries its own `n` -- a distribution statistic is never reported
    without the count it was computed over.
    """
    subset = predictions.loc[predictions["cutoff_view"] == cutoff_view]
    if complete_years_only:
        subset = subset.loc[~subset["is_provisional"]]
    if subset.empty:
        return pd.DataFrame()
    target_name = subset["target_name"].iloc[0]
    scale = _scale_for(target_name)

    rows = []
    for model_id in model_ids:
        model_df = subset.loc[subset["model_id"] == model_id]
        groups = model_df.groupby(group_col) if group_col else [(None, model_df)]
        for group_value, g in groups:
            signed_error = (g["forecast"] - g["actual"]) * scale
            row = {
                "model_id": model_id,
                "n": len(g),
                "unit": unit_label(target_name),
                "mean_signed_error_bias": float(signed_error.mean()) if len(g) else None,
                "median_signed_error": float(signed_error.median()) if len(g) else None,
                "std_signed_error": float(signed_error.std(ddof=1)) if len(g) > 1 else None,
                "skew_signed_error": float(signed_error.skew()) if len(g) > 2 else None,
            }
            if group_col:
                row[group_col] = group_value
            rows.append(row)
    table = pd.DataFrame(rows)
    if group_col and not table.empty:
        cols = [group_col, "model_id", *[c for c in table.columns if c not in (group_col, "model_id")]]
        table = table[cols]
    return table


CASE_STUDY_RULES: tuple[str, ...] = (
    "largest_adaptive_improvement",
    "largest_adaptive_deterioration",
    "typical_case_closest_to_median_improvement",
)


def select_case_studies(predictions: pd.DataFrame, *, cutoff_view: str, complete_years_only: bool = True) -> pd.DataFrame:
    """Transparent, code-defined case-selection rule, run identically
    regardless of which auctions it happens to surface -- never
    hand-picked to flatter the adaptive model. Among complete-year
    auctions with both `baseline_recent_history_frozen` and
    `baseline_recent_history_adaptive` predictions, selects exactly
    three cases by the adaptive model's improvement in absolute error
    over the frozen model (`ae_frozen - ae_adaptive`, positive =
    adaptive better):

    1. `largest_adaptive_improvement` -- the single largest positive value.
    2. `largest_adaptive_deterioration` -- the single largest negative value.
    3. `typical_case_closest_to_median_improvement` -- the auction whose
       improvement is closest to the median improvement across all
       matched auctions.

    Ties are broken deterministically (stable sort on an auction-key-
    sorted series), so re-running this on identical inputs always
    returns the identical three auctions. Every returned row reports
    both models' own forecasts (known at the prediction cutoff) and the
    settled `actual` outcome (known only afterward) as separate columns
    so the two are never conflated.
    """
    subset = predictions.loc[predictions["cutoff_view"] == cutoff_view]
    if complete_years_only:
        subset = subset.loc[~subset["is_provisional"]]
    if subset.empty:
        return pd.DataFrame()
    target_name = subset["target_name"].iloc[0]
    scale = _scale_for(target_name)

    frozen = subset.loc[subset["model_id"] == FROZEN_MODEL_ID].set_index("auction_key")
    adaptive = subset.loc[subset["model_id"] == ADAPTIVE_MODEL_ID].set_index("auction_key")
    common_keys = sorted(set(frozen.index) & set(adaptive.index))
    if not common_keys:
        return pd.DataFrame()
    frozen = frozen.loc[common_keys]
    adaptive = adaptive.loc[common_keys]
    if not np.array_equal(frozen["actual"].to_numpy(), adaptive["actual"].to_numpy()):
        raise AssertionError("select_case_studies: actual values disagree between frozen and adaptive rows for the same auction")

    ae_frozen = (frozen["forecast"] - frozen["actual"]).abs() * scale
    ae_adaptive = (adaptive["forecast"] - adaptive["actual"]).abs() * scale
    improvement = (ae_frozen - ae_adaptive)  # already indexed in sorted-auction_key order
    median_improvement = improvement.median()

    ordered_by_value = improvement.sort_values(kind="mergesort")
    most_improved_key = ordered_by_value.index[-1]
    most_deteriorated_key = ordered_by_value.index[0]
    typical_key = (improvement - median_improvement).abs().sort_values(kind="mergesort").index[0]

    def _row_for(auction_key: str, rule: str) -> dict:
        return {
            "case_selection_rule": rule,
            "auction_key": auction_key,
            "cutoff_view": cutoff_view,
            "test_year": int(frozen.loc[auction_key, "test_year"]),
            "tenor": frozen.loc[auction_key, "tenor"],
            "is_reopening": bool(frozen.loc[auction_key, "is_reopening"]),
            "known_at_cutoff__frozen_forecast_pct": float(frozen.loc[auction_key, "forecast"] * 100.0),
            "known_at_cutoff__adaptive_forecast_pct": float(adaptive.loc[auction_key, "forecast"] * 100.0),
            "learned_after_settlement__actual_pct": float(frozen.loc[auction_key, "actual"] * 100.0),
            "frozen_abs_error_pp": float(ae_frozen.loc[auction_key]),
            "adaptive_abs_error_pp": float(ae_adaptive.loc[auction_key]),
            "adaptive_improvement_pp": float(improvement.loc[auction_key]),
        }

    return pd.DataFrame(
        [
            _row_for(most_improved_key, "largest_adaptive_improvement"),
            _row_for(most_deteriorated_key, "largest_adaptive_deterioration"),
            _row_for(typical_key, "typical_case_closest_to_median_improvement"),
        ]
    )


def compute_quantile_clip_fraction(
    probabilistic_predictions: pd.DataFrame,
    *,
    cutoff_view: str,
    model_id: str,
    quantile_levels: tuple[float, ...],
    group_col: str | None = None,
    complete_years_only: bool = True,
) -> pd.DataFrame:
    """Fraction of rows for which AT LEAST ONE predicted quantile
    column touches the pre-declared valid-share clip bound (0.0 or
    1.0), optionally broken out by `group_col` (e.g. `"test_year"` or
    `"tenor"`). Every row carries its own `n` alongside the fraction.
    """
    subset = probabilistic_predictions.loc[
        (probabilistic_predictions["cutoff_view"] == cutoff_view) & (probabilistic_predictions["model_id"] == model_id)
    ]
    if complete_years_only:
        subset = subset.loc[~subset["is_provisional"]]
    if subset.empty:
        return pd.DataFrame()
    cols = [quantile_column_name(level) for level in quantile_levels]

    def _touches_bound(frame: pd.DataFrame) -> pd.Series:
        touches = pd.Series(False, index=frame.index)
        for col in cols:
            touches |= np.isclose(frame[col].to_numpy(dtype=float), 0.0) | np.isclose(frame[col].to_numpy(dtype=float), 1.0)
        return touches

    groups = subset.groupby(group_col) if group_col else [(None, subset)]
    rows = []
    for group_value, g in groups:
        touches = _touches_bound(g)
        row = {
            "n": len(g),
            "n_rows_touching_clip_bound": int(touches.sum()),
            "fraction_rows_touching_clip_bound": float(touches.mean()) if len(g) else None,
        }
        if group_col:
            row[group_col] = group_value
        rows.append(row)
    table = pd.DataFrame(rows)
    if group_col and not table.empty:
        table = table[[group_col, *[c for c in table.columns if c != group_col]]]
    return table


def build_stress_metrics_by_year_table(classifier_predictions: pd.DataFrame, *, cutoff_view: str) -> pd.DataFrame:
    """Per-complete-year classifier metrics (prevalence, Brier vs.
    no-skill, PR-AUC vs. base rate, confusion counts), for EVERY year
    with any labeled row -- unlike the Phase 7 gate report, this does
    not suppress a low-positive-count year's row entirely; it is
    included with its own small `n_positive` so a reader can judge
    reliability directly, never hidden.
    """
    from treasury_auction_stress.evaluation.stress_event import (
        compute_classifier_metrics,
    )

    if classifier_predictions.empty:
        return pd.DataFrame()
    subset = classifier_predictions.loc[
        (classifier_predictions["cutoff_view"] == cutoff_view) & (~classifier_predictions["is_provisional"])
    ]
    rows = []
    for year, group in subset.groupby("test_year"):
        metrics = compute_classifier_metrics(group)
        metrics["test_year"] = int(year)
        rows.append(metrics)
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table[["test_year", *[c for c in table.columns if c != "test_year"]]]
    return table


def year_block_bootstrap_ci(
    yearly_values: pd.Series,
    *,
    n_boot: int = 10000,
    seed: int = 42,
    ci: tuple[float, float] = (0.025, 0.975),
) -> dict:
    """Descriptive uncertainty around a paired performance difference
    that respects the chronological/yearly grouping: resamples which
    complete test YEARS contribute (with replacement), never individual
    auctions, because within one year, the same model's per-auction
    errors share a fold, a frozen model fit, and a regime -- they are
    not independent draws. `yearly_values` should be one summary value
    per complete test year (e.g. that year's own mean paired
    adaptive-minus-frozen difference).

    This is explicitly NOT a significance test: with only as many years
    as there are complete test years in this backtest (typically 11),
    this reports a percentile interval as descriptive uncertainty, not
    a claim of statistical significance. Deterministic given a fixed
    `seed` (matching this project's `random_state=42` convention
    everywhere else), so two runs on identical input produce an
    identical result -- required for the CLI's own reproducibility
    check.
    """
    values = yearly_values.dropna().to_numpy(dtype=float)
    n_years = len(values)
    if n_years < 2:
        return {
            "n_years": n_years,
            "point_estimate": float(values.mean()) if n_years else None,
            "ci_low": None,
            "ci_high": None,
            "n_boot": n_boot,
            "seed": seed,
            "method": "year-level block bootstrap (percentile interval), descriptive only -- too few years to bootstrap",
        }
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n_years, size=(n_boot, n_years))
    resample_means = values[resample_idx].mean(axis=1)
    lo, hi = np.quantile(resample_means, ci)
    return {
        "n_years": n_years,
        "point_estimate": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "seed": seed,
        "method": "year-level block bootstrap (percentile interval), descriptive only, not a significance test",
    }
