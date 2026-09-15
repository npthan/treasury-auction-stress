"""Candidate, past-only rate features built from the wide Treasury Par
Yield Curve table (`treasury_rates_normalize.pivot_wide_by_maturity`'s
output).

Same past-only discipline as
`treasury_auction_stress.features.dealer_candidate_features`: every
rolling computation uses `shift(1)` before `rolling`/`diff` so a value
at trading day *t* only ever depends on strictly-earlier days. The
wide table is one row per U.S. trading day (no weekend/holiday rows),
sorted ascending, so a plain `.diff(n)` is exactly "n trading days
ago."

## Percent vs. basis points -- explicit, tested conversion

Every rate is stored as `par_yield_percent` (Treasury's own unit, e.g.
`4.35` meaning 4.35%). A **one percentage-point** change equals **100
basis points**. Every "in basis points" feature below multiplies a
percentage-point difference by 100 explicitly -- see
`tests/test_treasury_rates_candidate_features.py` for a direct
arithmetic test of this conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.data.treasury_rates_schema import (
    MATURITY_LABELS_BY_YEARS_ASC,
    TENOR_TO_MATURITY_LABEL,
)

BASIS_POINTS_PER_PERCENTAGE_POINT = 100

DEFAULT_CHANGE_WINDOWS_TRADING_DAYS: tuple[int, ...] = (1, 5, 20)
DEFAULT_VOL_WINDOWS_TRADING_DAYS: tuple[int, ...] = (5, 20)
_NON_MATURITY_COLS = {"rate_date", "publication_date", "publication_safe_available_date"}


def _maturity_columns_present(wide_df: pd.DataFrame) -> list[str]:
    return [c for c in MATURITY_LABELS_BY_YEARS_ASC if c in wide_df.columns]


def add_rate_changes_and_volatility(
    wide_df: pd.DataFrame,
    *,
    change_windows: tuple[int, ...] = DEFAULT_CHANGE_WINDOWS_TRADING_DAYS,
    vol_windows: tuple[int, ...] = DEFAULT_VOL_WINDOWS_TRADING_DAYS,
) -> pd.DataFrame:
    """For every maturity column: `_chg_{n}d_bps` (n-trading-day change,
    in basis points) and `_vol_{n}d_bps` (n-day realized volatility --
    the standard deviation of daily changes over the trailing `n`
    trading days, in basis points, `shift(1)`'d so day *t*'s volatility
    figure never includes day *t*'s own change).
    """
    out = wide_df.sort_values("rate_date").reset_index(drop=True).copy()
    for maturity in _maturity_columns_present(out):
        daily_change_bps = out[maturity].diff() * BASIS_POINTS_PER_PERCENTAGE_POINT
        for window in change_windows:
            out[f"{maturity}_chg_{window}d_bps"] = (
                out[maturity] - out[maturity].shift(window)
            ) * BASIS_POINTS_PER_PERCENTAGE_POINT
        prior_daily_change = daily_change_bps.shift(1)
        for window in vol_windows:
            out[f"{maturity}_vol_{window}d_bps"] = prior_daily_change.rolling(
                window=window, min_periods=window
            ).std()
    return out


def add_curve_slopes_and_curvature(wide_df: pd.DataFrame) -> pd.DataFrame:
    """`slope_2s10s_bps` = 10yr - 2yr; `slope_5s30s_bps` = 30yr - 5yr;
    `curvature_2_10_30_bps` = a standard butterfly measure,
    `2*10yr - 2yr - 30yr` -- positive when the 10-year sits "rich"
    (low-yield) relative to a straight line between 2yr and 30yr.
    All in basis points, computed from the same day's levels only
    (never a rolling/lagged quantity itself, though the underlying
    levels are already past-only by construction of the as-of join).
    """
    out = wide_df.copy()
    if {"2 Yr", "10 Yr"}.issubset(out.columns):
        out["slope_2s10s_bps"] = (out["10 Yr"] - out["2 Yr"]) * BASIS_POINTS_PER_PERCENTAGE_POINT
    if {"5 Yr", "30 Yr"}.issubset(out.columns):
        out["slope_5s30s_bps"] = (out["30 Yr"] - out["5 Yr"]) * BASIS_POINTS_PER_PERCENTAGE_POINT
    if {"2 Yr", "10 Yr", "30 Yr"}.issubset(out.columns):
        out["curvature_2_10_30_bps"] = (
            2 * out["10 Yr"] - out["2 Yr"] - out["30 Yr"]
        ) * BASIS_POINTS_PER_PERCENTAGE_POINT
    return out


def build_rate_feature_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Apply every candidate transform once, in a fixed order."""
    out = add_rate_changes_and_volatility(wide_df)
    out = add_curve_slopes_and_curvature(out)
    return out


def add_tenor_matched_rate_features(joined_df: pd.DataFrame) -> pd.DataFrame:
    """`matched_tenor_par_yield_percent`, `adjacent_lower_maturity_*`,
    `adjacent_higher_maturity_*`, and `rate_age_calendar_days` -- the
    exact-tenor yield an auction's own maturity corresponds to, the
    curve points immediately below/above it (for local curve shape),
    and how stale the selected rate observation is relative to the
    cutoff. Requires the joined wide-table columns already be present
    (call after `treasury_rates_join.as_of_join`).
    """
    out = joined_df.copy()
    n = len(out)
    matched_yield = pd.Series(np.nan, index=out.index, dtype="float64")
    lower_label = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    higher_label = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    lower_yield = pd.Series(np.nan, index=out.index, dtype="float64")
    higher_yield = pd.Series(np.nan, index=out.index, dtype="float64")

    labels = MATURITY_LABELS_BY_YEARS_ASC
    for tenor, maturity_label in TENOR_TO_MATURITY_LABEL.items():
        mask = out["tenor"] == tenor
        if not mask.any() or maturity_label not in out.columns:
            continue
        matched_yield.loc[mask] = out.loc[mask, maturity_label].astype("float64")
        idx = labels.index(maturity_label)
        if idx > 0:
            lo = labels[idx - 1]
            if lo in out.columns:
                lower_label.loc[mask] = lo
                lower_yield.loc[mask] = out.loc[mask, lo].astype("float64")
        if idx < len(labels) - 1:
            hi = labels[idx + 1]
            if hi in out.columns:
                higher_label.loc[mask] = hi
                higher_yield.loc[mask] = out.loc[mask, hi].astype("float64")

    out["matched_tenor_maturity_label"] = out["tenor"].map(TENOR_TO_MATURITY_LABEL)
    out["matched_tenor_par_yield_percent"] = matched_yield
    out["adjacent_lower_maturity_label"] = lower_label.to_numpy()
    out["adjacent_lower_par_yield_percent"] = lower_yield
    out["adjacent_higher_maturity_label"] = higher_label.to_numpy()
    out["adjacent_higher_par_yield_percent"] = higher_yield
    return out


# Phase 5: which {maturity}_chg_{n}d_bps / {maturity}_vol_{n}d_bps columns
# (from add_rate_changes_and_volatility, computed once per maturity before
# the auction join) to surface as the auction's own *matched-tenor* change/
# volatility features -- mirrors add_tenor_matched_rate_features's level
# lookup, for the same reason: a flat `2 Yr_chg_5d_bps` column is not
# auction-tenor-aware on its own, so this selects the one column that
# actually corresponds to each row's own tenor.
_MATCHED_CHANGE_WINDOWS_TRADING_DAYS: tuple[int, ...] = (1, 5, 20)
_MATCHED_VOL_WINDOWS_TRADING_DAYS: tuple[int, ...] = (5, 20)


def add_tenor_matched_rate_change_features(joined_df: pd.DataFrame) -> pd.DataFrame:
    """`matched_tenor_chg_{n}d_bps` and `matched_tenor_vol_{n}d_bps`: the
    exact-tenor-matched trading-day change/realized-volatility columns
    already computed by `add_rate_changes_and_volatility` (called before
    the auction join, so every value is already past-only by
    construction) -- selected per row by `tenor`, exactly like
    `add_tenor_matched_rate_features` does for the level itself. Requires
    `add_tenor_matched_rate_features` to have been called already (uses
    its `matched_tenor_maturity_label` column).
    """
    out = joined_df.copy()
    if "matched_tenor_maturity_label" not in out.columns:
        out = add_tenor_matched_rate_features(out)

    for window in _MATCHED_CHANGE_WINDOWS_TRADING_DAYS:
        col_name = f"matched_tenor_chg_{window}d_bps"
        values = pd.Series(np.nan, index=out.index, dtype="float64")
        for tenor, label in TENOR_TO_MATURITY_LABEL.items():
            source_col = f"{label}_chg_{window}d_bps"
            if source_col not in out.columns:
                continue
            mask = out["tenor"] == tenor
            values.loc[mask] = out.loc[mask, source_col].astype("float64")
        out[col_name] = values

    for window in _MATCHED_VOL_WINDOWS_TRADING_DAYS:
        col_name = f"matched_tenor_vol_{window}d_bps"
        values = pd.Series(np.nan, index=out.index, dtype="float64")
        for tenor, label in TENOR_TO_MATURITY_LABEL.items():
            source_col = f"{label}_vol_{window}d_bps"
            if source_col not in out.columns:
                continue
            mask = out["tenor"] == tenor
            values.loc[mask] = out.loc[mask, source_col].astype("float64")
        out[col_name] = values

    return out
