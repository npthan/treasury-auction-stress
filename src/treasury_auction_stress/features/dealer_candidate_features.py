"""Candidate, past-only dealer features built from the wide Primary
Dealer Statistics table (`dealer_stats_normalize.pivot_wide`'s output).

Every transform in this module follows the same rule already
established by `treasury_auction_stress.features.dealer_absorption`
(`add_regime_feature`): `shift(1)` before any `rolling`/`diff` window,
so a value at observation week *t* only ever depends on weeks strictly
before *t*. Because the wide table has one row per calendar week with
**zero gaps** (verified for every selected series --
`artifacts/primary_dealer_data_quality.md`), a plain `.diff()`/`.shift()`
on the sorted table is exactly "the previous reporting week," not an
approximation. None of this depends on which auction cutoff a value
will later be joined to -- the safety comes from the dealer time
series' own construction, and the as-of join
(`treasury_auction_stress.features.dealer_join`) adds a second,
independent layer of safety on top (it can never select a value whose
*publication* postdates the cutoff). Raw levels are always kept
alongside every derived column, per `docs/point_in_time_rules.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.data.dealer_stats_schema import (
    DISCONTINUED_COMPONENT_STABLE_NAMES,
    HARMONIZED_BUCKETS,
    HARMONIZED_TOTAL_MODERN_STABLE_NAME,
    HARMONIZED_TOTAL_STABLE_NAME,
    LEGACY_COMPONENT_STABLE_NAMES,
    LEGACY_EXTENSION_BY_KEYID,
    SELECTED_SERIES,
    SERIES_BY_KEYID,
    TENOR_TO_MATURITY_BUCKET_KEYID,
)

_NON_FEATURE_COLS = {"observation_date", "publication_date", "publication_safe_available_date"}

TENOR_TO_STABLE_NAME: dict[str, str] = {
    tenor: SERIES_BY_KEYID[keyid].stable_name for tenor, keyid in TENOR_TO_MATURITY_BUCKET_KEYID.items()
}

LONG_END_POSITION_COLS = ("dealer_net_position_coupons_11y_21y", "dealer_net_position_coupons_gt_21y")
SHORT_END_POSITION_COLS = ("dealer_net_position_bills", "dealer_net_position_coupons_le_2y")

# Phase 3 acceptance review: the harmonized, cross-schema-period coarse
# buckets -- see dealer_stats_schema.py's historical-extension section.
HARMONIZED_STABLE_NAMES: tuple[str, ...] = tuple(b.stable_name for b in HARMONIZED_BUCKETS) + (
    HARMONIZED_TOTAL_STABLE_NAME,
)
# Only tenors covered by a harmonized bucket appear here (2/3/7/10/20/30-Year).
# 5-Year is intentionally absent: its own canonical column
# (`dealer_net_position_coupons_3y_6y`) was directly extended instead
# (decision 1, no coarsening needed) -- `TENOR_TO_STABLE_NAME` above
# already carries its full, now-longer history.
TENOR_TO_HARMONIZED_STABLE_NAME: dict[str, str] = {
    tenor: bucket.stable_name for bucket in HARMONIZED_BUCKETS for tenor in bucket.tenors
}

DEFAULT_TRAILING_WINDOWS_WEEKS: tuple[int, ...] = (4, 13)
DEFAULT_ROLLING_ZSCORE_WINDOW_WEEKS = 52
DEFAULT_ROLLING_ZSCORE_MIN_PERIODS = 13


def base_feature_columns(wide_df: pd.DataFrame) -> list[str]:
    """The raw series columns present in `wide_df` -- every selected
    series' stable name, excluding the observation/publication date
    columns `pivot_wide` always adds.
    """
    selected_names = {s.stable_name for s in SELECTED_SERIES}
    return [c for c in wide_df.columns if c in selected_names]


def add_weekly_change(wide_df: pd.DataFrame, *, base_columns: list[str] | None = None) -> pd.DataFrame:
    """`{col}_wow_change`: this week's level minus the immediately
    preceding reporting week's level.
    """
    out = wide_df.sort_values("observation_date").reset_index(drop=True).copy()
    columns = base_columns or base_feature_columns(out)
    for col in columns:
        out[f"{col}_wow_change"] = out[col].diff()
    return out


def add_trailing_changes(
    wide_df: pd.DataFrame,
    *,
    base_columns: list[str] | None = None,
    windows: tuple[int, ...] = DEFAULT_TRAILING_WINDOWS_WEEKS,
) -> pd.DataFrame:
    """`{col}_chg_{w}w`: this week's level minus the level `w` reporting
    weeks earlier.
    """
    out = wide_df.sort_values("observation_date").reset_index(drop=True).copy()
    columns = base_columns or base_feature_columns(out)
    for col in columns:
        for window in windows:
            out[f"{col}_chg_{window}w"] = out[col] - out[col].shift(window)
    return out


def add_rolling_zscore(
    wide_df: pd.DataFrame,
    *,
    base_columns: list[str] | None = None,
    window: int = DEFAULT_ROLLING_ZSCORE_WINDOW_WEEKS,
    min_periods: int = DEFAULT_ROLLING_ZSCORE_MIN_PERIODS,
) -> pd.DataFrame:
    """`{col}_zscore_{window}w`: this week's level standardized against
    the trailing mean/std of up to `window` **strictly preceding**
    weeks (`shift(1)` before `rolling`, exactly mirroring
    `dealer_absorption.add_regime_feature`). `pd.NA` until `min_periods`
    prior weeks exist -- never backfilled or assumed zero-variance.
    """
    out = wide_df.sort_values("observation_date").reset_index(drop=True).copy()
    columns = base_columns or base_feature_columns(out)
    for col in columns:
        prior = out[col].shift(1)
        rolling_mean = prior.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = prior.rolling(window=window, min_periods=min_periods).std()
        out[f"{col}_zscore_{window}w"] = (out[col] - rolling_mean) / rolling_std
    return out


def add_cross_maturity_imbalance(wide_df: pd.DataFrame) -> pd.DataFrame:
    """`dealer_long_short_imbalance`: total long-end net position
    (11-21y + >21y buckets) minus total short-end net position (bills +
    <=2y coupons). Plain `+`/`-` propagate missing values (unlike
    `.sum(skipna=True)`), so this is correctly `pd.NA` for every week
    before 2022-01-05, when the two long-end buckets do not yet exist
    -- never silently treated as zero. A positive value indicates
    dealers are structurally longer the long end than the short end
    relative to their own recent history; see
    `artifacts/primary_dealer_data_quality.md` for its realized range.
    """
    out = wide_df.copy()
    missing_cols = [c for c in (*LONG_END_POSITION_COLS, *SHORT_END_POSITION_COLS) if c not in out.columns]
    if missing_cols:
        raise KeyError(f"add_cross_maturity_imbalance: missing expected columns {missing_cols}")
    long_end = out[LONG_END_POSITION_COLS[0]] + out[LONG_END_POSITION_COLS[1]]
    short_end = out[SHORT_END_POSITION_COLS[0]] + out[SHORT_END_POSITION_COLS[1]]
    out["dealer_long_short_imbalance"] = long_end - short_end
    return out


def _legacy_stable_name_for_keyid(keyid: str) -> str:
    if keyid in LEGACY_COMPONENT_STABLE_NAMES:
        return LEGACY_COMPONENT_STABLE_NAMES[keyid]
    if keyid in LEGACY_EXTENSION_BY_KEYID:
        return LEGACY_EXTENSION_BY_KEYID[keyid].canonical_stable_name
    raise KeyError(f"_legacy_stable_name_for_keyid: unrecognized legacy keyid {keyid!r}")


def add_harmonized_positions(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3 acceptance review: build the 3 coarse, cross-schema-
    period harmonized maturity buckets, plus a harmonized whole-curve
    total, extending genuine coverage back to 2001-07-04 for every
    Treasury tenor's inventory feature -- see `dealer_stats_schema.py`
    (`HARMONIZED_BUCKETS`) for the full concept-by-concept
    justification of what was and wasn't extended, and why.

    Each harmonized column is built from up to three date-disjoint
    source windows -- legacy (2001-07-04 to 2013-03-27), an optional
    discontinued middle piece (2013-04-03 to 2021-12-29, long-end
    bucket only), and a modern-period sum of fine-grained buckets --
    combined with `combine_first`, which is safe here specifically
    because the windows never overlap (verified: this function raises
    if it ever finds two sources simultaneously non-null for the same
    week, rather than silently preferring one). Never interpolated,
    never backfilled, never assumes a value where none of the three
    sources has one.

    Gracefully degrades (all-`NaN` for a missing source, not an error)
    when a source column this function would otherwise use is absent
    from `wide_df` entirely -- e.g. a caller who fetched only the
    current-API series (`include_historical_extension=False`), or a
    test fixture that never included legacy data. A genuinely
    overlapping pair of *present* sources still raises.
    """
    out = wide_df.copy()

    def _col_or_nan(name: str) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series(np.nan, index=out.index, dtype="float64")

    for bucket in HARMONIZED_BUCKETS:
        legacy_col = _legacy_stable_name_for_keyid(bucket.legacy_keyid)
        result = _col_or_nan(legacy_col).copy()

        if bucket.discontinued_middle_keyid is not None:
            middle = _col_or_nan(DISCONTINUED_COMPONENT_STABLE_NAMES[bucket.discontinued_middle_keyid])
            if (result.notna() & middle.notna()).any():
                raise ValueError(
                    f"{bucket.stable_name}: legacy and discontinued-middle source "
                    "windows overlap -- expected disjoint, non-overlapping regimes"
                )
            result = result.combine_first(middle)

        modern_sum = _col_or_nan(bucket.modern_component_stable_names[0])
        for extra_col in bucket.modern_component_stable_names[1:]:
            modern_sum = modern_sum + _col_or_nan(extra_col)
        if (result.notna() & modern_sum.notna()).any():
            raise ValueError(
                f"{bucket.stable_name}: an earlier-regime source and the modern "
                "component sum overlap -- expected disjoint, non-overlapping regimes"
            )
        out[bucket.stable_name] = result.combine_first(modern_sum)

    legacy_total = (
        _col_or_nan("dealer_net_position_bills")
        + _col_or_nan("dealer_net_position_coupons_3y_6y")
        + _col_or_nan(LEGACY_COMPONENT_STABLE_NAMES["PDPUSGCS3LNOP"])
        + _col_or_nan(LEGACY_COMPONENT_STABLE_NAMES["PDPUSGCS611NOP"])
        + _col_or_nan(LEGACY_COMPONENT_STABLE_NAMES["PDPUSGCSM11NOP"])
    )
    modern_total = _col_or_nan(HARMONIZED_TOTAL_MODERN_STABLE_NAME)
    if (legacy_total.notna() & modern_total.notna()).any():
        raise ValueError(
            f"{HARMONIZED_TOTAL_STABLE_NAME}: legacy-component sum and the modern "
            "total overlap -- expected disjoint, non-overlapping regimes"
        )
    out[HARMONIZED_TOTAL_STABLE_NAME] = legacy_total.combine_first(modern_total)
    return out


def build_dealer_feature_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Apply every candidate transform once, in a fixed order, against
    the *original* raw-level columns only (so, e.g., `_chg_4w` is never
    computed on top of an already-derived `_wow_change` column).

    The raw levels of every selected series grew substantially over
    2013-2026 as the Treasury market itself grew, so
    `dealer_long_short_imbalance`'s raw level is dominated by that
    trend, not by a relative long/short tilt -- a final rolling
    z-score of the imbalance itself (still past-only, same `shift(1)`
    rule) is added specifically to give a scale-independent version.
    """
    base_columns = base_feature_columns(wide_df)
    out = wide_df.sort_values("observation_date").reset_index(drop=True).copy()
    out = add_weekly_change(out, base_columns=base_columns)
    out = add_trailing_changes(out, base_columns=base_columns)
    out = add_rolling_zscore(out, base_columns=base_columns)
    out = add_cross_maturity_imbalance(out)
    out = add_rolling_zscore(out, base_columns=["dealer_long_short_imbalance"])
    out = add_harmonized_positions(out)
    out = add_weekly_change(out, base_columns=list(HARMONIZED_STABLE_NAMES))
    out = add_trailing_changes(out, base_columns=list(HARMONIZED_STABLE_NAMES))
    out = add_rolling_zscore(out, base_columns=list(HARMONIZED_STABLE_NAMES))
    return out


def add_inventory_relative_to_offering(joined_df: pd.DataFrame) -> pd.DataFrame:
    """`dealer_inventory_to_offering_ratio`: the dealer community's
    current net position in the maturity bucket matching an auction's
    own tenor (`TENOR_TO_MATURITY_BUCKET_KEYID`), expressed as a
    multiple of that auction's own `offering_amt` -- "how large is
    dealers' existing inventory in this maturity segment relative to
    what is about to be newly supplied." Requires the joined dealer
    columns already be present (i.e. call after
    `treasury_auction_stress.features.dealer_join.as_of_join`).
    `offering_amt` is raw dollars; dealer levels are millions of
    dollars -- converted explicitly here, not silently assumed equal.
    """
    out = joined_df.copy()
    bucket_value_millions = pd.Series(np.nan, index=out.index, dtype="float64")
    for tenor, stable_name in TENOR_TO_STABLE_NAME.items():
        if stable_name not in out.columns:
            continue
        mask = out["tenor"] == tenor
        bucket_value_millions.loc[mask] = out.loc[mask, stable_name].astype("float64")
    out["dealer_tenor_bucket_inventory_millions"] = bucket_value_millions
    out["dealer_inventory_to_offering_ratio"] = (
        bucket_value_millions * 1_000_000
    ) / out["offering_amt"].astype("float64")
    return out


def add_inventory_relative_to_offering_harmonized(joined_df: pd.DataFrame) -> pd.DataFrame:
    """The harmonized counterpart of `add_inventory_relative_to_offering`
    (Phase 3 acceptance review): uses `TENOR_TO_HARMONIZED_STABLE_NAME`
    instead of the fine-grained bucket, so 2-Year, 3-Year, 7-Year,
    10-Year, 20-Year, and 30-Year auctions get an inventory-to-offering
    ratio back to 2001-07-04 (coarser -- shared between two tenors each
    -- before the relevant fine-bucket split existed). 5-Year is
    intentionally absent here: its fine-grained ratio from
    `add_inventory_relative_to_offering` already covers the same
    extended history, because `dealer_net_position_coupons_3y_6y`
    itself was directly extended (no coarsening needed).
    """
    out = joined_df.copy()
    bucket_value_millions = pd.Series(np.nan, index=out.index, dtype="float64")
    for tenor, stable_name in TENOR_TO_HARMONIZED_STABLE_NAME.items():
        if stable_name not in out.columns:
            continue
        mask = out["tenor"] == tenor
        bucket_value_millions.loc[mask] = out.loc[mask, stable_name].astype("float64")
    out["dealer_tenor_bucket_inventory_millions_harmonized"] = bucket_value_millions
    out["dealer_inventory_to_offering_ratio_harmonized"] = (
        bucket_value_millions * 1_000_000
    ) / out["offering_amt"].astype("float64")
    return out
