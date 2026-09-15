"""Release-aware as-of joins between the Treasury Daily Par Yield
Curve and Treasury auctions, for both prediction cutoffs.

Same `pandas.merge_asof(..., direction="backward")` mechanism as
`treasury_auction_stress.features.dealer_join.as_of_join`, keyed on
`publication_safe_available_date` (never `rate_date` itself) -- see
`treasury_rates_schema.py`'s publication-timing section for why a
same-day rate is not safely usable for a same-day cutoff. Robust to
`NaT` on either side, matching the fix made during the Phase 3
acceptance review (an internal `__as_of_join_row_order__` column,
reserved and guarded against collision, restores the caller's original
row order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.data.treasury_rates_schema import (
    MATURITY_LABELS_BY_YEARS_ASC,
    RATES_START_DATE,
)

STABLE_MATURITY_NAMES: tuple[str, ...] = MATURITY_LABELS_BY_YEARS_ASC
_COVERAGE_START = pd.Timestamp(RATES_START_DATE)

NO_COVERAGE_REASON = (
    f"no_rate_published_before_cutoff (this project's Treasury par-yield-curve "
    f"coverage begins {RATES_START_DATE}; see artifacts/treasury_rates_data_quality.md)"
)
SOURCE_MISSING_REASON = (
    "source_reported_missing_value_for_nearest_available_release "
    "(Treasury published an empty value for this maturity/date; not imputed)"
)


def as_of_join(auctions_df: pd.DataFrame, rates_wide_df: pd.DataFrame, *, cutoff_col: str) -> pd.DataFrame:
    """Backward as-of join of every maturity column present in
    `rates_wide_df` onto `auctions_df`, gated by `cutoff_col`. No
    auction is ever dropped; every row carries `rates_join_matched`,
    `rate_observation_age_calendar_days`,
    `rate_observation_age_business_days`, and (per maturity)
    `{maturity}_missing_reason`.
    """
    if "__as_of_join_row_order__" in auctions_df.columns:
        raise ValueError(
            "as_of_join: auctions_df already has a '__as_of_join_row_order__' column -- reserved internally"
        )
    left = auctions_df.reset_index(drop=True).copy()
    left["__as_of_join_row_order__"] = left.index

    right = (
        rates_wide_df.loc[
            rates_wide_df["rate_date"].notna() & rates_wide_df["publication_safe_available_date"].notna()
        ]
        # Phase 5 acceptance-review fix (see the identical, verified-live
        # fix in cftc_join.py): sorting on `publication_safe_available_
        # date` alone leaves tie-breaking among equal keys dependent on
        # this table's own (arbitrary) pre-sort row order. No such tie
        # is currently observed in this project's Treasury-rates data
        # (each trading day's own rate_date is naturally unique), but
        # the fix is applied defensively here too, with the same
        # deterministic rule (prefer the more recently observed release
        # among ties).
        .sort_values(["publication_safe_available_date", "rate_date"])
        .reset_index(drop=True)
    )

    has_cutoff = left[cutoff_col].notna()
    right_only_cols = [c for c in right.columns if c not in left.columns]
    right_only_empty = right[right_only_cols].iloc[0:0]

    def _with_right_columns(df_slice: pd.DataFrame) -> pd.DataFrame:
        filler = right_only_empty.reindex(df_slice.index)
        return pd.concat([df_slice, filler], axis=1)

    if has_cutoff.any():
        matched_part = pd.merge_asof(
            left.loc[has_cutoff].sort_values(cutoff_col),
            right,
            left_on=cutoff_col,
            right_on="publication_safe_available_date",
            direction="backward",
        )
    else:
        matched_part = _with_right_columns(left.iloc[0:0])
    unmatched_part = _with_right_columns(left.loc[~has_cutoff])

    merged = (
        pd.concat([matched_part, unmatched_part], ignore_index=True)
        .sort_values("__as_of_join_row_order__")
        .reset_index(drop=True)
        .drop(columns="__as_of_join_row_order__")
    )

    matched = merged["rate_date"].notna()
    merged["rates_join_matched"] = matched
    merged["rate_observation_age_calendar_days"] = (merged[cutoff_col] - merged["rate_date"]).dt.days

    # np.busday_count cannot accept NaT at all (raises), so it is only
    # ever called on the subset of rows with both a real matched
    # rate_date and a real cutoff -- every other row stays NaN, never 0.
    business_days = np.full(len(merged), np.nan)
    valid = (matched & merged[cutoff_col].notna()).to_numpy()
    if valid.any():
        rate_date_days = merged["rate_date"].to_numpy().astype("datetime64[D]")
        cutoff_days = merged[cutoff_col].to_numpy().astype("datetime64[D]")
        business_days[valid] = np.busday_count(rate_date_days[valid], cutoff_days[valid])
    merged["rate_observation_age_business_days"] = business_days

    maturities_present = [c for c in STABLE_MATURITY_NAMES if c in merged.columns]
    for maturity in maturities_present:
        missing = merged[maturity].isna()
        reason = np.full(len(merged), None, dtype=object)
        reason = np.where(~matched, NO_COVERAGE_REASON, reason)
        reason = np.where(missing & matched, SOURCE_MISSING_REASON, reason)
        merged[f"{maturity}_missing_reason"] = pd.array(reason, dtype="string")

    return merged
