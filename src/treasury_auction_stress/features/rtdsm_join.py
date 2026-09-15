"""Release-aware as-of joins between RTDSM macro vintages and Treasury
auctions, for both prediction cutoffs.

Unlike the other Phase 4 sources, an RTDSM "release" is a whole
*vintage column* (one per month or quarter, not one per auction-
relevant observation), so the join has two steps instead of one:

1. `pandas.merge_asof(..., direction="backward")` against a small
   *vintage index* (one row per vintage, keyed on
   `publication_safe_available_date`) to find which vintage was
   available as of the auction's own cutoff -- the same NaT-robust
   `__as_of_join_row_order__` pattern as every other join module here.
2. A plain merge of that matched `vintage_label` against a
   precomputed *snapshot table* (`rtdsm_normalize.build_snapshot_table`)
   giving the latest-known level and year-over-year comparison that
   vintage actually contained.

No macro variable is tenor-specific -- every auction, regardless of
tenor, gets the same national macro features.

## Direct observation vs. stale carry-forward vs. source gap
(Phase 4 acceptance review, issue 6)

`{mnemonic}_level` is **the latest observation known to be safely
available as of the cutoff** -- not necessarily "this period's"
observation. Three distinct situations all pass through the same
snapshot mechanism, and this module now labels them explicitly rather
than presenting them as one undifferentiated "level":

- **Direct observation**: the matched vintage's own latest non-null
  observation is exactly the period this project would *expect* to be
  current for a vintage of that age (one period behind the vintage's
  own reference month, the ordinary case).
- **Latest safely-available prior observation (stale carry-forward)**:
  the vintage exists and has a non-null latest observation, but that
  observation is *older* than expected -- `{mnemonic}_source_gap_detected`
  is `True`. This is a valid, honest "as-of" feature (never a
  fabricated value), but it is stale, and callers must not mistake it
  for a direct reading of the expected period. This is exactly what
  happens for RUC in October 2025 (see
  `artifacts/rtdsm_macro_vintage_data_quality.md`'s worked example).
- **Source gap**: the specific reason the carry-forward happened --
  `{mnemonic}_source_gap_detected=True` always accompanies a stale
  carry-forward; it is never set for an ordinary, on-time reading.

Every row also carries `{mnemonic}_expected_latest_observation_date`
(what this project would expect, given only the matched vintage's own
age) and `{mnemonic}_availability_precision` (the release-calendar
method/provenance behind the matched vintage's timing) for audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.data.rtdsm_schema import (
    QUARTER_TO_MIDDLE_MONTH,
    RtdsmVariable,
)

NO_COVERAGE_REASON_TEMPLATE = (
    "no_{mnemonic}_vintage_published_before_cutoff (see artifacts/rtdsm_macro_vintage_data_quality.md)"
)
NO_NONNULL_OBSERVATION_REASON = (
    "matched_vintage_had_no_non_null_observation (source-reported gap in this vintage's own series, not imputed)"
)


def _expected_latest_observation_date(vintage_year: pd.Series, vintage_period: pd.Series, variable: RtdsmVariable) -> pd.Series:
    """The observation period this project would expect to be the
    latest non-null value in a vintage of this age, under ordinary
    (non-disrupted) conditions -- one period behind the vintage's own
    reference month. Used only to *detect* a source gap (issue 6),
    never to fabricate or select a value.
    """
    if variable.vintage_frequency == "quarterly":
        reference_month = vintage_period.map(QUARTER_TO_MIDDLE_MONTH)
    else:
        reference_month = vintage_period
    reference_date = pd.to_datetime({"year": vintage_year, "month": reference_month, "day": 1})
    return reference_date - pd.DateOffset(months=1)


def as_of_join_variable(
    auctions_df: pd.DataFrame,
    vintage_index_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    *,
    variable: RtdsmVariable,
    cutoff_col: str,
) -> pd.DataFrame:
    """Attach `{mnemonic}_level`, `{mnemonic}_observation_date`,
    `{mnemonic}_pct_yoy_chg`, `{mnemonic}_level_yoy_chg`,
    `{mnemonic}_vintage_label`, `{mnemonic}_vintage_lag_calendar_days`,
    `{mnemonic}_observation_age_calendar_days`,
    `{mnemonic}_join_matched`, and `{mnemonic}_missing_reason` to every
    auction row. No auction is ever dropped.
    """
    m = variable.mnemonic
    if "__as_of_join_row_order__" in auctions_df.columns:
        raise ValueError(
            "as_of_join_variable: auctions_df already has a '__as_of_join_row_order__' column -- reserved internally"
        )
    left = auctions_df.reset_index(drop=True).copy()
    left["__as_of_join_row_order__"] = left.index

    right = (
        vintage_index_df.loc[vintage_index_df["publication_safe_available_date"].notna()]
        .merge(snapshot_df, on="vintage_label", how="left")
        # Phase 5 acceptance-review fix (see the identical, verified-live
        # fix in cftc_join.py): sorting on `publication_safe_available_
        # date` alone leaves tie-breaking among equal keys dependent on
        # this table's own (arbitrary) pre-sort row order. No such tie
        # is currently observed in this project's RTDSM vintage indices,
        # but the fix is applied defensively here too, using each
        # vintage's own nominal publication date as the deterministic
        # secondary key (prefer the more recently published vintage
        # among ties).
        .sort_values(["publication_safe_available_date", "nominal_publication_date"])
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

    vintage_matched = merged["vintage_label"].notna()
    has_observation = vintage_matched & merged["current_value"].notna()

    merged[f"{m}_join_matched"] = vintage_matched
    merged[f"{m}_vintage_label"] = merged["vintage_label"]
    merged[f"{m}_availability_precision"] = merged["availability_precision"]
    merged[f"{m}_vintage_lag_calendar_days"] = (merged[cutoff_col] - merged["publication_safe_available_date"]).dt.days
    merged[f"{m}_observation_date"] = merged["current_observation_date"]
    merged[f"{m}_observation_age_calendar_days"] = (merged[cutoff_col] - merged["current_observation_date"]).dt.days
    merged[f"{m}_level"] = merged["current_value"]
    merged[f"{m}_level_yoy_chg"] = merged["current_value"] - merged["yoy_value"]
    merged[f"{m}_pct_yoy_chg"] = (merged["current_value"] / merged["yoy_value"] - 1.0) * 100.0

    # -- Direct observation vs. stale carry-forward vs. source gap (issue 6) --
    expected_latest = pd.Series(pd.NaT, index=merged.index)
    if vintage_matched.any():
        expected_latest.loc[vintage_matched] = _expected_latest_observation_date(
            merged.loc[vintage_matched, "vintage_year"], merged.loc[vintage_matched, "vintage_period"], variable
        )
    merged[f"{m}_expected_latest_observation_date"] = expected_latest
    merged[f"{m}_source_gap_detected"] = has_observation & (merged["current_observation_date"] < expected_latest)

    reason = np.full(len(merged), None, dtype=object)
    reason = np.where(~vintage_matched, NO_COVERAGE_REASON_TEMPLATE.format(mnemonic=m), reason)
    reason = np.where(vintage_matched & ~has_observation, NO_NONNULL_OBSERVATION_REASON, reason)
    merged[f"{m}_missing_reason"] = pd.array(reason, dtype="string")

    drop_cols = [
        "vintage_label", "vintage_year", "vintage_period", "nominal_publication_date",
        "publication_safe_available_date", "availability_precision", "current_observation_date",
        "current_value", "yoy_observation_date", "yoy_value",
    ]
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns])


def as_of_join_all(
    auctions_df: pd.DataFrame,
    vintage_indices: dict[str, pd.DataFrame],
    snapshots: dict[str, pd.DataFrame],
    *,
    variables: tuple[RtdsmVariable, ...],
    cutoff_col: str,
) -> pd.DataFrame:
    """Fold `as_of_join_variable` over every selected variable in turn."""
    merged = auctions_df
    for variable in variables:
        merged = as_of_join_variable(
            merged,
            vintage_indices[variable.mnemonic],
            snapshots[variable.mnemonic],
            variable=variable,
            cutoff_col=cutoff_col,
        )
    return merged
