"""Phase 5: candidate, point-in-time auction-structure features.

Every feature here uses only fields known from the public auction
announcement (`announcemt_date`, `auction_date`, `tenor`, `cusip`,
`is_reopening`, `offering_amt`) -- never an auction-result field (see
`docs/target_specification.md` and `treasury_auction_stress.features.
phase4d_source_joins.FORBIDDEN_RESULT_COLUMNS`). Because these fields
are all fixed at (or before) announcement, this module's output is
identical for the announcement-cutoff and pre-auction-cutoff matrices
-- it is the shared, cutoff-independent "auction structure" block of
the Phase 5 core predictor schema.

## The same-day-announcement rule, made concrete

`docs/point_in_time_rules.md` and the Phase 5 task specification both
require that trailing/"previous auction" features never use same-day
row ordering to fake precedence: two auctions announced on the exact
same calendar date must never be treated as one preceding the other.
This is not a hypothetical edge case in this project's own data --
verified directly (`ANALYSIS_START_DATE` onward): 35 auctions share
their own tenor's announcement date with another same-tenor auction
(e.g. a reopening and a new issue of the same tenor announced the same
day), and 401 distinct announcement dates cover more than one auction
across tenors (routine multi-tenor "refunding" announcement bundles).

Every trailing/previous-auction feature below is therefore built via
one of two mechanisms, both structurally incapable of using a same-day
row to precede another same-day row:

- `pandas.merge_asof(..., direction="backward", allow_exact_matches=False)`
  for "the single most recent prior auction" lookups -- `allow_exact_matches
  =False` means a right-hand row whose key exactly equals the left-hand
  key is never selected, so a same-day announcement can never match
  itself or another same-day announcement as its own "previous" auction.
- `_distinct_date_trailing_sum` (below) for trailing-window sums: values
  are first aggregated to one slot per **distinct** `announcemt_date`
  (summing every auction announced that day into that single slot), a
  `shift(1)` + rolling window is computed over that per-date series (so
  the current date's own slot is always excluded), and the result is
  then broadcast back to every auction sharing that date. Two auctions
  announced the same day therefore always receive the identical
  trailing value, and neither one's own supply is included in it.

## No lookahead, no backfill, no target-performance selection

Every function here depends only on `announcemt_date`-ordered history
strictly before the current row. Windows (`TRAILING_ALL_TENOR_WINDOWS_
AUCTIONS`, `TRAILING_SAME_TENOR_WINDOWS_AUCTIONS`) are pre-specified,
justified by this project's own observed auction cadence (see their
docstrings), never chosen by looking at any target correlation. Rows
with no legitimate prior history (e.g. the first-ever auction of a
tenor, or a tenor's first several announcements) get `pd.NA`/`np.nan`,
never zero or a backfilled value from later history.

Includes a dedicated outcome-poisoning test in
`tests/test_auction_candidate_features.py`: replacing every auction-
result field with extreme/randomized values leaves every one of this
module's outputs bit-identical, because none of them ever reads a
result field in the first place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNOUNCEMENT_DATE_COL = "announcemt_date"
AUCTION_DATE_COL = "auction_date"
TENOR_COL = "tenor"
CUSIP_COL = "cusip"
REOPENING_COL = "is_reopening"
OFFERING_COL = "offering_amt"

# Cadence justification (verified against the ordinary modeling sample,
# 2010-01-01 onward): ~1279 nominal-coupon auctions announced over
# ~16.7 years is ~76-77 announcements/year, ~6.4/month. A window of 8
# distinct announcement dates approximates one month of aggregate
# nominal-coupon supply; 26 approximates one quarter. Chosen for
# cadence, never for any relationship with a target.
TRAILING_ALL_TENOR_WINDOWS_AUCTIONS: tuple[int, ...] = (8, 26)

# Same-tenor cadence varies materially by tenor (2-Year is roughly
# monthly; 20-Year is roughly quarterly), so these windows are
# expressed in "distinct prior same-tenor announcement dates," not
# calendar time: 4 approximates the last 4 months for a monthly tenor
# or the last year for a quarterly one; 8 doubles that lookback for
# every tenor. Both are small, fixed, and pre-specified -- never
# chosen by looking at target performance.
TRAILING_SAME_TENOR_WINDOWS_AUCTIONS: tuple[int, ...] = (4, 8)


def add_offering_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """`log_offering_amt` = `log1p(offering_amt)` -- a deterministic,
    fixed transform (never fit on the data), used because offering
    sizes span several orders of magnitude ($25mm for the two special
    auctions up to ~$70bn) and typical structural models of auction
    size effects are naturally expressed in log terms (see
    `treasury_auction_stress.features.dealer_absorption`, whose own
    `offering_slope` acts on `log(offering_amt)`).
    """
    out = df.copy()
    out["log_offering_amt"] = np.log1p(out[OFFERING_COL].astype("float64"))
    return out


def add_announcement_to_auction_gap(df: pd.DataFrame) -> pd.DataFrame:
    """`days_announcement_to_auction`: calendar days between the public
    announcement and the auction itself -- both dates are fixed and
    publicly known at announcement time (the announcement discloses the
    auction date), so this is safe for both prediction cutoffs.
    """
    out = df.copy()
    out["days_announcement_to_auction"] = (
        out[AUCTION_DATE_COL] - out[ANNOUNCEMENT_DATE_COL]
    ).dt.days
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar fields known before the auction: the auction's own
    month/quarter/day-of-week (all public at announcement, since the
    auction date itself is announced), and the announcement's own
    day-of-week. Plain integer encodings (1-12 for month, 1-4 for
    quarter, 0-6 for day-of-week per `pandas`' Monday=0 convention) --
    categorical encoding is explicitly deferred to Phase 6 training
    folds, per this phase's "no full-sample learned preprocessing" rule.
    """
    out = df.copy()
    out["auction_month"] = out[AUCTION_DATE_COL].dt.month
    out["auction_quarter"] = out[AUCTION_DATE_COL].dt.quarter
    out["auction_day_of_week"] = out[AUCTION_DATE_COL].dt.dayofweek
    out["announcement_day_of_week"] = out[ANNOUNCEMENT_DATE_COL].dt.dayofweek
    return out


def _distinct_date_trailing_sum(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    window: int,
    group_cols: tuple[str, ...] = (),
) -> np.ndarray:
    """Trailing sum of `value_col` over the `window` most recent
    **distinct** values of `date_col` strictly before each row's own
    `date_col` value, optionally computed within `group_cols`. See the
    module docstring's "same-day-announcement rule" section for why
    this is safe against same-day ties: every auction sharing a given
    `date_col` value (within `group_cols`, if any) is summed into one
    slot for that date before the rolling window is computed, and
    `shift(1)` guarantees the current date's own slot is excluded.
    """
    group_list = list(group_cols)
    working = df[[*group_list, date_col]].copy()
    working["_value"] = df[value_col].astype("float64")

    agg_keys = group_list + [date_col]
    agg = working.groupby(agg_keys, dropna=False)["_value"].sum().reset_index()
    agg = agg.sort_values(agg_keys).reset_index(drop=True)

    if group_list:
        agg["_trailing"] = agg.groupby(group_list, dropna=False)["_value"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).sum()
        )
    else:
        agg["_trailing"] = agg["_value"].shift(1).rolling(window=window, min_periods=1).sum()

    merged = working[agg_keys].merge(agg[agg_keys + ["_trailing"]], on=agg_keys, how="left")
    return merged["_trailing"].to_numpy()


def add_trailing_supply_features(
    df: pd.DataFrame,
    *,
    all_tenor_windows: tuple[int, ...] = TRAILING_ALL_TENOR_WINDOWS_AUCTIONS,
    same_tenor_windows: tuple[int, ...] = TRAILING_SAME_TENOR_WINDOWS_AUCTIONS,
) -> pd.DataFrame:
    """`trailing_nominal_coupon_supply_{w}` (all tenors pooled) and
    `trailing_same_tenor_supply_{w}` (within the row's own tenor) --
    the sum of `offering_amt` announced over the `w` most recent
    distinct prior announcement dates (see `_distinct_date_trailing_sum`).
    Both are `NaN` for the first announcement date(s) of the relevant
    history (pooled, or the row's own tenor) -- never zero, never
    backfilled.
    """
    out = df.copy()
    for window in all_tenor_windows:
        out[f"trailing_nominal_coupon_supply_{window}"] = _distinct_date_trailing_sum(
            out, date_col=ANNOUNCEMENT_DATE_COL, value_col=OFFERING_COL, window=window
        )
    for window in same_tenor_windows:
        out[f"trailing_same_tenor_supply_{window}"] = _distinct_date_trailing_sum(
            out,
            date_col=ANNOUNCEMENT_DATE_COL,
            value_col=OFFERING_COL,
            window=window,
            group_cols=(TENOR_COL,),
        )
    return out


def add_previous_same_tenor_features(df: pd.DataFrame) -> pd.DataFrame:
    """`previous_same_tenor_offering_amt`, `change_in_offering_amt_vs_
    previous_same_tenor`, and `days_since_prior_same_tenor_auction` --
    all derived from the single most recent *strictly earlier* same-
    tenor announcement, found via `merge_asof(..., direction="backward",
    allow_exact_matches=False)` per tenor. `allow_exact_matches=False`
    is what makes this safe against the verified same-day same-tenor
    ties in this project's own data (see module docstring): a same-day
    announcement is never selected as its own or another same-day
    announcement's "previous" auction.

    `pd.NA`/`NaN` for a tenor's first-ever announcement (e.g. the
    20-Year bond's 2020-05-20 reintroduction) -- never backfilled.
    """
    out = df.reset_index(drop=True).copy()
    out["__row_id__"] = out.index

    n = len(out)
    prev_offering = np.full(n, np.nan)
    prev_auction_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    for tenor, group in out.groupby(TENOR_COL, sort=False):
        left = group[["__row_id__", ANNOUNCEMENT_DATE_COL]].sort_values(ANNOUNCEMENT_DATE_COL)
        right = (
            group[[ANNOUNCEMENT_DATE_COL, OFFERING_COL, AUCTION_DATE_COL]]
            .sort_values(ANNOUNCEMENT_DATE_COL)
            .rename(
                columns={
                    OFFERING_COL: "_prev_offering",
                    AUCTION_DATE_COL: "_prev_auction_date",
                }
            )
        )
        matched = pd.merge_asof(
            left,
            right,
            on=ANNOUNCEMENT_DATE_COL,
            direction="backward",
            allow_exact_matches=False,
        )
        idx = matched["__row_id__"].to_numpy()
        prev_offering[idx] = matched["_prev_offering"].to_numpy()
        prev_auction_date[idx] = matched["_prev_auction_date"].to_numpy()

    out["previous_same_tenor_offering_amt"] = prev_offering
    out["change_in_offering_amt_vs_previous_same_tenor"] = (
        out[OFFERING_COL].astype("float64") - prev_offering
    )
    prev_auction_series = pd.Series(prev_auction_date, index=out.index)
    # The gap between this auction's own (announced, publicly known)
    # auction_date and the previous same-tenor auction's own
    # auction_date -- both dates are fixed and public no later than
    # this row's own announcement, so this is safe for both cutoffs.
    out["days_since_prior_same_tenor_auction"] = (
        out[AUCTION_DATE_COL] - prev_auction_series
    ).dt.days
    return out.drop(columns="__row_id__")


def add_prior_reopening_count(df: pd.DataFrame) -> pd.DataFrame:
    """`n_prior_reopenings_of_cusip`: how many strictly earlier-announced
    auctions share this row's own CUSIP and are themselves reopenings
    (CUSIPs are reused across reopenings of the same security -- see
    `treasury_auction_stress.features.dealer_join`'s module docstring).
    0 for a CUSIP's first (necessarily non-reopening) appearance, safe
    against same-day ties the same way as
    `add_previous_same_tenor_features` (per-CUSIP `merge_asof` with
    `allow_exact_matches=False` against a per-distinct-date cumulative
    count, never a same-day row treated as preceding another).
    """
    out = df.reset_index(drop=True).copy()
    out["__row_id__"] = out.index

    n = len(out)
    prior_count = np.zeros(n, dtype="int64")

    for _cusip, group in out.groupby(CUSIP_COL, sort=False):
        by_date = (
            group.groupby(ANNOUNCEMENT_DATE_COL, dropna=False)[REOPENING_COL]
            .sum()
            .reset_index()
            .sort_values(ANNOUNCEMENT_DATE_COL)
        )
        by_date["_cumulative_prior"] = by_date[REOPENING_COL].shift(1).fillna(0).cumsum()

        left = group[["__row_id__", ANNOUNCEMENT_DATE_COL]].sort_values(ANNOUNCEMENT_DATE_COL)
        matched = left.merge(
            by_date[[ANNOUNCEMENT_DATE_COL, "_cumulative_prior"]], on=ANNOUNCEMENT_DATE_COL, how="left"
        )
        idx = matched["__row_id__"].to_numpy()
        prior_count[idx] = matched["_cumulative_prior"].to_numpy()

    out["n_prior_reopenings_of_cusip"] = prior_count
    return out.drop(columns="__row_id__")


def build_auction_structure_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every candidate auction-structure transform once, in a
    fixed order. Cutoff-independent -- safe to call once and reuse for
    both the announcement and pre-auction predictor matrices.
    """
    out = add_offering_amount_features(df)
    out = add_announcement_to_auction_gap(out)
    out = add_calendar_features(out)
    out = add_trailing_supply_features(out)
    out = add_previous_same_tenor_features(out)
    out = add_prior_reopening_count(out)
    return out
