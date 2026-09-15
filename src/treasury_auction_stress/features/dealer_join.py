"""Release-aware as-of joins between NY Fed Primary Dealer Statistics
and Treasury auctions, for both planned prediction cutoffs.

## The two cutoffs (docs/point_in_time_rules.md's "Planned prediction cutoffs")

- **Announcement-time cutoff**: the auction's own `announcemt_date`,
  treated (per the existing, Phase-2-acceptance-reviewed rule) as
  "public information as of the end of that calendar date" -- a
  date-level cutoff, not an intraday timestamp.
- **Pre-auction cutoff**: this project's first concrete definition of
  it, per `docs/project_plan.md`'s "previous business day's market
  close" -- computed as the closest calendar day before `auction_date`
  that is not a weekend or U.S. federal holiday
  (`time_utils.previous_business_day`). This uses the same federal
  holiday calendar as the dealer-statistics publication rule, **not**
  a SIFMA bond-market holiday calendar -- the one documented
  divergence is Good Friday (a SIFMA-recommended bond-market close,
  not a Federal Reserve holiday). Because dealer statistics are
  *weekly*, this divergence essentially never changes which dealer
  release a cutoff resolves to (it would only matter if it moved the
  cutoff across a Wednesday/Thursday boundary), so it is disclosed
  here and in `artifacts/primary_dealer_data_quality.md` rather than
  engineered around.

## The join itself

Both cutoffs are matched against `publication_safe_available_date`
(see `dealer_stats_normalize.py`) using `pandas.merge_asof` with
`direction="backward"` -- which, by construction, can never select a
right-hand row with a later key than the left-hand row, i.e. it is
structurally incapable of selecting a not-yet-published observation or
backfilling from the future. Every row this project processes carries
its own `dealer_observation_age_days`, whether it matched at all
(`dealer_join_matched`), and, per series, a specific reason when the
value is missing (`{series}_missing_reason`) -- distinguishing "no
dealer data exists this early at all," "this specific series' regime
had not started yet," and "the source itself reported a missing
value for the nearest available release." No auction is ever dropped
for lacking dealer features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.data.dealer_stats_schema import (
    LEGACY_PERIOD_FIRST_OBSERVATION,
    REGIME_START_DATES,
    REGIME_START_OVERRIDES,
    SELECTED_SERIES,
)
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,  # noqa: F401 -- re-exported: existing callers/tests import this from here
)
from treasury_auction_stress.features.dealer_candidate_features import (
    HARMONIZED_STABLE_NAMES,
    add_inventory_relative_to_offering,
    add_inventory_relative_to_offering_harmonized,
)

# Every selected (current-API) series, plus the harmonized,
# cross-schema-period features added by the Phase 3 acceptance review
# -- both participate in the join and get their own per-row missing
# reason. The legacy-only raw component columns
# (`dealer_stats_schema.LEGACY_COMPONENT_STABLE_NAMES`) and the
# discontinued middle-piece column deliberately do NOT -- they are
# internal building blocks for the harmonized columns, not
# user-facing final features.
STABLE_SERIES_NAMES: tuple[str, ...] = tuple(s.stable_name for s in SELECTED_SERIES) + HARMONIZED_STABLE_NAMES
_REGIME_START_BY_STABLE_NAME: dict[str, pd.Timestamp] = {
    s.stable_name: pd.Timestamp(REGIME_START_OVERRIDES.get(s.stable_name, REGIME_START_DATES[s.regime_id]))
    for s in SELECTED_SERIES
}
_REGIME_START_BY_STABLE_NAME.update(
    {name: pd.Timestamp(LEGACY_PERIOD_FIRST_OBSERVATION) for name in HARMONIZED_STABLE_NAMES}
)

NO_COVERAGE_REASON = (
    "no_dealer_release_published_before_cutoff (this project's earliest dealer-data "
    "coverage, for the historically-extended and harmonized series, begins "
    "2001-07-04; most series begin 2013-04-03; see "
    "the Phase 3 acceptance review)"
)
REGIME_NOT_STARTED_REASON_TEMPLATE = "series_regime_not_yet_started (starts {start})"
SOURCE_MISSING_REASON = (
    "source_reported_missing_value_for_nearest_available_release "
    "(NY Fed published '*' for this week; not imputed or forward-filled)"
)


def as_of_join(
    auctions_df: pd.DataFrame,
    dealer_wide_df: pd.DataFrame,
    *,
    cutoff_col: str,
) -> pd.DataFrame:
    """Backward as-of join of every selected dealer series (plus any
    derived feature columns already present in `dealer_wide_df`) onto
    `auctions_df`, gated by `cutoff_col`.

    `dealer_wide_df` must carry `observation_date` and
    `publication_safe_available_date` (see
    `dealer_stats_normalize.pivot_wide`); every other column is joined
    as-is. Auctions are never dropped, never matched to a later
    release, and never backfilled across a missing week.

    Robust to `NaT` on either side (found by the Phase 3 acceptance
    review's boundary tests): `pandas.merge_asof` itself raises on a
    null merge key rather than treating it as "no match," so a `NaT`
    cutoff is set aside and reported as unmatched instead of being
    passed into the merge, and a dealer-table row with no real
    `observation_date`/`publication_safe_available_date` (corrupt or
    nonsensical -- never expected from this project's own pipeline,
    but not assumed) is excluded from the merge candidates entirely
    rather than being eligible to match anything. Output rows are
    restored to `auctions_df`'s original order.
    """
    if "__as_of_join_row_order__" in auctions_df.columns:
        raise ValueError(
            "as_of_join: auctions_df already has a '__as_of_join_row_order__' "
            "column -- this name is reserved internally"
        )
    left = auctions_df.reset_index(drop=True).copy()
    left["__as_of_join_row_order__"] = left.index

    right = (
        dealer_wide_df.loc[
            dealer_wide_df["observation_date"].notna()
            & dealer_wide_df["publication_safe_available_date"].notna()
        ]
        # Phase 5 acceptance-review fix (see the identical, verified-live
        # fix in cftc_join.py): sorting on `publication_safe_available_
        # date` alone leaves tie-breaking among equal keys dependent on
        # this table's own (arbitrary) pre-sort row order. No such tie
        # is currently observed in this project's dealer-statistics
        # data, but the fix is applied defensively here too, for the
        # same reason and with the same deterministic rule (prefer the
        # more recently observed release among ties).
        .sort_values(["publication_safe_available_date", "observation_date"])
        .reset_index(drop=True)
    )

    has_cutoff = left[cutoff_col].notna()
    right_only_cols = [c for c in right.columns if c not in left.columns]
    right_only_empty = right[right_only_cols].iloc[0:0]  # zero rows, dtypes preserved

    def _with_right_columns(df_slice: pd.DataFrame) -> pd.DataFrame:
        # Reindexing an empty-but-typed frame to a non-empty index
        # fills with NaN/NaT while preserving each column's real dtype
        # (datetime64 stays datetime64, Float64 stays Float64) --
        # unlike `DataFrame.reindex(columns=...)` on a non-empty frame,
        # which fills new columns with plain float NaN regardless of
        # the intended dtype and breaks downstream datetime arithmetic.
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
    matched = merged["observation_date"].notna()
    merged["dealer_join_matched"] = matched
    merged["dealer_observation_age_days"] = (
        merged[cutoff_col] - merged["observation_date"]
    ).dt.days

    joined_series_names = [c for c in STABLE_SERIES_NAMES if c in merged.columns]
    for name in joined_series_names:
        missing = merged[name].isna()
        before_regime = matched & (merged["observation_date"] < _REGIME_START_BY_STABLE_NAME[name])
        reason = np.full(len(merged), None, dtype=object)
        reason = np.where(~matched, NO_COVERAGE_REASON, reason)
        reason = np.where(
            missing & before_regime,
            REGIME_NOT_STARTED_REASON_TEMPLATE.format(
                start=_REGIME_START_BY_STABLE_NAME[name].date()
            ),
            reason,
        )
        reason = np.where(missing & matched & ~before_regime, SOURCE_MISSING_REASON, reason)
        merged[f"{name}_missing_reason"] = pd.array(reason, dtype="string")

    return merged


def coverage_summary(
    joined_df: pd.DataFrame, *, group_cols: tuple[str, ...] = ("tenor",)
) -> pd.DataFrame:
    """Per-group (e.g. tenor, or tenor+year) coverage: how many auctions
    matched any dealer release at all, and, for each series, how many
    had a genuine (non-missing) value versus each missing reason.
    """
    rows = []
    for keys, group in joined_df.groupby(list(group_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict = dict(zip(group_cols, keys, strict=True))
        row["n_auctions"] = len(group)
        row["n_matched_any_dealer_release"] = int(group["dealer_join_matched"].sum())
        row["median_dealer_observation_age_days"] = group["dealer_observation_age_days"].median()
        for name in STABLE_SERIES_NAMES:
            if name not in group.columns:
                continue
            row[f"{name}_n_available"] = int(group[name].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def worked_example(
    joined_announcement: pd.DataFrame,
    joined_pre_auction: pd.DataFrame,
    dealer_wide_df: pd.DataFrame,
    *,
    example_series: str,
    n_candidates: int = 4,
) -> dict:
    """Build one fully-explained worked example: an auction, its two
    cutoffs, the candidate dealer observations near it, which one each
    cutoff selected, and why the later ones were rejected. Picks the
    most recent auction with a matched release under both cutoffs so
    the example reflects current, real data rather than a synthetic
    case.
    """
    # CUSIPs are reused across reopenings, so candidates must be
    # matched on (cusip, auction_date) together, never cusip alone.
    key_cols = ["cusip", "auction_date"]
    both_matched_keys = joined_announcement.loc[
        joined_announcement["dealer_join_matched"], key_cols
    ].merge(joined_pre_auction.loc[joined_pre_auction["dealer_join_matched"], key_cols], on=key_cols)
    chosen_key = both_matched_keys.sort_values("auction_date").iloc[-1]

    example_row = joined_announcement.loc[
        (joined_announcement["cusip"] == chosen_key["cusip"])
        & (joined_announcement["auction_date"] == chosen_key["auction_date"])
    ].iloc[0]
    cusip = example_row["cusip"]
    pre_auction_row = joined_pre_auction.loc[
        (joined_pre_auction["cusip"] == chosen_key["cusip"])
        & (joined_pre_auction["auction_date"] == chosen_key["auction_date"])
    ].iloc[0]

    candidates = (
        dealer_wide_df.loc[dealer_wide_df["observation_date"] <= example_row["auction_date"]]
        .sort_values("observation_date")
        .tail(n_candidates)
        .copy()
    )
    candidates["selected_by_announcement_cutoff"] = (
        candidates["observation_date"] == example_row["observation_date"]
    )
    candidates["selected_by_pre_auction_cutoff"] = (
        candidates["observation_date"] == pre_auction_row["observation_date"]
    )
    candidates["rejected_because"] = candidates.apply(
        lambda r: (
            None
            if r["selected_by_announcement_cutoff"] or r["selected_by_pre_auction_cutoff"]
            else (
                "publication_safe_available_date is after at least one cutoff"
                if r["publication_safe_available_date"] > example_row[ANNOUNCEMENT_CUTOFF_COL]
                else "a more recent, still-safely-published release exists and is preferred"
            )
        ),
        axis=1,
    )

    return {
        "auction_date": str(example_row["auction_date"].date()),
        "tenor": example_row["tenor"],
        "cusip": cusip,
        "announcement_cutoff_date": str(example_row[ANNOUNCEMENT_CUTOFF_COL].date()),
        "pre_auction_cutoff_date": str(example_row[PRE_AUCTION_CUTOFF_COL].date()),
        "candidates": candidates[
            [
                "observation_date",
                "publication_date",
                "publication_safe_available_date",
                example_series,
                "selected_by_announcement_cutoff",
                "selected_by_pre_auction_cutoff",
                "rejected_because",
            ]
        ].to_dict("records"),
        "announcement_cutoff_selected_observation_date": str(example_row["observation_date"].date()),
        "pre_auction_cutoff_selected_observation_date": str(pre_auction_row["observation_date"].date()),
        "announcement_cutoff_selected_value": example_row[example_series],
        "pre_auction_cutoff_selected_value": pre_auction_row[example_series],
    }


# Phase 5: same (auction id columns) + (per-cutoff, prefixed feature
# columns) shape as
# `treasury_auction_stress.features.phase4d_source_joins`'s three
# market/macro join-table builders -- added here, in Phase 3's own join
# module, rather than in the Phase 4D module (whose docstring is
# explicit that it covers Treasury rates/CFTC/RTDSM only), so Phase 5's
# combined feature-matrix builder can treat all four source families
# uniformly without re-deriving this shape a fourth time. These
# constants are deliberately duplicated (not imported) from
# `phase4d_source_joins.AUCTION_ID_COLS`/`CUTOFF_COLS` -- both are
# derived from `treasury_auction_stress.features.auction_cutoffs`, and
# duplicating two small tuples of column names avoids an awkward
# Phase-3-imports-from-Phase-4 dependency direction for no real benefit.
DEALER_AUCTION_ID_COLS: tuple[str, ...] = ("cusip", "tenor", "auction_date", "announcemt_date", "is_reopening")
DEALER_CUTOFF_COLS: tuple[str, ...] = (ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL)


def build_dealer_join_table(auctions_df: pd.DataFrame, dealer_wide_df: pd.DataFrame) -> pd.DataFrame:
    """One row per `auctions_df` row: auction identifiers plus, for each
    of the two prediction cutoffs, every joined dealer-statistics
    feature column (raw series levels, weekly/trailing changes, rolling
    z-scores, harmonized buckets, and tenor-matched inventory ratios),
    prefixed `{cutoff_col}__`. Mirrors
    `phase4d_source_joins.build_rates_join_table` /
    `build_cftc_join_table` / `build_rtdsm_join_table` exactly, so
    `treasury_auction_stress.features.feature_matrix` can build all
    four source joins the same way.
    """
    out = auctions_df[list(DEALER_AUCTION_ID_COLS) + list(DEALER_CUTOFF_COLS)].copy()
    for cutoff_col in DEALER_CUTOFF_COLS:
        joined = as_of_join(auctions_df, dealer_wide_df, cutoff_col=cutoff_col)
        joined = add_inventory_relative_to_offering(joined)
        joined = add_inventory_relative_to_offering_harmonized(joined)
        new_cols = [c for c in joined.columns if c not in auctions_df.columns]
        prefixed = joined[new_cols].add_prefix(f"{cutoff_col}__")
        out = pd.concat([out.reset_index(drop=True), prefixed.reset_index(drop=True)], axis=1)
    return out
