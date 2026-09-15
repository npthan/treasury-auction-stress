"""Phase 7: the single shared, point-in-time-safe "trailing mean of the
N most recent same-group observations, as of THIS row's own cutoff"
primitive.

Two Phase 7 consumers share this exact implementation (never two
independently-written versions that could silently drift):

- `treasury_auction_stress.models.adaptive_baselines.
  AdaptiveRecentHistoryBaseline` -- the adaptive recent-history
  benchmark, which is allowed to update within a test year.
- `treasury_auction_stress.evaluation.stress_event`'s safe regime
  feature, replacing Phase 6's flagged-leaky
  `dealer_absorption.add_regime_feature`.

## Why this is safe by construction, not merely "usually safe"

For row *i* with its own prediction cutoff `cutoff_i` (announcement or
pre-auction, per view), a candidate row *j* (same group) is eligible
only if `j`'s own `safe_date_col` value is `<= cutoff_i`. This is
checked directly against an actual availability timestamp, never
against `date_col` sort order or row position. Two concrete
consequences, both load-bearing:

1. **Same-day announcements can never be given a manufactured
   sequence.** If two same-tenor auctions share an `announcemt_date`,
   neither can ever borrow the other's `announcement_cutoff_date`-view
   result, because both auctions' own results become safely available
   only after both of their own (necessarily later) auction dates --
   this falls out of the availability-timestamp comparison itself, not
   from any special-cased tie-break.
2. **A row can never see its own result.** `cutoff_i` is always
   `<= auction_date_i <= safe_date_i` by construction (the
   safe-availability rule only ever advances a date forward), so row
   *i* is never its own eligible candidate -- proven directly in
   `tests/test_phase7_safe_as_of_lookback.py`, and defended a second,
   redundant way here by explicitly excluding `key_col == own key`.

Because `safe_date_col` is a monotone-non-decreasing function of
`date_col` (the shared `next_full_business_day_after`/
`previous_business_day` rules never move a later auction's safe date
before an earlier auction's), sorting eligible candidates by
`safe_date_col` agrees with sorting by `date_col` -- so "the
n_lookback most recent eligible observations" is well-defined via a
single sort, with `(date_col, key_col)` as the deterministic tie-break
(never row position, which `pandas` does not otherwise guarantee is
stable across an arbitrary input order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_as_of_trailing_mean(
    pool: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    key_col: str,
    cutoff_col: str,
    safe_date_col: str,
    date_col: str = "auction_date",
    n_lookback: int = 8,
) -> pd.DataFrame:
    """For every row of `pool`, compute the mean of `value_col` over the
    up-to-`n_lookback` most recent same-`group_col` rows whose OWN
    `safe_date_col` is `<= this row's own cutoff_col` -- never the
    row's own value, never a row ordered only by position.

    Returns a copy of `pool` with two new columns:
    `f"{value_col}_safe_trailing_mean"` (NaN if zero eligible
    candidates exist) and `f"{value_col}_safe_trailing_n_used"` (0 in
    that case). Rows with a missing `value_col` are still valid
    candidates to exclude from *other* rows' windows (they simply
    never contribute a usable observation) and still receive their own
    output row.

    Row-order invariant: shuffling `pool`'s input order never changes
    any row's own output, because every row's eligible-candidate set
    and tie-break are defined by column values (`safe_date_col`,
    `date_col`, `key_col`), never by position.
    """
    mean_col = f"{value_col}_safe_trailing_mean"
    n_used_col = f"{value_col}_safe_trailing_n_used"
    if mean_col in pool.columns or n_used_col in pool.columns:
        raise ValueError(
            f"safe_as_of_trailing_mean: {mean_col!r}/{n_used_col!r} already present in `pool` -- this usually "
            "means the same pool was passed through this function twice (e.g. once before and once after "
            "concatenating train and test), which would silently produce duplicate columns. Compute this "
            "feature exactly once, over the full pool, before any train/test split."
        )

    out = pool.copy()
    out[mean_col] = np.nan
    out[n_used_col] = 0

    for _, group in pool.groupby(group_col):
        candidates = group.dropna(subset=[value_col]).sort_values([date_col, key_col])
        cand_safe_dates = candidates[safe_date_col].to_numpy()
        cand_values = candidates[value_col].to_numpy(dtype="float64")
        cand_keys = candidates[key_col].to_numpy()

        for idx, row in group.iterrows():
            # searchsorted requires ascending array; cand_safe_dates is
            # ascending because candidates are sorted by (date_col,
            # key_col) and safe_date_col is monotone non-decreasing in
            # date_col (see module docstring).
            pos = int(np.searchsorted(cand_safe_dates, np.datetime64(row[cutoff_col]), side="right"))
            window_keys = cand_keys[max(0, pos - n_lookback) : pos]
            window_values = cand_values[max(0, pos - n_lookback) : pos]

            # Defensive, redundant self-exclusion (see module docstring
            # point 2) -- structurally should never fire, but a single
            # dropped self-reference must never silently poison a mean.
            self_mask = window_keys != row[key_col]
            usable = window_values[self_mask]

            if usable.size:
                out.at[idx, mean_col] = float(usable.mean())
                out.at[idx, n_used_col] = int(usable.size)

    return out
