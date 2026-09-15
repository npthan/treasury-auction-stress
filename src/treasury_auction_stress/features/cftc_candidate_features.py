"""Candidate, past-only positioning features built from the normalized
CFTC TFF Futures Only table (`cftc_normalize.normalize_cftc_positioning`'s
output).

## Units and semantics discipline (task requirement, verbatim)

- Positions stay in **contracts** the entire way through this module.
  Nothing here relabels a contract count as a dollar amount, notional
  exposure, duration, or DV01 -- that would require a per-contract
  multiplier this project has not verified and does not apply.
- `spreading` positions are **never** added to either the long or the
  short side of a category's net position -- a spread position has no
  net directional exposure by definition, and CFTC reports it
  separately for exactly that reason.
- CFTC's `Dealer/Intermediary` category is **not** NY Fed's primary
  dealer universe (see `cftc_schema.py`'s module docstring) -- nothing
  here merges or compares the two.
- No full-sample z-score, percentile, or rank is computed anywhere in
  this module (the task's explicit prohibition) -- only past-only
  level, share, and change features, each depending solely on that
  contract's own history up to and including the current report.
"""

from __future__ import annotations

import pandas as pd

CATEGORY_DIRECTIONAL_FIELDS: dict[str, tuple[str, str, str]] = {
    "dealer": ("dealer_positions_long_all", "dealer_positions_short_all", "dealer_positions_spread_all"),
    "asset_mgr": ("asset_mgr_positions_long", "asset_mgr_positions_short", "asset_mgr_positions_spread"),
    "lev_money": ("lev_money_positions_long", "lev_money_positions_short", "lev_money_positions_spread"),
    "other_rept": ("other_rept_positions_long", "other_rept_positions_short", "other_rept_positions_spread"),
}
TOTAL_REPORTABLE_FIELDS = ("tot_rept_positions_long_all", "tot_rept_positions_short")
NONREPORTABLE_FIELDS = ("nonrept_positions_long_all", "nonrept_positions_short_all")

DEFAULT_CHANGE_WINDOWS_WEEKS: tuple[int, ...] = (1, 4)


def add_net_positions(df: pd.DataFrame) -> pd.DataFrame:
    """`{category}_net_contracts` = long - short, in contracts.
    Spreading is left as its own untouched column, never folded in.
    """
    out = df.copy()
    for category, (long_col, short_col, _spread_col) in CATEGORY_DIRECTIONAL_FIELDS.items():
        out[f"{category}_net_contracts"] = out[long_col] - out[short_col]
    out["tot_rept_net_contracts"] = out[TOTAL_REPORTABLE_FIELDS[0]] - out[TOTAL_REPORTABLE_FIELDS[1]]
    out["nonrept_net_contracts"] = out[NONREPORTABLE_FIELDS[0]] - out[NONREPORTABLE_FIELDS[1]]
    return out


def add_share_of_open_interest(df: pd.DataFrame) -> pd.DataFrame:
    """`{category}_{side}_pct_oi` = that field divided by
    `open_interest_all` for the *same row* (same contract, same
    report week) -- a same-week ratio of two already-published
    figures, not a rolling or cross-week computation, so it carries no
    additional leakage risk beyond the row's own publication timing.
    """
    out = df.copy()
    oi = out["open_interest_all"]
    for category, (long_col, short_col, spread_col) in CATEGORY_DIRECTIONAL_FIELDS.items():
        out[f"{category}_long_pct_oi"] = out[long_col] / oi
        out[f"{category}_short_pct_oi"] = out[short_col] / oi
        out[f"{category}_spread_pct_oi"] = out[spread_col] / oi
    return out


def add_week_over_week_changes(
    df: pd.DataFrame, *, change_windows_weeks: tuple[int, ...] = DEFAULT_CHANGE_WINDOWS_WEEKS
) -> pd.DataFrame:
    """`{category}_net_contracts_chg_{n}w` -- the change in net
    position over the trailing `n` *observation weeks* (not calendar
    weeks -- a disrupted report published late still occupies its own
    original weekly observation slot, so this is robust to publication
    delays). Computed within each `contract_code` group, sorted by
    `report_date` ascending, via a plain `.diff(n)` -- since this
    project's TFF ingestion has no missing observation weeks within a
    contract's series (delayed reports still arrive, just later), a
    week-indexed `diff` is exact, not an approximation.
    """
    out = df.sort_values(["contract_code", "report_date"]).reset_index(drop=True).copy()
    net_cols = [f"{category}_net_contracts" for category in CATEGORY_DIRECTIONAL_FIELDS] + [
        "tot_rept_net_contracts",
        "nonrept_net_contracts",
    ]
    grouped = out.groupby("contract_code", sort=False)
    for col in net_cols:
        if col not in out.columns:
            continue
        for window in change_windows_weeks:
            out[f"{col}_chg_{window}w"] = grouped[col].diff(window)
    return out


def build_positioning_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every candidate transform once, in a fixed order."""
    out = add_net_positions(df)
    out = add_share_of_open_interest(out)
    out = add_week_over_week_changes(out)
    return out
