"""Release-aware as-of joins between CFTC TFF Futures Only positioning
and Treasury auctions, for both prediction cutoffs.

Same `pandas.merge_asof(..., direction="backward")` mechanism, keyed
on `publication_safe_available_date` (never `report_date` itself), and
the same NaT-robust `__as_of_join_row_order__` pattern as
`treasury_rates_join.as_of_join` / `dealer_join.as_of_join`.

## Tenor matching is honest about being many-to-one, not forced 1:1

CFTC Treasury futures do not line up one-to-one with Treasury auction
tenors (`cftc_schema.TENOR_TO_CONTRACT_CODES`): 3-Year and 7-Year have
no futures contract at all, and 20-Year has no *direct* contract, only
bond-future context. `as_of_join` attaches every selected contract's
features (via the wide, contract-prefixed table from
`cftc_normalize.pivot_wide_by_contract`) to every auction row --
`add_tenor_matched_positioning_features` then explicitly selects the
*primary, direct-match* contract's features for an auction's own
tenor (`cftc_schema.TENOR_PRIMARY_CONTRACT_CODE`), and separately
surfaces which (if any) secondary/context contracts exist, rather than
silently picking one or averaging across them.

## Source coverage vs. direct-contract mapping are two independent axes
(Phase 4 acceptance review, issue 9)

**Whether a safely-available CFTC report exists at all** and
**whether the auction's own tenor has a defensible direct futures
contract** are never conflated here. A 3-Year/7-Year/20-Year auction
can (and, in this project's data, does) still have a fully available
CFTC report -- it simply has no *tenor-specific* contract to select
from it. `add_tenor_matched_positioning_features` therefore exposes
both axes as separate columns:

- `cftc_report_available` / `cftc_source_join_status`: whether *any*
  safely-available CFTC report existed by the cutoff, independent of
  tenor.
- `direct_contract_mapping_status` / `direct_contract_code` /
  `direct_contract_missing_reason`: whether *this auction's own
  tenor* has a direct-match futures contract, and, if so, whether that
  specific contract actually had data in the matched report.

Reports and tests must never count "no direct contract" (a property of
the tenor) as if it were "no source coverage" (a property of the
report). `positioning_missing_reason` is retained for backward
compatibility with earlier Phase 4B code and reports, but conflates
the two -- new code should prefer the separated fields above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.data.cftc_schema import (
    CFTC_START_DATE,
    CONTRACT_BY_CODE,
    TENOR_PRIMARY_CONTRACT_CODE,
    TENOR_TO_CONTRACT_CODES,
)

_COVERAGE_START = pd.Timestamp(CFTC_START_DATE)

NO_COVERAGE_REASON = (
    f"no_cftc_report_published_before_cutoff (this project's CFTC TFF Futures Only "
    f"ingestion begins {CFTC_START_DATE}; see artifacts/cftc_positioning_data_quality.md)"
)
NO_CONTRACT_FOR_TENOR_REASON = (
    "no_direct_futures_contract_for_this_tenor (verified absence -- see "
    "cftc_schema.TENORS_WITH_NO_CONTRACT / TENOR_PRIMARY_CONTRACT_CODE)"
)

# direct_contract_mapping_status values -- see module docstring, issue 9.
DIRECT_CONTRACT_STATUS_AVAILABLE = "direct_contract_available"
DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR = "no_direct_contract_for_tenor"
DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE = "direct_contract_mapped_but_data_unavailable"

# cftc_source_join_status values -- independent of tenor/direct-contract status.
SOURCE_JOIN_STATUS_MATCHED = "matched"
SOURCE_JOIN_STATUS_NO_REPORT_BEFORE_CUTOFF = "no_report_before_cutoff"

_MATCHED_FEATURE_SUFFIXES: tuple[str, ...] = (
    "dealer_net_contracts",
    "asset_mgr_net_contracts",
    "lev_money_net_contracts",
    "other_rept_net_contracts",
    "tot_rept_net_contracts",
    "nonrept_net_contracts",
    "dealer_long_pct_oi",
    "dealer_short_pct_oi",
    "dealer_spread_pct_oi",
    "open_interest_all",
    "conc_gross_le_4_tdr_long",
    "conc_gross_le_4_tdr_short",
    "conc_gross_le_8_tdr_long",
    "conc_gross_le_8_tdr_short",
    "dealer_net_contracts_chg_1w",
    "dealer_net_contracts_chg_4w",
)


def as_of_join(auctions_df: pd.DataFrame, positioning_wide_df: pd.DataFrame, *, cutoff_col: str) -> pd.DataFrame:
    """Backward as-of join of every contract-prefixed feature column in
    `positioning_wide_df` onto `auctions_df`, gated by `cutoff_col`. No
    auction is ever dropped; every row carries `cftc_join_matched` and
    `cftc_observation_age_calendar_days`/`..._business_days`.
    """
    if "__as_of_join_row_order__" in auctions_df.columns:
        raise ValueError(
            "as_of_join: auctions_df already has a '__as_of_join_row_order__' column -- reserved internally"
        )
    left = auctions_df.reset_index(drop=True).copy()
    left["__as_of_join_row_order__"] = left.index

    right = (
        positioning_wide_df.loc[
            positioning_wide_df["report_date"].notna()
            & positioning_wide_df["publication_safe_available_date"].notna()
        ]
        # Phase 5 acceptance-review fix: sorting on `publication_safe_
        # available_date` alone is not enough -- a real, verified tie
        # exists in this project's own live CFTC data (the 2025-12-23
        # and 2026-01-06 reports both become safely available on
        # 2026-01-12, a holiday-adjacent compression of two release
        # dates onto one day). `pandas.merge_asof`'s tie-breaking among
        # equal keys depends on each tied row's *relative order after
        # sorting*, which a plain single-column `sort_values` leaves
        # dependent on the input's own (arbitrary) pre-sort row order --
        # a real, input-order-dependent nondeterminism, not merely a
        # theoretical risk (proven by
        # tests/test_cftc_join.py::test_tied_safe_available_dates_break_deterministically_by_report_date
        # and by leakage_audit.per_source_input_order_invariance against
        # this project's real data). Sorting by `report_date` as an
        # explicit secondary key makes the tie-break deterministic and
        # economically sensible: among two reports safely available on
        # the same day, prefer the more recently observed one.
        .sort_values(["publication_safe_available_date", "report_date"])
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

    matched = merged["report_date"].notna()
    merged["cftc_join_matched"] = matched
    merged["cftc_observation_age_calendar_days"] = (merged[cutoff_col] - merged["report_date"]).dt.days

    business_days = np.full(len(merged), np.nan)
    valid = (matched & merged[cutoff_col].notna()).to_numpy()
    if valid.any():
        report_date_days = merged["report_date"].to_numpy().astype("datetime64[D]")
        cutoff_days = merged[cutoff_col].to_numpy().astype("datetime64[D]")
        business_days[valid] = np.busday_count(report_date_days[valid], cutoff_days[valid])
    merged["cftc_observation_age_business_days"] = business_days

    return merged


def add_tenor_matched_positioning_features(joined_df: pd.DataFrame) -> pd.DataFrame:
    """For each auction row, select the *primary direct-match*
    contract's positioning features for that auction's own tenor
    (`{code}__{suffix}` -> `matched_{suffix}`), plus
    `matched_contract_code`, `matched_contract_official_name`,
    `secondary_context_contract_codes`.

    Also adds the source-coverage-vs-direct-mapping fields required by
    the Phase 4 acceptance review (issue 9): `cftc_report_available`,
    `cftc_source_join_status`, `direct_contract_mapping_status`,
    `direct_contract_code`, `direct_contract_missing_reason`.
    `positioning_missing_reason` is retained for backward compatibility
    but conflates the two axes -- prefer the separated fields in new
    code.
    """
    out = joined_df.copy()
    n = len(out)
    matched_code = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    matched_name = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    secondary_codes = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    missing_reason = np.full(n, None, dtype=object)
    direct_status = np.full(n, DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR, dtype=object)
    direct_missing_reason = np.full(n, None, dtype=object)
    matched_values: dict[str, pd.Series] = {
        suffix: pd.Series(pd.NA, index=out.index, dtype="Float64") for suffix in _MATCHED_FEATURE_SUFFIXES
    }

    for tenor, all_codes in TENOR_TO_CONTRACT_CODES.items():
        mask = out["tenor"] == tenor
        if not mask.any():
            continue
        secondary = tuple(c for c in all_codes if c != TENOR_PRIMARY_CONTRACT_CODE.get(tenor))
        secondary_codes.loc[mask] = ",".join(secondary) if secondary else pd.NA

        primary = TENOR_PRIMARY_CONTRACT_CODE.get(tenor)
        if primary is None:
            missing_reason[mask.to_numpy()] = NO_CONTRACT_FOR_TENOR_REASON
            direct_missing_reason[mask.to_numpy()] = NO_CONTRACT_FOR_TENOR_REASON
            continue

        matched_code.loc[mask] = primary
        matched_name.loc[mask] = CONTRACT_BY_CODE[primary].official_name
        presence_col = f"{primary}__open_interest_all"
        contract_has_data = out[presence_col].notna() if presence_col in out.columns else pd.Series(False, index=out.index)
        has_data_np = (mask & contract_has_data).to_numpy()
        no_data_np = (mask & ~contract_has_data).to_numpy()
        direct_status[has_data_np] = DIRECT_CONTRACT_STATUS_AVAILABLE
        direct_status[no_data_np] = DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE
        for suffix in _MATCHED_FEATURE_SUFFIXES:
            col = f"{primary}__{suffix}"
            if col in out.columns:
                matched_values[suffix].loc[mask] = out.loc[mask, col].astype("Float64")

    no_mapping_mask = ~out["tenor"].isin(TENOR_TO_CONTRACT_CODES)
    missing_reason[no_mapping_mask.to_numpy()] = NO_CONTRACT_FOR_TENOR_REASON
    direct_missing_reason[no_mapping_mask.to_numpy()] = NO_CONTRACT_FOR_TENOR_REASON

    report_available = out["cftc_join_matched"].to_numpy()
    has_primary = matched_code.notna().to_numpy()
    no_release_yet_mask = has_primary & ~report_available
    missing_reason[no_release_yet_mask] = NO_COVERAGE_REASON
    direct_missing_reason[no_release_yet_mask] = NO_COVERAGE_REASON
    # No safely-available report at all overrides any tenor-specific
    # direct-contract status computed above -- there is nothing to select from.
    direct_status[~report_available] = DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE
    direct_status[~report_available & ~has_primary] = DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR

    out["matched_contract_code"] = matched_code.to_numpy()
    out["matched_contract_official_name"] = matched_name.to_numpy()
    out["secondary_context_contract_codes"] = secondary_codes.to_numpy()
    for suffix, series in matched_values.items():
        out[f"matched_{suffix}"] = series
    out["positioning_missing_reason"] = pd.array(missing_reason, dtype="string")

    # -- Separated source-coverage vs. direct-contract-mapping fields (issue 9) --
    out["cftc_report_available"] = report_available
    out["cftc_source_join_status"] = np.where(report_available, SOURCE_JOIN_STATUS_MATCHED, SOURCE_JOIN_STATUS_NO_REPORT_BEFORE_CUTOFF)
    out["direct_contract_mapping_status"] = direct_status
    out["direct_contract_code"] = matched_code.to_numpy()
    out["direct_contract_missing_reason"] = pd.array(direct_missing_reason, dtype="string")
    return out
