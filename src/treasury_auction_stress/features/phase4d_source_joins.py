"""Phase 4D: source-specific auction join tables and a leakage/
row-count audit.

Builds one processed join table per Phase 4 market/macro source
(Treasury rates, CFTC TFF positioning, RTDSM macro vintages) --
**never a single combined feature matrix** (that is explicitly
deferred to Phase 5). Each table carries only auction identifiers, the
two prediction cutoffs, and that source's own joined feature columns
-- never a single auction-result/outcome column from the nominal-
coupon auctions table, by construction (selected before any join, not
filtered out after).

## Forbidden auction-result fields

Every column below is set at or after the auction itself (yields,
prices, accepted/tendered amounts by bidder category, allocation
percentages, settlement-time accrued interest, the coupon rate itself)
and is therefore never a legitimate *pre-auction* feature. This list
is used only defensively here -- to assert none of them leaked into a
per-source join table -- since the tables are built by selecting a
fixed, small set of identifier columns before joining, not by
filtering a wide table after the fact.
"""

from __future__ import annotations

import pandas as pd

from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)
from treasury_auction_stress.features.cftc_join import (
    add_tenor_matched_positioning_features,
)
from treasury_auction_stress.features.cftc_join import as_of_join as cftc_as_of_join
from treasury_auction_stress.features.eligibility import (
    select_analysis_sample,
    select_modeling_sample,
)
from treasury_auction_stress.features.rtdsm_join import as_of_join_all
from treasury_auction_stress.features.treasury_rates_candidate_features import (
    add_tenor_matched_rate_change_features,
    add_tenor_matched_rate_features,
    build_rate_feature_table,
)
from treasury_auction_stress.features.treasury_rates_join import (
    as_of_join as rates_as_of_join,
)

AUCTION_ID_COLS: tuple[str, ...] = ("cusip", "tenor", "auction_date", "announcemt_date", "is_reopening")
CUTOFF_COLS: tuple[str, ...] = (ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL)

FORBIDDEN_RESULT_COLUMNS: tuple[str, ...] = (
    "high_yield", "low_yield", "avg_med_yield", "high_yield_raw", "low_yield_raw", "avg_med_yield_raw",
    "high_price", "low_price", "avg_med_price", "price_per100", "adj_price", "unadj_price",
    "high_discnt_rate", "low_discnt_rate", "avg_med_discnt_rate",
    "high_investment_rate", "low_investment_rate", "avg_med_investment_rate",
    "high_discnt_margin", "low_discnt_margin", "avg_med_discnt_margin",
    "bid_to_cover_ratio", "bid_to_cover_ratio_raw",
    "total_accepted", "total_tendered", "total_accepted_raw", "total_tendered_raw",
    "comp_accepted", "comp_tendered", "comp_tenders_accepted", "comp_accepted_raw", "comp_tendered_raw",
    "noncomp_accepted", "noncomp_tenders_accepted", "noncomp_accepted_raw",
    "primary_dealer_accepted", "primary_dealer_tendered", "primary_dealer_accepted_raw", "primary_dealer_tendered_raw",
    "direct_bidder_accepted", "direct_bidder_tendered", "direct_bidder_accepted_raw", "direct_bidder_tendered_raw",
    "indirect_bidder_accepted", "indirect_bidder_tendered", "indirect_bidder_accepted_raw", "indirect_bidder_tendered_raw",
    "soma_accepted", "soma_tendered", "soma_accepted_raw", "soma_tendered_raw",
    "fima_noncomp_accepted", "fima_noncomp_tendered", "fima_noncomp_accepted_raw", "fima_noncomp_tendered_raw",
    "treas_retail_accepted", "treas_retail_tenders_accepted", "treas_retail_accepted_raw",
    "allocation_pctage", "allocation_pctage_decimals", "allocation_pctage_raw",
    "accrued_int_per100", "accrued_int_per1000", "adj_accrued_int_per1000", "unadj_accrued_int_per1000",
    "int_rate", "std_int_payment_per1000", "spread",
    "frn_index_determination_rate", "ref_cpi_on_issue_date", "ref_cpi_on_dated_date", "index_ratio_on_issue_date",
    "pdf_filenm_comp_results", "xml_filenm_comp_results", "pdf_filenm_noncomp_results",
    "results_available",
    # targets.py-derived outcome fields (never present in the raw nominal_df
    # table, but forbidden defensively in case a caller merges them in later).
    "public_accepted_amount", "fed_reserve_addon_amount", "primary_dealer_share",
    "direct_bidder_share", "indirect_bidder_share", "noncompetitive_share", "fima_share",
    "bid_to_cover_calculated", "bid_to_cover_reconciliation_diff",
)


def build_samples(nominal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    settled, pending = select_analysis_sample(nominal_df)
    normalized_complete_sample = add_cutoff_dates(pd.concat([settled, pending], ignore_index=True))
    ordinary_modeling_sample = add_cutoff_dates(select_modeling_sample(settled))
    return normalized_complete_sample, ordinary_modeling_sample


def _ids_only(auctions_df: pd.DataFrame) -> pd.DataFrame:
    return auctions_df[list(AUCTION_ID_COLS) + list(CUTOFF_COLS)].copy()


def build_rates_join_table(auctions_df: pd.DataFrame, rates_wide_df: pd.DataFrame) -> pd.DataFrame:
    feat = build_rate_feature_table(rates_wide_df)
    out = _ids_only(auctions_df)
    for cutoff_col in CUTOFF_COLS:
        joined = rates_as_of_join(auctions_df, feat, cutoff_col=cutoff_col)
        joined = add_tenor_matched_rate_features(joined)
        joined = add_tenor_matched_rate_change_features(joined)
        new_cols = [c for c in joined.columns if c not in auctions_df.columns]
        prefixed = joined[new_cols].add_prefix(f"{cutoff_col}__")
        out = pd.concat([out.reset_index(drop=True), prefixed.reset_index(drop=True)], axis=1)
    return out


def build_cftc_join_table(auctions_df: pd.DataFrame, positioning_wide_df: pd.DataFrame) -> pd.DataFrame:
    out = _ids_only(auctions_df)
    for cutoff_col in CUTOFF_COLS:
        joined = cftc_as_of_join(auctions_df, positioning_wide_df, cutoff_col=cutoff_col)
        joined = add_tenor_matched_positioning_features(joined)
        new_cols = [c for c in joined.columns if c not in auctions_df.columns]
        prefixed = joined[new_cols].add_prefix(f"{cutoff_col}__")
        out = pd.concat([out.reset_index(drop=True), prefixed.reset_index(drop=True)], axis=1)
    return out


def build_rtdsm_join_table(
    auctions_df: pd.DataFrame, vintage_indices: dict[str, pd.DataFrame], snapshots: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    out = _ids_only(auctions_df)
    for cutoff_col in CUTOFF_COLS:
        joined = as_of_join_all(auctions_df, vintage_indices, snapshots, variables=VARIABLE_REGISTRY, cutoff_col=cutoff_col)
        new_cols = [c for c in joined.columns if c not in auctions_df.columns]
        prefixed = joined[new_cols].add_prefix(f"{cutoff_col}__")
        out = pd.concat([out.reset_index(drop=True), prefixed.reset_index(drop=True)], axis=1)
    return out


def audit_join_table(name: str, df: pd.DataFrame, *, expected_rows: int) -> dict:
    leaked = sorted(set(df.columns) & set(FORBIDDEN_RESULT_COLUMNS))
    dup_keys = df.duplicated(subset=["cusip", "auction_date"]).sum()
    return {
        "table": name,
        "row_count": len(df),
        "expected_row_count": expected_rows,
        "row_count_matches": len(df) == expected_rows,
        "duplicate_cusip_auction_date_rows": int(dup_keys),
        "forbidden_columns_present": leaked,
        "passed": len(df) == expected_rows and dup_keys == 0 and not leaked,
    }
