"""Verified schema information for the Treasury Fiscal Data auctions API.

Everything in this module was captured by directly querying the live
endpoint on 2026-09-10 (see docs/data_dictionary.md for how), not
guessed from the project plan. If the live API's field set changes,
`data_quality.check_schema_drift` (in quality.py) will flag it by
comparing a fresh response's columns against `EXPECTED_COLUMNS` below.
"""

from __future__ import annotations

API_BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
AUCTIONS_ENDPOINT = "/v1/accounting/od/auctions_query"

# The API's hard maximum page size, confirmed empirically: requesting
# page[size]=10001 returns HTTP 400 "Expected an integer between 1 and
# 10000". Requesting more than this is a bug, not a feature to retry.
MAX_PAGE_SIZE = 10000

# The literal string the API uses to represent a missing value. Note
# this is the JSON string "null", not a JSON null literal -- naive
# numeric parsing (e.g. float(x)) would raise on it, and naive "does
# this look falsy" checks would wrongly treat it as compatible with 0.
API_NULL_TOKEN = "null"

# Every field the live API returned as of 2026-09-10, mapped to the
# API's own self-reported data type (from the response's
# meta.dataTypes). This is used only for schema-drift detection --
# NOT as an authoritative parsing guide, because the API's own type
# metadata is internally inconsistent (e.g. "high_price" is reported
# as STRING while the structurally identical "high_yield" is reported
# as NUMBER, even though both hold decimal strings like "99.759").
# Numeric parsing in normalize.py uses its own hand-verified field
# lists below instead of trusting this dict.
EXPECTED_COLUMNS: dict[str, str] = {
    "accrued_int_per100": "NUMBER",
    "accrued_int_per1000": "NUMBER",
    "adj_accrued_int_per1000": "NUMBER",
    "adj_price": "NUMBER",
    "allocation_pctage": "NUMBER",
    "allocation_pctage_decimals": "NUMBER",
    "announcemt_date": "DATE",
    "announcemtd_cusip": "STRING",
    "auction_date": "DATE",
    "auction_format": "STRING",
    "avg_med_discnt_margin": "NUMBER",
    "avg_med_discnt_rate": "NUMBER",
    "avg_med_investment_rate": "NUMBER",
    "avg_med_price": "STRING",
    "avg_med_yield": "NUMBER",
    "back_dated": "STRING",
    "back_dated_date": "DATE",
    "bid_to_cover_ratio": "NUMBER",
    "call_date": "DATE",
    "callable": "STRING",
    "called_date": "DATE",
    "cash_management_bill_cmb": "STRING",
    "closing_time_comp": "STRING",
    "closing_time_noncomp": "STRING",
    "comp_accepted": "NUMBER",
    "comp_bid_decimals": "NUMBER",
    "comp_tendered": "NUMBER",
    "comp_tenders_accepted": "STRING",
    "corpus_cusip": "STRING",
    "cpi_base_reference_period": "STRING",
    "currently_outstanding": "NUMBER",
    "cusip": "STRING",
    "dated_date": "DATE",
    "direct_bidder_accepted": "NUMBER",
    "direct_bidder_tendered": "NUMBER",
    "est_pub_held_mat_by_type_amt": "NUMBER",
    "fima_included": "STRING",
    "fima_noncomp_accepted": "NUMBER",
    "fima_noncomp_tendered": "NUMBER",
    "first_int_payment_date": "DATE",
    "first_int_period": "STRING",
    "floating_rate": "STRING",
    "frn_index_determination_date": "DATE",
    "frn_index_determination_rate": "NUMBER",
    "high_discnt_margin": "NUMBER",
    "high_discnt_rate": "NUMBER",
    "high_investment_rate": "NUMBER",
    "high_price": "STRING",
    "high_yield": "NUMBER",
    "index_ratio_on_issue_date": "NUMBER",
    "indirect_bidder_accepted": "NUMBER",
    "indirect_bidder_tendered": "NUMBER",
    "inflation_index_security": "STRING",
    "int_payment_frequency": "STRING",
    "int_rate": "NUMBER",
    "issue_date": "DATE",
    "low_discnt_margin": "NUMBER",
    "low_discnt_rate": "NUMBER",
    "low_investment_rate": "NUMBER",
    "low_price": "STRING",
    "low_yield": "NUMBER",
    "mat_date": "DATE",
    "maturity_date": "DATE",
    "max_comp_award": "NUMBER",
    "max_noncomp_award": "NUMBER",
    "max_single_bid": "NUMBER",
    "min_bid_amt": "NUMBER",
    "min_strip_amt": "NUMBER",
    "min_to_issue": "NUMBER",
    "multiples_to_bid": "NUMBER",
    "multiples_to_issue": "NUMBER",
    "nlp_exclusion_amt": "NUMBER",
    "nlp_reporting_threshold": "NUMBER",
    "noncomp_accepted": "NUMBER",
    "noncomp_tenders_accepted": "STRING",
    "offering_amt": "CURRENCY0",
    "original_cusip": "STRING",
    "original_dated_date": "DATE",
    "original_issue_date": "DATE",
    "original_security_term": "STRING",
    "pdf_filenm_announcemt": "STRING",
    "pdf_filenm_comp_results": "STRING",
    "pdf_filenm_noncomp_results": "STRING",
    "pdf_filenm_spec_announcemt": "STRING",
    "price_per100": "STRING",
    "primary_dealer_accepted": "NUMBER",
    "primary_dealer_tendered": "NUMBER",
    "record_date": "DATE",
    "ref_cpi_on_dated_date": "NUMBER",
    "ref_cpi_on_issue_date": "NUMBER",
    "reopening": "STRING",
    "security_term": "STRING",
    "security_term_day_month": "STRING",
    "security_term_week_year": "STRING",
    "security_type": "STRING",
    "series": "STRING",
    "soma_accepted": "NUMBER",
    "soma_holdings": "NUMBER",
    "soma_included": "STRING",
    "soma_tendered": "NUMBER",
    "spread": "NUMBER",
    "std_int_payment_per1000": "NUMBER",
    "strippable": "STRING",
    "tiin_conversion_factor_per1000": "NUMBER",
    "tint_cusip_1": "STRING",
    "tint_cusip_2": "STRING",
    "total_accepted": "NUMBER",
    "total_tendered": "NUMBER",
    "treas_retail_accepted": "NUMBER",
    "treas_retail_tenders_accepted": "STRING",
    "unadj_accrued_int_per1000": "NUMBER",
    "unadj_price": "NUMBER",
    "xml_filenm_announcemt": "STRING",
    "xml_filenm_comp_results": "STRING",
}

# Fields this project actually depends on (identity, timing, targets).
# Hand-verified against live data, independent of EXPECTED_COLUMNS.
DATE_FIELDS: tuple[str, ...] = (
    "record_date",
    "announcemt_date",  # NOTE: verified API field name -- "announcement" without the second "n"
    "auction_date",
    "issue_date",
    "maturity_date",
    "dated_date",
)

NUMERIC_FIELDS: tuple[str, ...] = (
    "offering_amt",
    "total_accepted",
    "total_tendered",
    "bid_to_cover_ratio",
    "primary_dealer_accepted",
    "primary_dealer_tendered",
    "direct_bidder_accepted",
    "direct_bidder_tendered",
    "indirect_bidder_accepted",
    "indirect_bidder_tendered",
    "noncomp_accepted",
    "comp_accepted",
    "comp_tendered",
    "allocation_pctage",
    "high_yield",
    "low_yield",
    "avg_med_yield",
    # Added in Phase 2: verified numeric fields needed to reconcile
    # total_accepted/total_tendered against the bidder-category
    # breakdown (see docs/target_specification.md). Federal Reserve
    # System Open Market Account (SOMA) and Foreign and International
    # Monetary Authorities (FIMA) add-ons are awarded outside
    # competitive bidding, on top of the publicly announced offering,
    # and are NOT allocated to the primary_dealer/direct_bidder/
    # indirect_bidder categories -- verified by exact-dollar identity
    # over the full settled nominal-coupon sample (see
    # docs/data_dictionary.md).
    "soma_accepted",
    "soma_tendered",
    "fima_noncomp_accepted",
    "fima_noncomp_tendered",
    "treas_retail_accepted",
)

CATEGORICAL_IDENTITY_FIELDS: tuple[str, ...] = (
    "cusip",
    "security_type",
    "security_term",
    "original_security_term",
    "reopening",
    "inflation_index_security",
    "floating_rate",
    # Added in the Phase 2 acceptance review: "No" identifies auctions
    # restricted to primary-dealer-only bidding (customer/noncompetitive/
    # FIMA tenders not accepted) -- verified against official Treasury
    # announcement PDFs for the two $25,000,000 auctions this review
    # investigated. See treasury_auction_stress.features.eligibility.
    "noncomp_tenders_accepted",
)

# Every field this project preserves through normalization, i.e. the
# union of the fields explicitly requested in docs/project_plan.md
# plus the fields normalization derives its logic from.
CORE_FIELDS: tuple[str, ...] = (
    *DATE_FIELDS,
    *NUMERIC_FIELDS,
    *CATEGORICAL_IDENTITY_FIELDS,
)

# Nominal coupon identification, verified against live 2010+ data:
# security_type in {"Note", "Bond"} AND inflation_index_security == "No"
# AND floating_rate == "No" isolates exactly the 2/3/5/7/10/20/30-year
# nominal coupon universe (TIPS carry inflation_index_security == "Yes";
# the 2-year FRN carries floating_rate == "Yes"; Bills are a distinct
# security_type). This was confirmed by cross-tabulating all three
# fields over every 2010+ record, not assumed.
NOMINAL_COUPON_SECURITY_TYPES = frozenset({"Note", "Bond"})

# The seven nominal coupon tenors in scope, in canonical order. Verified
# to be exactly the set of original_security_term values present once
# the nominal-coupon filter above is applied.
NOMINAL_COUPON_TENORS: tuple[str, ...] = (
    "2-Year",
    "3-Year",
    "5-Year",
    "7-Year",
    "10-Year",
    "20-Year",
    "30-Year",
)
