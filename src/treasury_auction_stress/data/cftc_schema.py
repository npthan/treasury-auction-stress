"""Verified schema and configuration for CFTC's Traders in Financial
Futures (TFF) "Futures Only" report.

Everything here was captured by directly querying the live Socrata API
at `publicreporting.cftc.gov` on 2026-09-11, and by reading CFTC's own
explanatory notes and press releases -- not guessed.

## The dataset, verified

CFTC publishes several distinct Commitments-of-Traders report
families through the same Socrata "Public Reporting Environment"
(`https://publicreporting.cftc.gov`), each its own dataset id:

- `gpe5-46if` -- **Traders in Financial Futures, Futures Only** (the
  one this project uses -- verified via `futonly_or_combined` field,
  which is literally `"FutOnly"` for every row in this dataset).
- `yw9f-hn96` -- TFF Futures-and-Options Combined (verified via the
  same field, `"Combined"`) -- **never used or mixed with the above.**
- Legacy COT and Disaggregated (physical-commodity) COT are separate
  datasets entirely, not queried by this project at all.

No auth/token is required for the query volumes this project needs
(CFTC: "generally used ... successfully without using a token, as long
as you are not overusing the API").

## Verified contract registry (Treasury note/bond futures only)

Queried live: `SELECT DISTINCT contract_market_name,
cftc_contract_market_code WHERE commodity_subgroup_name = 'Interest
Rates - U.S. Treasury'`. Of the 9 rows returned, 6 are genuine
Treasury note/bond futures (the other 3 -- `MICRO 10 YEAR YIELD` and a
DTCC Treasury repo series -- are not note/bond futures and are out of
scope). **There is no 3-Year Treasury note futures contract in this
data at all** -- verified by its complete absence from the query
result, not assumed. Per the task's own instruction ("do not force a
one-to-one mapping when none exists"), 3-Year and 7-Year auctions get
no direct contract match; see `TENOR_TO_CONTRACT_CODES` below for the
context-only (never a forced 1:1) mapping used instead.

| Code | Name(s) verified | First TFF obs (verified) | Notes |
|---|---|---|---|
| `020601` | UST BOND | 2006-06-13 | Classic 30-Year bond futures |
| `020604` | ULTRA UST BOND (2010-2025), ULTRA US T BOND (2 weeks, Sep 2025) | 2010-03-02 | Verified name variant for the same code, Sep 2025 only |
| `042601` | UST 2Y NOTE | 2006-06-13 | |
| `044601` | UST 5Y NOTE | 2006-06-13 | |
| `043602` | UST 10Y NOTE | 2006-06-13 | |
| `043607` | ULTRA UST 10Y | 2016-03-08 | Verified real introduction date (Ultra 10-Year futures launched 2016) |

The 2006-06-13 start predates the TFF report's own 2010 introduction
(per CFTC's historical-compressed-file page, TFF Futures Only annual
files begin "starting July 20, 2010") -- the Socrata API evidently
serves TFF-formatted records back to 2006 for these base contracts.
This project ingests 2010-01-01 onward regardless, matching the
auction universe, so this discrepancy does not affect coverage.

## CFTC's own category definitions (verified, explanatory notes)

TFF splits reportable positions into four trader categories, **not**
the same thing as this project's own "primary dealer" concept from
Phase 3 (NY Fed Primary Dealer Statistics) -- **the CFTC "Dealer/
Intermediary" category is a broader, differently-defined regulatory
classification (any registered swap dealer/futures commission
merchant acting as an intermediary), not identical to, and not a
verified subset or superset of, NY Fed's designated primary dealers.**
This project never conflates the two.

- `dealer` (Dealer/Intermediary): dealers and intermediaries.
- `asset_mgr` (Asset Manager/Institutional): institutional money
  managers, pension funds, insurance companies, mutual funds.
- `lev_money` (Leveraged Funds): hedge funds and other leveraged money.
- `other_rept` (Other Reportables): reportable traders not in the
  above three.
- `nonrept` (Nonreportable): everyone below CFTC's reporting
  threshold -- long/short only, no spreading breakout exists for this
  category (CFTC does not collect it).

Each of the first four categories reports `long`, `short`, and
`spreading` (spread) positions separately -- **spreading positions are
never assigned to either directional side** in this project's
candidate features (see `cftc_candidate_features.py`).
"""

from __future__ import annotations

SOCRATA_BASE_URL = "https://publicreporting.cftc.gov"
TFF_FUTURES_ONLY_DATASET_ID = "gpe5-46if"
TFF_COMBINED_DATASET_ID_EXCLUDED = "yw9f-hn96"  # verified to exist; never queried

EXPLANATORY_NOTES_URL = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/index.htm"
RELEASE_SCHEDULE_URL = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm"
HISTORICAL_SPECIAL_ANNOUNCEMENTS_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm"
)

CFTC_START_DATE = "2010-01-01"
UNITS = "contracts"

TRADER_CATEGORIES: tuple[str, ...] = ("dealer", "asset_mgr", "lev_money", "other_rept")
TRADER_CATEGORY_FIELD_PREFIX: dict[str, str] = {
    "dealer": "dealer_positions",
    "asset_mgr": "asset_mgr_positions",
    "lev_money": "lev_money_positions",
    "other_rept": "other_rept_positions",
}
TRADER_CATEGORY_LABELS: dict[str, str] = {
    "dealer": "Dealer/Intermediary",
    "asset_mgr": "Asset Manager/Institutional",
    "lev_money": "Leveraged Funds",
    "other_rept": "Other Reportables",
}

# Fields fetched per row, verified present in a live query.
FIELDS_TO_KEEP: tuple[str, ...] = (
    "report_date_as_yyyy_mm_dd",
    "contract_market_name",
    "cftc_contract_market_code",
    "open_interest_all",
    "dealer_positions_long_all",
    "dealer_positions_short_all",
    "dealer_positions_spread_all",
    "asset_mgr_positions_long",
    "asset_mgr_positions_short",
    "asset_mgr_positions_spread",
    "lev_money_positions_long",
    "lev_money_positions_short",
    "lev_money_positions_spread",
    "other_rept_positions_long",
    "other_rept_positions_short",
    "other_rept_positions_spread",
    "nonrept_positions_long_all",
    "nonrept_positions_short_all",
    "tot_rept_positions_long_all",
    "tot_rept_positions_short",
    "traders_tot_all",
    "conc_gross_le_4_tdr_long",
    "conc_gross_le_4_tdr_short",
    "conc_gross_le_8_tdr_long",
    "conc_gross_le_8_tdr_short",
    "contract_units",
    "futonly_or_combined",
)


class ContractDefinition:
    __slots__ = ("code", "first_verified_observation", "name_aliases", "official_name", "tenor_role")

    def __init__(self, *, code: str, official_name: str, name_aliases: tuple[str, ...], first_verified_observation: str, tenor_role: str) -> None:
        self.code = code
        self.official_name = official_name
        self.name_aliases = name_aliases
        self.first_verified_observation = first_verified_observation
        self.tenor_role = tenor_role


CONTRACT_REGISTRY: tuple[ContractDefinition, ...] = (
    ContractDefinition(
        code="042601", official_name="UST 2Y NOTE", name_aliases=(),
        first_verified_observation="2006-06-13", tenor_role="2-Year direct match",
    ),
    ContractDefinition(
        code="044601", official_name="UST 5Y NOTE", name_aliases=(),
        first_verified_observation="2006-06-13", tenor_role="5-Year direct match",
    ),
    ContractDefinition(
        code="043602", official_name="UST 10Y NOTE", name_aliases=(),
        first_verified_observation="2006-06-13", tenor_role="10-Year direct match",
    ),
    ContractDefinition(
        code="043607", official_name="ULTRA UST 10Y", name_aliases=(),
        first_verified_observation="2016-03-08", tenor_role="10-Year secondary context (optional, per task instructions)",
    ),
    ContractDefinition(
        code="020601", official_name="UST BOND", name_aliases=(),
        first_verified_observation="2006-06-13", tenor_role="30-Year direct match; 20-Year context",
    ),
    ContractDefinition(
        code="020604", official_name="ULTRA UST BOND", name_aliases=("ULTRA US T BOND",),
        first_verified_observation="2010-03-02", tenor_role="30-Year/20-Year secondary context",
    ),
)
CONTRACT_CODES: tuple[str, ...] = tuple(c.code for c in CONTRACT_REGISTRY)
CONTRACT_BY_CODE: dict[str, ContractDefinition] = {c.code: c for c in CONTRACT_REGISTRY}

# Auction tenor -> contract code(s). A tuple, never a forced single
# match -- 3-Year and 7-Year map to NO contract (none exists) and are
# absent from this dict entirely, not mapped to a misleading proxy.
TENOR_TO_CONTRACT_CODES: dict[str, tuple[str, ...]] = {
    "2-Year": ("042601",),
    "5-Year": ("044601",),
    "10-Year": ("043602", "043607"),  # direct + optional Ultra context
    "20-Year": ("020601", "020604"),  # no direct contract; classic + ultra bond context
    "30-Year": ("020601", "020604"),  # direct (classic) + optional Ultra context
}
# Tenors with NO contract mapping at all -- verified absence, not an oversight.
TENORS_WITH_NO_CONTRACT: tuple[str, ...] = ("3-Year", "7-Year")

# The single *direct-match* contract per tenor (a strict subset of
# TENOR_TO_CONTRACT_CODES, excluding secondary/context-only entries).
# 20-Year has no direct futures contract at all -- deliberately absent
# from this dict, not mapped to either bond contract as a stand-in.
TENOR_PRIMARY_CONTRACT_CODE: dict[str, str] = {
    "2-Year": "042601",
    "5-Year": "044601",
    "10-Year": "043602",
    "30-Year": "020601",
}
