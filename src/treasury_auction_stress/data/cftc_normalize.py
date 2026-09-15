"""Turn raw CFTC TFF Futures Only JSON responses into a tidy,
release-aware table -- one row per (report_date, contract_code),
matching the shape the source itself already reports in (unlike the
weekly dealer-statistics or daily-rates sources, TFF rows are already
"one observation per entity per week," so no wide/long pivot is
needed here beyond parsing types and attaching timing).

Every numeric field in the raw Socrata JSON arrives as a **string**
(verified live) -- parsed explicitly here, never left as text.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from treasury_auction_stress.data.cftc_release_calendar import compute_release_dates
from treasury_auction_stress.data.cftc_schema import CONTRACT_BY_CODE, FIELDS_TO_KEEP

_NUMERIC_FIELDS = tuple(f for f in FIELDS_TO_KEEP if f not in ("report_date_as_yyyy_mm_dd", "contract_market_name", "cftc_contract_market_code", "contract_units", "futonly_or_combined"))


def pages_to_records(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for contract in raw_payload.get("contracts", []):
        url = contract["url"]
        for row in contract.get("body", []):
            record = {field: row.get(field) for field in FIELDS_TO_KEEP}
            record["source_url"] = url
            records.append(record)
    return records


def normalize_cftc_positioning(raw_payload: dict[str, Any], *, retrieval_timestamp_utc: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse raw records into a tidy table with release-aware timing.

    Every field this project promised to preserve is a column:
    report observation date, contract identity, every trader-category
    position field, open interest, concentration, release rule and
    both nominal/safe-available dates, and per-row missingness.
    """
    records = pages_to_records(raw_payload)
    anomalies: dict[str, Any] = {"unrecognized_contract_codes": [], "duplicate_report_rows": 0}
    if not records:
        return pd.DataFrame(), anomalies

    df = pd.DataFrame.from_records(records)
    unrecognized = sorted(set(df["cftc_contract_market_code"]) - set(CONTRACT_BY_CODE))
    if unrecognized:
        anomalies["unrecognized_contract_codes"] = unrecognized

    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    dup_mask = df.duplicated(subset=["report_date", "cftc_contract_market_code"], keep=False)
    anomalies["duplicate_report_rows"] = int(dup_mask.sum())

    for field in _NUMERIC_FIELDS:
        df[f"{field}_raw"] = df[field]
        df[field] = pd.to_numeric(df[field], errors="coerce").astype("Float64")

    df = df.rename(columns={"cftc_contract_market_code": "contract_code", "contract_market_name": "contract_name_as_reported"})

    release = compute_release_dates(df["report_date"])
    df = pd.concat([df.reset_index(drop=True), release.reset_index(drop=True)], axis=1)

    df["source_id"] = "cftc_tff_futures_only"
    df["retrieval_timestamp_utc"] = retrieval_timestamp_utc
    df["units"] = "contracts"

    ordered = [
        "report_date",
        "contract_code",
        "contract_name_as_reported",
        "units",
        *_NUMERIC_FIELDS,
        *(f"{field}_raw" for field in _NUMERIC_FIELDS),
        "source_id",
        "retrieval_timestamp_utc",
        "release_rule_source",
        "nominal_publication_date",
        "publication_safe_available_date",
        "availability_precision",
        "source_url",
    ]
    out = df[ordered].sort_values(["contract_code", "report_date"]).reset_index(drop=True)
    return out, anomalies


_RELEASE_TIMING_COLS = (
    "nominal_publication_date",
    "publication_safe_available_date",
    "availability_precision",
    "release_rule_source",
)


def pivot_wide_by_contract(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per `report_date`, every per-contract feature column
    prefixed `{contract_code}__{column}` -- mirrors
    `treasury_rates_normalize.pivot_wide_by_maturity`'s wide-by-entity
    shape, so `cftc_join.as_of_join` can reuse the same
    `merge_asof`-on-a-single-timeline pattern used for rates and dealer
    statistics.

    Release timing (`publication_safe_available_date` etc.) is
    computed purely from `report_date` (see
    `cftc_release_calendar.py`), so it is identical across every
    contract for a given report week -- taken once per `report_date`,
    not duplicated per contract prefix.
    """
    if long_df.empty:
        return long_df

    value_cols = [
        c
        for c in long_df.columns
        if c not in {"report_date", "contract_code", "contract_name_as_reported", "source_url", *_RELEASE_TIMING_COLS}
    ]
    pivoted = long_df.pivot(index="report_date", columns="contract_code", values=value_cols)
    pivoted.columns = [f"{contract_code}__{col}" for col, contract_code in pivoted.columns]
    pivoted = pivoted.reset_index()

    timing = (
        long_df[["report_date", *_RELEASE_TIMING_COLS]]
        .drop_duplicates(subset="report_date")
        .sort_values("report_date")
    )
    out = pivoted.merge(timing, on="report_date", how="left").sort_values("report_date").reset_index(drop=True)
    return out
