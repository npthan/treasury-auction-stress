"""Turn raw Treasury auctions API responses into an analysis-ready table.

This module never mutates or looks at the original raw JSON file on
disk; it only reads the payload built by raw_store.build_raw_payload
and returns pandas DataFrames. Nothing here decides what the final
research targets are (see docs/target_specification.md) -- it only
preserves the fields those targets will eventually be built from,
with an honest, auditable representation of what is and isn't known
for each auction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from treasury_auction_stress.data.schema import (
    API_NULL_TOKEN,
    CATEGORICAL_IDENTITY_FIELDS,
    DATE_FIELDS,
    EXPECTED_COLUMNS,
    NOMINAL_COUPON_SECURITY_TYPES,
    NOMINAL_COUPON_TENORS,
    NUMERIC_FIELDS,
)

CANDIDATE_PRIMARY_KEY: tuple[str, str] = ("cusip", "auction_date")


def pages_to_records(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every page's `body.data` list into one list of records."""
    records: list[dict[str, Any]] = []
    for page in raw_payload.get("pages", []):
        records.extend(page.get("body", {}).get("data", []))
    return records


def to_raw_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """The rawest useful representation: one row per record, every
    column a plain string exactly as the API sent it (including the
    literal "null" token for missing values). No parsing, no derived
    columns.
    """
    return pd.DataFrame.from_records(records)


def check_schema_drift(df: pd.DataFrame) -> dict[str, list[str]]:
    """Compare a raw dataframe's columns against the columns observed
    when this project last inspected the live API (schema.py).

    Returns a dict with "new_columns" (present now, not seen before)
    and "missing_columns" (seen before, absent now) -- both empty
    means no drift detected.
    """
    actual = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    return {
        "new_columns": sorted(actual - expected),
        "missing_columns": sorted(expected - actual),
    }


def _null_to_na(series: pd.Series) -> pd.Series:
    """Replace the API's literal "null" string token with a real
    missing value, without touching any other value (in particular,
    a legitimate "0" stays "0", not missing).
    """
    return series.mask(series == API_NULL_TOKEN)


def _parse_date_column(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    cleaned = _null_to_na(raw)
    parsed = pd.to_datetime(cleaned, format="%Y-%m-%d", errors="coerce")
    invalid = parsed.isna() & cleaned.notna()
    return parsed, invalid


def _parse_numeric_column(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    cleaned = _null_to_na(raw)
    parsed = pd.to_numeric(cleaned, errors="coerce").astype("Float64")
    invalid = parsed.isna() & cleaned.notna()
    return parsed, invalid


def normalize_auctions(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse dates and numbers, flag anomalies, and add the derived
    identity columns (`is_nominal_coupon`, `tenor`, `is_reopening`,
    `results_available`) the rest of the project depends on.

    Every field this touches gets a same-named `<field>_raw` sibling
    column holding the untouched original string, so nothing is lost
    even if a parse is wrong.

    Returns (normalized_dataframe, anomalies) where `anomalies` is a
    plain dict suitable for feeding straight into the data-quality
    report -- it is never used to silently drop or "fix" rows here.
    """
    df = raw_df.copy()
    anomalies: dict[str, Any] = {
        "invalid_dates": {},
        "invalid_numbers": {},
        "unexpected_categories": {},
    }

    for col in DATE_FIELDS:
        if col not in df.columns:
            continue
        parsed, invalid = _parse_date_column(df[col])
        df[f"{col}_raw"] = df[col]
        df[col] = parsed
        n_invalid = int(invalid.sum())
        if n_invalid:
            anomalies["invalid_dates"][col] = n_invalid

    for col in NUMERIC_FIELDS:
        if col not in df.columns:
            continue
        parsed, invalid = _parse_numeric_column(df[col])
        df[f"{col}_raw"] = df[col]
        df[col] = parsed
        n_invalid = int(invalid.sum())
        if n_invalid:
            anomalies["invalid_numbers"][col] = n_invalid

    for col in ("reopening", "inflation_index_security", "floating_rate"):
        if col in df.columns:
            observed = set(df[col].dropna().unique()) - {API_NULL_TOKEN}
            unexpected = observed - {"Yes", "No"}
            if unexpected:
                anomalies["unexpected_categories"][col] = sorted(unexpected)

    df["is_reopening"] = df.get("reopening") == "Yes"
    df["is_nominal_coupon"] = (
        df.get("security_type").isin(NOMINAL_COUPON_SECURITY_TYPES)
        & (df.get("inflation_index_security") == "No")
        & (df.get("floating_rate") == "No")
    )
    df["tenor"] = df["original_security_term"].where(df["is_nominal_coupon"])

    observed_tenors = set(df.loc[df["is_nominal_coupon"], "tenor"].dropna().unique())
    unexpected_tenors = observed_tenors - set(NOMINAL_COUPON_TENORS)
    if unexpected_tenors:
        anomalies["unexpected_categories"]["tenor"] = sorted(unexpected_tenors)

    # An auction whose result fields are still "null" hasn't happened
    # yet or hasn't been fully reported yet (e.g. an auction scheduled
    # for today or the near future). This is expected, not a data
    # quality problem -- see artifacts/auction_data_quality.md.
    df["results_available"] = df["total_accepted"].notna()

    for col in CATEGORICAL_IDENTITY_FIELDS:
        if col in df.columns:
            df[col] = _null_to_na(df[col])

    return df, anomalies


def find_duplicate_keys(
    df: pd.DataFrame, key: tuple[str, ...] = CANDIDATE_PRIMARY_KEY
) -> pd.DataFrame:
    """Rows sharing the candidate primary key (cusip, auction_date).

    Empty result is the expected, healthy case -- this function does
    not raise, it just reports, since duplicates are a data-quality
    finding to be written up, not silently dropped.
    """
    mask = df.duplicated(subset=list(key), keep=False)
    return df.loc[mask].sort_values(list(key))


def nominal_coupon_subset(df: pd.DataFrame) -> pd.DataFrame:
    """The 2/3/5/7/10/20/30-year nominal coupon analytical subset.

    Excluded rows (Bills, TIPS, FRNs) are not dropped from the full
    normalized table -- only this subset omits them, and only for
    analysis convenience.
    """
    return df.loc[df["is_nominal_coupon"]].copy()
