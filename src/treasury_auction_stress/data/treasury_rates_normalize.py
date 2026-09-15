"""Turn raw Treasury Daily Par Yield Curve CSV text into a tidy,
release-aware long-format table -- one row per (rate_date, maturity).

Mirrors `treasury_auction_stress.data.dealer_stats_normalize`'s shape
and conventions (stable field names, explicit regime id, publication-
availability columns computed at normalization time, never deferred).
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from treasury_auction_stress.data.time_utils import (
    next_full_business_day_after,
    us_federal_holidays,
)
from treasury_auction_stress.data.treasury_rates_schema import (
    DATE_COLUMN,
    MATURITY_LABEL_TO_YEARS,
    REGIME_20Y_COMPOSITE,
    REGIME_20Y_REAL_BOND,
    REGIME_HS_SPLINE,
    REGIME_MC_SPLINE,
    TWENTY_YEAR_METHODOLOGY_CHANGE_DATE,
    UNITS,
    WHOLE_CURVE_METHODOLOGY_CHANGE_DATE,
)

_MC_CHANGE = pd.Timestamp(WHOLE_CURVE_METHODOLOGY_CHANGE_DATE)
_20Y_CHANGE = pd.Timestamp(TWENTY_YEAR_METHODOLOGY_CHANGE_DATE)


def parse_csv_to_wide(raw_text: str, *, source_year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one year's raw CSV text into a wide DataFrame (one row per
    `rate_date`, one column per maturity label, as strings). Detects
    genuinely unrecognized maturity columns (schema drift) without
    guessing what they mean.
    """
    anomalies: dict[str, Any] = {"unrecognized_columns": [], "duplicate_dates": 0}
    df = pd.read_csv(io.StringIO(raw_text), dtype=str, keep_default_na=False)
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"treasury_rates_normalize: expected a '{DATE_COLUMN}' column, got {list(df.columns)}")

    maturity_cols = [c for c in df.columns if c != DATE_COLUMN]
    unrecognized = sorted(set(maturity_cols) - set(MATURITY_LABEL_TO_YEARS))
    if unrecognized:
        anomalies["unrecognized_columns"] = unrecognized

    df["rate_date"] = pd.to_datetime(df[DATE_COLUMN], format="%m/%d/%Y", errors="coerce")
    invalid_dates = int(df["rate_date"].isna().sum())
    if invalid_dates:
        anomalies["invalid_dates"] = invalid_dates

    duplicate_mask = df["rate_date"].duplicated(keep=False)
    anomalies["duplicate_dates"] = int(df.loc[duplicate_mask, "rate_date"].nunique())

    df["source_year"] = source_year
    return df, anomalies


def _methodology_regime_for(maturity_label: str, rate_date: pd.Series) -> pd.Series:
    """One combined regime id per row: the whole-curve spline regime,
    plus (20-Year only) the construction-method sub-regime. See
    `treasury_rates_schema.py`'s module docstring for both boundaries.
    """
    whole_curve = pd.Series(
        [REGIME_MC_SPLINE if d >= _MC_CHANGE else REGIME_HS_SPLINE for d in rate_date],
        index=rate_date.index,
    )
    if maturity_label != "20 Yr":
        return whole_curve
    twenty_year = pd.Series(
        [REGIME_20Y_REAL_BOND if d >= _20Y_CHANGE else REGIME_20Y_COMPOSITE for d in rate_date],
        index=rate_date.index,
    )
    return whole_curve.str.cat(twenty_year, sep="+")


def to_long(wide_df: pd.DataFrame, *, retrieval_timestamp_utc: str, raw_artifact_id: str) -> pd.DataFrame:
    """Melt the wide table into the tidy long format, one row per
    (rate_date, maturity), with parsed values, missingness, regime,
    and publication-availability columns attached.
    """
    maturity_cols = [c for c in wide_df.columns if c in MATURITY_LABEL_TO_YEARS]
    long_rows = []
    for maturity_label in maturity_cols:
        sub = wide_df[["rate_date", maturity_label]].rename(columns={maturity_label: "value_raw"}).copy()
        sub["maturity_label"] = maturity_label
        long_rows.append(sub)
    long_df = pd.concat(long_rows, ignore_index=True)

    long_df["value_raw"] = long_df["value_raw"].astype(str).str.strip()
    cleaned = long_df["value_raw"].mask(long_df["value_raw"] == "")
    long_df["par_yield_percent"] = pd.to_numeric(cleaned, errors="coerce").astype("Float64")
    long_df["is_missing"] = long_df["par_yield_percent"].isna()
    long_df["missing_reason"] = pd.array([pd.NA] * len(long_df), dtype="string")
    long_df.loc[long_df["is_missing"], "missing_reason"] = (
        "source reported an empty value for this maturity/date -- not imputed or forward-filled "
        "(e.g. a maturity not auctioned/quoted that day, such as the 30-Year during its 2002-2006 "
        "issuance suspension)"
    )

    long_df["maturity_years"] = long_df["maturity_label"].map(MATURITY_LABEL_TO_YEARS)
    long_df["units"] = UNITS

    regimes = []
    for maturity_label, group in long_df.groupby("maturity_label", sort=False):
        regimes.append(_methodology_regime_for(maturity_label, group["rate_date"]))
    long_df["methodology_regime"] = pd.concat(regimes).reindex(long_df.index)

    long_df["publication_date"] = long_df["rate_date"]
    holidays = us_federal_holidays(
        start=(long_df["rate_date"].min() - pd.Timedelta(days=7)).isoformat(),
        end=(long_df["rate_date"].max() + pd.Timedelta(days=21)).isoformat(),
    )
    unique_dates = long_df["rate_date"].dropna().drop_duplicates()
    safe_by_date = {d: next_full_business_day_after(d, holidays) for d in unique_dates}
    long_df["publication_safe_available_date"] = long_df["rate_date"].map(safe_by_date)
    long_df["publication_rule"] = (
        "same_trading_day_by_approximately_6pm_et_then_next_full_us_business_day"
    )
    long_df["availability_precision"] = "documented_approximate_time_no_exact_historical_timestamp"

    long_df["source_id"] = "treasury_daily_par_yield_curve"
    long_df["retrieval_timestamp_utc"] = retrieval_timestamp_utc
    long_df["raw_artifact_id"] = raw_artifact_id

    ordered_cols = [
        "rate_date",
        "maturity_label",
        "maturity_years",
        "par_yield_percent",
        "value_raw",
        "units",
        "source_id",
        "methodology_regime",
        "retrieval_timestamp_utc",
        "raw_artifact_id",
        "publication_rule",
        "publication_date",
        "publication_safe_available_date",
        "availability_precision",
        "is_missing",
        "missing_reason",
    ]
    return (
        long_df[ordered_cols]
        .sort_values(["maturity_label", "rate_date"])
        .reset_index(drop=True)
    )


def pivot_wide_by_maturity(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per `rate_date`, one column per maturity label (percent
    values) -- convenient shape for rolling-window candidate features.
    """
    wide = long_df.pivot_table(index="rate_date", columns="maturity_label", values="par_yield_percent", aggfunc="first")
    wide.columns.name = None
    extras = long_df.groupby("rate_date")[
        ["publication_date", "publication_safe_available_date"]
    ].first()
    return wide.join(extras).reset_index().sort_values("rate_date").reset_index(drop=True)
