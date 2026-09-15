"""Parse an RTDSM xlsx workbook into a tidy long-format vintage table,
and attach release-timing metadata to each vintage.

Every numeric value is read exactly as pandas reads a blank Excel
cell: `NaN` -- never imputed, forward-filled, or treated as zero (see
`rtdsm_schema.py`'s module docstring).
"""

from __future__ import annotations

import calendar
import io
import re

import pandas as pd

from treasury_auction_stress.data.rtdsm_schema import (
    PRECISION_VERIFIED_EXACT_DAY,
    QUARTER_TO_MIDDLE_MONTH,
    QUARTERLY_VINTAGE_NOMINAL_DAY,
    YOY_LAG_PERIODS_BY_OBSERVATION_FREQUENCY,
    RtdsmVariable,
    resolve_vintage_year,
)
from treasury_auction_stress.data.time_utils import (
    next_full_business_day_after,
    us_federal_holidays,
)

_MONTHLY_VINTAGE_RE = re.compile(r"^(\d{2})M(\d{1,2})$")
_QUARTERLY_VINTAGE_RE = re.compile(r"^(\d{2})Q(\d)$")
_MONTHLY_OBS_RE = re.compile(r"^(\d{4}):(\d{2})$")
_QUARTERLY_OBS_RE = re.compile(r"^(\d{4}):Q(\d)$")


def _parse_observation_period(period: str, observation_frequency: str) -> pd.Timestamp:
    if observation_frequency == "monthly":
        m = _MONTHLY_OBS_RE.match(period)
        if not m:
            return pd.NaT
        year, month = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(year=year, month=month, day=1)
    m = _QUARTERLY_OBS_RE.match(period)
    if not m:
        return pd.NaT
    year, quarter = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year=year, month=QUARTER_TO_MIDDLE_MONTH[quarter] - 1, day=1)


def _parse_vintage_suffix(suffix: str, vintage_frequency: str) -> tuple[int, int] | None:
    pattern = _MONTHLY_VINTAGE_RE if vintage_frequency == "monthly" else _QUARTERLY_VINTAGE_RE
    m = pattern.match(suffix)
    if not m:
        return None
    return resolve_vintage_year(int(m.group(1))), int(m.group(2))


def parse_workbook(raw_bytes: bytes, variable: RtdsmVariable) -> tuple[pd.DataFrame, dict]:
    """Melt the wide vintage workbook into long format: one row per
    (observation_period, vintage_label). Columns not matching this
    variable's own vintage-naming pattern are dropped and reported in
    `anomalies["unrecognized_vintage_columns"]`, never silently kept.
    """
    anomalies: dict = {"unrecognized_vintage_columns": [], "unparseable_observation_periods": 0, "observation_period_gaps": 0}
    xls = pd.ExcelFile(io.BytesIO(raw_bytes))
    raw_df = xls.parse(xls.sheet_names[0])

    prefix = variable.mnemonic
    records = []
    for col in raw_df.columns:
        if col == "DATE":
            continue
        if not col.startswith(prefix):
            anomalies["unrecognized_vintage_columns"].append(col)
            continue
        parsed = _parse_vintage_suffix(col[len(prefix) :], variable.vintage_frequency)
        if parsed is None:
            anomalies["unrecognized_vintage_columns"].append(col)
            continue
        vintage_year, vintage_period = parsed
        records.append((col, vintage_year, vintage_period))

    obs_dates = raw_df["DATE"].map(lambda p: _parse_observation_period(p, variable.observation_frequency))
    anomalies["unparseable_observation_periods"] = int(obs_dates.isna().sum())

    # Phase 4 acceptance review, issue 7: the year-over-year feature
    # relies on a *complete* period grid (one row per calendar period,
    # blank cells for missing values -- never a missing ROW) so that a
    # positional shift is also a calendar-exact shift. Detected, not
    # assumed, for every workbook actually parsed.
    valid_dates = obs_dates.dropna().sort_values()
    if len(valid_dates) > 1:
        freq = "MS" if variable.observation_frequency == "monthly" else "QS"
        expected = pd.date_range(valid_dates.iloc[0], valid_dates.iloc[-1], freq=freq)
        anomalies["observation_period_gaps"] = int(len(expected) - len(valid_dates.unique()))

    frames = []
    for col, vintage_year, vintage_period in records:
        frames.append(
            pd.DataFrame(
                {
                    "mnemonic": variable.mnemonic,
                    "observation_period": raw_df["DATE"],
                    "observation_date": obs_dates,
                    "vintage_label": col[len(prefix) :],
                    "vintage_year": vintage_year,
                    "vintage_period": vintage_period,
                    "value": pd.to_numeric(raw_df[col], errors="coerce"),
                }
            )
        )
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long_df, anomalies


def attach_release_dates(long_df: pd.DataFrame, variable: RtdsmVariable) -> pd.DataFrame:
    """Attach `nominal_publication_date`, `publication_safe_available_date`,
    and `availability_precision` to every row, computed purely from
    `vintage_year`/`vintage_period` (see `rtdsm_schema.py`'s module
    docstring for the exact/generalized precision distinction).
    """
    if long_df.empty:
        return long_df
    out = long_df.copy()

    # Computed once per distinct vintage (not once per melted row -- a
    # monthly-vintage variable with a long history can have hundreds of
    # thousands of (observation, vintage) rows sharing only a few
    # hundred distinct vintages) and merged back.
    vintages = out[["vintage_year", "vintage_period"]].drop_duplicates().reset_index(drop=True)

    if variable.vintage_frequency == "quarterly":
        middle_month = vintages["vintage_period"].map(QUARTER_TO_MIDDLE_MONTH)
        nominal = pd.to_datetime(
            {"year": vintages["vintage_year"], "month": middle_month, "day": QUARTERLY_VINTAGE_NOMINAL_DAY}
        )
        precision = PRECISION_VERIFIED_EXACT_DAY
    else:
        # Phase 4 acceptance review, issue 3: each monthly-vintage
        # variable uses its OWN evidence-derived nominal day (never
        # borrowed from an unrelated variable's release schedule -- see
        # rtdsm_schema.py's module docstring), clamped to the true
        # number of days in that vintage's own month (e.g. ROUTPUT's
        # day 30 in a vintage month of February).
        nominal_day = variable.monthly_vintage_nominal_day
        days_in_month = vintages.apply(
            lambda row: calendar.monthrange(int(row["vintage_year"]), int(row["vintage_period"]))[1], axis=1
        )
        clamped_day = days_in_month.clip(upper=nominal_day)
        nominal = pd.to_datetime(
            {"year": vintages["vintage_year"], "month": vintages["vintage_period"], "day": clamped_day}
        )
        precision = variable.monthly_vintage_precision_label

    holidays = us_federal_holidays(
        start=(nominal.min() - pd.Timedelta(days=7)).isoformat(),
        end=(nominal.max() + pd.Timedelta(days=21)).isoformat(),
    )
    vintages["nominal_publication_date"] = nominal
    vintages["publication_safe_available_date"] = nominal.map(lambda d: next_full_business_day_after(d, holidays))
    vintages["availability_precision"] = precision

    out = out.merge(vintages, on=["vintage_year", "vintage_period"], how="left")
    return out


def vintage_index(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct vintage, for the as-of join against
    auctions -- `cftc_join`/`treasury_rates_join`'s pattern applied to
    a coarser (monthly/quarterly) release cadence.
    """
    if long_df.empty:
        return long_df
    return (
        long_df[["vintage_label", "vintage_year", "vintage_period", "nominal_publication_date", "publication_safe_available_date", "availability_precision"]]
        .drop_duplicates(subset="vintage_label")
        .sort_values("publication_safe_available_date")
        .reset_index(drop=True)
    )


def build_snapshot_table(long_df: pd.DataFrame, variable: RtdsmVariable) -> pd.DataFrame:
    """One row per vintage: the most recent non-null observation known
    as of that vintage (`current_observation_date`/`current_value`),
    plus the observation exactly one year earlier in that same
    vintage's own period sequence (`yoy_observation_date`/`yoy_value`)
    -- computed via `shift` over the *full* period sequence (including
    any null rows) so "one year ago" means one year of periods, not
    one year of non-null observations.
    """
    if long_df.empty:
        return long_df
    yoy_lag = YOY_LAG_PERIODS_BY_OBSERVATION_FREQUENCY[variable.observation_frequency]
    df = long_df.sort_values(["vintage_label", "observation_date"]).reset_index(drop=True).copy()
    grouped = df.groupby("vintage_label", sort=False)
    df["yoy_value"] = grouped["value"].shift(yoy_lag)
    df["yoy_observation_date"] = grouped["observation_date"].shift(yoy_lag)

    valid = df.loc[df["value"].notna()]
    if valid.empty:
        return pd.DataFrame(
            columns=["vintage_label", "current_observation_date", "current_value", "yoy_observation_date", "yoy_value"]
        )
    idx = valid.groupby("vintage_label")["observation_date"].idxmax()
    current = df.loc[idx, ["vintage_label", "observation_date", "value", "yoy_observation_date", "yoy_value"]]
    return current.rename(columns={"observation_date": "current_observation_date", "value": "current_value"}).reset_index(drop=True)


def latest_known_value(long_df: pd.DataFrame, vintage_label: str) -> tuple[pd.Timestamp, float]:
    """Within one vintage's own column, the most recent non-null
    observation -- i.e. exactly what a real-time forecaster would have
    known as "the latest {variable} reading" as of that vintage.
    """
    rows = long_df.loc[(long_df["vintage_label"] == vintage_label) & long_df["value"].notna()]
    if rows.empty:
        return pd.NaT, float("nan")
    row = rows.loc[rows["observation_date"].idxmax()]
    return row["observation_date"], float(row["value"])
