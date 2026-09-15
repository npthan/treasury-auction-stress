from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.data.treasury_rates_normalize import (
    parse_csv_to_wide,
    pivot_wide_by_maturity,
    to_long,
)
from treasury_auction_stress.data.treasury_rates_schema import (
    REGIME_20Y_REAL_BOND,
    REGIME_MC_SPLINE,
)


def test_parse_modern_schema_correctly(treasury_rates_2024_csv):
    wide, anomalies = parse_csv_to_wide(treasury_rates_2024_csv, source_year=2024)
    assert anomalies == {"unrecognized_columns": [], "duplicate_dates": 0}
    assert len(wide) == 10
    assert "20 Yr" in wide.columns and "1.5 Month" not in wide.columns


def test_older_schema_has_no_phantom_columns(treasury_rates_1990_csv):
    wide, anomalies = parse_csv_to_wide(treasury_rates_1990_csv, source_year=1990)
    assert anomalies["unrecognized_columns"] == []
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    maturities = set(long_df["maturity_label"])
    assert "1 Mo" not in maturities
    assert "20 Yr" not in maturities
    assert "30 Yr" in maturities


def test_current_schema_includes_one_point_five_month(treasury_rates_2026_csv):
    wide, anomalies = parse_csv_to_wide(treasury_rates_2026_csv, source_year=2026)
    assert anomalies["unrecognized_columns"] == []
    assert "1.5 Month" in wide.columns


def test_30yr_suspension_boundary_is_missing_not_zero(treasury_rates_2002_suspension_csv):
    wide, _ = parse_csv_to_wide(treasury_rates_2002_suspension_csv, source_year=2002)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    thirty = long_df.loc[long_df["maturity_label"] == "30 Yr"].set_index("rate_date")

    missing_row = thirty.loc[pd.Timestamp("2002-02-19")]
    assert missing_row["is_missing"]
    assert pd.isna(missing_row["par_yield_percent"])
    assert "empty value" in missing_row["missing_reason"]

    present_row = thirty.loc[pd.Timestamp("2002-02-15")]
    assert not present_row["is_missing"]
    assert present_row["par_yield_percent"] == pytest.approx(5.37)


def test_percent_values_preserved_verbatim(treasury_rates_2024_csv):
    wide, _ = parse_csv_to_wide(treasury_rates_2024_csv, source_year=2024)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    row = long_df.loc[
        (long_df["rate_date"] == pd.Timestamp("2024-12-31")) & (long_df["maturity_label"] == "10 Yr")
    ].iloc[0]
    assert row["par_yield_percent"] == 4.58
    assert row["value_raw"] == "4.58"
    assert row["units"] == "percent"


def test_methodology_regime_mc_spline_for_2024(treasury_rates_2024_csv):
    wide, _ = parse_csv_to_wide(treasury_rates_2024_csv, source_year=2024)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    tenyr = long_df.loc[long_df["maturity_label"] == "10 Yr"]
    assert (tenyr["methodology_regime"] == REGIME_MC_SPLINE).all()


def test_methodology_regime_20yr_combines_both_regimes(treasury_rates_2024_csv):
    wide, _ = parse_csv_to_wide(treasury_rates_2024_csv, source_year=2024)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    twentyyr = long_df.loc[long_df["maturity_label"] == "20 Yr"]
    assert (twentyyr["methodology_regime"] == f"{REGIME_MC_SPLINE}+{REGIME_20Y_REAL_BOND}").all()


def test_publication_date_equals_rate_date(treasury_rates_2024_csv):
    wide, _ = parse_csv_to_wide(treasury_rates_2024_csv, source_year=2024)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    row = long_df.iloc[0]
    assert row["publication_date"] == row["rate_date"]


def test_ordinary_weekday_safe_available_date_is_next_calendar_day():
    # A Tuesday with no adjacent holiday -- the +1-calendar-day and
    # next-full-business-day rules happen to agree here.
    csv_text = 'Date,"2 Yr"\n12/17/2024,4.25\n'
    wide, _ = parse_csv_to_wide(csv_text, source_year=2024)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    row = long_df.iloc[0]
    assert row["publication_safe_available_date"] == pd.Timestamp("2024-12-18")


def test_friday_rate_date_gets_monday_safe_available_date_not_saturday():
    """Phase 4 acceptance review, issue 2: a Friday `rate_date` must
    never receive a Saturday `publication_safe_available_date`. Fixed
    from a plain `rate_date + 1 calendar day` to
    `time_utils.next_full_business_day_after`.
    """
    csv_text = 'Date,"2 Yr"\n09/11/2026,3.50\n'  # 2026-09-11 is a Friday
    wide, _ = parse_csv_to_wide(csv_text, source_year=2026)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    row = long_df.iloc[0]
    assert row["publication_safe_available_date"] == pd.Timestamp("2026-09-14")  # Monday
    assert row["publication_safe_available_date"].weekday() < 5


def test_friday_rate_date_with_monday_holiday_skips_to_tuesday():
    csv_text = 'Date,"2 Yr"\n01/16/2026,3.60\n'  # Friday; Monday 2026-01-19 is MLK Day
    wide, _ = parse_csv_to_wide(csv_text, source_year=2026)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    row = long_df.iloc[0]
    assert row["publication_safe_available_date"] == pd.Timestamp("2026-01-20")  # Tuesday


def test_year_end_boundary_safe_available_date_skips_new_years_day():
    csv_text = 'Date,"2 Yr"\n12/31/2025,3.70\n'  # Wednesday; Thursday 2026-01-01 is a holiday
    wide, _ = parse_csv_to_wide(csv_text, source_year=2025)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    row = long_df.iloc[0]
    assert row["publication_safe_available_date"] == pd.Timestamp("2026-01-02")


def test_no_rate_date_ever_produces_a_weekend_safe_available_date(treasury_rates_2024_csv):
    wide, _ = parse_csv_to_wide(treasury_rates_2024_csv, source_year=2024)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    assert (long_df["publication_safe_available_date"].dt.weekday < 5).all()


def test_duplicate_dates_detected():
    csv_text = 'Date,"1 Mo"\n01/02/2024,4.00\n01/02/2024,4.01\n'
    _wide, anomalies = parse_csv_to_wide(csv_text, source_year=2024)
    assert anomalies["duplicate_dates"] == 1


def test_pivot_wide_by_maturity_shape(treasury_rates_2024_csv):
    wide, _ = parse_csv_to_wide(treasury_rates_2024_csv, source_year=2024)
    long_df = to_long(wide, retrieval_timestamp_utc="x", raw_artifact_id="y")
    pivoted = pivot_wide_by_maturity(long_df)
    assert len(pivoted) == 10
    assert pivoted["rate_date"].is_monotonic_increasing
    assert "10 Yr" in pivoted.columns
    assert "publication_safe_available_date" in pivoted.columns
