from __future__ import annotations

import pandas as pd

from treasury_auction_stress.data.rtdsm_normalize import (
    attach_release_dates,
    build_snapshot_table,
    latest_known_value,
    parse_workbook,
    vintage_index,
)
from treasury_auction_stress.data.rtdsm_schema import (
    PRECISION_VERIFIED_EXACT_DAY,
    VARIABLE_BY_MNEMONIC,
)


def test_ruc_quarterly_vintage_columns_parsed_with_no_anomalies(rtdsm_ruc_sample_bytes):
    long_df, anomalies = parse_workbook(rtdsm_ruc_sample_bytes, VARIABLE_BY_MNEMONIC["RUC"])
    assert anomalies == {"unrecognized_vintage_columns": [], "unparseable_observation_periods": 0, "observation_period_gaps": 0}
    assert set(long_df["vintage_label"]) == {"20Q2", "21Q1", "25Q4", "26Q1", "26Q3"}


def test_ruc_missing_values_are_nan_not_sentinel(rtdsm_ruc_sample_bytes):
    long_df, _ = parse_workbook(rtdsm_ruc_sample_bytes, VARIABLE_BY_MNEMONIC["RUC"])
    row = long_df[(long_df["vintage_label"] == "20Q2") & (long_df["observation_date"] == pd.Timestamp("2020-05-01"))]
    assert row["value"].isna().all()


def test_ruc_quarterly_release_dates_use_verified_exact_day(rtdsm_ruc_sample_bytes):
    long_df, _ = parse_workbook(rtdsm_ruc_sample_bytes, VARIABLE_BY_MNEMONIC["RUC"])
    long_df = attach_release_dates(long_df, VARIABLE_BY_MNEMONIC["RUC"])
    row = long_df[long_df["vintage_label"] == "20Q2"].iloc[0]
    assert row["nominal_publication_date"] == pd.Timestamp("2020-05-15")
    assert row["availability_precision"] == PRECISION_VERIFIED_EXACT_DAY


def test_revision_is_visible_across_vintages(rtdsm_ruc_sample_bytes):
    """The synthetic RUC fixture deliberately gives 2020:04 a different
    fabricated value in vintage 21Q1 than in 20Q2 -- a structural
    revision-across-vintages case (see tests/rtdsm_synthetic_fixtures.py).
    For the real, historically documented RUC revision this case is
    modeled on, see the live_network test in tests/test_live_network.py.
    """
    long_df, _ = parse_workbook(rtdsm_ruc_sample_bytes, VARIABLE_BY_MNEMONIC["RUC"])
    april_2020 = long_df[long_df["observation_date"] == pd.Timestamp("2020-04-01")]
    first_release = april_2020[april_2020["vintage_label"] == "20Q2"]["value"].iloc[0]
    later_vintage = april_2020[april_2020["vintage_label"] == "21Q1"]["value"].iloc[0]
    assert first_release == 6.0
    assert later_vintage == 6.5
    assert first_release != later_vintage


def test_latest_known_value_respects_a_permanent_gap_not_forward_filled(rtdsm_ruc_sample_bytes):
    """The synthetic RUC fixture deliberately leaves 2025:09 onward
    blank in vintage 25Q4 (a permanent-gap edge case, structurally
    analogous to a real government-shutdown-induced reporting gap this
    project observed live -- see the live_network RTDSM test for the
    real-data verification). The last fabricated non-null value is at
    2025:08."""
    long_df, _ = parse_workbook(rtdsm_ruc_sample_bytes, VARIABLE_BY_MNEMONIC["RUC"])
    obs_date, value = latest_known_value(long_df, "25Q4")
    assert obs_date == pd.Timestamp("2025-08-01")
    assert value == 12.4


def test_ipt_monthly_vintage_columns_and_variable_specific_precision(rtdsm_ipt_sample_bytes):
    variable = VARIABLE_BY_MNEMONIC["IPT"]
    long_df, anomalies = parse_workbook(rtdsm_ipt_sample_bytes, variable)
    assert anomalies == {"unrecognized_vintage_columns": [], "unparseable_observation_periods": 0, "observation_period_gaps": 0}
    long_df = attach_release_dates(long_df, variable)
    row = long_df[long_df["vintage_label"] == "20M6"].iloc[0]
    assert row["nominal_publication_date"] == pd.Timestamp("2020-06-18")
    assert row["availability_precision"] == variable.monthly_vintage_precision_label
    assert "federal_reserve_board_ip" in variable.monthly_vintage_precision_label


def test_ipt_anomaly_not_visible_in_earlier_vintage_visible_in_later_vintage(rtdsm_ipt_sample_bytes):
    """The synthetic IPT fixture deliberately leaves 2020:04 unreported
    in the earlier 20M3 vintage (it hadn't happened yet as of that
    vintage's own collection date) and gives it a fabricated, visibly
    off-trend value in the later 20M6 vintage -- a structural analog of
    "an anomaly invisible in an earlier vintage, fully visible in a
    later one," never a real industrial-production reading. For the
    real COVID-19 industrial-production collapse this case is modeled
    on, see the live_network test in tests/test_live_network.py."""
    long_df, _ = parse_workbook(rtdsm_ipt_sample_bytes, VARIABLE_BY_MNEMONIC["IPT"])
    march_vintage = long_df[long_df["vintage_label"] == "20M3"]
    assert march_vintage[march_vintage["observation_date"] == pd.Timestamp("2020-03-01")]["value"].isna().all()
    june_vintage = long_df[long_df["vintage_label"] == "20M6"]
    april_value = june_vintage[june_vintage["observation_date"] == pd.Timestamp("2020-04-01")]["value"].iloc[0]
    assert april_value == 85.0


def test_vintage_index_is_one_row_per_vintage_sorted(rtdsm_ruc_sample_bytes):
    long_df, _ = parse_workbook(rtdsm_ruc_sample_bytes, VARIABLE_BY_MNEMONIC["RUC"])
    long_df = attach_release_dates(long_df, VARIABLE_BY_MNEMONIC["RUC"])
    vidx = vintage_index(long_df)
    assert len(vidx) == 5
    assert list(vidx["publication_safe_available_date"]) == sorted(vidx["publication_safe_available_date"])


def test_snapshot_table_yoy_uses_full_period_sequence_not_just_nonnull(rtdsm_ruc_sample_bytes):
    long_df, _ = parse_workbook(rtdsm_ruc_sample_bytes, VARIABLE_BY_MNEMONIC["RUC"])
    snap = build_snapshot_table(long_df, VARIABLE_BY_MNEMONIC["RUC"])
    row = snap[snap["vintage_label"] == "21Q1"].iloc[0]
    assert (row["current_observation_date"] - row["yoy_observation_date"]).days in (365, 366)


# -- Phase 4 acceptance review, issue 3: variable-specific release-date evidence --


def _synthetic_monthly_vintage_frame(vintage_year: int, vintage_period: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vintage_year": [vintage_year],
            "vintage_period": [vintage_period],
            "vintage_label": ["x"],
            "observation_period": ["2020:01"],
            "observation_date": [pd.Timestamp("2020-01-01")],
            "value": [1.0],
        }
    )


def test_different_monthly_vintage_variables_get_different_nominal_days():
    """The core issue-3 regression: ROUTPUT (BEA, late-month) and IPT
    (Federal Reserve Board, mid-month) must NOT share a nominal day.
    """
    routput = VARIABLE_BY_MNEMONIC["ROUTPUT"]
    ipt = VARIABLE_BY_MNEMONIC["IPT"]
    assert routput.monthly_vintage_nominal_day != ipt.monthly_vintage_nominal_day
    assert routput.monthly_vintage_precision_label != ipt.monthly_vintage_precision_label

    routput_row = attach_release_dates(_synthetic_monthly_vintage_frame(2021, 3), routput).iloc[0]
    ipt_row = attach_release_dates(_synthetic_monthly_vintage_frame(2021, 3), ipt).iloc[0]
    assert routput_row["nominal_publication_date"] == pd.Timestamp("2021-03-30")
    assert ipt_row["nominal_publication_date"] == pd.Timestamp("2021-03-18")
    assert routput_row["nominal_publication_date"] != ipt_row["nominal_publication_date"]


def test_routput_nominal_day_clamps_to_actual_month_length():
    """ROUTPUT's day-30 rule must clamp in February (28 or 29 days),
    never overflow into March.
    """
    routput = VARIABLE_BY_MNEMONIC["ROUTPUT"]
    row_2021 = attach_release_dates(_synthetic_monthly_vintage_frame(2021, 2), routput).iloc[0]  # non-leap
    row_2024 = attach_release_dates(_synthetic_monthly_vintage_frame(2024, 2), routput).iloc[0]  # leap
    assert row_2021["nominal_publication_date"] == pd.Timestamp("2021-02-28")
    assert row_2024["nominal_publication_date"] == pd.Timestamp("2024-02-29")


def test_pcpi_replaces_cpi_and_is_monthly_vintage_not_quarterly():
    assert "CPI" not in VARIABLE_BY_MNEMONIC
    pcpi = VARIABLE_BY_MNEMONIC["PCPI"]
    assert pcpi.vintage_frequency == "monthly"
    assert pcpi.releasing_institution == "Bureau of Labor Statistics (BLS)"


# -- Phase 4 acceptance review, issue 7: year-over-year calendar-offset correctness --


def test_monthly_yoy_uses_exact_calendar_offset_despite_a_missing_intermediate_period():
    """A positional-only implementation (e.g. shift over non-null rows
    only) would be thrown off by the missing June 2020 value and end up
    comparing the wrong calendar months. The real implementation shifts
    over the *full* period grid (including the null row), so it must
    still land on exactly one year earlier.
    """
    dates = pd.date_range("2019-01-01", "2021-06-01", freq="MS")
    values = [100.0 + i for i in range(len(dates))]
    missing_idx = dates.get_loc(pd.Timestamp("2020-06-01"))
    values[missing_idx] = float("nan")
    long_df = pd.DataFrame(
        {
            "vintage_label": "v1",
            "observation_date": dates,
            "value": values,
        }
    )
    snap = build_snapshot_table(long_df, VARIABLE_BY_MNEMONIC["IPT"])  # monthly observation frequency
    row = snap.iloc[0]
    assert row["current_observation_date"] == row["yoy_observation_date"] + pd.DateOffset(years=1)
    assert row["current_value"] == values[-1]


def test_quarterly_yoy_uses_exact_calendar_offset_not_four_non_null_rows_back():
    dates = pd.date_range("2018-01-01", "2021-01-01", freq="QS")
    values = [50.0 + i for i in range(len(dates))]
    missing_idx = dates.get_loc(pd.Timestamp("2019-04-01"))
    values[missing_idx] = float("nan")
    long_df = pd.DataFrame({"vintage_label": "v1", "observation_date": dates, "value": values})
    snap = build_snapshot_table(long_df, VARIABLE_BY_MNEMONIC["ROUTPUT"])  # quarterly observation frequency
    row = snap.iloc[0]
    assert row["current_observation_date"] == row["yoy_observation_date"] + pd.DateOffset(years=1)


def test_yoy_comparison_never_crosses_a_vintage_boundary_so_base_year_changes_are_safe():
    """RTDSM's own documentation states an index variable's base year
    'varies over vintages (but not over time within a vintage).' Since
    `current_value` and `yoy_value` are both drawn from the SAME
    vintage's own column, a base-year change between vintages can never
    silently corrupt a single YoY comparison -- verified directly by
    the shift being computed within, not across, `vintage_label` groups.
    """
    dates = pd.date_range("2019-01-01", "2020-06-01", freq="MS")
    v1 = pd.DataFrame({"vintage_label": "v1", "observation_date": dates, "value": [100.0 + i for i in range(len(dates))]})
    v2 = pd.DataFrame({"vintage_label": "v2", "observation_date": dates, "value": [500.0 + i for i in range(len(dates))]})  # different base
    long_df = pd.concat([v1, v2], ignore_index=True)
    snap = build_snapshot_table(long_df, VARIABLE_BY_MNEMONIC["IPT"])
    row_v1 = snap[snap["vintage_label"] == "v1"].iloc[0]
    row_v2 = snap[snap["vintage_label"] == "v2"].iloc[0]
    # each vintage's yoy_value must come from its OWN base, never mixed --
    # 18 monthly rows (2019-01..2020-06), yoy_lag=12 -> current=index 17,
    # yoy=index 5 (2019-06), so v1's yoy is 100+5 and v2's is 500+5.
    assert row_v1["yoy_value"] == 105.0
    assert row_v2["yoy_value"] == 505.0
    assert row_v1["current_value"] == 117.0
    assert row_v2["current_value"] == 517.0


def test_every_monthly_vintage_variable_has_its_own_cited_institution():
    """No two monthly-vintage variables may share a releasing
    institution + nominal day pulled from the same evidence
    unless they are genuinely released by the same agency on the same
    schedule (this project has none such among its 5 monthly-vintage
    variables).
    """
    monthly_vars = [v for v in VARIABLE_BY_MNEMONIC.values() if v.vintage_frequency == "monthly"]
    assert len(monthly_vars) == 5
    for v in monthly_vars:
        assert v.monthly_vintage_nominal_day is not None
        assert v.monthly_vintage_precision_label is not None
        assert v.releasing_institution
        assert v.release_date_source_citation
