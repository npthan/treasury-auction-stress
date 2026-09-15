"""Tests that actually call real external APIs (Treasury Fiscal Data,
NY Fed Primary Dealer Statistics, Treasury daily par yield curve, CFTC
TFF Futures Only, and Philadelphia Fed RTDSM). Never FRED/ALFRED.

Excluded from the default `pytest` run (see the `addopts` in
pyproject.toml) so the rest of the suite stays deterministic and
offline. Run explicitly with:

    uv run pytest -o addopts="" -m live_network

(`-o addopts=""` clears pyproject.toml's `addopts = "-m 'not
live_network'"` first -- passing a second `-m` on the command line
alone does not reliably override it.)
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from treasury_auction_stress.data.cftc_client import fetch_contract
from treasury_auction_stress.data.client import fetch_page
from treasury_auction_stress.data.dealer_stats_client import (
    fetch_legacy_series,
    fetch_series,
    fetch_series_list,
)
from treasury_auction_stress.data.rtdsm_client import (
    EXPECTED_CONTENT_TYPE,
    fetch_variable,
)
from treasury_auction_stress.data.rtdsm_normalize import (
    attach_release_dates,
    latest_known_value,
    parse_workbook,
)
from treasury_auction_stress.data.rtdsm_schema import VARIABLE_BY_MNEMONIC
from treasury_auction_stress.data.treasury_rates_client import fetch_year
from treasury_auction_stress.data.treasury_rates_normalize import parse_csv_to_wide


@pytest.mark.live_network
def test_live_fetch_single_small_page():
    page = fetch_page(
        filter_str="auction_date:gte:2024-01-01",
        sort="auction_date",
        page_number=1,
        page_size=2,
    )
    assert page.http_status == 200
    assert "data" in page.parsed
    assert len(page.parsed["data"]) <= 2


@pytest.mark.live_network
def test_live_fetch_dealer_stats_series_list():
    result = fetch_series_list()
    assert result.http_status == 200
    keyids = {e["keyid"] for e in result.parsed["pd"]["timeseries"]}
    assert "PDPOSGS-B" in keyids  # a series this project actually selects


@pytest.mark.live_network
def test_live_fetch_dealer_stats_single_series():
    result = fetch_series("PDPOSGS-B")
    assert result.http_status == 200
    obs = result.parsed["pd"]["timeseries"]
    assert len(obs) > 0
    assert obs[0]["asofdate"] == "2013-04-03"  # verified fixed start of this keyid's history


@pytest.mark.live_network
def test_live_fetch_legacy_pre_2013_series():
    """Phase 3 acceptance review: proves the legacy, period-scoped
    endpoint genuinely serves pre-2013 data live, not from a fixture --
    this is the primary-source evidence the historical extension
    depends on.
    """
    result = fetch_legacy_series("SBP2013", "PDPUSGTBNOP")
    assert result.http_status == 200
    obs = result.parsed["pd"]["timeseries"]
    assert len(obs) > 0
    assert obs[0]["asofdate"] == "2001-07-04"
    assert obs[-1]["asofdate"] == "2013-03-27"


@pytest.mark.live_network
def test_live_fetch_discontinued_long_bucket_still_reachable():
    """PDPOSGSC-G11 was superseded by PDPOSGSC-G11L21/PDPOSGSC-G21 on
    2022-01-05 and no longer appears in the active series list, but
    remains fetchable via the plain endpoint -- this is what the
    harmonized long-duration feature's 2013-2021 middle piece depends on.
    """
    result = fetch_series("PDPOSGSC-G11")
    assert result.http_status == 200
    obs = result.parsed["pd"]["timeseries"]
    assert obs[0]["asofdate"] == "2013-04-03"
    assert obs[-1]["asofdate"] == "2021-12-29"


# -- Phase 4 acceptance review, issue 1: one meaningful live test per new Phase 4 source --


@pytest.mark.live_network
def test_live_fetch_treasury_rates_year_has_expected_structure():
    """A real, closed historical year (2024) -- small, stable, and
    deterministic. Validates more than an HTTP 200: expected tenor
    columns, parseability, and a plausible full-year trading-day count.
    """
    result = fetch_year(2024)
    assert result.http_status == 200
    assert '"2 Yr"' in result.raw_text
    assert '"10 Yr"' in result.raw_text

    wide, anomalies = parse_csv_to_wide(result.raw_text, source_year=2024)
    assert anomalies["unrecognized_columns"] == []
    assert anomalies["duplicate_dates"] == 0
    assert 240 <= len(wide) <= 260  # a full year of U.S. trading days, not a truncated response
    assert wide["rate_date"].min().year == 2024
    assert wide["rate_date"].max().year == 2024


@pytest.mark.live_network
def test_live_fetch_cftc_contract_has_expected_report_type_and_structure():
    """A single, small (limit=5) query against the live CFTC Socrata
    endpoint for one selected contract -- validates the report type
    (Futures Only, never Combined) and expected fields, not just a 200.
    """
    result = fetch_contract("042601", limit=5)  # UST 2Y NOTE
    assert result.http_status == 200
    rows = json.loads(result.raw_text)
    assert len(rows) == 5
    row = rows[0]
    assert row["cftc_contract_market_code"] == "042601"
    assert row["futonly_or_combined"] == "FutOnly"  # never the Combined report
    assert "dealer_positions_long_all" in row
    assert "open_interest_all" in row
    report_date = pd.Timestamp(row["report_date_as_yyyy_mm_dd"])
    assert report_date.year >= 2006  # this project's verified earliest live observation


@pytest.mark.live_network
def test_live_fetch_rtdsm_variable_is_a_valid_workbook_with_expected_sheet():
    """RUC -- the smallest RTDSM workbook this project selects.
    Validates the file is a genuine, parseable xlsx workbook with the
    expected sheet name and vintage-column structure, not just a 200
    and a content-type header.
    """
    variable = VARIABLE_BY_MNEMONIC["RUC"]
    result = fetch_variable(variable)
    assert result.http_status == 200
    assert result.content_type == EXPECTED_CONTENT_TYPE

    xls = pd.ExcelFile(io.BytesIO(result.raw_bytes))
    assert xls.sheet_names == ["ruc"]

    long_df, anomalies = parse_workbook(result.raw_bytes, variable)
    assert anomalies["unrecognized_vintage_columns"] == []
    assert anomalies["unparseable_observation_periods"] == 0
    assert len(long_df) > 0
    assert long_df["vintage_label"].str.match(r"^\d{2}Q\d$").all()  # RUC is quarterly-vintage


# -- Phase 9 acceptance-review remediation: genuine RTDSM observations
# moved here from the always-run offline suite (which now uses wholly
# synthetic fixtures, see tests/rtdsm_synthetic_fixtures.py). These
# three tests assert real, historically documented facts against
# live-fetched data -- never a committed fixture -- and were actually
# run and observed to pass against live data during this review.


@pytest.mark.live_network
def test_live_fetch_rtdsm_ruc_reflects_the_verified_2010_march_revision():
    """The real, historically documented RUC revision this project's
    point-in-time join logic depends on: the March 2010 unemployment
    rate was first reported in vintage 10Q2 and later revised in
    vintage 13Q1 to a different value -- and 10Q2 became safely
    available well before 13Q1 did, so a June 2010 auction's
    point-in-time feature must be able to see only the first-known
    value. This is the real-data counterpart of the synthetic
    structural test in tests/test_rtdsm_join.py.
    """
    variable = VARIABLE_BY_MNEMONIC["RUC"]
    result = fetch_variable(variable)
    assert result.http_status == 200

    long_df, _ = parse_workbook(result.raw_bytes, variable)
    long_df = attach_release_dates(long_df, variable)

    def value_at(vintage_label: str, observation: str):
        row = long_df[
            (long_df["vintage_label"] == vintage_label)
            & (long_df["observation_date"] == pd.Timestamp(observation))
        ]
        return None if row.empty else row["value"].iloc[0]

    first_reported = value_at("10Q2", "2010-03-01")
    later_revision = value_at("13Q1", "2010-03-01")
    assert first_reported is not None
    assert later_revision is not None
    assert first_reported != later_revision

    row_10q2 = long_df[long_df["vintage_label"] == "10Q2"].iloc[0]
    row_13q1 = long_df[long_df["vintage_label"] == "13Q1"].iloc[0]
    real_auction_cutoff = pd.Timestamp("2010-06-03")  # CUSIP 912828ND8, 10-Year, auction_date 2010-06-09
    assert row_10q2["publication_safe_available_date"] <= real_auction_cutoff
    assert row_13q1["publication_safe_available_date"] > real_auction_cutoff


@pytest.mark.live_network
def test_live_fetch_rtdsm_ipt_covid_collapse_not_visible_in_march_vintage():
    """The real, historically documented COVID-19 industrial-production
    collapse: not yet visible in the 20M3 vintage (collected in March
    2020, before April's own data existed), fully visible in the 20M6
    vintage. This is the real-data counterpart of the synthetic
    structural test in tests/test_rtdsm_normalize.py.
    """
    variable = VARIABLE_BY_MNEMONIC["IPT"]
    result = fetch_variable(variable)
    assert result.http_status == 200

    long_df, _ = parse_workbook(result.raw_bytes, variable)
    march_vintage = long_df[long_df["vintage_label"] == "20M3"]
    assert not march_vintage.empty
    assert march_vintage[march_vintage["observation_date"] == pd.Timestamp("2020-03-01")]["value"].isna().all()

    june_vintage = long_df[long_df["vintage_label"] == "20M6"]
    assert not june_vintage.empty
    april_value = june_vintage[june_vintage["observation_date"] == pd.Timestamp("2020-04-01")]["value"]
    assert not april_value.empty
    assert april_value.notna().all()


@pytest.mark.live_network
def test_live_fetch_rtdsm_ruc_shutdown_gap_still_present_in_25q4_vintage():
    """The real, October-2025 government-shutdown-induced RUC reporting
    gap this project observed live: vintage 25Q4's own latest non-null
    observation is 2025-08-01, two months earlier than the 2025-10-01 a
    vintage of its age would ordinarily be expected to carry. This is
    the real-data counterpart of the synthetic structural test in
    tests/test_rtdsm_normalize.py and tests/test_rtdsm_join.py. If this
    ever starts failing because the gap was later backfilled, that is
    good news, not a bug -- update or remove this test at that point.
    """
    variable = VARIABLE_BY_MNEMONIC["RUC"]
    result = fetch_variable(variable)
    assert result.http_status == 200

    long_df, _ = parse_workbook(result.raw_bytes, variable)
    if "25Q4" not in set(long_df["vintage_label"]):
        pytest.skip("vintage 25Q4 no longer present in the live RUC workbook")
    obs_date, value = latest_known_value(long_df, "25Q4")
    assert obs_date == pd.Timestamp("2025-08-01")
    assert value is not None
