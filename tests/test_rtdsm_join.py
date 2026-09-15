from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.data.rtdsm_normalize import (
    attach_release_dates,
    build_snapshot_table,
    parse_workbook,
    vintage_index,
)
from treasury_auction_stress.data.rtdsm_schema import VARIABLE_BY_MNEMONIC
from treasury_auction_stress.features.rtdsm_join import (
    NO_COVERAGE_REASON_TEMPLATE,
    NO_NONNULL_OBSERVATION_REASON,
    as_of_join_variable,
)


@pytest.fixture
def ruc_pieces(rtdsm_ruc_sample_bytes):
    variable = VARIABLE_BY_MNEMONIC["RUC"]
    long_df, _ = parse_workbook(rtdsm_ruc_sample_bytes, variable)
    long_df = attach_release_dates(long_df, variable)
    return variable, vintage_index(long_df), build_snapshot_table(long_df, variable)


def _auctions(rows):
    return pd.DataFrame(rows)


def test_never_selects_a_later_vintage(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    for cutoff in pd.date_range("2020-01-01", "2026-08-01", freq="30D"):
        auctions = _auctions([{"cusip": "X", "cutoff": cutoff}])
        result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
        if result.loc[0, "RUC_join_matched"]:
            matched_vintage = vidx[vidx["vintage_label"] == result.loc[0, "RUC_vintage_label"]].iloc[0]
            assert matched_vintage["publication_safe_available_date"] <= cutoff


def test_no_auction_dropped_for_lacking_coverage(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    auctions = _auctions([{"cusip": "X", "cutoff": pd.Timestamp("2000-01-01")}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert len(result) == 1
    assert not result.loc[0, "RUC_join_matched"]
    assert result.loc[0, "RUC_missing_reason"] == NO_COVERAGE_REASON_TEMPLATE.format(mnemonic="RUC")


def test_matched_level_and_yoy_change_are_populated_from_the_matched_vintage(ruc_pieces):
    """RUC_level below is a fabricated value from the synthetic fixture
    (tests/rtdsm_synthetic_fixtures.py), not a real observation."""
    variable, vidx, snap = ruc_pieces
    row_20q2 = vidx[vidx["vintage_label"] == "20Q2"].iloc[0]
    auctions = _auctions([{"cusip": "X", "cutoff": row_20q2["publication_safe_available_date"]}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert result.loc[0, "RUC_join_matched"]
    assert result.loc[0, "RUC_vintage_label"] == "20Q2"
    assert result.loc[0, "RUC_level"] == 6.0  # synthetic first-released April-2020-analog value
    assert result.loc[0, "RUC_observation_date"] == pd.Timestamp("2020-04-01")


def test_nat_cutoff_does_not_crash_and_is_unmatched(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    auctions = _auctions([{"cusip": "X", "cutoff": pd.NaT}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert len(result) == 1
    assert not result.loc[0, "RUC_join_matched"]


def test_reserved_column_name_raises(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    auctions = _auctions([{"cusip": "X", "cutoff": pd.Timestamp("2025-01-01")}])
    auctions["__as_of_join_row_order__"] = 0
    with pytest.raises(ValueError):
        as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")


def test_matched_vintage_with_shutdown_gap_reports_stale_observation_not_synthesized(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    row_25q4 = vidx[vidx["vintage_label"] == "25Q4"].iloc[0]
    auctions = _auctions([{"cusip": "X", "cutoff": row_25q4["publication_safe_available_date"]}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert result.loc[0, "RUC_join_matched"]
    assert result.loc[0, "RUC_observation_date"] == pd.Timestamp("2025-08-01")
    assert pd.isna(result.loc[0, "RUC_missing_reason"])


# -- Phase 4 acceptance review, issue 6: gap/staleness semantics --


def test_shutdown_gap_sets_source_gap_detected_flag(ruc_pieces):
    """The synthetic fixture's permanent-gap edge case (structurally
    analogous to a real, October-2025 government-shutdown-induced RUC
    gap this project observed live -- see the live_network RTDSM test
    for the real-data verification): expected latest observation is
    2025-10-01 (one period behind the 25Q4 vintage's own November
    reference month), but the actual latest known value is 2025-08-01
    -- this must be flagged as a detected source gap, not silently
    presented as an ordinary, on-time reading.
    """
    variable, vidx, snap = ruc_pieces
    row_25q4 = vidx[vidx["vintage_label"] == "25Q4"].iloc[0]
    auctions = _auctions([{"cusip": "X", "cutoff": row_25q4["publication_safe_available_date"]}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert result.loc[0, "RUC_source_gap_detected"]
    assert result.loc[0, "RUC_expected_latest_observation_date"] == pd.Timestamp("2025-10-01")
    assert result.loc[0, "RUC_observation_date"] < result.loc[0, "RUC_expected_latest_observation_date"]


def test_ordinary_vintage_never_flags_a_source_gap(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    row_20q2 = vidx[vidx["vintage_label"] == "20Q2"].iloc[0]
    auctions = _auctions([{"cusip": "X", "cutoff": row_20q2["publication_safe_available_date"]}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert not result.loc[0, "RUC_source_gap_detected"]
    assert result.loc[0, "RUC_observation_date"] == result.loc[0, "RUC_expected_latest_observation_date"]


def test_availability_precision_provenance_survives_the_join(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    row_20q2 = vidx[vidx["vintage_label"] == "20Q2"].iloc[0]
    auctions = _auctions([{"cusip": "X", "cutoff": row_20q2["publication_safe_available_date"]}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert result.loc[0, "RUC_availability_precision"] == row_20q2["availability_precision"]


def test_no_gap_flag_or_provenance_when_unmatched(ruc_pieces):
    variable, vidx, snap = ruc_pieces
    auctions = _auctions([{"cusip": "X", "cutoff": pd.Timestamp("2000-01-01")}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert not result.loc[0, "RUC_join_matched"]
    assert not result.loc[0, "RUC_source_gap_detected"]
    assert pd.isna(result.loc[0, "RUC_expected_latest_observation_date"])


def test_future_vintages_never_change_an_earlier_auctions_matched_features(ruc_pieces):
    """Future-row invariance: adding a later vintage to the vintage
    index must never change what an earlier-cutoff auction resolves
    to (merge_asof direction='backward' structurally guarantees this,
    but this test proves it end-to-end through as_of_join_variable).
    """
    variable, vidx, snap = ruc_pieces
    row_20q2 = vidx[vidx["vintage_label"] == "20Q2"].iloc[0]
    auctions = _auctions([{"cusip": "X", "cutoff": row_20q2["publication_safe_available_date"]}])

    before = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")

    future_vintage = pd.DataFrame(
        {
            "vintage_label": ["99Q9"],
            "vintage_year": [2099],
            "vintage_period": [1],
            "nominal_publication_date": [pd.Timestamp("2099-02-15")],
            "publication_safe_available_date": [pd.Timestamp("2099-02-16")],
            "availability_precision": ["x"],
        }
    )
    future_snap = pd.DataFrame(
        {
            "vintage_label": ["99Q9"],
            "current_observation_date": [pd.Timestamp("2099-01-01")],
            "current_value": [123.4],
            "yoy_observation_date": [pd.Timestamp("2098-01-01")],
            "yoy_value": [100.0],
        }
    )
    vidx_with_future = pd.concat([vidx, future_vintage], ignore_index=True)
    snap_with_future = pd.concat([snap, future_snap], ignore_index=True)
    after = as_of_join_variable(auctions, vidx_with_future, snap_with_future, variable=variable, cutoff_col="cutoff")

    pd.testing.assert_frame_equal(before, after)


def test_matched_vintage_with_no_observation_at_all_gets_explicit_reason():
    variable = VARIABLE_BY_MNEMONIC["RUC"]
    vidx = pd.DataFrame(
        {
            "vintage_label": ["99X1"],
            "vintage_year": [1999],
            "vintage_period": [1],
            "nominal_publication_date": [pd.Timestamp("1999-02-15")],
            "publication_safe_available_date": [pd.Timestamp("1999-02-16")],
            "availability_precision": ["x"],
        }
    )
    snap = pd.DataFrame(
        {
            "vintage_label": ["99X1"],
            "current_observation_date": [pd.NaT],
            "current_value": [float("nan")],
            "yoy_observation_date": [pd.NaT],
            "yoy_value": [float("nan")],
        }
    )
    auctions = pd.DataFrame([{"cusip": "X", "cutoff": pd.Timestamp("2000-01-01")}])
    result = as_of_join_variable(auctions, vidx, snap, variable=variable, cutoff_col="cutoff")
    assert result.loc[0, "RUC_join_matched"]
    assert result.loc[0, "RUC_missing_reason"] == NO_NONNULL_OBSERVATION_REASON


def test_in_sample_revision_worked_example_uses_only_the_safely_available_vintage():
    """Phase 4 acceptance review, issue 5, updated for Phase 9
    acceptance-review remediation: a structural, synthetic-value
    in-sample (2010-present) revision example, tied to a real auction.
    The two `value`s below are fabricated (not the real RUC readings
    this scenario was originally modeled on -- see the live_network
    RTDSM test for the real, historically documented March-2010 RUC
    revision this structurally mirrors). A real 10-Year auction (CUSIP
    912828ND8, 2010-06-09, announcement cutoff 2010-06-03) has a cutoff
    after vintage 10Q2's safe-available date (2010-05-17) but nearly 3
    years before vintage 13Q1's -- so this auction's point-in-time
    feature must reflect the *first-known* value for that period, not
    the later revision, even though this test looks the historical
    observation up directly (not merely the "current level" feature,
    which is a different, later observation period).
    """
    vintage_year = pd.Series([2010, 2013], name="vintage_year")
    vintage_period = pd.Series([2, 1], name="vintage_period")
    variable = VARIABLE_BY_MNEMONIC["RUC"]

    long_df = pd.DataFrame(
        {
            "mnemonic": "RUC",
            "observation_period": ["2010:03"] * 2,
            "observation_date": [pd.Timestamp("2010-03-01")] * 2,
            "vintage_label": ["10Q2", "13Q1"],
            "vintage_year": vintage_year,
            "vintage_period": vintage_period,
            "value": [8.0, 8.5],  # fabricated: first-reported vs. later-revised (synthetic)
        }
    )
    long_df = attach_release_dates(long_df, variable)
    vidx = vintage_index(long_df)

    real_auction_cutoff = pd.Timestamp("2010-06-03")  # CUSIP 912828ND8, 10-Year, auction_date 2010-06-09
    assert vidx.loc[vidx["vintage_label"] == "10Q2", "publication_safe_available_date"].iloc[0] <= real_auction_cutoff
    assert vidx.loc[vidx["vintage_label"] == "13Q1", "publication_safe_available_date"].iloc[0] > real_auction_cutoff

    # The join must select 10Q2, not 13Q1.
    matched = pd.merge_asof(
        pd.DataFrame({"cutoff": [real_auction_cutoff]}).sort_values("cutoff"),
        vidx.sort_values("publication_safe_available_date"),
        left_on="cutoff",
        right_on="publication_safe_available_date",
        direction="backward",
    )
    matched_vintage = matched.loc[0, "vintage_label"]
    assert matched_vintage == "10Q2"

    historical_value_as_known = long_df.loc[
        (long_df["vintage_label"] == matched_vintage) & (long_df["observation_date"] == pd.Timestamp("2010-03-01")),
        "value",
    ].iloc[0]
    assert historical_value_as_known == 8.0  # the first-known (synthetic) value, never the 8.5 later revision
