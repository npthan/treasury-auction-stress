from __future__ import annotations

import pandas as pd

from treasury_auction_stress.data.cftc_normalize import normalize_cftc_positioning
from treasury_auction_stress.data.cftc_release_calendar import (
    PRECISION_VERIFIED_EXACT_DATE,
    PRECISION_VERIFIED_STANDARD,
)


def test_numeric_fields_parsed_from_strings(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    assert df["open_interest_all"].dtype == pd.Float64Dtype()
    row = df[(df["contract_code"] == "043602") & (df["report_date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert row["open_interest_all"] > 0
    assert isinstance(row["open_interest_all_raw"], str)


def test_no_unrecognized_contract_codes(cftc_tff_sample_payload):
    _, anomalies = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    assert anomalies["unrecognized_contract_codes"] == []
    assert anomalies["duplicate_report_rows"] == 0


def test_units_are_contracts_not_relabeled(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    assert set(df["units"]) == {"contracts"}


def test_multi_contract_rows_all_present(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    assert set(df["contract_code"]) == {"043602", "042601"}
    assert len(df) == 6


def test_ordinary_week_gets_standard_precision(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    row = df[(df["contract_code"] == "043602") & (df["report_date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert row["availability_precision"] == PRECISION_VERIFIED_STANDARD
    assert row["nominal_publication_date"] == pd.Timestamp("2024-01-05")


def test_2023_ion_disruption_window_gets_exact_per_report_dates(cftc_tff_sample_payload):
    """Phase 4 acceptance review, issue 10: each affected report gets
    its own actual (per-report) publication date, not one bulk
    end-of-window date -- verified against CFTC's own Historical
    Special Announcements page.
    """
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    expected = {"2023-01-31": "2023-02-27", "2023-02-07": "2023-03-06"}
    for obs_date, safe_date in expected.items():
        row = df[(df["contract_code"] == "043602") & (df["report_date"] == pd.Timestamp(obs_date))].iloc[0]
        assert row["publication_safe_available_date"] == pd.Timestamp(safe_date)
        assert row["availability_precision"] == PRECISION_VERIFIED_EXACT_DATE
        assert not pd.isna(row["nominal_publication_date"])  # a real per-report date, not the old NaT placeholder


def test_2025_shutdown_window_gets_exact_per_report_dates(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    expected = {"2025-09-30": "2025-11-20", "2025-10-07": "2025-11-24"}
    for obs_date, safe_date in expected.items():
        row = df[(df["contract_code"] == "043602") & (df["report_date"] == pd.Timestamp(obs_date))].iloc[0]
        assert row["availability_precision"] == PRECISION_VERIFIED_EXACT_DATE
        assert row["publication_safe_available_date"] == pd.Timestamp(safe_date)


def test_no_disrupted_report_ever_gets_a_weekend_safe_available_date(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    assert (df["publication_safe_available_date"].dt.weekday < 5).all()


def test_spreading_kept_as_its_own_field_not_merged_into_directional(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    for cat in ("dealer_positions", "asset_mgr_positions", "lev_money_positions", "other_rept_positions"):
        cols = {c for c in df.columns if c.startswith(cat)}
        assert any(c.endswith(("spread_all", "spread")) for c in cols)


def test_empty_payload_returns_empty_frame():
    df, anomalies = normalize_cftc_positioning({"contracts": []}, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    assert df.empty
    assert anomalies == {"unrecognized_contract_codes": [], "duplicate_report_rows": 0}


def test_rows_sorted_by_contract_then_date(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    for _, group in df.groupby("contract_code"):
        assert list(group["report_date"]) == sorted(group["report_date"])
