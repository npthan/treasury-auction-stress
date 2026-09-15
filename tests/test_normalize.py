from __future__ import annotations

import re

import pandas as pd

from treasury_auction_stress.data.normalize import (
    check_schema_drift,
    find_duplicate_keys,
    nominal_coupon_subset,
    normalize_auctions,
    pages_to_records,
    to_raw_dataframe,
)


def _raw_df(sample_page_body):
    records = pages_to_records({"pages": [{"body": sample_page_body}]})
    return to_raw_dataframe(records)


def test_pages_to_records_flattens_multiple_pages(two_page_fixture):
    payload = {
        "pages": [
            {"body": two_page_fixture["page1"]},
            {"body": two_page_fixture["page2"]},
        ]
    }
    records = pages_to_records(payload)
    assert len(records) == len(two_page_fixture["page1"]["data"]) + len(
        two_page_fixture["page2"]["data"]
    )


def test_pages_to_records_handles_empty_response(empty_page_body):
    records = pages_to_records({"pages": [{"body": empty_page_body}]})
    assert records == []
    df = to_raw_dataframe(records)
    assert df.empty


def test_normalize_parses_dates_and_preserves_raw_strings(sample_page_body):
    df = _raw_df(sample_page_body)
    normalized, _ = normalize_auctions(df)
    assert pd.api.types.is_datetime64_any_dtype(normalized["auction_date"])
    # the raw string form must survive alongside the parsed value
    assert normalized["auction_date_raw"].iloc[0] == df["auction_date"].iloc[0]


def test_normalize_parses_numbers_without_coercing_missing_to_zero(sample_page_body):
    df = _raw_df(sample_page_body)
    normalized, _ = normalize_auctions(df)

    pending_row = normalized.loc[normalized["total_accepted_raw"] == "null"]
    assert len(pending_row) == 1
    assert pending_row["total_accepted"].isna().all()  # missing, not 0
    assert not (pending_row["total_accepted"] == 0).any()

    settled_rows = normalized.loc[normalized["total_accepted_raw"] != "null"]
    assert (settled_rows["total_accepted"] > 0).all()


def test_normalize_flags_no_invalid_dates_or_numbers_on_clean_fixture(sample_page_body):
    df = _raw_df(sample_page_body)
    _, anomalies = normalize_auctions(df)
    assert anomalies["invalid_dates"] == {}
    assert anomalies["invalid_numbers"] == {}


def test_normalize_detects_invalid_date_value(sample_page_body):
    df = _raw_df(sample_page_body)
    df.loc[df.index[0], "auction_date"] = "not-a-date"
    _, anomalies = normalize_auctions(df)
    assert anomalies["invalid_dates"].get("auction_date") == 1


def test_normalize_detects_invalid_numeric_value(sample_page_body):
    df = _raw_df(sample_page_body)
    df.loc[df.index[0], "offering_amt"] = "not-a-number"
    _, anomalies = normalize_auctions(df)
    assert anomalies["invalid_numbers"].get("offering_amt") == 1


def test_normalize_detects_unexpected_category_value(sample_page_body):
    df = _raw_df(sample_page_body)
    df.loc[df.index[0], "reopening"] = "Maybe"
    _, anomalies = normalize_auctions(df)
    assert anomalies["unexpected_categories"].get("reopening") == ["Maybe"]


def test_security_type_and_tenor_and_reopening_classification(sample_page_body):
    df = _raw_df(sample_page_body)
    normalized, _ = normalize_auctions(df)

    bond_note = normalized.loc[normalized["security_type"] == "Bill"]
    assert not bond_note["is_nominal_coupon"].any()

    nominal = nominal_coupon_subset(normalized)
    # the 2-year new issue, the 10-year reopening, and the pending 30-year reopening
    assert set(nominal["tenor"]) == {"2-Year", "10-Year", "30-Year"}
    assert bool(nominal.loc[nominal["tenor"] == "10-Year", "is_reopening"].iloc[0])
    assert not bool(nominal.loc[nominal["tenor"] == "2-Year", "is_reopening"].iloc[0])

    tips_rows = normalized.loc[normalized["inflation_index_security"] == "Yes"]
    assert not tips_rows["is_nominal_coupon"].any()

    frn_rows = normalized.loc[normalized["floating_rate"] == "Yes"]
    assert not frn_rows["is_nominal_coupon"].any()


def test_pending_auction_flagged_as_results_unavailable(sample_page_body):
    df = _raw_df(sample_page_body)
    normalized, _ = normalize_auctions(df)
    pending = normalized.loc[normalized["total_accepted"].isna()]
    assert len(pending) == 1
    assert not pending["results_available"].iloc[0]


def test_find_duplicate_keys_is_empty_on_clean_fixture(sample_page_body):
    df = _raw_df(sample_page_body)
    normalized, _ = normalize_auctions(df)
    dupes = find_duplicate_keys(normalized)
    assert dupes.empty


def test_find_duplicate_keys_detects_injected_duplicate(sample_page_body):
    df = _raw_df(sample_page_body)
    normalized, _ = normalize_auctions(df)
    duplicated = pd.concat([normalized, normalized.iloc[[0]]], ignore_index=True)
    dupes = find_duplicate_keys(duplicated)
    assert len(dupes) == 2  # the original row and its injected duplicate


def test_schema_drift_detects_unexpected_and_missing_columns(sample_page_body):
    df = _raw_df(sample_page_body)
    df["some_brand_new_field"] = "x"
    df = df.drop(columns=["cusip"])
    drift = check_schema_drift(df)
    assert "some_brand_new_field" in drift["new_columns"]
    assert "cusip" in drift["missing_columns"]


def test_schema_drift_is_empty_on_clean_fixture(sample_page_body):
    df = _raw_df(sample_page_body)
    drift = check_schema_drift(df)
    assert drift["new_columns"] == []
    assert drift["missing_columns"] == []


def test_announcement_date_is_date_only_never_an_invented_timestamp(sample_page_body):
    """Guardrail for docs/point_in_time_rules.md's date-level cutoff rule:
    the source data itself never carries a time-of-day for announcemt_date,
    and normalization must not fabricate one.
    """
    df = _raw_df(sample_page_body)
    date_only_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    raw_values = df["announcemt_date"].dropna()
    assert all(date_only_pattern.match(v) for v in raw_values)

    normalized, _ = normalize_auctions(df)
    parsed = normalized["announcemt_date"].dropna()
    assert (parsed.dt.normalize() == parsed).all()  # every value is midnight, i.e. date-only
