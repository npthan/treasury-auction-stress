from __future__ import annotations

import pandas as pd

from treasury_auction_stress.features.phase4d_source_joins import (
    AUCTION_ID_COLS,
    CUTOFF_COLS,
    FORBIDDEN_RESULT_COLUMNS,
    audit_join_table,
    build_cftc_join_table,
    build_rates_join_table,
    build_rtdsm_join_table,
)


def _auctions():
    df = pd.DataFrame(
        {
            "cusip": ["A", "B"],
            "tenor": ["2-Year", "10-Year"],
            "announcemt_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "auction_date": pd.to_datetime(["2024-01-10", "2024-01-17"]),
            "is_reopening": [False, True],
            # A forbidden result column deliberately present on the input --
            # the join-table builders must not carry it through.
            "high_yield": [4.5, 4.2],
        }
    )
    from treasury_auction_stress.features.auction_cutoffs import add_cutoff_dates

    return add_cutoff_dates(df)


def test_rates_join_table_excludes_forbidden_columns_and_keeps_ids():
    auctions = _auctions()
    rates_wide = pd.DataFrame(
        {
            "rate_date": pd.bdate_range("2024-01-01", periods=20),
            "publication_date": pd.bdate_range("2024-01-01", periods=20),
            "publication_safe_available_date": pd.bdate_range("2024-01-01", periods=20) + pd.Timedelta(days=1),
            "2 Yr": [4.3] * 20,
            "10 Yr": [4.1] * 20,
        }
    )
    out = build_rates_join_table(auctions, rates_wide)
    assert set(AUCTION_ID_COLS) | set(CUTOFF_COLS) <= set(out.columns)
    assert not set(out.columns) & set(FORBIDDEN_RESULT_COLUMNS)
    assert "high_yield" not in out.columns
    assert len(out) == len(auctions)


def test_cftc_join_table_excludes_forbidden_columns():
    auctions = _auctions()
    positioning_wide = pd.DataFrame(
        {
            "report_date": pd.date_range("2024-01-02", periods=6, freq="7D"),
            "publication_safe_available_date": pd.date_range("2024-01-02", periods=6, freq="7D") + pd.Timedelta(days=4),
            "042601__dealer_net_contracts": [1.0] * 6,
            "043602__dealer_net_contracts": [2.0] * 6,
        }
    )
    out = build_cftc_join_table(auctions, positioning_wide)
    assert not set(out.columns) & set(FORBIDDEN_RESULT_COLUMNS)
    assert "high_yield" not in out.columns


def test_rtdsm_join_table_excludes_forbidden_columns():
    from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY

    auctions = _auctions()
    vintage_indices = {}
    snapshots = {}
    for variable in VARIABLE_REGISTRY:
        vintage_indices[variable.mnemonic] = pd.DataFrame(
            {
                "vintage_label": ["24Q1"],
                "vintage_year": [2024],
                "vintage_period": [1],
                "nominal_publication_date": [pd.Timestamp("2024-02-15")],
                "publication_safe_available_date": [pd.Timestamp("2024-02-16")],
                "availability_precision": ["x"],
            }
        )
        snapshots[variable.mnemonic] = pd.DataFrame(
            {
                "vintage_label": ["24Q1"],
                "current_observation_date": [pd.Timestamp("2024-01-01")],
                "current_value": [1.0],
                "yoy_observation_date": [pd.Timestamp("2023-01-01")],
                "yoy_value": [0.9],
            }
        )
    out = build_rtdsm_join_table(auctions, vintage_indices, snapshots)
    assert not set(out.columns) & set(FORBIDDEN_RESULT_COLUMNS)
    assert "high_yield" not in out.columns
    assert len(out) == len(auctions)


def test_audit_join_table_flags_row_count_mismatch():
    df = pd.DataFrame({"cusip": ["A"], "auction_date": pd.to_datetime(["2024-01-10"])})
    result = audit_join_table("dummy", df, expected_rows=2)
    assert result["passed"] == False
    assert result["row_count_matches"] == False


def test_audit_join_table_flags_duplicate_keys():
    df = pd.DataFrame(
        {"cusip": ["A", "A"], "auction_date": pd.to_datetime(["2024-01-10", "2024-01-10"])}
    )
    result = audit_join_table("dummy", df, expected_rows=2)
    assert result["duplicate_cusip_auction_date_rows"] == 1
    assert result["passed"] == False


def test_audit_join_table_flags_leaked_forbidden_column():
    df = pd.DataFrame({"cusip": ["A"], "auction_date": pd.to_datetime(["2024-01-10"]), "high_yield": [4.5]})
    result = audit_join_table("dummy", df, expected_rows=1)
    assert result["forbidden_columns_present"] == ["high_yield"]
    assert result["passed"] == False


def test_audit_join_table_passes_clean_table():
    df = pd.DataFrame({"cusip": ["A", "B"], "auction_date": pd.to_datetime(["2024-01-10", "2024-01-17"])})
    result = audit_join_table("dummy", df, expected_rows=2)
    assert result["passed"] == True
