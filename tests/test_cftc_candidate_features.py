from __future__ import annotations

import pandas as pd

from treasury_auction_stress.data.cftc_normalize import normalize_cftc_positioning
from treasury_auction_stress.features.cftc_candidate_features import (
    add_net_positions,
    add_share_of_open_interest,
    add_week_over_week_changes,
    build_positioning_feature_table,
)


def _normalized(cftc_tff_sample_payload):
    df, _ = normalize_cftc_positioning(cftc_tff_sample_payload, retrieval_timestamp_utc="2026-09-11T00:00:00Z")
    return df


def test_net_position_is_long_minus_short_excludes_spread(cftc_tff_sample_payload):
    df = _normalized(cftc_tff_sample_payload)
    out = add_net_positions(df)
    row = out[(out["contract_code"] == "043602") & (out["report_date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    expected = row["dealer_positions_long_all"] - row["dealer_positions_short_all"]
    assert row["dealer_net_contracts"] == expected
    assert "dealer_positions_spread_all" not in {"dealer_net_contracts"}


def test_share_of_open_interest_is_row_local_ratio(cftc_tff_sample_payload):
    df = _normalized(cftc_tff_sample_payload)
    out = add_share_of_open_interest(df)
    row = out[(out["contract_code"] == "043602") & (out["report_date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    expected = row["dealer_positions_long_all"] / row["open_interest_all"]
    assert row["dealer_long_pct_oi"] == expected


def test_week_over_week_change_is_within_contract_only(cftc_tff_sample_payload):
    df = _normalized(cftc_tff_sample_payload)
    out = add_net_positions(df)
    out = add_week_over_week_changes(out)
    first_042601 = out[out["contract_code"] == "042601"].sort_values("report_date").iloc[0]
    assert pd.isna(first_042601["dealer_net_contracts_chg_1w"])

    ust10y = out[out["contract_code"] == "043602"].sort_values("report_date").reset_index(drop=True)
    chg = ust10y.loc[1, "dealer_net_contracts_chg_1w"]
    expected = ust10y.loc[1, "dealer_net_contracts"] - ust10y.loc[0, "dealer_net_contracts"]
    assert chg == expected


def test_units_stay_in_contracts_no_dollar_relabeling(cftc_tff_sample_payload):
    df = _normalized(cftc_tff_sample_payload)
    out = build_positioning_feature_table(df)
    forbidden_substrings = ("dollar", "notional", "duration", "dv01")
    for col in out.columns:
        lowered = col.lower()
        assert not any(term in lowered for term in forbidden_substrings)


def test_no_full_sample_zscore_or_percentile_columns(cftc_tff_sample_payload):
    df = _normalized(cftc_tff_sample_payload)
    out = build_positioning_feature_table(df)
    forbidden_substrings = ("zscore", "z_score", "percentile", "pctile", "rank")
    for col in out.columns:
        lowered = col.lower()
        assert not any(term in lowered for term in forbidden_substrings)


def test_build_positioning_feature_table_preserves_row_count(cftc_tff_sample_payload):
    df = _normalized(cftc_tff_sample_payload)
    out = build_positioning_feature_table(df)
    assert len(out) == len(df)
