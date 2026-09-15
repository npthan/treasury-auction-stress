from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)
from treasury_auction_stress.features.cftc_join import (
    NO_CONTRACT_FOR_TENOR_REASON,
    NO_COVERAGE_REASON,
    add_tenor_matched_positioning_features,
    as_of_join,
)


def _positioning_wide(n_weeks: int = 6, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_weeks, freq="7D")  # weekly Tuesdays
    return pd.DataFrame(
        {
            "report_date": dates,
            "publication_safe_available_date": dates + pd.Timedelta(days=4),
            "nominal_publication_date": dates + pd.Timedelta(days=3),
            "availability_precision": ["documented_standard_rule_inferred_for_this_week"] * n_weeks,
            "043602__dealer_net_contracts": [float(-100_000 + 1000 * i) for i in range(n_weeks)],
            "043602__open_interest_all": [float(4_000_000 + 1000 * i) for i in range(n_weeks)],
            "043607__dealer_net_contracts": [float(-5_000 + 100 * i) for i in range(n_weeks)],
        }
    )


def _auctions(rows):
    defaults = {"cusip": "X", "tenor": "10-Year"}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_same_day_report_not_used_for_a_same_day_cutoff():
    positioning = _positioning_wide()
    same_day_cutoff = positioning.loc[3, "report_date"]
    auctions = _auctions([{"cutoff": same_day_cutoff}])
    result = as_of_join(auctions, positioning, cutoff_col="cutoff")
    assert result.loc[0, "report_date"] < same_day_cutoff


def test_cutoff_on_safe_available_date_selects_that_weeks_report():
    positioning = _positioning_wide()
    safe_date = positioning.loc[3, "publication_safe_available_date"]
    auctions = _auctions([{"cutoff": safe_date}])
    result = as_of_join(auctions, positioning, cutoff_col="cutoff")
    assert result.loc[0, "report_date"] == positioning.loc[3, "report_date"]


def test_never_selects_a_later_report():
    positioning = _positioning_wide()
    for cutoff in pd.date_range("2024-01-01", "2024-03-01", freq="5D"):
        auctions = _auctions([{"cutoff": cutoff}])
        result = as_of_join(auctions, positioning, cutoff_col="cutoff")
        if result.loc[0, "cftc_join_matched"]:
            assert result.loc[0, "publication_safe_available_date"] <= cutoff


def test_tied_safe_available_dates_break_deterministically_by_report_date():
    """Phase 5 acceptance review: a real tie exists in this project's
    own live CFTC data -- the 2025-12-23 and 2026-01-06 reports both
    become safely available on 2026-01-12 (a holiday-adjacent
    compression of two release dates onto one day). Before the fix,
    `merge_asof`'s tie-breaking depended on the two tied rows' relative
    order in the INPUT (pre-sort) dataframe, which is not something
    this project's own construction guarantees -- verified directly by
    building the tied fixture in both possible input orders and
    checking the result is identical (the more recent report_date, per
    the documented deterministic tie-break rule) either way.
    """
    tied_date = pd.Timestamp("2026-01-12")
    rows = [
        {
            "report_date": pd.Timestamp("2025-12-23"),
            "publication_safe_available_date": tied_date,
            "nominal_publication_date": tied_date - pd.Timedelta(days=1),
            "availability_precision": "test",
            "043602__dealer_net_contracts": 111.0,
            "043602__open_interest_all": 1000.0,
        },
        {
            "report_date": pd.Timestamp("2026-01-06"),
            "publication_safe_available_date": tied_date,
            "nominal_publication_date": tied_date - pd.Timedelta(days=1),
            "availability_precision": "test",
            "043602__dealer_net_contracts": 222.0,
            "043602__open_interest_all": 2000.0,
        },
    ]
    auctions = _auctions([{"cutoff": tied_date + pd.Timedelta(days=3)}])

    forward_order = pd.DataFrame(rows)
    reversed_order = pd.DataFrame(list(reversed(rows)))

    result_forward = as_of_join(auctions, forward_order, cutoff_col="cutoff")
    result_reversed = as_of_join(auctions, reversed_order, cutoff_col="cutoff")

    # Both input orders must select the SAME (more recent) report.
    assert result_forward.loc[0, "report_date"] == pd.Timestamp("2026-01-06")
    assert result_reversed.loc[0, "report_date"] == pd.Timestamp("2026-01-06")
    assert result_forward.loc[0, "043602__dealer_net_contracts"] == 222.0
    assert result_reversed.loc[0, "043602__dealer_net_contracts"] == 222.0


def test_no_auction_dropped_for_lacking_coverage():
    positioning = _positioning_wide(start="2026-06-01")
    auctions = _auctions([{"cutoff": pd.Timestamp("2020-01-01")}])
    result = as_of_join(auctions, positioning, cutoff_col="cutoff")
    assert len(result) == 1
    assert not result.loc[0, "cftc_join_matched"]


def test_nat_cutoff_does_not_crash_and_is_unmatched():
    positioning = _positioning_wide()
    auctions = _auctions([{"cutoff": pd.NaT}])
    result = as_of_join(auctions, positioning, cutoff_col="cutoff")
    assert len(result) == 1
    assert not result.loc[0, "cftc_join_matched"]


def test_reserved_column_name_raises():
    positioning = _positioning_wide()
    auctions = _auctions([{"cutoff": pd.Timestamp("2024-01-10")}])
    auctions["__as_of_join_row_order__"] = 0
    with pytest.raises(ValueError):
        as_of_join(auctions, positioning, cutoff_col="cutoff")


def test_announcement_and_pre_auction_cutoffs_can_differ():
    positioning = _positioning_wide(n_weeks=10)
    auctions = pd.DataFrame(
        {
            "cusip": ["A"],
            "tenor": ["10-Year"],
            "announcemt_date": [pd.Timestamp("2024-01-07")],
            "auction_date": [pd.Timestamp("2024-02-16")],
        }
    )
    auctions = add_cutoff_dates(auctions)
    ann = as_of_join(auctions, positioning, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    pre = as_of_join(auctions, positioning, cutoff_col=PRE_AUCTION_CUTOFF_COL)
    assert ann.loc[0, "report_date"] < pre.loc[0, "report_date"]


def test_matched_features_use_primary_direct_contract_not_secondary():
    positioning = _positioning_wide()
    auctions = _auctions([{"tenor": "10-Year", "cutoff": positioning.loc[3, "publication_safe_available_date"]}])
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    out = add_tenor_matched_positioning_features(joined)
    assert out.loc[0, "matched_contract_code"] == "043602"
    assert out.loc[0, "matched_dealer_net_contracts"] == positioning.loc[3, "043602__dealer_net_contracts"]
    assert out.loc[0, "secondary_context_contract_codes"] == "043607"


def test_tenor_with_no_contract_gets_explicit_reason_not_forced_match():
    positioning = _positioning_wide()
    auctions = _auctions([{"tenor": "3-Year", "cutoff": positioning.loc[3, "publication_safe_available_date"]}])
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    out = add_tenor_matched_positioning_features(joined)
    assert pd.isna(out.loc[0, "matched_contract_code"])
    assert out.loc[0, "positioning_missing_reason"] == NO_CONTRACT_FOR_TENOR_REASON


def test_20year_gets_secondary_context_but_no_primary_match():
    positioning = _positioning_wide()
    positioning["020601__dealer_net_contracts"] = 1.0
    positioning["020604__dealer_net_contracts"] = 2.0
    auctions = _auctions([{"tenor": "20-Year", "cutoff": positioning.loc[3, "publication_safe_available_date"]}])
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    out = add_tenor_matched_positioning_features(joined)
    assert pd.isna(out.loc[0, "matched_contract_code"])
    assert out.loc[0, "secondary_context_contract_codes"] == "020601,020604"
    assert out.loc[0, "positioning_missing_reason"] == NO_CONTRACT_FOR_TENOR_REASON


def test_no_report_before_cutoff_gets_no_coverage_reason():
    positioning = _positioning_wide(start="2026-06-01")
    auctions = _auctions([{"tenor": "10-Year", "cutoff": pd.Timestamp("2020-01-01")}])
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    out = add_tenor_matched_positioning_features(joined)
    assert out.loc[0, "positioning_missing_reason"] == NO_COVERAGE_REASON


# -- Phase 4 acceptance review, issue 9: source coverage vs. direct-contract mapping --


def test_source_coverage_and_direct_mapping_fields_are_independent():
    positioning = _positioning_wide()
    cutoff = positioning.loc[3, "publication_safe_available_date"]
    auctions = _auctions(
        [
            {"tenor": "10-Year", "cutoff": cutoff},  # has a direct contract
            {"tenor": "3-Year", "cutoff": cutoff},  # no direct contract, but report is available
        ]
    )
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    out = add_tenor_matched_positioning_features(joined)

    # Both rows see the same report as available -- tenor never affects source coverage.
    assert out.loc[0, "cftc_report_available"]
    assert out.loc[1, "cftc_report_available"]
    assert out.loc[0, "cftc_source_join_status"] == "matched"
    assert out.loc[1, "cftc_source_join_status"] == "matched"

    # Direct-contract mapping differs by tenor, independent of source coverage.
    assert out.loc[0, "direct_contract_mapping_status"] == "direct_contract_available"
    assert out.loc[1, "direct_contract_mapping_status"] == "no_direct_contract_for_tenor"
    assert out.loc[0, "direct_contract_code"] == "043602"
    assert pd.isna(out.loc[1, "direct_contract_code"])


def test_no_direct_contract_is_never_reported_as_a_source_coverage_failure():
    positioning = _positioning_wide()
    cutoff = positioning.loc[3, "publication_safe_available_date"]
    auctions = _auctions([{"tenor": "7-Year", "cutoff": cutoff}])
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    out = add_tenor_matched_positioning_features(joined)
    assert out.loc[0, "cftc_report_available"]  # report IS available
    assert out.loc[0, "direct_contract_mapping_status"] == "no_direct_contract_for_tenor"  # merely no direct contract


def test_direct_contract_mapped_but_no_data_is_distinguished_from_no_report():
    """A tenor's primary contract can be defined but have no data for a
    specific matched report (e.g. a contract that started later than
    the matched report date). That must be reported as
    'mapped_but_data_unavailable', distinct from 'no report at all'.
    """
    positioning = _positioning_wide()
    cutoff = positioning.loc[3, "publication_safe_available_date"]
    positioning = positioning.copy()
    positioning["043602__open_interest_all"] = float("nan")  # simulate no data for this contract that week
    auctions = _auctions([{"tenor": "10-Year", "cutoff": cutoff}])
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    out = add_tenor_matched_positioning_features(joined)
    assert out.loc[0, "cftc_report_available"]
    assert out.loc[0, "direct_contract_mapping_status"] == "direct_contract_mapped_but_data_unavailable"


# -- Phase 4 acceptance review, issue 8: pre-2010 lookback for early-2010 auctions --


def test_pre_2010_source_history_supports_a_january_2010_auction_cutoff():
    """The CFTC source-history window may (and, per the acceptance
    review, should) begin before the auction modeling sample's own
    2010-01-01 start -- an auction with an early-2010 cutoff must be
    able to match a 2009 report rather than being artificially
    unmatched at the sample boundary.
    """
    positioning = _positioning_wide(n_weeks=10, start="2009-11-03")  # starts well before 2010
    early_2010_cutoff = pd.Timestamp("2010-01-05")
    auctions = _auctions([{"tenor": "10-Year", "cutoff": early_2010_cutoff}])
    joined = as_of_join(auctions, positioning, cutoff_col="cutoff")
    assert joined.loc[0, "cftc_join_matched"]
    assert joined.loc[0, "report_date"] < pd.Timestamp("2010-01-01")
