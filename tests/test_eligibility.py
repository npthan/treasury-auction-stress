import pandas as pd

from treasury_auction_stress.features.eligibility import (
    RESTRICTED_PRIMARY_DEALER_ONLY,
    SMALL_OFFERING_THRESHOLD,
    SPECIAL_AUCTION_TYPE_COL,
    classify_special_auctions,
    describe_eligibility,
    select_analysis_sample,
    select_modeling_sample,
)


def _nominal_df():
    return pd.DataFrame(
        {
            "auction_date": pd.to_datetime(
                [
                    "2008-06-01",  # before start date -- excluded
                    "2019-06-21",  # settled, restricted primary-dealer-only reopening
                    "2024-01-23",  # settled, ordinary offering
                    "2026-09-10",  # pending
                ]
            ),
            "cusip": ["OLD00000", "9128286T2", "91282CJV4", "912810XX0"],
            "tenor": ["10-Year", "10-Year", "2-Year", "30-Year"],
            "is_reopening": [False, True, False, True],
            "offering_amt": pd.array([10e9, 25_000_000, 60e9, 22e9], dtype="Float64"),
            "results_available": [True, True, True, False],
            "noncomp_tenders_accepted": ["Yes", "No", "Yes", "Yes"],
        }
    )


def test_select_analysis_sample_excludes_before_start_date():
    settled, pending = select_analysis_sample(_nominal_df(), start_date="2010-01-01")
    all_rows = pd.concat([settled, pending])
    assert (all_rows["auction_date"] >= pd.Timestamp("2010-01-01")).all()
    assert len(all_rows) == 3  # the 2008 row is dropped entirely


def test_select_analysis_sample_splits_settled_and_pending():
    settled, pending = select_analysis_sample(_nominal_df())
    assert len(settled) == 2
    assert len(pending) == 1
    assert pending.iloc[0]["auction_date"] == pd.Timestamp("2026-09-10")
    assert set(settled["results_available"]) == {True}
    assert set(pending["results_available"]) == {False}


def test_small_offering_flag():
    settled, _pending = select_analysis_sample(_nominal_df())
    small = settled.loc[settled["offering_amt"] == 25_000_000]
    assert small.iloc[0]["is_unusually_small_offering"]
    ordinary = settled.loc[settled["offering_amt"] == 60e9]
    assert not ordinary.iloc[0]["is_unusually_small_offering"]
    assert 25_000_000 < SMALL_OFFERING_THRESHOLD


def test_classify_special_auctions_uses_noncomp_tenders_accepted_field():
    classified = classify_special_auctions(_nominal_df())
    restricted = classified.loc[classified["cusip"] == "9128286T2"]
    ordinary = classified.loc[classified["cusip"] == "91282CJV4"]
    assert restricted.iloc[0][SPECIAL_AUCTION_TYPE_COL] == RESTRICTED_PRIMARY_DEALER_ONLY
    assert pd.isna(ordinary.iloc[0][SPECIAL_AUCTION_TYPE_COL])


def test_special_auctions_preserved_in_settled_but_excluded_from_modeling_sample():
    settled, _pending = select_analysis_sample(_nominal_df())
    modeling_sample = select_modeling_sample(settled)

    # Preserved: still present, and findable, in the settled audit frame.
    assert "9128286T2" in set(settled["cusip"])
    # Excluded: cannot silently enter the main modeling sample.
    assert "9128286T2" not in set(modeling_sample["cusip"])
    # The ordinary auction is unaffected either way.
    assert "91282CJV4" in set(settled["cusip"])
    assert "91282CJV4" in set(modeling_sample["cusip"])
    assert len(modeling_sample) == len(settled) - 1


def test_describe_eligibility_counts():
    summary = describe_eligibility(_nominal_df())
    assert summary["rows_before_start_date_excluded"] == 1
    assert summary["settled_analysis_sample_rows"] == 2
    assert summary["modeling_sample_rows"] == 1
    assert summary["special_auction_rows"] == 1
    assert summary["special_auction_type"] == RESTRICTED_PRIMARY_DEALER_ONLY
    assert summary["special_auction_records"][0]["cusip"] == "9128286T2"
    assert "primary dealers only" in summary["special_auction_exclusion_reason"]
    assert summary["pending_rows"] == 1
    assert summary["pending_auction_dates"] == ["2026-09-10"]
    assert len(summary["unusually_small_offering_rows"]) == 1
    assert summary["unusually_small_offering_rows"][0]["auction_date"] == "2019-06-21"
