from __future__ import annotations

import pandas as pd

from treasury_auction_stress.features.dealer_join import (
    ANNOUNCEMENT_CUTOFF_COL,
    NO_COVERAGE_REASON,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
    as_of_join,
    coverage_summary,
    worked_example,
)


def _dealer_wide(n_weeks: int = 8, start: str = "2026-01-07", value_start: float = 100.0) -> pd.DataFrame:
    """Synthetic wide dealer table: Wednesday observations, one week
    apart, publication Thursday, safe-available Friday -- matching the
    real, verified NY Fed cadence used elsewhere in this project.
    """
    obs = pd.date_range(start, periods=n_weeks, freq="7D")
    pub = obs + pd.Timedelta(days=1)
    safe = pub + pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "observation_date": obs,
            "publication_date": pub,
            "publication_safe_available_date": safe,
            "dealer_net_position_bills": [value_start + 10 * i for i in range(n_weeks)],
        }
    )


def _auctions(rows: list[dict]) -> pd.DataFrame:
    defaults = {"cusip": "X", "tenor": "2-Year", "offering_amt": 50_000_000_000.0}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_leaky_join_on_observation_date_would_have_selected_unpublished_value():
    """Deliberately leaky example: a join keyed on `observation_date`
    directly ("the report exists as of its own observation date") would
    wrongly find a value on the very day it was observed, days before
    it was ever actually published. The real `as_of_join` (keyed on
    `publication_safe_available_date`) must reject this.
    """
    dealer = _dealer_wide()
    cutoff = dealer.loc[0, "observation_date"]  # 2026-01-07, the week's own obs date

    # The leaky approach: an equality/nearest match on observation_date.
    leaky_match = dealer.loc[dealer["observation_date"] == cutoff, "dealer_net_position_bills"]
    assert not leaky_match.empty  # confirms the leak really would have "found" a value

    auctions = _auctions([{"leaky_cutoff": cutoff}]).rename(columns={"leaky_cutoff": "cutoff"})
    result = as_of_join(auctions, dealer, cutoff_col="cutoff")

    assert not bool(result.loc[0, "dealer_join_matched"])
    assert pd.isna(result.loc[0, "dealer_net_position_bills"])
    assert result.loc[0, "dealer_net_position_bills_missing_reason"] == NO_COVERAGE_REASON


def test_never_selects_an_observation_published_after_cutoff():
    dealer = _dealer_wide()
    for cutoff in pd.date_range("2026-01-01", "2026-03-01", freq="3D"):
        auctions = _auctions([{"cutoff": cutoff}])
        result = as_of_join(auctions, dealer, cutoff_col="cutoff")
        if result.loc[0, "dealer_join_matched"]:
            assert result.loc[0, "publication_safe_available_date"] <= cutoff


def test_ambiguous_same_day_publication_falls_back_to_previous_release():
    """A cutoff that lands exactly on a week's *publication_date* (not
    its safe-available date) is the documented ambiguous same-day
    case -- the join must fall back to the previous week's release,
    never the ambiguous same-day one.
    """
    dealer = _dealer_wide()
    week1_pub_date = dealer.loc[1, "publication_date"]  # exactly the raw publication date
    auctions = _auctions([{"cutoff": week1_pub_date}])
    result = as_of_join(auctions, dealer, cutoff_col="cutoff")
    assert result.loc[0, "observation_date"] == dealer.loc[0, "observation_date"]  # previous week, not week1
    assert result.loc[0, "dealer_net_position_bills"] == dealer.loc[0, "dealer_net_position_bills"]


def test_cutoff_on_safe_available_date_selects_that_weeks_release():
    dealer = _dealer_wide()
    week1_safe_date = dealer.loc[1, "publication_safe_available_date"]
    auctions = _auctions([{"cutoff": week1_safe_date}])
    result = as_of_join(auctions, dealer, cutoff_col="cutoff")
    assert result.loc[0, "observation_date"] == dealer.loc[1, "observation_date"]


def test_missing_week_does_not_cause_a_future_backfill():
    """Dropping a week from the dealer table (a genuine reporting gap)
    must not make a cutoff shortly after it reach forward into a later
    week that happens to still be available -- it must fall back to
    the last real release *before* the gap, i.e. the join always moves
    strictly backward.
    """
    dealer = _dealer_wide(n_weeks=8)
    with_gap = dealer.drop(index=3).reset_index(drop=True)  # remove week index 3 entirely
    cutoff = dealer.loc[3, "publication_safe_available_date"]  # would have matched the missing week exactly
    auctions = _auctions([{"cutoff": cutoff}])
    result = as_of_join(auctions, with_gap, cutoff_col="cutoff")
    assert result.loc[0, "observation_date"] == dealer.loc[2, "observation_date"]  # falls back, not forward


def test_no_auction_is_dropped_for_lacking_coverage():
    dealer = _dealer_wide(start="2026-06-01")  # far in the future relative to the auction below
    auctions = _auctions([{"cutoff": pd.Timestamp("2020-01-01")}])
    result = as_of_join(auctions, dealer, cutoff_col="cutoff")
    assert len(result) == 1  # preserved, not dropped
    assert not result.loc[0, "dealer_join_matched"]
    assert result.loc[0, "dealer_net_position_bills_missing_reason"] == NO_COVERAGE_REASON


def test_coverage_summary_reports_every_auction_not_just_matched_ones():
    dealer = _dealer_wide(start="2026-06-01")
    auctions = _auctions(
        [
            {"cutoff": pd.Timestamp("2020-01-01"), "tenor": "2-Year"},
            {"cutoff": pd.Timestamp("2020-01-01"), "tenor": "2-Year"},
            {"cutoff": pd.Timestamp("2026-06-11"), "tenor": "10-Year"},
        ]
    )
    joined = as_of_join(auctions, dealer, cutoff_col="cutoff")
    summary = coverage_summary(joined, group_cols=("tenor",))
    row_2y = summary.loc[summary["tenor"] == "2-Year"].iloc[0]
    assert row_2y["n_auctions"] == 2
    assert row_2y["n_matched_any_dealer_release"] == 0
    row_10y = summary.loc[summary["tenor"] == "10-Year"].iloc[0]
    assert row_10y["n_auctions"] == 1
    assert row_10y["n_matched_any_dealer_release"] == 1


def test_add_cutoff_dates_pre_auction_is_a_business_day_before_auction_date():
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2026-09-10", "2026-09-08"]),  # Thu, Tue
            "announcemt_date": pd.to_datetime(["2026-09-03", "2026-08-27"]),
        }
    )
    out = add_cutoff_dates(auctions)
    assert out.loc[0, PRE_AUCTION_CUTOFF_COL] == pd.Timestamp("2026-09-09")  # Wed before Thu
    assert out.loc[1, PRE_AUCTION_CUTOFF_COL] == pd.Timestamp("2026-09-04")  # Fri before Tue (skips weekend)
    assert out.loc[0, ANNOUNCEMENT_CUTOFF_COL] == pd.Timestamp("2026-09-03")


def test_announcement_and_pre_auction_cutoffs_can_select_different_observations():
    dealer = _dealer_wide(n_weeks=4, start="2026-01-07")
    # observation weeks: 01-07 (safe 01-09), 01-14 (safe 01-16), 01-21 (safe 01-23), 01-28 (safe 01-30)
    auctions = pd.DataFrame(
        {
            "cusip": ["A"],
            "tenor": ["2-Year"],
            "offering_amt": [50_000_000_000.0],
            "announcemt_date": [pd.Timestamp("2026-01-10")],  # only week 0 is safely available
            "auction_date": [pd.Timestamp("2026-01-24")],  # pre-auction cutoff reaches week 2
        }
    )
    auctions = add_cutoff_dates(auctions)
    ann = as_of_join(auctions, dealer, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    pre = as_of_join(auctions, dealer, cutoff_col=PRE_AUCTION_CUTOFF_COL)
    assert ann.loc[0, "observation_date"] < pre.loc[0, "observation_date"]
    assert ann.loc[0, "dealer_net_position_bills"] != pre.loc[0, "dealer_net_position_bills"]


def test_worked_example_matches_between_the_two_cutoffs(dealer_stats_raw_payload):
    from treasury_auction_stress.data.dealer_stats_normalize import (
        normalize_dealer_stats,
        pivot_wide,
    )
    from treasury_auction_stress.features.dealer_candidate_features import (
        build_dealer_feature_table,
    )

    long_df, _ = normalize_dealer_stats(dealer_stats_raw_payload)
    feat = build_dealer_feature_table(pivot_wide(long_df))

    auctions = pd.DataFrame(
        {
            "cusip": ["ABC123"],
            "tenor": ["2-Year"],
            "offering_amt": [50_000_000_000.0],
            "announcemt_date": [pd.Timestamp("2022-01-20")],
            "auction_date": [pd.Timestamp("2022-01-25")],
        }
    )
    auctions = add_cutoff_dates(auctions)
    ann = as_of_join(auctions, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    pre = as_of_join(auctions, feat, cutoff_col=PRE_AUCTION_CUTOFF_COL)

    example = worked_example(ann, pre, feat, example_series="dealer_net_position_bills")
    assert example["cusip"] == "ABC123"
    assert example["announcement_cutoff_selected_observation_date"] <= example["pre_auction_cutoff_selected_observation_date"]
    assert len(example["candidates"]) > 0


def test_build_dealer_join_table_has_per_cutoff_prefixed_columns_and_no_forbidden_leak():
    from treasury_auction_stress.features.dealer_join import build_dealer_join_table

    auctions = _auctions(
        [
            {
                "announcemt_date": pd.Timestamp("2026-01-08"),
                "auction_date": pd.Timestamp("2026-01-13"),
                "is_reopening": False,
                "high_yield": 4.5,  # a forbidden result column, deliberately present on the input
            }
        ]
    )
    auctions = add_cutoff_dates(auctions)
    dealer_wide = _dealer_wide()

    out = build_dealer_join_table(auctions, dealer_wide)
    assert len(out) == len(auctions)
    assert "high_yield" not in out.columns
    for cutoff_col in (ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL):
        assert f"{cutoff_col}__dealer_net_position_bills" in out.columns
        assert f"{cutoff_col}__dealer_join_matched" in out.columns
        assert f"{cutoff_col}__dealer_inventory_to_offering_ratio_harmonized" in out.columns
