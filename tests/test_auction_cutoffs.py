from __future__ import annotations

import pandas as pd

from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)


def test_announcement_cutoff_is_the_announcement_date_normalized():
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2026-09-10"]),
            "announcemt_date": pd.to_datetime(["2026-09-03"]),
        }
    )
    out = add_cutoff_dates(auctions)
    assert out.loc[0, ANNOUNCEMENT_CUTOFF_COL] == pd.Timestamp("2026-09-03")


def test_pre_auction_cutoff_skips_weekend():
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2026-09-08"]),  # Tuesday
            "announcemt_date": pd.to_datetime(["2026-08-27"]),
        }
    )
    out = add_cutoff_dates(auctions)
    assert out.loc[0, PRE_AUCTION_CUTOFF_COL] == pd.Timestamp("2026-09-04")  # Friday before


def test_pre_auction_cutoff_skips_a_federal_holiday():
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2026-09-08"]),  # day after Labor Day (Mon 2026-09-07)
            "announcemt_date": pd.to_datetime(["2026-08-27"]),
        }
    )
    out = add_cutoff_dates(auctions)
    assert out.loc[0, PRE_AUCTION_CUTOFF_COL] == pd.Timestamp("2026-09-04")  # Friday before Labor Day


def test_multiple_rows_use_independent_cutoffs():
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2026-01-13", "2026-06-11"]),
            "announcemt_date": pd.to_datetime(["2026-01-08", "2026-06-04"]),
        }
    )
    out = add_cutoff_dates(auctions)
    assert out[ANNOUNCEMENT_CUTOFF_COL].tolist() == [pd.Timestamp("2026-01-08"), pd.Timestamp("2026-06-04")]
