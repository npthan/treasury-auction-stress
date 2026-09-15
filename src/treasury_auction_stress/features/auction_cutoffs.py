"""The two prediction cutoffs, shared by every Phase 3/4 source-specific
join module (`dealer_join.py`, `treasury_rates_join.py`, `cftc_join.py`,
`rtdsm_join.py`).

Extracted out of `dealer_join.py` (Phase 4A) so all four source
families use the exact same cutoff computation -- these are properties
of the *auction*, not of any one data source, and must never drift
between source-specific join modules.

- **Announcement-time cutoff**: the auction's own `announcemt_date`,
  treated (per the Phase-2-acceptance-reviewed rule) as "public
  information as of the end of that calendar date" -- a date-level
  cutoff, not an intraday timestamp.
- **Pre-auction cutoff**: "the previous business day's market close"
  (`docs/project_plan.md`) -- the closest calendar day before
  `auction_date` that is not a weekend or U.S. federal holiday
  (`time_utils.previous_business_day`). This is date-level too: "close"
  means "as of the end of that calendar day," never a specific
  intraday hour -- see each source's own publication-timing section
  for how that interacts with same-day-published data.
"""

from __future__ import annotations

import pandas as pd

from treasury_auction_stress.data.time_utils import (
    previous_business_day,
    us_federal_holidays,
)

ANNOUNCEMENT_CUTOFF_COL = "announcement_cutoff_date"
PRE_AUCTION_CUTOFF_COL = "pre_auction_cutoff_date"


def add_cutoff_dates(auctions_df: pd.DataFrame) -> pd.DataFrame:
    """Add `announcement_cutoff_date` and `pre_auction_cutoff_date`.
    Both are plain calendar dates (midnight `Timestamp`s), per this
    project's date-level cutoff convention -- see module docstring.
    """
    out = auctions_df.copy()
    out[ANNOUNCEMENT_CUTOFF_COL] = pd.to_datetime(out["announcemt_date"]).dt.normalize()
    holidays = us_federal_holidays(
        start=(out["auction_date"].min() - pd.Timedelta(days=14)).isoformat(),
        end=(out["auction_date"].max() + pd.Timedelta(days=1)).isoformat(),
    )
    out[PRE_AUCTION_CUTOFF_COL] = out["auction_date"].dt.normalize().map(
        lambda d: previous_business_day(d, holidays)
    )
    return out
