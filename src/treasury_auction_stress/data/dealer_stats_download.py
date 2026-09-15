"""Orchestrates downloading the full set of selected NY Fed Primary
Dealer Statistics series, reusing a cached raw artifact when one
already exists for today's (UTC) retrieval date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.dealer_stats_client import (
    fetch_all_legacy_series,
    fetch_all_selected_series,
    fetch_series_list,
)
from treasury_auction_stress.data.dealer_stats_raw_store import (
    RawArtifactMetadata,
    is_cached,
    load_metadata,
    raw_artifact_paths,
    write_raw_artifact,
)
from treasury_auction_stress.data.dealer_stats_schema import (
    ALL_LEGACY_KEYIDS,
    DISCONTINUED_MIDDLE_KEYIDS,
    LEGACY_PERIOD_KEY,
    SELECTED_KEYIDS,
)
from treasury_auction_stress.data.time_utils import utc_today_date


@dataclass(frozen=True)
class DealerStatsDownloadResult:
    metadata: RawArtifactMetadata
    raw_path: Path
    meta_path: Path
    retrieval_date: str
    was_cached: bool


def download_dealer_stats(
    *,
    raw_dir: Path,
    retrieval_date: str | None = None,
    force_refresh: bool = False,
    fetch_series_list_for_drift_check: bool = True,
    include_historical_extension: bool = True,
    **fetch_kwargs: Any,
) -> DealerStatsDownloadResult:
    """Download (or reuse a cached copy of) every series in
    `dealer_stats_schema.SELECTED_KEYIDS`, plus (by default, since the
    Phase 3 acceptance review) the pre-2013 legacy series and the
    discontinued 2013-2021 combined long-maturity bucket needed to
    build the historical extension -- see
    `dealer_stats_schema.py`'s historical-extension section for why.

    `retrieval_date` defaults to today's UTC calendar date -- the same
    operational-bookkeeping convention as
    `treasury_auction_stress.data.download.download_auctions`; see
    `time_utils.py` for why this is UTC, not America/New_York.
    """
    retrieval_date = retrieval_date or utc_today_date()
    raw_path, meta_path = raw_artifact_paths(raw_dir, retrieval_date)

    if not force_refresh and is_cached(raw_dir, retrieval_date):
        return DealerStatsDownloadResult(
            metadata=load_metadata(meta_path),
            raw_path=raw_path,
            meta_path=meta_path,
            retrieval_date=retrieval_date,
            was_cached=True,
        )

    series_list = fetch_series_list(**fetch_kwargs) if fetch_series_list_for_drift_check else None
    series = fetch_all_selected_series(SELECTED_KEYIDS, **fetch_kwargs)
    if include_historical_extension:
        series += fetch_all_selected_series(DISCONTINUED_MIDDLE_KEYIDS, **fetch_kwargs)
        series += fetch_all_legacy_series(LEGACY_PERIOD_KEY, ALL_LEGACY_KEYIDS, **fetch_kwargs)
    metadata = write_raw_artifact(
        raw_dir,
        retrieval_date=retrieval_date,
        series_list=series_list,
        series=series,
    )
    return DealerStatsDownloadResult(
        metadata=metadata,
        raw_path=raw_path,
        meta_path=meta_path,
        retrieval_date=retrieval_date,
        was_cached=False,
    )
