"""Orchestrates downloading the Treasury auctions dataset for a date
range, reusing a cached raw artifact when one already exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.client import fetch_all_pages
from treasury_auction_stress.data.raw_store import (
    RawArtifactMetadata,
    is_cached,
    load_metadata,
    raw_artifact_paths,
    write_raw_artifact,
)
from treasury_auction_stress.data.schema import MAX_PAGE_SIZE
from treasury_auction_stress.data.time_utils import utc_today_date


def build_date_range_filter(start_date: str, end_date: str) -> str:
    """Build the API's `filter=` query value for an auction_date range.

    Dates must already be in YYYY-MM-DD form, per the API's documented
    date format.
    """
    return f"auction_date:gte:{start_date},auction_date:lte:{end_date}"


@dataclass(frozen=True)
class DownloadResult:
    metadata: RawArtifactMetadata
    raw_path: Path
    meta_path: Path
    retrieval_date: str
    was_cached: bool


def download_auctions(
    *,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    retrieval_date: str | None = None,
    force_refresh: bool = False,
    page_size: int = MAX_PAGE_SIZE,
    **fetch_kwargs: Any,
) -> DownloadResult:
    """Download (or reuse a cached copy of) auctions for [start_date, end_date].

    `retrieval_date` defaults to today's calendar date in **UTC**
    (`treasury_auction_stress.data.time_utils.utc_today_date`) -- this is
    operational bookkeeping (when did this pipeline run), not a Treasury
    market date, so UTC is the deliberate, unambiguous choice; see
    `time_utils.py` for why this must not be confused with `end_date`'s
    own default (America/New_York), which is a market-date decision made
    by the caller (see `treasury_auction_stress.data.cli`). Passing
    `retrieval_date` explicitly is mainly useful for tests, where a fixed
    value keeps the cache filename deterministic.
    """
    retrieval_date = retrieval_date or utc_today_date()
    raw_path, meta_path = raw_artifact_paths(raw_dir, start_date, end_date, retrieval_date)

    if not force_refresh and is_cached(raw_dir, start_date, end_date, retrieval_date):
        return DownloadResult(
            metadata=load_metadata(meta_path),
            raw_path=raw_path,
            meta_path=meta_path,
            retrieval_date=retrieval_date,
            was_cached=True,
        )

    filter_str = build_date_range_filter(start_date, end_date)
    pages = fetch_all_pages(
        filter_str=filter_str,
        sort="auction_date",
        page_size=page_size,
        **fetch_kwargs,
    )
    metadata = write_raw_artifact(
        raw_dir,
        start_date=start_date,
        end_date=end_date,
        retrieval_date=retrieval_date,
        pages=pages,
    )
    return DownloadResult(
        metadata=metadata,
        raw_path=raw_path,
        meta_path=meta_path,
        retrieval_date=retrieval_date,
        was_cached=False,
    )
