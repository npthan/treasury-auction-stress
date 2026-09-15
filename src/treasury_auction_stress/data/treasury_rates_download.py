"""Orchestrates downloading every calendar year of Treasury Daily Par
Yield Curve rates from `treasury_rates_schema.RATES_START_DATE` through
the current America/New_York calendar year, reusing cached raw
artifacts already on disk for a given (year, retrieval date).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.time_utils import market_today_date, utc_today_date
from treasury_auction_stress.data.treasury_rates_client import fetch_year
from treasury_auction_stress.data.treasury_rates_raw_store import (
    RawArtifactMetadata,
    is_cached,
    load_metadata,
    raw_artifact_paths,
    write_raw_artifact,
)
from treasury_auction_stress.data.treasury_rates_schema import RATES_START_DATE


@dataclass(frozen=True)
class YearDownloadResult:
    year: int
    metadata: RawArtifactMetadata
    raw_path: Path
    meta_path: Path
    was_cached: bool


def years_to_fetch(start_date: str = RATES_START_DATE, end_date: str | None = None) -> tuple[int, ...]:
    end_date = end_date or market_today_date()
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    return tuple(range(start_year, end_year + 1))


def download_treasury_rates(
    *,
    raw_dir: Path,
    retrieval_date: str | None = None,
    force_refresh: bool = False,
    years: tuple[int, ...] | None = None,
    **fetch_kwargs: Any,
) -> list[YearDownloadResult]:
    """Download (or reuse cached copies of) every year in `years`
    (default: `years_to_fetch()`, i.e. `RATES_START_DATE` through the
    current America/New_York calendar year -- the market's own
    timezone, matching the rest of this project's "through today"
    conventions).
    """
    retrieval_date = retrieval_date or utc_today_date()
    years = years or years_to_fetch()
    results: list[YearDownloadResult] = []
    for year in years:
        raw_path, meta_path = raw_artifact_paths(raw_dir, year, retrieval_date)
        if not force_refresh and is_cached(raw_dir, year, retrieval_date):
            results.append(
                YearDownloadResult(
                    year=year,
                    metadata=load_metadata(meta_path),
                    raw_path=raw_path,
                    meta_path=meta_path,
                    was_cached=True,
                )
            )
            continue
        fetched = fetch_year(year, **fetch_kwargs)
        metadata = write_raw_artifact(raw_dir, year=year, retrieval_date=retrieval_date, fetched=fetched)
        results.append(
            YearDownloadResult(
                year=year, metadata=metadata, raw_path=raw_path, meta_path=meta_path, was_cached=False
            )
        )
    return results
