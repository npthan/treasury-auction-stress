"""Orchestrates downloading every selected CFTC TFF Futures Only
contract, reusing a cached raw artifact when one already exists for
today's (UTC) retrieval date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.cftc_client import fetch_contracts
from treasury_auction_stress.data.cftc_raw_store import (
    RawArtifactMetadata,
    is_cached,
    load_metadata,
    raw_artifact_paths,
    write_raw_artifact,
)
from treasury_auction_stress.data.cftc_schema import CONTRACT_CODES
from treasury_auction_stress.data.time_utils import utc_today_date


@dataclass(frozen=True)
class CftcDownloadResult:
    metadata: RawArtifactMetadata
    raw_path: Path
    meta_path: Path
    retrieval_date: str
    was_cached: bool


def download_cftc_positioning(
    *, raw_dir: Path, retrieval_date: str | None = None, force_refresh: bool = False, **fetch_kwargs: Any
) -> CftcDownloadResult:
    retrieval_date = retrieval_date or utc_today_date()
    raw_path, meta_path = raw_artifact_paths(raw_dir, retrieval_date)

    if not force_refresh and is_cached(raw_dir, retrieval_date):
        return CftcDownloadResult(
            metadata=load_metadata(meta_path), raw_path=raw_path, meta_path=meta_path,
            retrieval_date=retrieval_date, was_cached=True,
        )

    contracts = fetch_contracts(CONTRACT_CODES, **fetch_kwargs)
    metadata = write_raw_artifact(raw_dir, retrieval_date=retrieval_date, contracts=contracts)
    return CftcDownloadResult(
        metadata=metadata, raw_path=raw_path, meta_path=meta_path, retrieval_date=retrieval_date, was_cached=False
    )
