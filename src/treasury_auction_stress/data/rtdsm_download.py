"""Orchestrates downloading every selected RTDSM variable, reusing a
cached raw artifact when one already exists for today's (UTC)
retrieval date -- one artifact per (mnemonic, retrieval date).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.rtdsm_client import fetch_variable
from treasury_auction_stress.data.rtdsm_raw_store import (
    RawArtifactMetadata,
    is_cached,
    load_metadata,
    raw_artifact_paths,
    write_raw_artifact,
)
from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY, RtdsmVariable
from treasury_auction_stress.data.time_utils import utc_today_date


@dataclass(frozen=True)
class RtdsmDownloadResult:
    metadata: RawArtifactMetadata
    raw_path: Path
    meta_path: Path
    retrieval_date: str
    was_cached: bool


def download_variable(
    variable: RtdsmVariable,
    *,
    raw_dir: Path,
    retrieval_date: str | None = None,
    force_refresh: bool = False,
    **fetch_kwargs: Any,
) -> RtdsmDownloadResult:
    retrieval_date = retrieval_date or utc_today_date()
    raw_path, meta_path = raw_artifact_paths(raw_dir, variable.mnemonic, retrieval_date)

    if not force_refresh and is_cached(raw_dir, variable.mnemonic, retrieval_date):
        return RtdsmDownloadResult(
            metadata=load_metadata(meta_path), raw_path=raw_path, meta_path=meta_path,
            retrieval_date=retrieval_date, was_cached=True,
        )

    fetched = fetch_variable(variable, **fetch_kwargs)
    metadata = write_raw_artifact(raw_dir, retrieval_date=retrieval_date, fetched=fetched)
    return RtdsmDownloadResult(
        metadata=metadata, raw_path=raw_path, meta_path=meta_path, retrieval_date=retrieval_date, was_cached=False
    )


def download_rtdsm(
    *, raw_dir: Path, retrieval_date: str | None = None, force_refresh: bool = False, **fetch_kwargs: Any
) -> list[RtdsmDownloadResult]:
    session = fetch_kwargs.pop("session", None)
    if session is None:
        import requests

        session = requests.Session()
    return [
        download_variable(v, raw_dir=raw_dir, retrieval_date=retrieval_date, force_refresh=force_refresh, session=session, **fetch_kwargs)
        for v in VARIABLE_REGISTRY
    ]
