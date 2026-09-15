"""Persisting raw NY Fed Primary Dealer Statistics API responses to
disk as immutable, checksummed artifacts.

Structurally a near-twin of `treasury_auction_stress.data.raw_store`
(same immutability/idempotence/checksum contract, same sidecar
metadata shape) but bundles *all* selected series' raw responses from
one pipeline run into a single artifact, because this source has no
date-range parameter to key a filename on -- a "pull" here means "the
full history of every selected keyid, as of this retrieval," not a
particular [start, end] window. See raw_store.py's own module
docstring for the shared vocabulary (raw/checksum/idempotent/sidecar).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.dealer_stats_client import (
    FetchedSeries,
    FetchedSeriesList,
)

RAW_ARTIFACT_FORMAT_VERSION = "1"
SOURCE_NAME = "ny_fed_primary_dealer_statistics"


class RawArtifactConflictError(Exception):
    """A raw artifact already exists on disk with *different* content
    than what we are about to write for the same retrieval date.
    """


@dataclass(frozen=True)
class RawArtifactMetadata:
    source_name: str
    request_urls: list[str]
    query_parameters: dict[str, Any]
    retrieval_timestamp_utc: str
    http_status_codes: list[int]
    response_format: str
    row_count: int
    dataset_start_date: str | None
    dataset_end_date: str | None
    checksum_sha256: str
    raw_artifact_format_version: str
    keyids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_artifact_paths(raw_dir: Path, retrieval_date: str) -> tuple[Path, Path]:
    """Deterministic filenames for a given retrieval day -- same-day
    reruns are idempotent (see module docstring)."""
    stem = f"ny_fed_primary_dealer_stats_raw_retrieved_{retrieval_date}"
    return raw_dir / f"{stem}.json", raw_dir / f"{stem}.meta.json"


def build_raw_payload(
    series_list: FetchedSeriesList | None, series: list[FetchedSeries]
) -> dict[str, Any]:
    """Assemble the on-disk raw JSON: the untouched body of the series
    list response (if fetched) plus every requested series' untouched
    body, with enough per-request context to audit it later.
    """
    return {
        "source_name": SOURCE_NAME,
        "raw_artifact_format_version": RAW_ARTIFACT_FORMAT_VERSION,
        "series_list": (
            {
                "url": series_list.url,
                "http_status": series_list.http_status,
                "retrieved_at_utc": series_list.retrieved_at_utc,
                "body": series_list.parsed,
            }
            if series_list is not None
            else None
        ),
        "series": [
            {
                "keyid": s.keyid,
                "url": s.url,
                "http_status": s.http_status,
                "retrieved_at_utc": s.retrieved_at_utc,
                "body": s.parsed,
                "period": s.period,
            }
            for s in series
        ],
    }


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _observation_dates(series: list[FetchedSeries]) -> list[str]:
    dates: list[str] = []
    for s in series:
        for obs in s.parsed.get("pd", {}).get("timeseries", []):
            date = obs.get("asofdate")
            if date:
                dates.append(date)
    return dates


def is_cached(raw_dir: Path, retrieval_date: str) -> bool:
    """True if a valid (checksum-matching) raw artifact already exists
    for this exact retrieval day."""
    raw_path, meta_path = raw_artifact_paths(raw_dir, retrieval_date)
    if not raw_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = load_metadata(meta_path)
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    actual_checksum = _sha256_of_text(raw_path.read_text(encoding="utf-8"))
    return actual_checksum == metadata.checksum_sha256


def write_raw_artifact(
    raw_dir: Path,
    *,
    retrieval_date: str,
    series_list: FetchedSeriesList | None,
    series: list[FetchedSeries],
) -> RawArtifactMetadata:
    """Write the raw artifact and its metadata sidecar. Raises
    `RawArtifactConflictError` if a *different* artifact already exists
    for this retrieval day; returns the existing metadata without
    writing anything if an identical artifact is already cached.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = raw_artifact_paths(raw_dir, retrieval_date)

    payload = build_raw_payload(series_list, series)
    serialized = _serialize(payload)
    checksum = _sha256_of_text(serialized)

    if raw_path.exists():
        existing_checksum = _sha256_of_text(raw_path.read_text(encoding="utf-8"))
        if existing_checksum == checksum:
            return load_metadata(meta_path)
        raise RawArtifactConflictError(
            f"{raw_path} already exists with different content "
            f"(existing checksum {existing_checksum[:12]}..., new checksum "
            f"{checksum[:12]}...). Raw artifacts are immutable; this should "
            "only be possible if retrieval_date was reused incorrectly."
        )

    dates = _observation_dates(series)
    row_count = len(dates)
    request_urls = ([series_list.url] if series_list is not None else []) + [
        s.url for s in series
    ]
    http_status_codes = ([series_list.http_status] if series_list is not None else []) + [
        s.http_status for s in series
    ]
    retrieval_timestamp_utc = (
        series_list.retrieved_at_utc if series_list is not None else (series[0].retrieved_at_utc if series else "")
    )

    metadata = RawArtifactMetadata(
        source_name=SOURCE_NAME,
        request_urls=request_urls,
        query_parameters={"keyids": [s.keyid for s in series]},
        retrieval_timestamp_utc=retrieval_timestamp_utc,
        http_status_codes=http_status_codes,
        response_format="json",
        row_count=row_count,
        dataset_start_date=min(dates) if dates else None,
        dataset_end_date=max(dates) if dates else None,
        checksum_sha256=checksum,
        raw_artifact_format_version=RAW_ARTIFACT_FORMAT_VERSION,
        keyids=[s.keyid for s in series],
    )

    raw_path.write_text(serialized, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return metadata


def load_metadata(meta_path: Path) -> RawArtifactMetadata:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return RawArtifactMetadata(**data)


def load_raw_payload(raw_path: Path) -> dict[str, Any]:
    return json.loads(raw_path.read_text(encoding="utf-8"))
