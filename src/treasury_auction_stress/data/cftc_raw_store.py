"""Persisting raw CFTC TFF Futures Only JSON responses to disk as
immutable, checksummed artifacts -- one artifact per retrieval date,
bundling every selected contract's raw response (mirrors
`dealer_stats_raw_store.py`'s pattern).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.cftc_client import FetchedContract

RAW_ARTIFACT_FORMAT_VERSION = "1"
SOURCE_NAME = "cftc_tff_futures_only"


class RawArtifactConflictError(Exception):
    pass


@dataclass(frozen=True)
class RawArtifactMetadata:
    source_name: str
    contract_codes: list[str]
    request_urls: list[str]
    retrieval_timestamp_utc: str
    http_status_codes: list[int]
    response_format: str
    row_count: int
    dataset_date_range: list[str]
    checksum_sha256: str
    raw_artifact_format_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_artifact_paths(raw_dir: Path, retrieval_date: str) -> tuple[Path, Path]:
    stem = f"cftc_tff_futures_only_raw_retrieved_{retrieval_date}"
    return raw_dir / f"{stem}.json", raw_dir / f"{stem}.meta.json"


def build_raw_payload(contracts: list[FetchedContract]) -> dict[str, Any]:
    return {
        "source_name": SOURCE_NAME,
        "raw_artifact_format_version": RAW_ARTIFACT_FORMAT_VERSION,
        "contracts": [
            {
                "contract_code": c.contract_code,
                "url": c.url,
                "http_status": c.http_status,
                "retrieved_at_utc": c.retrieved_at_utc,
                "body": json.loads(c.raw_text),
            }
            for c in contracts
        ],
    }


def is_cached(raw_dir: Path, retrieval_date: str) -> bool:
    raw_path, meta_path = raw_artifact_paths(raw_dir, retrieval_date)
    if not raw_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = load_metadata(meta_path)
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    actual_checksum = _sha256_of_text(raw_path.read_text(encoding="utf-8"))
    return actual_checksum == metadata.checksum_sha256


def write_raw_artifact(raw_dir: Path, *, retrieval_date: str, contracts: list[FetchedContract]) -> RawArtifactMetadata:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = raw_artifact_paths(raw_dir, retrieval_date)

    payload = build_raw_payload(contracts)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    checksum = _sha256_of_text(serialized)

    if raw_path.exists():
        existing_checksum = _sha256_of_text(raw_path.read_text(encoding="utf-8"))
        if existing_checksum == checksum:
            return load_metadata(meta_path)
        raise RawArtifactConflictError(
            f"{raw_path} already exists with different content -- raw artifacts are immutable."
        )

    dates = [row["report_date_as_yyyy_mm_dd"] for c in contracts for row in json.loads(c.raw_text)]
    metadata = RawArtifactMetadata(
        source_name=SOURCE_NAME,
        contract_codes=[c.contract_code for c in contracts],
        request_urls=[c.url for c in contracts],
        retrieval_timestamp_utc=contracts[0].retrieved_at_utc if contracts else "",
        http_status_codes=[c.http_status for c in contracts],
        response_format="json",
        row_count=len(dates),
        dataset_date_range=[min(dates), max(dates)] if dates else [],
        checksum_sha256=checksum,
        raw_artifact_format_version=RAW_ARTIFACT_FORMAT_VERSION,
    )

    tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    tmp_path.write_text(serialized, encoding="utf-8")
    tmp_path.replace(raw_path)
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return metadata


def load_metadata(meta_path: Path) -> RawArtifactMetadata:
    return RawArtifactMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))


def load_raw_payload(raw_path: Path) -> dict[str, Any]:
    return json.loads(raw_path.read_text(encoding="utf-8"))
