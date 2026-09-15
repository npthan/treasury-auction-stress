"""Persisting raw RTDSM xlsx workbooks to disk as immutable,
checksummed artifacts -- one artifact per (mnemonic, retrieval date).
Unlike this project's other raw stores, the artifact itself is the
binary `.xlsx` file (not JSON); a `.meta.json` sidecar carries the
required retrieval metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.rtdsm_client import FetchedVariable

RAW_ARTIFACT_FORMAT_VERSION = "1"
SOURCE_NAME = "philadelphia_fed_rtdsm"


class RawArtifactConflictError(Exception):
    pass


@dataclass(frozen=True)
class RawArtifactMetadata:
    source_name: str
    mnemonic: str
    request_url: str
    retrieval_timestamp_utc: str
    http_status: int
    response_format: str
    content_length_bytes: int
    checksum_sha256: str
    raw_artifact_format_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_artifact_paths(raw_dir: Path, mnemonic: str, retrieval_date: str) -> tuple[Path, Path]:
    stem = f"rtdsm_{mnemonic.lower()}_raw_retrieved_{retrieval_date}"
    return raw_dir / f"{stem}.xlsx", raw_dir / f"{stem}.meta.json"


def is_cached(raw_dir: Path, mnemonic: str, retrieval_date: str) -> bool:
    raw_path, meta_path = raw_artifact_paths(raw_dir, mnemonic, retrieval_date)
    if not raw_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = load_metadata(meta_path)
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    actual_checksum = _sha256_of_bytes(raw_path.read_bytes())
    return actual_checksum == metadata.checksum_sha256


def write_raw_artifact(raw_dir: Path, *, retrieval_date: str, fetched: FetchedVariable) -> RawArtifactMetadata:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = raw_artifact_paths(raw_dir, fetched.mnemonic, retrieval_date)

    checksum = _sha256_of_bytes(fetched.raw_bytes)
    if raw_path.exists():
        existing_checksum = _sha256_of_bytes(raw_path.read_bytes())
        if existing_checksum == checksum:
            return load_metadata(meta_path)
        raise RawArtifactConflictError(
            f"{raw_path} already exists with different content -- raw artifacts are immutable."
        )

    metadata = RawArtifactMetadata(
        source_name=SOURCE_NAME,
        mnemonic=fetched.mnemonic,
        request_url=fetched.url,
        retrieval_timestamp_utc=fetched.retrieved_at_utc,
        http_status=fetched.http_status,
        response_format="xlsx",
        content_length_bytes=len(fetched.raw_bytes),
        checksum_sha256=checksum,
        raw_artifact_format_version=RAW_ARTIFACT_FORMAT_VERSION,
    )

    tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    tmp_path.write_bytes(fetched.raw_bytes)
    tmp_path.replace(raw_path)
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return metadata


def load_metadata(meta_path: Path) -> RawArtifactMetadata:
    return RawArtifactMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))


def load_raw_bytes(raw_path: Path) -> bytes:
    return raw_path.read_bytes()
