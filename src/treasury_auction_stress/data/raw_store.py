"""Persisting raw API responses to disk as immutable, checksummed artifacts.

Vocabulary, for anyone new to this:

- "Raw" means exactly what the server sent back, before this project
  does anything to interpret it (that interpretation happens later, in
  normalize.py). Keeping an untouched raw copy means that if a later
  processing step turns out to have a bug, the original data can be
  re-processed without needing to re-download anything.
- A "checksum" is a short fingerprint (here, SHA-256) computed from a
  file's exact bytes. Two files with the same checksum are guaranteed
  (for all practical purposes) to have identical contents. It is used
  here to detect whether a "cached" file has been tampered with or
  corrupted, and to let tests assert a raw file was never rewritten.
- "Idempotent" means running the same operation twice has the same
  effect as running it once -- rerunning today's download should not
  create a second, different copy of today's data.
- A "sidecar" file is a small companion file (here, `<name>.meta.json`)
  that sits next to the main file and describes it, rather than being
  merged into it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.client import FetchedPage

RAW_ARTIFACT_FORMAT_VERSION = "1"
SOURCE_NAME = "treasury_fiscal_data_auctions"


class RawArtifactConflictError(Exception):
    """Raised when a raw artifact already exists on disk with
    *different* content than what we are about to write.

    Raw artifacts are immutable once written -- this error means the
    caller is trying to overwrite history rather than write a new,
    distinctly-named snapshot.
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
    dataset_start_date: str
    dataset_end_date: str
    checksum_sha256: str
    raw_artifact_format_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_artifact_paths(
    raw_dir: Path, start_date: str, end_date: str, retrieval_date: str
) -> tuple[Path, Path]:
    """Deterministic filenames for a given (start, end, retrieval day).

    Same inputs always produce the same path, which is what makes
    same-day reruns idempotent (see module docstring).
    """
    stem = f"treasury_auctions_raw_{start_date}_to_{end_date}_retrieved_{retrieval_date}"
    return raw_dir / f"{stem}.json", raw_dir / f"{stem}.meta.json"


def build_raw_payload(pages: list[FetchedPage]) -> dict[str, Any]:
    """Assemble the on-disk raw JSON: the untouched parsed body of every
    page, plus enough per-page request context to audit it later.
    """
    return {
        "source_name": SOURCE_NAME,
        "raw_artifact_format_version": RAW_ARTIFACT_FORMAT_VERSION,
        "pages": [
            {
                "url": page.url,
                "params": page.params,
                "http_status": page.http_status,
                "retrieved_at_utc": page.retrieved_at_utc,
                "body": page.parsed,
            }
            for page in pages
        ],
    }


def _serialize(payload: dict[str, Any]) -> str:
    # Fixed separators and no key sorting: this is a faithful transcript
    # of what was received and assembled, not a normalized re-encoding.
    return json.dumps(payload, ensure_ascii=False, indent=2)


def is_cached(raw_dir: Path, start_date: str, end_date: str, retrieval_date: str) -> bool:
    """True if a valid (checksum-matching) raw artifact already exists
    for this exact (start, end, retrieval day) triple.
    """
    raw_path, meta_path = raw_artifact_paths(raw_dir, start_date, end_date, retrieval_date)
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
    start_date: str,
    end_date: str,
    retrieval_date: str,
    pages: list[FetchedPage],
) -> RawArtifactMetadata:
    """Write the raw artifact and its metadata sidecar.

    Raises RawArtifactConflictError if a *different* artifact already
    exists at this path (raw artifacts are never mutated in place).
    Returns the existing metadata without writing anything if an
    identical artifact is already cached (idempotence).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = raw_artifact_paths(raw_dir, start_date, end_date, retrieval_date)

    payload = build_raw_payload(pages)
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

    row_count = sum(len(page.parsed.get("data", [])) for page in pages)
    metadata = RawArtifactMetadata(
        source_name=SOURCE_NAME,
        request_urls=[page.url for page in pages],
        query_parameters={
            k: v for k, v in pages[0].params.items() if k not in ("page[number]",)
        }
        if pages
        else {},
        retrieval_timestamp_utc=pages[0].retrieved_at_utc if pages else "",
        http_status_codes=[page.http_status for page in pages],
        response_format="json",
        row_count=row_count,
        dataset_start_date=start_date,
        dataset_end_date=end_date,
        checksum_sha256=checksum,
        raw_artifact_format_version=RAW_ARTIFACT_FORMAT_VERSION,
    )

    raw_path.write_text(serialized, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return metadata


def load_metadata(meta_path: Path) -> RawArtifactMetadata:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return RawArtifactMetadata(**data)


def load_raw_payload(raw_path: Path) -> dict[str, Any]:
    return json.loads(raw_path.read_text(encoding="utf-8"))
