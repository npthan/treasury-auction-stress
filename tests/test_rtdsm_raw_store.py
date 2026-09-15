from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from treasury_auction_stress.data.rtdsm_client import FetchedVariable
from treasury_auction_stress.data.rtdsm_raw_store import (
    RawArtifactConflictError,
    is_cached,
    load_metadata,
    load_raw_bytes,
    raw_artifact_paths,
    write_raw_artifact,
)

_BYTES = b"PK\x03\x04fake-xlsx-bytes"


def _fetched(data: bytes = _BYTES) -> FetchedVariable:
    return FetchedVariable(
        mnemonic="RUC", url="https://example.invalid/ruc.xlsx", http_status=200,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        retrieved_at_utc="2026-09-11T00:00:00+00:00", raw_bytes=data,
    )


def test_write_raw_artifact_creates_files_and_correct_metadata(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, meta_path = raw_artifact_paths(tmp_path, "RUC", "2026-09-11")
    assert raw_path.exists() and meta_path.exists()
    assert metadata.content_length_bytes == len(_BYTES)
    assert metadata.source_name == "philadelphia_fed_rtdsm"
    assert metadata.mnemonic == "RUC"


def test_checksum_matches_file_contents(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, "RUC", "2026-09-11")
    actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert actual == metadata.checksum_sha256


def test_repeat_write_is_idempotent(tmp_path: Path):
    first = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, "RUC", "2026-09-11")
    mtime = raw_path.stat().st_mtime_ns
    second = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    assert second.checksum_sha256 == first.checksum_sha256
    assert raw_path.stat().st_mtime_ns == mtime


def test_raw_artifact_is_immutable(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    with pytest.raises(RawArtifactConflictError):
        write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched(b"different-bytes"))


def test_is_cached_true_after_write_false_before(tmp_path: Path):
    assert is_cached(tmp_path, "RUC", "2026-09-11") is False
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    assert is_cached(tmp_path, "RUC", "2026-09-11") is True


def test_is_cached_false_if_corrupted(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, "RUC", "2026-09-11")
    raw_path.write_bytes(b"corrupted")
    assert is_cached(tmp_path, "RUC", "2026-09-11") is False


def test_load_raw_bytes_round_trips(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, meta_path = raw_artifact_paths(tmp_path, "RUC", "2026-09-11")
    assert load_raw_bytes(raw_path) == _BYTES
    metadata = load_metadata(meta_path)
    assert metadata.mnemonic == "RUC"


def test_no_leftover_tmp_file_after_write(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, "RUC", "2026-09-11")
    tmp_marker = raw_path.with_suffix(raw_path.suffix + ".tmp")
    assert not tmp_marker.exists()
