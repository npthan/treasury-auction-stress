from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from treasury_auction_stress.data.dealer_stats_client import (
    FetchedSeries,
    FetchedSeriesList,
)
from treasury_auction_stress.data.dealer_stats_raw_store import (
    RawArtifactConflictError,
    is_cached,
    load_metadata,
    load_raw_payload,
    raw_artifact_paths,
    write_raw_artifact,
)


def _series(keyid: str, obs: list[dict], url: str | None = None) -> FetchedSeries:
    return FetchedSeries(
        keyid=keyid,
        url=url or f"https://example.invalid/{keyid}",
        http_status=200,
        retrieved_at_utc="2026-09-11T00:00:00+00:00",
        raw_text="{}",
        parsed={"pd": {"timeseries": obs}},
    )


def _sample_series() -> list[FetchedSeries]:
    return [
        _series(
            "PDPOSGS-B",
            [
                {"asofdate": "2013-04-03", "keyid": "PDPOSGS-B", "value": "46324"},
                {"asofdate": "2013-04-10", "keyid": "PDPOSGS-B", "value": "53585"},
            ],
        ),
        _series(
            "PDSOOS-UTSETTOT",
            [{"asofdate": "2013-04-03", "keyid": "PDSOOS-UTSETTOT", "value": "*"}],
        ),
    ]


def test_write_raw_artifact_creates_raw_file_and_sidecar(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", series_list=None, series=_sample_series())
    raw_path, meta_path = raw_artifact_paths(tmp_path, "2026-09-11")
    assert raw_path.exists()
    assert meta_path.exists()
    assert metadata.row_count == 3
    assert metadata.source_name == "ny_fed_primary_dealer_statistics"
    assert metadata.dataset_start_date == "2013-04-03"
    assert metadata.dataset_end_date == "2013-04-10"
    assert set(metadata.keyids) == {"PDPOSGS-B", "PDSOOS-UTSETTOT"}


def test_raw_artifact_checksum_matches_file_contents(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", series_list=None, series=_sample_series())
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    actual = hashlib.sha256(raw_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert actual == metadata.checksum_sha256


def test_repeat_run_with_identical_data_is_idempotent(tmp_path: Path):
    kwargs = {"retrieval_date": "2026-09-11", "series_list": None, "series": _sample_series()}
    first = write_raw_artifact(tmp_path, **kwargs)
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    mtime_after_first = raw_path.stat().st_mtime_ns

    second = write_raw_artifact(tmp_path, **kwargs)

    assert second.checksum_sha256 == first.checksum_sha256
    assert raw_path.stat().st_mtime_ns == mtime_after_first


def test_raw_artifact_is_immutable_once_written(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", series_list=None, series=_sample_series())
    with pytest.raises(RawArtifactConflictError):
        write_raw_artifact(
            tmp_path,
            retrieval_date="2026-09-11",
            series_list=None,
            series=[_series("PDPOSGS-B", [{"asofdate": "2013-04-03", "keyid": "PDPOSGS-B", "value": "999"}])],
        )


def test_is_cached_true_after_write_and_false_before(tmp_path: Path):
    assert is_cached(tmp_path, "2026-09-11") is False
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", series_list=None, series=_sample_series())
    assert is_cached(tmp_path, "2026-09-11") is True


def test_is_cached_false_if_raw_file_corrupted(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", series_list=None, series=_sample_series())
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    raw_path.write_text("corrupted", encoding="utf-8")
    assert is_cached(tmp_path, "2026-09-11") is False


def test_load_raw_payload_round_trips(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", series_list=None, series=_sample_series())
    raw_path, meta_path = raw_artifact_paths(tmp_path, "2026-09-11")
    payload = load_raw_payload(raw_path)
    assert payload["series"][0]["keyid"] == "PDPOSGS-B"

    metadata = load_metadata(meta_path)
    assert metadata.dataset_start_date == "2013-04-03"


def test_write_raw_artifact_includes_series_list_when_provided(tmp_path: Path):
    series_list = FetchedSeriesList(
        url="https://markets.newyorkfed.org/api/pd/list/timeseries.json",
        http_status=200,
        retrieved_at_utc="2026-09-11T00:00:00+00:00",
        raw_text="{}",
        parsed={"pd": {"timeseries": [{"keyid": "PDPOSGS-B", "description": "x", "seriesbreak": "SBN2024"}]}},
    )
    metadata = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", series_list=series_list, series=_sample_series())
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    payload = load_raw_payload(raw_path)
    assert payload["series_list"]["body"] == series_list.parsed
    assert series_list.url in metadata.request_urls
