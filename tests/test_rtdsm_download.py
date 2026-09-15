from __future__ import annotations

from pathlib import Path

from treasury_auction_stress.data import rtdsm_download as download_module
from treasury_auction_stress.data.rtdsm_client import (
    EXPECTED_CONTENT_TYPE,
    FetchedVariable,
)
from treasury_auction_stress.data.rtdsm_download import download_variable
from treasury_auction_stress.data.rtdsm_schema import VARIABLE_BY_MNEMONIC

_RUC = VARIABLE_BY_MNEMONIC["RUC"]


def _fake_fetch_variable(variable, **_kwargs) -> FetchedVariable:
    return FetchedVariable(
        mnemonic=variable.mnemonic, url=variable.xlsx_url, http_status=200,
        content_type=EXPECTED_CONTENT_TYPE, retrieved_at_utc="2026-09-11T00:00:00+00:00", raw_bytes=b"fake",
    )


def test_first_download_fetches_live(tmp_path: Path, monkeypatch):
    calls = []

    def fake_fetch_variable(variable, **kwargs):
        calls.append(variable.mnemonic)
        return _fake_fetch_variable(variable, **kwargs)

    monkeypatch.setattr(download_module, "fetch_variable", fake_fetch_variable)
    result = download_variable(_RUC, raw_dir=tmp_path, retrieval_date="2026-09-11")
    assert result.was_cached is False
    assert calls == ["RUC"]
    assert result.raw_path.exists()


def test_second_download_reuses_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(download_module, "fetch_variable", _fake_fetch_variable)
    download_variable(_RUC, raw_dir=tmp_path, retrieval_date="2026-09-11")

    calls = []

    def fail_if_called(variable, **kwargs):
        calls.append(variable.mnemonic)
        raise AssertionError("should not re-fetch when cached")

    monkeypatch.setattr(download_module, "fetch_variable", fail_if_called)
    result = download_variable(_RUC, raw_dir=tmp_path, retrieval_date="2026-09-11")
    assert result.was_cached is True
    assert calls == []


def test_force_refresh_bypasses_cache(tmp_path: Path, monkeypatch):
    calls = []

    def fake_fetch_variable(variable, **kwargs):
        calls.append(variable.mnemonic)
        return _fake_fetch_variable(variable, **kwargs)

    monkeypatch.setattr(download_module, "fetch_variable", fake_fetch_variable)
    download_variable(_RUC, raw_dir=tmp_path, retrieval_date="2026-09-11")
    result = download_variable(_RUC, raw_dir=tmp_path, retrieval_date="2026-09-11", force_refresh=True)
    assert result.was_cached is False
    assert len(calls) == 2
