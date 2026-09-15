from __future__ import annotations

import json
from pathlib import Path

from treasury_auction_stress.data import cftc_download as download_module
from treasury_auction_stress.data.cftc_client import FetchedContract
from treasury_auction_stress.data.cftc_download import download_cftc_positioning


def _fake_contracts(contract_codes, **_kwargs) -> list[FetchedContract]:
    return [
        FetchedContract(
            contract_code=code, url=f"https://example.invalid/{code}", http_status=200,
            retrieved_at_utc="2026-09-11T00:00:00+00:00",
            raw_text=json.dumps([{"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000"}]),
        )
        for code in contract_codes
    ]


def test_first_download_fetches_live(tmp_path: Path, monkeypatch):
    calls = []

    def fake_fetch_contracts(contract_codes, **kwargs):
        calls.append(contract_codes)
        return _fake_contracts(contract_codes)

    monkeypatch.setattr(download_module, "fetch_contracts", fake_fetch_contracts)
    result = download_cftc_positioning(raw_dir=tmp_path, retrieval_date="2026-09-11")
    assert result.was_cached is False
    assert len(calls) == 1
    assert result.raw_path.exists()


def test_second_download_reuses_cache(tmp_path: Path, monkeypatch):
    def fake_fetch_contracts(contract_codes, **kwargs):
        return _fake_contracts(contract_codes)

    monkeypatch.setattr(download_module, "fetch_contracts", fake_fetch_contracts)
    download_cftc_positioning(raw_dir=tmp_path, retrieval_date="2026-09-11")

    calls = []

    def fail_if_called(contract_codes, **kwargs):
        calls.append(contract_codes)
        raise AssertionError("should not re-fetch when cached")

    monkeypatch.setattr(download_module, "fetch_contracts", fail_if_called)
    result = download_cftc_positioning(raw_dir=tmp_path, retrieval_date="2026-09-11")
    assert result.was_cached is True
    assert calls == []


def test_force_refresh_bypasses_cache(tmp_path: Path, monkeypatch):
    calls = []

    def fake_fetch_contracts(contract_codes, **kwargs):
        calls.append(contract_codes)
        return _fake_contracts(contract_codes)

    monkeypatch.setattr(download_module, "fetch_contracts", fake_fetch_contracts)
    download_cftc_positioning(raw_dir=tmp_path, retrieval_date="2026-09-11")
    result = download_cftc_positioning(raw_dir=tmp_path, retrieval_date="2026-09-11", force_refresh=True)
    assert result.was_cached is False
    assert len(calls) == 2
