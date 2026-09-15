from __future__ import annotations

import pytest
import requests

from treasury_auction_stress.data import cftc_client as client_module
from treasury_auction_stress.data.cftc_client import (
    CftcApiConnectionError,
    CftcApiHTTPError,
    fetch_contract,
    fetch_contracts,
)


class FakeResponse:
    def __init__(self, status_code, text, url="https://example.invalid/"):
        self.status_code = status_code
        self.text = text
        self.url = url


class FakeSession:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.params_requested = []

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        self.params_requested.append(params)
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_contract_success():
    body = '[{"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000"}]'
    session = FakeSession([FakeResponse(200, body)])
    result = fetch_contract("043602", session=session)
    assert result.raw_text == body
    assert result.contract_code == "043602"
    assert session.params_requested[0]["cftc_contract_market_code"] == "043602"
    assert session.params_requested[0]["$order"] == "report_date_as_yyyy_mm_dd ASC"


def test_fetch_contract_client_error_does_not_retry():
    session = FakeSession([FakeResponse(404, "not found")])
    with pytest.raises(CftcApiHTTPError):
        fetch_contract("043602", session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 1


def test_fetch_contract_server_error_retries_then_raises():
    session = FakeSession([FakeResponse(503, "unavailable")])
    with pytest.raises(CftcApiHTTPError):
        fetch_contract("043602", session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3


def test_fetch_contract_connection_error_retries_then_raises():
    session = FakeSession([requests.exceptions.ConnectionError("boom")])
    with pytest.raises(CftcApiConnectionError) as exc_info:
        fetch_contract("043602", session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3
    assert exc_info.value.attempts == 3


def test_fetch_contracts_reuses_one_session(monkeypatch):
    calls = []

    def fake_fetch_contract(contract_code, *, session, **kwargs):
        calls.append((contract_code, session))
        return client_module.FetchedContract(
            contract_code=contract_code, url=f"https://example.invalid/{contract_code}",
            http_status=200, retrieved_at_utc="2026-09-11T00:00:00+00:00", raw_text="[]",
        )

    monkeypatch.setattr(client_module, "fetch_contract", fake_fetch_contract)
    results = fetch_contracts(("042601", "043602"))
    assert [r.contract_code for r in results] == ["042601", "043602"]
    assert calls[0][1] is calls[1][1]
