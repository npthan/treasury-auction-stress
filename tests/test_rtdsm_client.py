from __future__ import annotations

import pytest
import requests

from treasury_auction_stress.data import rtdsm_client as client_module
from treasury_auction_stress.data.rtdsm_client import (
    EXPECTED_CONTENT_TYPE,
    RtdsmApiConnectionError,
    RtdsmApiContentTypeError,
    RtdsmApiHTTPError,
    fetch_variable,
    fetch_variables,
)
from treasury_auction_stress.data.rtdsm_schema import VARIABLE_BY_MNEMONIC

_RUC = VARIABLE_BY_MNEMONIC["RUC"]


class FakeResponse:
    def __init__(self, status_code, content=b"", url="https://example.invalid/ruc.xlsx", content_type=EXPECTED_CONTENT_TYPE):
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self.url = url
        self.headers = {"Content-Type": content_type}


class FakeSession:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_variable_success():
    session = FakeSession([FakeResponse(200, content=b"PK\x03\x04fake-xlsx-bytes")])
    result = fetch_variable(_RUC, session=session)
    assert result.raw_bytes == b"PK\x03\x04fake-xlsx-bytes"
    assert result.mnemonic == "RUC"


def test_fetch_variable_client_error_does_not_retry():
    session = FakeSession([FakeResponse(404, content=b"not found")])
    with pytest.raises(RtdsmApiHTTPError):
        fetch_variable(_RUC, session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 1


def test_fetch_variable_server_error_retries_then_raises():
    session = FakeSession([FakeResponse(503, content=b"unavailable")])
    with pytest.raises(RtdsmApiHTTPError):
        fetch_variable(_RUC, session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3


def test_fetch_variable_connection_error_retries_then_raises():
    session = FakeSession([requests.exceptions.ConnectionError("boom")])
    with pytest.raises(RtdsmApiConnectionError) as exc_info:
        fetch_variable(_RUC, session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3
    assert exc_info.value.attempts == 3


def test_fetch_variable_rejects_unexpected_content_type():
    session = FakeSession([FakeResponse(200, content=b"<html>error</html>", content_type="text/html")])
    with pytest.raises(RtdsmApiContentTypeError):
        fetch_variable(_RUC, session=session)


def test_fetch_variables_reuses_one_session(monkeypatch):
    calls = []

    def fake_fetch_variable(variable, *, session, **kwargs):
        calls.append((variable.mnemonic, session))
        return client_module.FetchedVariable(
            mnemonic=variable.mnemonic, url=variable.xlsx_url, http_status=200,
            content_type=EXPECTED_CONTENT_TYPE, retrieved_at_utc="2026-09-11T00:00:00+00:00", raw_bytes=b"x",
        )

    monkeypatch.setattr(client_module, "fetch_variable", fake_fetch_variable)
    from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY

    results = fetch_variables(VARIABLE_REGISTRY[:2])
    assert [r.mnemonic for r in results] == [v.mnemonic for v in VARIABLE_REGISTRY[:2]]
    assert calls[0][1] is calls[1][1]
