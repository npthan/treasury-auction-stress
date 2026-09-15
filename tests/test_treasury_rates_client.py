from __future__ import annotations

import pytest
import requests

from treasury_auction_stress.data import treasury_rates_client as client_module
from treasury_auction_stress.data.treasury_rates_client import (
    TreasuryRatesApiConnectionError,
    TreasuryRatesApiContentTypeError,
    TreasuryRatesApiHTTPError,
    fetch_year,
    fetch_years,
)


class FakeResponse:
    def __init__(self, status_code, text, url="https://example.invalid/", content_type="text/csv; charset=UTF-8"):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.headers = {"Content-Type": content_type}


class FakeSession:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.urls_requested = []

    def get(self, url, timeout=None):
        self.calls += 1
        self.urls_requested.append(url)
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_year_success():
    body = 'Date,"1 Mo"\n01/02/2024,4.00\n'
    session = FakeSession([FakeResponse(200, body)])
    result = fetch_year(2024, session=session)
    assert result.raw_text == body
    assert result.year == 2024
    assert "2024" in session.urls_requested[0]


def test_fetch_year_client_error_does_not_retry():
    session = FakeSession([FakeResponse(404, "not found")])
    with pytest.raises(TreasuryRatesApiHTTPError):
        fetch_year(2024, session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 1


def test_fetch_year_server_error_retries_then_raises():
    session = FakeSession([FakeResponse(503, "unavailable")])
    with pytest.raises(TreasuryRatesApiHTTPError):
        fetch_year(2024, session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3


def test_fetch_year_connection_error_retries_then_raises():
    session = FakeSession([requests.exceptions.ConnectionError("boom")])
    with pytest.raises(TreasuryRatesApiConnectionError) as exc_info:
        fetch_year(2024, session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3
    assert exc_info.value.attempts == 3


def test_fetch_year_rejects_unexpected_content_type():
    session = FakeSession([FakeResponse(200, "<html>error</html>", content_type="text/html")])
    with pytest.raises(TreasuryRatesApiContentTypeError):
        fetch_year(2024, session=session)


def test_fetch_years_reuses_one_session(monkeypatch):
    calls = []

    def fake_fetch_year(year, *, session, **kwargs):
        calls.append((year, session))
        return client_module.FetchedYear(
            year=year, url=f"https://example.invalid/{year}", http_status=200,
            content_type="text/csv", retrieved_at_utc="2026-09-11T00:00:00+00:00", raw_text="Date\n",
        )

    monkeypatch.setattr(client_module, "fetch_year", fake_fetch_year)
    results = fetch_years((2023, 2024))
    assert [r.year for r in results] == [2023, 2024]
    assert calls[0][1] is calls[1][1]
