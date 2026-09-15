"""Tests for the Treasury Fiscal Data API HTTP client.

None of these tests make a real network call -- they use fake
`requests.Session`-like objects so retry/backoff/error-handling logic
can be tested deterministically and offline. See test_live_network.py
for the one test that actually hits the real API.
"""

from __future__ import annotations

import pytest
import requests

from treasury_auction_stress.data import client as client_module
from treasury_auction_stress.data.client import (
    TreasuryApiConnectionError,
    TreasuryApiHTTPError,
    fetch_all_pages,
    fetch_page,
)


class FakeResponse:
    def __init__(self, status_code: int, body: dict, url: str = "https://example.invalid/") -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)
        self.url = url

    def json(self) -> dict:
        return self._body


class FakeSession:
    """A fake requests.Session whose .get() replays a scripted sequence
    of responses/exceptions, one per call, and counts calls made.
    """

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_page_success_returns_parsed_body(sample_page_body):
    session = FakeSession([FakeResponse(200, sample_page_body)])
    page = fetch_page(
        filter_str="auction_date:gte:2010-01-01",
        sort="auction_date",
        page_number=1,
        page_size=10,
        session=session,
    )
    assert page.http_status == 200
    assert page.parsed == sample_page_body
    assert session.calls == 1


def test_fetch_page_rejects_out_of_range_page_size():
    with pytest.raises(ValueError):
        fetch_page(filter_str=None, sort="auction_date", page_number=1, page_size=10001)


def test_fetch_page_client_error_does_not_retry():
    session = FakeSession([FakeResponse(400, {"error": "Invalid Query Param"})])
    with pytest.raises(TreasuryApiHTTPError) as exc_info:
        fetch_page(
            filter_str=None,
            sort="auction_date",
            page_number=1,
            page_size=10,
            session=session,
            max_retries=3,
            backoff_base_seconds=0,
        )
    assert exc_info.value.status_code == 400
    assert session.calls == 1  # no retry on a 4xx


def test_fetch_page_server_error_retries_then_raises():
    session = FakeSession([FakeResponse(503, {"error": "unavailable"})])
    with pytest.raises(TreasuryApiHTTPError):
        fetch_page(
            filter_str=None,
            sort="auction_date",
            page_number=1,
            page_size=10,
            session=session,
            max_retries=3,
            backoff_base_seconds=0,
        )
    assert session.calls == 3  # retried up to the bound, then gave up


def test_fetch_page_connection_error_retries_then_raises_clear_error():
    session = FakeSession([requests.exceptions.ConnectionError("boom")])
    with pytest.raises(TreasuryApiConnectionError) as exc_info:
        fetch_page(
            filter_str=None,
            sort="auction_date",
            page_number=1,
            page_size=10,
            session=session,
            max_retries=3,
            backoff_base_seconds=0,
        )
    assert session.calls == 3
    assert exc_info.value.attempts == 3
    assert "boom" in str(exc_info.value)


def test_fetch_all_pages_merges_multiple_pages(monkeypatch, two_page_fixture):
    page1_body = two_page_fixture["page1"]
    page2_body = two_page_fixture["page2"]

    def fake_fetch_page(*, page_number, **kwargs):
        body = page1_body if page_number == 1 else page2_body
        return client_module.FetchedPage(
            url=f"https://example.invalid/?page={page_number}",
            params={},
            http_status=200,
            retrieved_at_utc="2026-09-10T00:00:00+00:00",
            raw_text=str(body),
            parsed=body,
        )

    monkeypatch.setattr(client_module, "fetch_page", fake_fetch_page)
    pages = fetch_all_pages(filter_str="auction_date:gte:2010-01-01", sort="auction_date", page_size=4)

    assert len(pages) == 2
    assert pages[0].parsed == page1_body
    assert pages[1].parsed == page2_body
