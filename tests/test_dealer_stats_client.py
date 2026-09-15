"""Tests for the NY Fed Primary Dealer Statistics HTTP client. No real
network calls -- see test_live_network.py for the one that hits the
real API.
"""

from __future__ import annotations

import pytest
import requests

from treasury_auction_stress.data import dealer_stats_client as client_module
from treasury_auction_stress.data.dealer_stats_client import (
    DealerStatsApiConnectionError,
    DealerStatsApiHTTPError,
    fetch_all_legacy_series,
    fetch_all_selected_series,
    fetch_legacy_series,
    fetch_series,
    fetch_series_list,
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
    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.calls = 0
        self.urls_requested: list[str] = []

    def get(self, url, timeout=None):
        self.calls += 1
        self.urls_requested.append(url)
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_series_success_returns_parsed_body():
    body = {"pd": {"timeseries": [{"asofdate": "2013-04-03", "keyid": "PDPOSGS-B", "value": "46324"}]}}
    session = FakeSession([FakeResponse(200, body)])
    series = fetch_series("PDPOSGS-B", session=session)
    assert series.http_status == 200
    assert series.parsed == body
    assert series.keyid == "PDPOSGS-B"
    assert session.calls == 1


def test_fetch_series_client_error_does_not_retry():
    session = FakeSession([FakeResponse(404, {"error": "not found"})])
    with pytest.raises(DealerStatsApiHTTPError) as exc_info:
        fetch_series("BOGUS-KEYID", session=session, max_retries=3, backoff_base_seconds=0)
    assert exc_info.value.status_code == 404
    assert session.calls == 1


def test_fetch_series_server_error_retries_then_raises():
    session = FakeSession([FakeResponse(503, {"error": "unavailable"})])
    with pytest.raises(DealerStatsApiHTTPError):
        fetch_series("PDPOSGS-B", session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3


def test_fetch_series_connection_error_retries_then_raises_clear_error():
    session = FakeSession([requests.exceptions.ConnectionError("boom")])
    with pytest.raises(DealerStatsApiConnectionError) as exc_info:
        fetch_series("PDPOSGS-B", session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 3
    assert exc_info.value.attempts == 3
    assert "boom" in str(exc_info.value)


def test_fetch_series_list_hits_list_endpoint():
    body = {"pd": {"timeseries": [{"seriesbreak": "SBN2024", "keyid": "PDPOSGS-B", "description": "x"}]}}
    session = FakeSession([FakeResponse(200, body)])
    result = fetch_series_list(session=session)
    assert result.parsed == body
    assert session.urls_requested[0].endswith("/list/timeseries.json")


def test_fetch_all_selected_series_reuses_one_session(monkeypatch):
    calls = []

    def fake_fetch_series(keyid, *, session, **kwargs):
        calls.append((keyid, session))
        return client_module.FetchedSeries(
            keyid=keyid,
            url=f"https://example.invalid/{keyid}",
            http_status=200,
            retrieved_at_utc="2026-09-11T00:00:00+00:00",
            raw_text="{}",
            parsed={"pd": {"timeseries": []}},
        )

    monkeypatch.setattr(client_module, "fetch_series", fake_fetch_series)
    results = fetch_all_selected_series(("PDPOSGS-B", "PDPOSGSC-L2"))
    assert [r.keyid for r in results] == ["PDPOSGS-B", "PDPOSGSC-L2"]
    assert calls[0][1] is calls[1][1]  # same session object reused


def test_fetch_legacy_series_hits_the_period_scoped_endpoint():
    body = {"pd": {"timeseries": [{"asofdate": "2001-07-04", "keyid": "PDPUSGTBNOP", "value": "7110"}]}}
    session = FakeSession([FakeResponse(200, body)])
    series = fetch_legacy_series("SBP2013", "PDPUSGTBNOP", session=session)
    assert series.period == "SBP2013"
    assert series.parsed == body
    assert session.urls_requested[0].endswith("/get/SBP2013/timeseries/PDPUSGTBNOP.json")


def test_plain_fetch_series_leaves_period_as_none():
    body = {"pd": {"timeseries": [{"asofdate": "2013-04-03", "keyid": "PDPOSGS-B", "value": "46324"}]}}
    session = FakeSession([FakeResponse(200, body)])
    series = fetch_series("PDPOSGS-B", session=session)
    assert series.period is None


def test_fetch_legacy_series_client_error_does_not_retry():
    session = FakeSession([FakeResponse(404, {"error": "not found"})])
    with pytest.raises(DealerStatsApiHTTPError):
        fetch_legacy_series("SBP2013", "BOGUS", session=session, max_retries=3, backoff_base_seconds=0)
    assert session.calls == 1


def test_fetch_all_legacy_series_reuses_one_session_and_tags_period(monkeypatch):
    calls = []

    def fake_fetch_legacy_series(period, keyid, *, session, **kwargs):
        calls.append((period, keyid, session))
        return client_module.FetchedSeries(
            keyid=keyid,
            url=f"https://example.invalid/{period}/{keyid}",
            http_status=200,
            retrieved_at_utc="2026-09-11T00:00:00+00:00",
            raw_text="{}",
            parsed={"pd": {"timeseries": []}},
            period=period,
        )

    monkeypatch.setattr(client_module, "fetch_legacy_series", fake_fetch_legacy_series)
    results = fetch_all_legacy_series("SBP2013", ("PDPUSGTBNOP", "PDPUSGCS36NOP"))
    assert [r.keyid for r in results] == ["PDPUSGTBNOP", "PDPUSGCS36NOP"]
    assert all(r.period == "SBP2013" for r in results)
    assert calls[0][2] is calls[1][2]  # same session object reused
