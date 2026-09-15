"""Low-level HTTP client for the New York Fed Primary Dealer Statistics
API (`treasury_auction_stress.data.dealer_stats_schema`).

Structurally a near-twin of `treasury_auction_stress.data.client` (the
Fiscal Data auctions client): same bounded-retry, bounded-backoff,
explicit-timeout policy, same "4xx never retries, 5xx/connection
errors do" split. Kept as a separate module (not a shared base class)
because the two APIs have different pagination models -- this one has
none: `/get/{keyid}.json` always returns a series' entire history in
one response, with no page/date-range parameter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from treasury_auction_stress.data.dealer_stats_schema import (
    API_BASE_URL,
    GET_ENDPOINT_TEMPLATE,
    LEGACY_GET_ENDPOINT_TEMPLATE,
    LIST_ENDPOINT,
)
from treasury_auction_stress.data.time_utils import utc_now_iso

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_READ_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0


class DealerStatsApiError(Exception):
    """Base class for every error this client can raise."""


class DealerStatsApiHTTPError(DealerStatsApiError):
    """The server responded with an error status. Not retried -- the
    request itself would need to change.
    """

    def __init__(self, status_code: int, message: str, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {message}")


class DealerStatsApiConnectionError(DealerStatsApiError):
    """The request never got a response, even after bounded retries."""

    def __init__(self, url: str, attempts: int, last_error: Exception) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Failed to reach {url} after {attempts} attempt(s): {last_error!r}"
        )


@dataclass(frozen=True)
class FetchedSeries:
    """One raw HTTP response, unmodified, for a single `keyid`.

    `period` is `None` for a plain `/get/{keyid}.json` fetch (the
    current-API keyids this project originally selected, and the one
    discontinued-but-still-fetchable long-bucket keyid), or a schema-
    period key (e.g. `"SBP2013"`) when fetched via the legacy,
    period-scoped endpoint -- added during the Phase 3 acceptance
    review. See `dealer_stats_schema.py`'s historical-extension
    section.
    """

    keyid: str
    url: str
    http_status: int
    retrieved_at_utc: str
    raw_text: str
    parsed: dict[str, Any]
    period: str | None = None


@dataclass(frozen=True)
class FetchedSeriesList:
    """One raw HTTP response from the `/list/timeseries.json` endpoint."""

    url: str
    http_status: int
    retrieved_at_utc: str
    raw_text: str
    parsed: dict[str, Any]


def _get_with_retries(
    url: str,
    *,
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    backoff_base_seconds: float,
    session: requests.Session,
) -> tuple[int, str, dict[str, Any], str]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        retrieved_at = utc_now_iso()
        try:
            response = session.get(url, timeout=(connect_timeout, read_timeout))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise DealerStatsApiConnectionError(
                url=url, attempts=attempt, last_error=exc
            ) from exc

        if response.status_code >= 500:
            last_error = DealerStatsApiHTTPError(
                response.status_code, response.text[:500], url
            )
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise last_error

        if response.status_code >= 400:
            raise DealerStatsApiHTTPError(response.status_code, response.text[:500], url)

        return response.status_code, retrieved_at, response.json(), response.text

    raise DealerStatsApiConnectionError(
        url=url, attempts=max_retries, last_error=last_error or RuntimeError("unknown")
    )


def fetch_series_list(
    *,
    base_url: str = API_BASE_URL,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> FetchedSeriesList:
    """Fetch the list of every currently-active series (used for
    schema-drift detection against `dealer_stats_schema.SELECTED_SERIES`,
    not for the actual data pull).
    """
    url = f"{base_url.rstrip('/')}{LIST_ENDPOINT}"
    http = session or requests.Session()
    status, retrieved_at, parsed, raw_text = _get_with_retries(
        url,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        session=http,
    )
    return FetchedSeriesList(
        url=url,
        http_status=status,
        retrieved_at_utc=retrieved_at,
        raw_text=raw_text,
        parsed=parsed,
    )


def fetch_series(
    keyid: str,
    *,
    base_url: str = API_BASE_URL,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> FetchedSeries:
    """Fetch one series' complete history."""
    url = f"{base_url.rstrip('/')}{GET_ENDPOINT_TEMPLATE.format(keyid=keyid)}"
    http = session or requests.Session()
    status, retrieved_at, parsed, raw_text = _get_with_retries(
        url,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        session=http,
    )
    return FetchedSeries(
        keyid=keyid,
        url=url,
        http_status=status,
        retrieved_at_utc=retrieved_at,
        raw_text=raw_text,
        parsed=parsed,
    )


def fetch_legacy_series(
    period: str,
    keyid: str,
    *,
    base_url: str = API_BASE_URL,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> FetchedSeries:
    """Fetch one series' complete history from a specific historical
    schema period, via the period-scoped legacy endpoint (discovered
    by inspecting the GSDS UI's own JS bundle -- see
    `dealer_stats_schema.LEGACY_GET_ENDPOINT_TEMPLATE`). Used for
    pre-2013 data; the plain `fetch_series` above cannot reach it.
    """
    url = f"{base_url.rstrip('/')}{LEGACY_GET_ENDPOINT_TEMPLATE.format(period=period, keyid=keyid)}"
    http = session or requests.Session()
    status, retrieved_at, parsed, raw_text = _get_with_retries(
        url,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        session=http,
    )
    return FetchedSeries(
        keyid=keyid,
        url=url,
        http_status=status,
        retrieved_at_utc=retrieved_at,
        raw_text=raw_text,
        parsed=parsed,
        period=period,
    )


def fetch_all_selected_series(
    keyids: tuple[str, ...],
    **fetch_kwargs: Any,
) -> list[FetchedSeries]:
    """Fetch every selected series, reusing one `requests.Session`."""
    session = fetch_kwargs.pop("session", None) or requests.Session()
    return [fetch_series(keyid, session=session, **fetch_kwargs) for keyid in keyids]


def fetch_all_legacy_series(
    period: str,
    keyids: tuple[str, ...],
    **fetch_kwargs: Any,
) -> list[FetchedSeries]:
    """Fetch every given keyid from one historical schema period,
    reusing one `requests.Session`.
    """
    session = fetch_kwargs.pop("session", None) or requests.Session()
    return [fetch_legacy_series(period, keyid, session=session, **fetch_kwargs) for keyid in keyids]
