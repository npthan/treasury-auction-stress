"""Low-level HTTP client for the Treasury Fiscal Data auctions API.

A "client" here just means: the piece of code responsible for talking
to the outside API over HTTP -- building the request URL, sending it,
and handling network problems -- as opposed to the code that later
makes sense of the data it returns (that's normalize.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from treasury_auction_stress.data.schema import (
    API_BASE_URL,
    AUCTIONS_ENDPOINT,
    MAX_PAGE_SIZE,
)
from treasury_auction_stress.data.time_utils import utc_now_iso

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_READ_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0


class TreasuryApiError(Exception):
    """Base class for every error this client can raise."""


class TreasuryApiHTTPError(TreasuryApiError):
    """The server responded, but with an error status (e.g. bad query param).

    Retrying will not fix this -- the request itself needs to change --
    so the client raises this immediately without retrying.
    """

    def __init__(self, status_code: int, message: str, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {message}")


class TreasuryApiConnectionError(TreasuryApiError):
    """The request never got a response (timeout, DNS failure, etc.),
    even after the bounded number of retries.
    """

    def __init__(self, url: str, attempts: int, last_error: Exception) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Failed to reach {url} after {attempts} attempt(s): {last_error!r}"
        )


@dataclass(frozen=True)
class FetchedPage:
    """One raw HTTP response from the auctions endpoint, unmodified.

    `raw_text` is kept verbatim (not just the parsed dict) so the raw
    artifact written to disk is the actual bytes the server sent, not
    a Python-reserialized approximation of them.
    """

    url: str
    params: dict[str, Any]
    http_status: int
    retrieved_at_utc: str
    raw_text: str
    parsed: dict[str, Any]


def _build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}{endpoint}"


def fetch_page(
    *,
    filter_str: str | None,
    sort: str,
    page_number: int,
    page_size: int,
    base_url: str = API_BASE_URL,
    endpoint: str = AUCTIONS_ENDPOINT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> FetchedPage:
    """Fetch exactly one page of the auctions endpoint.

    Retries a bounded number of times, with exponential backoff, but
    only for transient problems (connection failures, timeouts, and
    HTTP 5xx server errors). A 4xx response (e.g. an invalid page
    size) is raised immediately, since retrying an invalid request
    just repeats the same failure.
    """
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(
            f"page_size must be between 1 and {MAX_PAGE_SIZE}, got {page_size}"
        )

    url = _build_url(base_url, endpoint)
    params: dict[str, Any] = {
        "page[number]": page_number,
        "page[size]": page_size,
        "sort": sort,
    }
    if filter_str:
        params["filter"] = filter_str

    http = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        retrieved_at = utc_now_iso()
        try:
            response = http.get(
                url,
                params=params,
                timeout=(connect_timeout, read_timeout),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise TreasuryApiConnectionError(
                url=url,
                attempts=attempt,
                last_error=exc,
            ) from exc

        if response.status_code >= 500:
            last_error = TreasuryApiHTTPError(
                response.status_code, response.text[:500], response.url
            )
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise last_error

        if response.status_code >= 400:
            # Client error: our request was wrong. Do not retry.
            raise TreasuryApiHTTPError(
                response.status_code, response.text[:500], response.url
            )

        return FetchedPage(
            url=response.url,
            params=dict(params),
            http_status=response.status_code,
            retrieved_at_utc=retrieved_at,
            raw_text=response.text,
            parsed=response.json(),
        )

    # Unreachable in practice: the loop above always returns or raises.
    raise TreasuryApiConnectionError(url=url, attempts=max_retries, last_error=last_error or RuntimeError("unknown"))


def fetch_all_pages(
    *,
    filter_str: str | None,
    sort: str = "auction_date",
    page_size: int = MAX_PAGE_SIZE,
    **fetch_kwargs: Any,
) -> list[FetchedPage]:
    """Fetch every page for a query, following meta.total-pages.

    Uses a single `requests.Session` across pages so the same TCP
    connection can be reused instead of opening a new one per page.
    """
    session = fetch_kwargs.pop("session", None) or requests.Session()
    first = fetch_page(
        filter_str=filter_str,
        sort=sort,
        page_number=1,
        page_size=page_size,
        session=session,
        **fetch_kwargs,
    )
    pages = [first]
    total_pages = int(first.parsed.get("meta", {}).get("total-pages", 1))
    for page_number in range(2, total_pages + 1):
        pages.append(
            fetch_page(
                filter_str=filter_str,
                sort=sort,
                page_number=page_number,
                page_size=page_size,
                session=session,
                **fetch_kwargs,
            )
        )
    return pages
