"""Low-level HTTP client for the U.S. Treasury Daily Par Yield Curve
Rates CSV export (`treasury_auction_stress.data.treasury_rates_schema`).

Same bounded-retry, bounded-backoff, explicit-timeout policy as the
other two clients in this project (`data.client`, `data.dealer_stats_client`).
One request = one full calendar year of daily rates (verified: this is
the source's own natural pagination unit, not a choice this project
imposed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from treasury_auction_stress.data.time_utils import utc_now_iso
from treasury_auction_stress.data.treasury_rates_schema import CSV_URL_TEMPLATE

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_READ_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0


class TreasuryRatesApiError(Exception):
    """Base class for every error this client can raise."""


class TreasuryRatesApiHTTPError(TreasuryRatesApiError):
    def __init__(self, status_code: int, message: str, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {message}")


class TreasuryRatesApiConnectionError(TreasuryRatesApiError):
    def __init__(self, url: str, attempts: int, last_error: Exception) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Failed to reach {url} after {attempts} attempt(s): {last_error!r}")


class TreasuryRatesApiContentTypeError(TreasuryRatesApiError):
    """The response's Content-Type header was not a CSV/text type --
    the endpoint may have returned an HTML error page instead of data.
    """

    def __init__(self, content_type: str, url: str) -> None:
        self.content_type = content_type
        self.url = url
        super().__init__(f"Unexpected Content-Type {content_type!r} from {url}")


@dataclass(frozen=True)
class FetchedYear:
    """One raw HTTP response, unmodified, for a single calendar year."""

    year: int
    url: str
    http_status: int
    content_type: str
    retrieved_at_utc: str
    raw_text: str


def fetch_year(
    year: int,
    *,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> FetchedYear:
    """Fetch one calendar year's daily par yield curve CSV."""
    url = CSV_URL_TEMPLATE.format(year=year)
    http = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        retrieved_at = utc_now_iso()
        try:
            response = http.get(url, timeout=(connect_timeout, read_timeout))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise TreasuryRatesApiConnectionError(url=url, attempts=attempt, last_error=exc) from exc

        if response.status_code >= 500:
            last_error = TreasuryRatesApiHTTPError(response.status_code, response.text[:500], url)
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise last_error

        if response.status_code >= 400:
            raise TreasuryRatesApiHTTPError(response.status_code, response.text[:500], url)

        content_type = response.headers.get("Content-Type", "")
        if not any(token in content_type.lower() for token in ("csv", "text/plain", "text/csv")):
            raise TreasuryRatesApiContentTypeError(content_type, url)

        return FetchedYear(
            year=year,
            url=response.url,
            http_status=response.status_code,
            content_type=content_type,
            retrieved_at_utc=retrieved_at,
            raw_text=response.text,
        )

    raise TreasuryRatesApiConnectionError(url=url, attempts=max_retries, last_error=last_error or RuntimeError("unknown"))


def fetch_years(years: tuple[int, ...], **fetch_kwargs) -> list[FetchedYear]:
    """Fetch every given year, reusing one `requests.Session`."""
    session = fetch_kwargs.pop("session", None) or requests.Session()
    return [fetch_year(year, session=session, **fetch_kwargs) for year in years]
