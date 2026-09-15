"""Low-level HTTP client for the Philadelphia Fed's RTDSM xlsx
workbooks (`philadelphiafed.org`, never FRED/ALFRED).

Same bounded-retry, bounded-backoff, explicit-timeout, and
content-type-verification policy as this project's other clients
(`treasury_rates_client.py`, `cftc_client.py`) -- except the payload
here is **binary** (an .xlsx workbook), not text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from treasury_auction_stress.data.rtdsm_schema import RtdsmVariable
from treasury_auction_stress.data.time_utils import utc_now_iso

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_READ_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
EXPECTED_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class RtdsmApiError(Exception):
    pass


class RtdsmApiHTTPError(RtdsmApiError):
    def __init__(self, status_code: int, message: str, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {message}")


class RtdsmApiConnectionError(RtdsmApiError):
    def __init__(self, url: str, attempts: int, last_error: Exception) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Failed to reach {url} after {attempts} attempt(s): {last_error!r}")


class RtdsmApiContentTypeError(RtdsmApiError):
    def __init__(self, content_type: str, url: str) -> None:
        self.content_type = content_type
        self.url = url
        super().__init__(f"Unexpected content-type {content_type!r} from {url} -- expected an xlsx workbook")


@dataclass(frozen=True)
class FetchedVariable:
    """One raw HTTP response, unmodified, for a single RTDSM variable."""

    mnemonic: str
    url: str
    http_status: int
    content_type: str
    retrieved_at_utc: str
    raw_bytes: bytes


def fetch_variable(
    variable: RtdsmVariable,
    *,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> FetchedVariable:
    """Fetch one RTDSM variable's complete vintage-history workbook."""
    http = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        retrieved_at = utc_now_iso()
        try:
            response = http.get(variable.xlsx_url, timeout=(connect_timeout, read_timeout))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise RtdsmApiConnectionError(url=variable.xlsx_url, attempts=attempt, last_error=exc) from exc

        if response.status_code >= 500:
            last_error = RtdsmApiHTTPError(response.status_code, response.text[:500], response.url)
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise last_error

        if response.status_code >= 400:
            raise RtdsmApiHTTPError(response.status_code, response.text[:500], response.url)

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        if content_type != EXPECTED_CONTENT_TYPE:
            raise RtdsmApiContentTypeError(content_type, response.url)

        return FetchedVariable(
            mnemonic=variable.mnemonic,
            url=response.url,
            http_status=response.status_code,
            content_type=content_type,
            retrieved_at_utc=retrieved_at,
            raw_bytes=response.content,
        )

    raise RtdsmApiConnectionError(url=variable.xlsx_url, attempts=max_retries, last_error=last_error or RuntimeError("unknown"))


def fetch_variables(variables: tuple[RtdsmVariable, ...], **fetch_kwargs) -> list[FetchedVariable]:
    session = fetch_kwargs.pop("session", None) or requests.Session()
    return [fetch_variable(v, session=session, **fetch_kwargs) for v in variables]
