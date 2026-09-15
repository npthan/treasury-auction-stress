"""Low-level HTTP client for CFTC's TFF Futures Only Socrata dataset
(`treasury_auction_stress.data.cftc_schema.TFF_FUTURES_ONLY_DATASET_ID`).

Same bounded-retry, bounded-backoff, explicit-timeout policy as this
project's other clients. One request = one contract's complete
history (Socrata's `$limit`/`$where` query parameters, no auth token
required at this project's request volume).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from treasury_auction_stress.data.cftc_schema import (
    SOCRATA_BASE_URL,
    TFF_FUTURES_ONLY_DATASET_ID,
)
from treasury_auction_stress.data.time_utils import utc_now_iso

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_READ_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_LIMIT = 5000  # comfortably above any single contract's full row count (verified max ~1057)


class CftcApiError(Exception):
    pass


class CftcApiHTTPError(CftcApiError):
    def __init__(self, status_code: int, message: str, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {message}")


class CftcApiConnectionError(CftcApiError):
    def __init__(self, url: str, attempts: int, last_error: Exception) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Failed to reach {url} after {attempts} attempt(s): {last_error!r}")


@dataclass(frozen=True)
class FetchedContract:
    """One raw HTTP response, unmodified, for a single contract code."""

    contract_code: str
    url: str
    http_status: int
    retrieved_at_utc: str
    raw_text: str


def fetch_contract(
    contract_code: str,
    *,
    dataset_id: str = TFF_FUTURES_ONLY_DATASET_ID,
    limit: int = DEFAULT_LIMIT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> FetchedContract:
    """Fetch one contract's complete TFF Futures Only history, sorted
    by report date ascending.
    """
    url = f"{SOCRATA_BASE_URL}/resource/{dataset_id}.json"
    params = {
        "cftc_contract_market_code": contract_code,
        "$limit": limit,
        "$order": "report_date_as_yyyy_mm_dd ASC",
    }
    http = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        retrieved_at = utc_now_iso()
        try:
            response = http.get(url, params=params, timeout=(connect_timeout, read_timeout))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise CftcApiConnectionError(url=url, attempts=attempt, last_error=exc) from exc

        if response.status_code >= 500:
            last_error = CftcApiHTTPError(response.status_code, response.text[:500], response.url)
            if attempt < max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            raise last_error

        if response.status_code >= 400:
            raise CftcApiHTTPError(response.status_code, response.text[:500], response.url)

        return FetchedContract(
            contract_code=contract_code,
            url=response.url,
            http_status=response.status_code,
            retrieved_at_utc=retrieved_at,
            raw_text=response.text,
        )

    raise CftcApiConnectionError(url=url, attempts=max_retries, last_error=last_error or RuntimeError("unknown"))


def fetch_contracts(contract_codes: tuple[str, ...], **fetch_kwargs) -> list[FetchedContract]:
    session = fetch_kwargs.pop("session", None) or requests.Session()
    return [fetch_contract(code, session=session, **fetch_kwargs) for code in contract_codes]
