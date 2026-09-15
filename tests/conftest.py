import json
from pathlib import Path

import pytest
from rtdsm_synthetic_fixtures import (
    build_ipt_synthetic_workbook,
    build_ruc_synthetic_workbook,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def load_text_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def sample_page_body() -> dict:
    return load_fixture("sample_auctions_page.json")


@pytest.fixture
def two_page_fixture() -> dict:
    return load_fixture("two_page_auctions_fixture.json")


@pytest.fixture
def empty_page_body() -> dict:
    return load_fixture("empty_auctions_page.json")


@pytest.fixture
def dealer_stats_raw_payload() -> dict:
    """A real, small excerpt (2021-12-01 through 2022-02-16) of every
    selected NY Fed Primary Dealer Statistics series, captured live
    2026-09-11. Deliberately spans: two Fed-holiday-adjacent Thursdays
    (Christmas/New Year observed dates), the verified 2022-01-05
    long-maturity-bucket regime start, and real `"*"` (missing) weeks
    in the financing series. See tests/fixtures/README.md.
    """
    return load_fixture("dealer_stats_sample_series.json")


@pytest.fixture
def dealer_stats_series_list_body() -> dict:
    """A real excerpt of `/api/pd/list/timeseries.json`, containing
    every selected keyid plus two deliberately-unselected ones."""
    return load_fixture("dealer_stats_series_list_sample.json")


@pytest.fixture
def dealer_stats_historical_extension_payload() -> dict:
    """Phase 3 acceptance review: a real excerpt (2013-02-01 through
    2013-05-01) spanning the verified April-2013 schema-redesign
    boundary, captured live 2026-09-11 via both fetch mechanisms: the
    5 legacy (`period="SBP2013"`) keyids fetched through the
    period-scoped endpoint, and the 2 directly-extended concepts'
    modern keyids (`PDPOSGS-B`, `PDPOSGSC-G3L6`) fetched through the
    plain endpoint, for the same window. See tests/fixtures/README.md.
    """
    return load_fixture("dealer_stats_historical_extension_sample.json")


@pytest.fixture
def treasury_rates_2024_csv() -> str:
    """10 real rows, modern 13-maturity schema, captured live 2026-09-11."""
    return load_text_fixture("treasury_rates_2024_sample.csv")


@pytest.fixture
def treasury_rates_2002_suspension_csv() -> str:
    """4 real rows spanning the verified 2002-02-19 start of the
    30-Year issuance suspension (30 Yr present-but-empty from that
    date, populated before it)."""
    return load_text_fixture("treasury_rates_2002_suspension_sample.csv")


@pytest.fixture
def treasury_rates_1990_csv() -> str:
    """5 real rows, older 9-maturity schema (no 1 Mo/2 Mo/4 Mo/20 Yr)."""
    return load_text_fixture("treasury_rates_1990_sample.csv")


@pytest.fixture
def treasury_rates_2026_csv() -> str:
    """5 real rows, current 14-maturity schema including 1.5 Month."""
    return load_text_fixture("treasury_rates_2026_sample.csv")


@pytest.fixture
def rtdsm_ruc_sample_bytes() -> bytes:
    """A wholly SYNTHETIC quarterly-vintage RUC-shaped workbook (Phase 9
    acceptance-review remediation -- this project no longer commits or
    generates any real RTDSM observation; see
    tests/rtdsm_synthetic_fixtures.py's module docstring and
    tests/fixtures/README.md). Structurally shaped like RTDSM's real RUC
    file (86 monthly observation periods, 5 vintage columns) with 3
    fabricated edge cases: an ordinary vintage-coverage boundary, a
    revision-across-vintages case, and a permanent gap shared by the two
    most recent vintages. No value here is a real unemployment reading.
    Deterministic: calling the builder twice returns identical bytes.
    """
    return build_ruc_synthetic_workbook()


@pytest.fixture
def rtdsm_ipt_sample_bytes() -> bytes:
    """A wholly SYNTHETIC monthly-vintage IPT-shaped workbook (Phase 9
    acceptance-review remediation -- see rtdsm_ruc_sample_bytes above
    and tests/rtdsm_synthetic_fixtures.py's module docstring).
    Structurally shaped like RTDSM's real IPT file (80 monthly
    observation periods, 4 vintage columns) with a fabricated
    not-yet-visible-in-an-earlier-vintage / visible-in-a-later-vintage
    edge case. No value here is a real industrial-production reading.
    Deterministic: calling the builder twice returns identical bytes.
    """
    return build_ipt_synthetic_workbook()


@pytest.fixture
def cftc_tff_sample_payload() -> dict:
    """Real CFTC TFF Futures Only rows, captured live 2026-09-11:
    5 UST 10Y NOTE (043602) observation weeks deliberately spanning an
    ordinary week (2024-01-02), the 2023 ION-incident disruption window
    (2023-01-31, 2023-02-07), and the 2025 shutdown window (2025-09-30,
    2025-10-07); plus 1 UST 2Y NOTE (042601) row for multi-contract
    coverage. See tests/fixtures/README.md."""
    return load_fixture("cftc_tff_sample_payload.json")
