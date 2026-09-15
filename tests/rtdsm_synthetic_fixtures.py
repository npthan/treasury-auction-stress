"""Deterministic, wholly synthetic Philadelphia Fed RTDSM-shaped test
workbooks (Phase 9 acceptance-review remediation).

The two real committed fixtures this module replaces
(`tests/fixtures/rtdsm_ruc_sample.xlsx`, `rtdsm_ipt_sample.xlsx`) were
small but genuine, verbatim excerpts of RTDSM's own copyrighted-status-
ambiguous data (see `docs/data_sources_and_licensing.md` Section 5).
Rather than publish that ambiguity, no real RTDSM value appears
anywhere in this repository's publishable tree: every number below is
fabricated by the deterministic formulas in this file, never copied,
perturbed, rescaled, or derived from any real observation. Only the
STRUCTURAL shape -- column-naming convention, sheet layout, observation-
period grid, vintage labels, and specific edge cases (a missing/blank
cell, a value that differs across two vintages for the same period, a
permanent multi-month gap in the two most recent vintages) -- is
preserved, because that shape is RTDSM's own published *format*
(`gen_doc_GDI.pdf`/`doc_ip.pdf`), not its data.

Tests that need to verify an actual, real-world RTDSM fact (the exact
verified release day against the live calendar, a genuine historical
revision, a genuine reporting gap) now do so against live-fetched data
in `tests/test_live_network.py`, not against these synthetic fixtures --
see that file's RTDSM section.

Calling either builder function twice returns byte-identical output
(no wall-clock, no randomness -- `_FIXED_TIMESTAMP` pins the workbook's
own metadata so this holds at the byte level, not just the parsed-data
level); `tests/test_rtdsm_synthetic_fixtures.py` asserts this
determinism directly.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

from openpyxl import Workbook

_FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)  # pins workbook metadata for byte-determinism
_SYNTHETIC_MARKER = "SYNTHETIC RTDSM-shaped test data -- no real observation; see tests/rtdsm_synthetic_fixtures.py"


def _write_workbook(sheet_name: str, headers: list[str], rows: list[list[object]]) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    wb.properties.title = _SYNTHETIC_MARKER
    wb.properties.creator = "synthetic-fixture-generator"
    wb.properties.created = _FIXED_TIMESTAMP
    wb.properties.modified = _FIXED_TIMESTAMP
    ws.append(headers)
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _monthly_period_labels(start_year: int, start_month: int, count: int) -> list[str]:
    labels = []
    y, m = start_year, start_month
    for _ in range(count):
        labels.append(f"{y}:{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return labels


def _ruc_value(t: int) -> float:
    """A smooth, obviously-fabricated synthetic ramp -- not derived
    from any real unemployment-rate reading."""
    return round(5.0 + 0.1 * t, 1)


def build_ruc_synthetic_workbook() -> bytes:
    """A quarterly-vintage, monthly-observation workbook shaped like
    RTDSM's real RUC (unemployment rate) file: `DATE` column
    `YYYY:MM`, 86 months (2019:06..2026:07), 5 vintage columns.
    Preserves 3 structural edge cases with fabricated numbers:
    - RUC20Q2 ends at its own vintage's expected last period (2020:04),
      leaving 2020:05 onward blank (an ordinary, non-gap vintage edge).
    - RUC21Q1 (a later vintage) reports the SAME 2020:04 period with a
      DIFFERENT fabricated value than RUC20Q2 did -- a revision-across-
      vintages case.
    - RUC25Q4 and RUC26Q1 both stop short of their own expected last
      period (2025:08 instead of the expected 2025:10) -- a permanent-
      gap-in-two-consecutive-vintages case. RUC26Q3 has no such gap.
    """
    periods = _monthly_period_labels(2019, 6, 86)  # 2019:06 .. 2026:07
    idx = {p: i for i, p in enumerate(periods)}
    headers = ["DATE", "RUC20Q2", "RUC21Q1", "RUC25Q4", "RUC26Q1", "RUC26Q3"]

    def cell(col: str, t: int) -> float | None:
        last_ok = {
            "RUC20Q2": idx["2020:04"],
            "RUC21Q1": idx["2021:01"],
            "RUC25Q4": idx["2025:08"],
            "RUC26Q1": idx["2025:08"],
            "RUC26Q3": len(periods) - 1,
        }[col]
        if t > last_ok:
            return None
        if col == "RUC21Q1" and t == idx["2020:04"]:
            return 6.5  # deliberate revision: differs from RUC20Q2's value at the same period
        return _ruc_value(t)

    rows = [[periods[t], *(cell(c, t) for c in headers[1:])] for t in range(len(periods))]
    return _write_workbook("ruc", headers, rows)


def _ipt_value(t: int) -> float:
    """A smooth, obviously-fabricated synthetic ramp -- not derived
    from any real industrial-production reading."""
    return round(90.0 + 0.2 * t, 1)


def build_ipt_synthetic_workbook() -> bytes:
    """A monthly-vintage, monthly-observation workbook shaped like
    RTDSM's real IPT (industrial production) file: `DATE` column
    `YYYY:MM`, 80 months (2019:12..2026:07), 4 vintage columns.
    Preserves 2 structural edge cases with fabricated numbers:
    - IPT20M3 ends at its own vintage's expected last period (2020:02),
      leaving 2020:03 onward blank -- so a later vintage's report of
      2020:04 (below) is genuinely absent from this earlier vintage.
    - IPT20M6 (a later vintage) DOES report 2020:04, with a deliberate,
      visibly-off-trend fabricated value (a structural analog of "an
      anomaly not yet visible in an earlier vintage, fully visible in a
      later one" -- never a real economic reading).
    """
    periods = _monthly_period_labels(2019, 12, 80)  # 2019:12 .. 2026:07
    idx = {p: i for i, p in enumerate(periods)}
    headers = ["DATE", "IPT20M3", "IPT20M6", "IPT26M7", "IPT26M8"]

    def cell(col: str, t: int) -> float | None:
        last_ok = {
            "IPT20M3": idx["2020:02"],
            "IPT20M6": idx["2020:05"],
            "IPT26M7": idx["2026:06"],
            "IPT26M8": len(periods) - 1,
        }[col]
        if t > last_ok:
            return None
        if col == "IPT20M6" and t == idx["2020:04"]:
            return 85.0  # deliberate off-trend fabricated value, not a real reading
        return _ipt_value(t)

    rows = [[periods[t], *(cell(c, t) for c in headers[1:])] for t in range(len(periods))]
    return _write_workbook("ipt", headers, rows)
