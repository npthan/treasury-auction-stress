# Test fixtures

These files are small, hand-picked excerpts of **real** responses from
the live Treasury Fiscal Data auctions API
(`https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query`),
captured on 2026-09-10. They are committed to the repository so the
test suite can run fully offline and deterministically.

- `sample_auctions_page.json` -- one API-response-shaped page
  containing 6 real auction records, chosen to cover: a nominal
  2-year new issue, a nominal 10-year reopening, a Treasury Bill, a
  TIPS note, a 2-year FRN, and a pending 30-year bond auction whose
  result fields are still `null` (auction date = the day this fixture
  was captured).
- `two_page_auctions_fixture.json` -- the same 6 records split across
  two pages (`page1`: first 4, `page2`: last 2), for testing
  pagination-merging logic without needing a live multi-page query.
- `empty_auctions_page.json` -- a structurally valid page with zero
  records, for testing the empty-response case.

The `meta`/`links` envelopes were reconstructed to be internally
consistent for a fixture of this size (e.g. `total-count` reflects the
6 records actually present, not the full live dataset's count); the
`data` records themselves are verbatim real values, not fabricated.

## Phase 3: NY Fed Primary Dealer Statistics fixtures

- `dealer_stats_sample_series.json` -- a real excerpt (weeks
  2021-12-01 through 2022-02-16) of every one of this project's 16
  selected series, captured live on 2026-09-11 from
  `https://markets.newyorkfed.org/api/pd/get/{keyid}.json`, already
  shaped as this project's raw-artifact payload (see
  `dealer_stats_raw_store.build_raw_payload`). Chosen specifically to
  cover, with **real, unmodified values** (no fabricated numbers):
  - Two ordinary weekly releases immediately adjacent to the
    Christmas/New Year federal holidays (observed 2021-12-24 and
    2021-12-31, both Fridays that year -- neither holiday happens to
    fall on the Thursday nominal release day itself in this
    particular window). The Thursday-holiday shift case itself
    (`compute_publication_dates` advancing past a holiday that *does*
    land on the nominal Thursday, e.g. Thanksgiving) is exercised with
    real Federal Reserve holiday-calendar dates directly in
    `tests/test_dealer_stats_normalize.py` and
    `tests/test_time_utils.py`, not via this fixture.
  - The verified 2022-01-05 start of `PDPOSGSC-G11L21` and
    `PDPOSGSC-G21` (11-21y / >21y position buckets): 12 weeks of
    history for every other series in this window, but only 7 for
    these two -- a real (not synthetic) demonstration of the
    documented schema-regime break.
  - Real, scattered `"*"` (source-missing) weeks in
    `PDSOOS-UTSETTOT` (2021-12-01, 2021-12-08, 2021-12-15, 2021-12-22,
    2022-01-19, 2022-01-26).
- `dealer_stats_series_list_sample.json` -- a real excerpt of
  `/api/pd/list/timeseries.json`: all 16 selected keyids' entries plus
  two deliberately-unselected ones (`PDTIPSTOT`, `PDTRGST-TOT` -- the
  latter has an unresolved name/description/value ambiguity documented
  in `dealer_stats_schema.py` and is never fetched for data).

### Phase 3 acceptance review: pre-2013 historical extension fixture

- `dealer_stats_historical_extension_sample.json` -- a real excerpt
  (weeks 2013-02-01 through 2013-05-01) spanning the verified
  April-2013 FR 2004 schema-redesign boundary, captured live
  2026-09-11 through **both** fetch mechanisms this project uses:
  - The 5 legacy keyids (`PDPUSGTBNOP`, `PDPUSGCS36NOP`,
    `PDPUSGCS3LNOP`, `PDPUSGCS611NOP`, `PDPUSGCSM11NOP`), each tagged
    `"period": "SBP2013"`, fetched via the period-scoped legacy
    endpoint (`/api/pd/get/SBP2013/timeseries/{keyid}.json`) -- 8 real
    weekly observations each, ending exactly 2013-03-27.
  - The modern keyids for the two directly-extended concepts
    (`PDPOSGS-B`, `PDPOSGSC-G3L6`), tagged `"period": null`, fetched
    via the plain endpoint -- 5 real weekly observations each,
    starting exactly 2013-04-03 (no gap, no overlap with the legacy
    side above).
  - Deliberately does **not** include the discontinued
    `PDPOSGSC-G11` middle-piece series (its own regime starts
    2013-04-03 too, well-covered by other tests) or any of the 16
    originally-selected current-API series (already covered by
    `dealer_stats_sample_series.json`) -- this fixture exists
    specifically to exercise the (keyid, period) metadata-resolution
    logic across the legacy/modern boundary with real values, not to
    duplicate coverage that already exists.

### Phase 4A: Treasury Daily Par Yield Curve fixtures

Real excerpts of the official CSV export
(`https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve...`),
captured live 2026-09-11, chosen for specific verified schema/
missingness properties:

- `treasury_rates_2024_sample.csv` -- 10 real rows, the "modern"
  13-maturity schema (no `1.5 Month` yet), all values present.
- `treasury_rates_2002_suspension_sample.csv` -- 4 real rows spanning
  the exact verified start of the 30-Year issuance suspension: `30 Yr`
  present and non-empty through 2002-02-15, present but **empty**
  starting 2002-02-19 -- a genuine, source-native missing-value case.
- `treasury_rates_1990_sample.csv` -- 5 real rows on the older,
  9-maturity schema (no `1 Mo`, `2 Mo`, `4 Mo`, `20 Yr` columns at
  all) -- exercises schema evolution, not just missing values within
  an existing column.
- `treasury_rates_2026_sample.csv` -- 5 real rows on the current,
  14-maturity schema including `1.5 Month` (introduced 2025-02-18).

### Phase 4B: CFTC TFF Futures Only fixture

- `cftc_tff_sample_payload.json` -- real CFTC TFF Futures Only rows,
  captured live 2026-09-11 from
  `https://publicreporting.cftc.gov/resource/gpe5-46if.json`, already
  shaped as this project's raw-artifact payload (see
  `cftc_raw_store.build_raw_payload`): 5 UST 10Y NOTE (`043602`)
  observation weeks deliberately spanning an ordinary week
  (2024-01-02), the 2023 ION-incident disruption window (2023-01-31,
  2023-02-07), and the 2025 shutdown window (2025-09-30, 2025-10-07);
  plus 1 UST 2Y NOTE (`042601`) row for multi-contract coverage. All
  numeric values are real, unmodified strings as returned by the API.

### Phase 4C / Phase 9 remediation: Philadelphia Fed RTDSM fixtures

**No RTDSM fixture file is committed here.** Through Phase 8, this
directory held `rtdsm_ruc_sample.xlsx` and `rtdsm_ipt_sample.xlsx` --
real, small excerpts of the official xlsx vintage-history workbooks
(`philadelphiafed.org`, captured live 2026-09-11). The Phase 9
acceptance review found RTDSM's public-redistribution status genuinely
ambiguous (general copyright caution vs. a stated research purpose --
see `docs/data_sources_and_licensing.md` Section 5) and, per an explicit
user decision, both files were **removed** rather than published.

They are replaced by `tests/rtdsm_synthetic_fixtures.py`: two
deterministic Python functions (`build_ruc_synthetic_workbook`,
`build_ipt_synthetic_workbook`) that generate RTDSM-shaped xlsx bytes
at test time -- same column-naming convention, sheet layout, and
observation-period grid as the real files, same structural edge cases
(a missing/blank cell, a value that differs across two vintages for the
same period, a permanent multi-month gap shared by two vintages), but
**every value is fabricated**, never copied, perturbed, or derived from
a real observation. See that module's own docstring for the full
design, and `tests/test_rtdsm_synthetic_fixtures.py` for the tests
proving determinism and the absence of any of the removed files' real
values.

Tests that need to verify an actual, real-world RTDSM fact (the
verified 2010 March RUC revision, the COVID-19 industrial-production
collapse, the October 2025 reporting gap) now do so against
live-fetched data in `tests/test_live_network.py`'s RTDSM section,
opt-in only (`uv run pytest -o addopts="" -m live_network`), never
against a committed fixture.
