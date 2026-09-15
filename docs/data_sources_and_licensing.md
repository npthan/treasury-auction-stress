# Data Sources and Licensing

**This document is a factual record of what was checked and when, plus
this project's interpretation of what was found. It is documentation
of a project decision, not a legal opinion.** Where a page's own words
are quoted or closely paraphrased below, that is marked "documented
terms." Anything beyond the literal text of a source's own page is
marked as "this project's interpretation," and no claim of legal
certainty is made anywhere in this document. This document was written
during Phase 9 (2026-09-14), re-checking and consolidating the
per-source research already recorded in `configs/sources.yml` and
`docs/data_source_governance.md`, per `docs/phase_9_protocol.md`
section 9B and the `phase_9b_provenance_and_licensing` block of
`configs/phase_9_release.yml`.

Verified directly in this session:

- `git ls-files data/` returns only three tracked files: `data/interim/.gitkeep`,
  `data/processed/.gitkeep`, `data/raw/.gitkeep`. No raw, interim, or
  processed dataset is committed to git for any source.
- `tests/fixtures/` contains small, named sample files (JSON/CSV/XLSX)
  plus its own `README.md`, which documents each fixture's provenance.
  Spot-checking `sample_auctions_page.json`, `treasury_rates_2024_sample.csv`,
  and `cftc_tff_sample_payload.json` directly in this session confirms
  the fixtures README's own description: these are small, hand-picked,
  **real** excerpts of live API/CSV responses (verbatim data values),
  with only the response *envelope* (`meta`/`links`/page counts)
  reconstructed to be internally consistent for a fixture of this
  size. None of the sampled files look synthetic or fabricated at the
  data-value level.
- `dashboard/data/phase8_dashboard_data.json` (826 KB) was inspected
  directly: its top-level keys are `generated_from` (input-file
  digests only, no raw values), `meta` (model IDs, test years, cutoff
  views), `point`, `probabilistic`, `stress`, `uncertainty`, and
  `warnings` — all aggregated backtest metrics and model outputs, no
  per-auction, per-week, or per-record raw data from any source.

---

## 1. Treasury Fiscal Data — Treasury Securities Auctions

- **Publisher:** U.S. Department of the Treasury, Bureau of the Fiscal
  Service.
- **Canonical pages:** dataset landing page
  `https://fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/`;
  API documentation `https://fiscaldata.treasury.gov/api-documentation/`.
- **What this repo downloads/derives:** per-auction records from the
  `/v1/accounting/od/auctions_query` endpoint — CUSIP, security type/
  term, auction/announcement/issue/maturity dates, offering amount,
  bid-to-cover ratio, dealer/direct/indirect allotment percentages,
  award rates/yields, and related auction-result fields (see
  `configs/sources.yml`'s `treasury_fiscal_data_auctions` entry and
  `docs/data_dictionary.md`).
- **Raw data committed to git:** No — verified above (`git ls-files
  data/` shows only `.gitkeep` placeholders).
- **Fixture provenance:** Real, verbatim data values from a live query
  captured 2026-09-10, per `tests/fixtures/README.md`; envelope
  metadata (`total-count`, etc.) was reconstructed to be internally
  consistent for a 6-record fixture. Not synthetic at the data-value
  level.
- **Attribution requirement (documented terms):** The dataset landing
  page fetched today did not surface an explicit public-domain or
  attribution statement in the fetched content; footer links to a
  Privacy Policy and a FOIA page were present but not resolved
  further. The **API documentation page**, fetched separately today,
  contains a "License and Authorization" section stating (as
  summarized by the fetch): "The data is offered free, without
  restriction, and available to copy, adapt, redistribute, or
  otherwise use for non-commercial or commercial purposes," with no
  explicit attribution mandate stated on that page.
- **This project's interpretation:** U.S. federal government works are
  generally public domain under 17 U.S.C. §105 (not eligible for U.S.
  copyright protection), and the API documentation's "License and
  Authorization" language is consistent with that — but this document
  does not independently verify the §105 analysis against the fetched
  page text; the fetched page did not use the phrase "public domain"
  or cite §105 verbatim. Attribution to "U.S. Department of the
  Treasury" is recommended here as good practice even though no
  specific template was found required.
- **Redistribution/usage restriction:** None found beyond the general
  "free... for non-commercial or commercial purposes" language above.
- **Dashboard/report suitability:** Suitable. Figures derived from
  this source in the dashboard JSON and Phase 8 reports are aggregated
  backtest statistics (errors, calibration, coverage), never raw
  per-auction CUSIP-level rows.
- **Date checked:** 2026-09-14 (both the dataset landing page and the
  API documentation page were fetched live this session).
- **Open uncertainty:** The dataset landing page itself did not
  surface explicit terms language in this session's fetch (it may
  exist elsewhere on the site, e.g. under Fiscal Service's own privacy
  policy at `fiscal.treasury.gov/about-us/privacy-policy`, which was
  not separately fetched). Rely on the API documentation page's
  "License and Authorization" language as the operative statement
  found.

---

## 2. NY Fed Primary Dealer Statistics

- **Publisher:** Federal Reserve Bank of New York.
- **Canonical page:**
  `https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics`;
  Terms of Use at `https://www.newyorkfed.org/privacy/termsofuse`.
- **What this repo downloads/derives:** weekly FR 2004A/B/C series
  (Treasury-excl.-TIPS dealer positions by maturity bucket, aggregate
  transaction volume, repo, reverse repo, securities borrowed/lent,
  fails to deliver/receive) via `https://markets.newyorkfed.org/api/pd`
  — see `configs/sources.yml`'s `ny_fed_primary_dealer_statistics`
  entry for the full 16-series list.
- **Raw data committed to git:** No (verified above).
- **Fixture provenance:** Real excerpts (specific week ranges, e.g.
  2021-12-01 through 2022-02-16) captured live 2026-09-11, described
  in `tests/fixtures/README.md` as real, unmodified series values,
  including genuine missing-value ("*") occurrences and genuine
  schema-break windows (e.g. the 2013 FR 2004 redesign). Not
  synthetic at the data-value level.
- **Attribution requirement (documented terms):** The Primary Dealer
  Statistics landing page itself, fetched today, did not contain
  specific redistribution/attribution language for the data; it links
  to a general "Terms of Use" that governs the site. Fetching the
  Terms of Use page (`/privacy/termsofuse`) directly today found: the
  NY Fed or its licensors own all website content/data and grant "a
  non-exclusive license... to use, copy, and distribute Content for
  your personal or business purposes," conditioned on including any
  copyright notice/source identifiers the NY Fed provides, in the
  suggested form "© [year] Federal Reserve Bank of New York. Content
  from the New York Fed subject to the Terms of Use at
  newyorkfed.org," clearly labeling any modifications, and not stating
  or implying NY Fed endorsement or using the NY Fed name for
  advertising/commercial endorsement purposes.
- **Redistribution/usage restriction:** Redistribution is permitted
  under the Terms of Use's non-exclusive license, subject to the
  attribution/no-endorsement conditions above. This project has not
  independently confirmed whether machine-learning model training
  specifically falls within "personal or business purposes" — the
  Terms of Use language is broad but does not use ML-specific
  language.
- **Dashboard/report suitability:** Suitable. All figures derived from
  this source in the dashboard/reports are aggregated model
  inputs/outputs (e.g. feature importances, backtest metrics), never
  raw weekly series values reproduced wholesale.
- **Date checked:** 2026-09-14 (landing page and Terms of Use page
  both fetched live this session).
- **Open uncertainty:** No source-specific carve-out or restriction on
  ML training was found or ruled out; the Terms of Use is a
  general-purpose website terms document, not a data-specific license.
  If this project is distributed publicly, including an attribution
  line in the recommended NY Fed format above is a reasonable,
  low-effort step.

---

## 3. Treasury Daily Par Yield Curve

- **Publisher:** U.S. Department of the Treasury.
- **Canonical pages:** data page
  `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve`;
  the `terms_or_usage_policy_url` previously recorded in
  `configs/sources.yml`
  (`https://home.treasury.gov/about/general-information/privacy-and-legal-notices`)
  returned **HTTP 404** when fetched live today. A follow-up check
  (later in this same Phase 9 pass) located and confirmed a working
  replacement, `https://home.treasury.gov/subfooter/site-policies-and-notices`
  (loads successfully, HTTP 200), and `configs/sources.yml` has been
  updated to point at it, with `terms_reviewed_date` advanced to
  **2026-09-14**.
- **What this repo downloads/derives:** daily par yield curve rates
  (2/3/5/7/10/20/30-Year nominal tenors, plus shorter-maturity bill
  tenors) via the site's CSV export — see `configs/sources.yml`'s
  `treasury_daily_par_yield_curve` entry.
- **Raw data committed to git:** No (verified above).
- **Fixture provenance:** Real CSV rows captured live 2026-09-11,
  chosen for specific verified schema/missingness properties (e.g. the
  2002 30-Year issuance-suspension gap, the pre-2020 9-maturity
  schema), per `tests/fixtures/README.md`. Not synthetic.
- **Attribution requirement (documented terms):** The replacement page
  (`.../subfooter/site-policies-and-notices`) was fetched live and
  confirmed to load (HTTP 200), but its content is a federal-policy
  index page (Privacy Act, No FEAR Act, accessibility, etc.) — it does
  not itself contain an explicit data-reuse or redistribution-license
  section. `configs/sources.yml`'s prior `attribution_requirement`
  ("cite 'U.S. Department of the Treasury' as the source; no specific
  attribution template mandated," last substantively reviewed
  2026-09-11) is therefore carried forward unchanged in substance —
  only the dead link was repaired, not a new terms statement found.
- **Redistribution/usage restriction:** Not re-confirmed this session,
  for the reason above; `configs/sources.yml` records
  `permitted_project_use`: "public U.S. government data... research
  use," last reviewed 2026-09-11.
  This project's interpretation, consistent with the treasury auctions
  source above, is that Treasury data is generally U.S. government
  work and not restricted, but this specific page's current terms
  text was not re-read live today.
- **Dashboard/report suitability:** Suitable. Only aggregated
  backtest/feature statistics derived from this source appear in the
  dashboard/reports, never raw daily rate rows reproduced wholesale.
- **Date checked:** Landing/methodology pages exist and were not
  re-fetched this session beyond the 404 check above; last actual
  terms review recorded in `configs/sources.yml` is **2026-09-11**,
  carried forward, not re-verified 2026-09-14.
- **Open uncertainty:** The dead link has been repaired
  (`configs/sources.yml` now points at a live page), but no page found
  so far states explicit data-redistribution terms for this specific
  dataset — the interpretation that Treasury rate data is unrestricted
  U.S. government work (consistent with the auctions source above)
  remains this project's interpretation, not a terms statement quoted
  verbatim from an authoritative page. A human should still confirm
  this before public release if a stronger guarantee is wanted.

---

## 4. CFTC Traders in Financial Futures (TFF) — Futures Only

- **Publisher:** U.S. Commodity Futures Trading Commission.
- **Canonical pages:** report landing page
  `https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm`;
  the `terms_or_usage_policy_url` recorded in `configs/sources.yml`
  (`https://www.cftc.gov/PrivacyPolicy/index.htm`) returned **HTTP
  404** when fetched live today. The CFTC's general Web Policy page,
  `https://www.cftc.gov/WebPolicy/index.htm`, was fetched live today
  as a substitute and does contain a copyright/reuse section (below).
- **What this repo downloads/derives:** weekly Traders in Financial
  Futures "Futures Only" positioning data (contract counts by trader
  category) for specific UST futures contract codes, via
  `https://publicreporting.cftc.gov` — see `configs/sources.yml`'s
  `cftc_tff_futures_only` entry.
- **Raw data committed to git:** No (verified above).
- **Fixture provenance:** Real rows captured live 2026-09-11 from the
  live Socrata endpoint, spanning ordinary weeks, the 2023 ION
  incident, and 2025 shutdown windows, per `tests/fixtures/README.md`.
  Values are described there as "real, unmodified strings as returned
  by the API." Not synthetic.
- **Attribution requirement (documented terms):** The CFTC Web Policy
  page fetched today, under "Copyright," states (as summarized by the
  fetch): "Government information at the CFTC website is in the
  public domain. Public domain information may be freely distributed
  and copied, but it is requested that in any subsequent use the CFTC
  be given appropriate acknowledgement." It separately notes
  third-party materials on the site may be copyrighted and require
  permission, and that CFTC's own data is distinct from any such
  third-party content.
- **Redistribution/usage restriction:** None beyond the
  requested-not-required acknowledgement above, per the fetched Web
  Policy page. This is a different specific URL than the one recorded
  in `configs/sources.yml` (`/PrivacyPolicy/index.htm`, now 404); the
  Web Policy page is the substitute actually confirmed live today.
- **Dashboard/report suitability:** Suitable. Only aggregated
  backtest/feature statistics derived from this source appear in the
  dashboard/reports, never raw weekly positioning rows reproduced
  wholesale.
- **Date checked:** 2026-09-14 (Web Policy substitute page fetched
  live; the originally-recorded PrivacyPolicy URL confirmed 404 live
  today).
- **Open uncertainty:** The specific URL recorded in
  `configs/sources.yml` no longer resolves and should be updated to
  `https://www.cftc.gov/WebPolicy/index.htm` (or a further-verified
  current URL) before public release. The public-domain/
  acknowledgement-requested language found today is consistent with
  what a U.S. federal agency's own data is generally expected to
  carry, but was not previously recorded verbatim in
  `configs/sources.yml`, only inferred.

---

## 5. Philadelphia Fed Real-Time Data Set for Macroeconomists (RTDSM)

**Phase 9 acceptance-review remediation update (user decision, this
review):** the two real `.xlsx` fixtures this section originally
described below were **removed** from the repository rather than
published, because the redistribution ambiguity described in this
section's "Open uncertainty" bullet was judged not worth risking on a
public snapshot. They are replaced by deterministic, wholly synthetic,
schema-equivalent fixtures generated in Python
(`tests/rtdsm_synthetic_fixtures.py`) -- no real RTDSM observation
appears anywhere in this repository's current tree. Tests that need to
verify a genuine RTDSM fact now do so against live-fetched data,
opt-in only (`tests/test_live_network.py`, `-m live_network`), never a
committed file. This is the **safer-remediation** choice the task
instructions describe, not a resolution of the underlying terms
ambiguity — that ambiguity, and this project's own aggregate
model-output/report use of RTDSM-derived figures (unaffected by the
fixture change), are exactly as described below, unchanged and still
disclosed as interpretation. The rest of this section is the original,
unedited Phase 9 research (kept for its historical/evidentiary value).

- **Publisher:** Federal Reserve Bank of Philadelphia.
- **Canonical pages:** RTDSM landing page
  `https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists`;
  privacy notice `https://www.philadelphiafed.org/about-us/privacy-notice`.
- **What this repo downloads/derives:** vintage-history workbooks for
  specific macro series (e.g. unemployment rate RUC, industrial
  production IPT) — wide observation-period x vintage-column tables —
  see `configs/sources.yml`'s `philadelphia_fed_rtdsm` entry.
- **Raw data committed to git:** No (verified above).
- **Fixture provenance:** Real excerpts of the official `.xlsx`
  vintage-history workbooks captured live 2026-09-11, chosen to cover
  specific genuine revisions and a genuine data gap (per
  `tests/fixtures/README.md`, e.g. the April 2020 unemployment-rate
  revision and an October 2025 shutdown-adjacent gap). Not synthetic.
- **Attribution requirement (documented terms):** The RTDSM landing
  page, fetched live today, confirms the stated research purpose is
  still present: researchers may use RTDSM to "verify empirical
  results, to analyze policy, or to forecast." The page also still
  carries the recommended citation: "A Real-Time Data Set for
  Macroeconomists," Dean Croushore and Tom Stark, *Journal of
  Econometrics* 105 (November 2001), pp. 111-30, described on the page
  as "the preferred article to cite for technical journal articles."
  Both match what `docs/data_source_governance.md` and
  `configs/sources.yml` already recorded from the 2026-09-11 review.
  The privacy-notice page, fetched live today, states under
  "Appropriate Use": "Some of the content on this website may be
  copyrighted. Permission to use such copyrighted material must be
  obtained from the owner," and separately prohibits using the
  Philadelphia Fed's name "in any advertising, as an endorsement for
  any product or service, or for any other commercial purpose," and
  states its logos "may not be used without permission."
- **Redistribution/usage restriction:** The privacy-notice page's
  general copyrighted-content caution is a website-wide statement, not
  RTDSM-specific, and does not explicitly say whether RTDSM's own
  data tables are considered "copyrighted content" requiring
  permission versus a public research dataset offered for the stated
  research purposes above. This project's interpretation, consistent
  with `docs/data_source_governance.md`, is that RTDSM's explicitly
  stated research/forecasting purpose covers this project's use, and
  this project does not use the Philadelphia Fed's name for
  advertising/endorsement/commercial purposes. This is an
  interpretation, not a resolution of the ambiguity between the
  general copyright caution and the stated research purpose.
- **Dashboard/report suitability:** Suitable. Only aggregated
  backtest/feature statistics derived from RTDSM vintages appear in
  the dashboard/reports, never raw vintage-table cells reproduced
  wholesale.
- **Date checked:** 2026-09-14 (both the RTDSM landing page and the
  privacy-notice page fetched live this session).
- **Open uncertainty:** As already flagged in
  `docs/data_source_governance.md`: if RTDSM's stated research
  purpose were ever found not to clearly cover this project's actual
  use (including any future public redistribution of RTDSM-derived
  figures), this project's stated policy is to stop and seek written
  clarification from the Philadelphia Fed, the same way it already
  treats FRED/ALFRED. The attribution requirement (Croushore and
  Stark citation, no Federal Reserve System/Board endorsement implied)
  is already honored in `docs/data_source_governance.md` and
  `artifacts/rtdsm_macro_vintage_data_quality.md`.

---

## Fetch summary (what succeeded vs. failed live today, 2026-09-14)

| Source | Page fetched live today | Result |
|---|---|---|
| Treasury Fiscal Data auctions | dataset landing page | fetched, no explicit terms text found |
| Treasury Fiscal Data auctions | API documentation page | fetched, found "License and Authorization" section |
| NY Fed Primary Dealer Stats | landing page | fetched, no data-specific terms found |
| NY Fed Primary Dealer Stats | Terms of Use (`/privacy/termsofuse`) | fetched, found attribution/redistribution language |
| Treasury daily par yield curve | recorded `terms_or_usage_policy_url` | **404 — not found**; replacement located and confirmed live (200), `configs/sources.yml` updated |
| CFTC TFF | recorded `terms_or_usage_policy_url` (`/PrivacyPolicy/index.htm`) | **404 — not found** |
| CFTC TFF | Web Policy page (substitute, found via search) | fetched, found public-domain/copyright section |
| Philadelphia Fed RTDSM | privacy-notice page | fetched, found general copyright caution |
| Philadelphia Fed RTDSM | RTDSM landing page | fetched, confirmed stated purpose and citation still present |

Two of the five sources' originally-recorded terms URLs (Treasury
par-yield-curve privacy/legal-notices page, and the CFTC PrivacyPolicy
page) no longer resolved as of 2026-09-14. Both have since been
repaired in `configs/sources.yml`: the CFTC entry now points at the
Web Policy page fetched and quoted above; the Treasury entry now
points at a confirmed-live replacement page, though that replacement
does not itself add new data-reuse terms language beyond what was
already recorded from the 2026-09-11 review.

---

## 6. Dependency license review

Reviewed via `importlib.metadata` against the installed environment
(`uv run python -c "..."`), using each package's own declared
`License-Expression` (PEP 639) or classic `License`/`Classifier`
metadata field, whichever was populated — several modern packages
(numpy, pyarrow, scikit-learn, statsmodels, pydantic, pytest, ruff)
only populate the newer `License-Expression` field and report `(unset)`
on the legacy field, which is why both were checked.

| Package | Declared license | Notes |
|---|---|---|
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | composite of permissive licenses across bundled components |
| pandas | BSD 3-Clause | |
| pyarrow | Apache-2.0 | |
| pydantic | MIT | |
| PyYAML | MIT | |
| requests | Apache-2.0 | |
| scikit-learn | BSD-3-Clause | |
| scipy | BSD-3-Clause (classifier: BSD License) | |
| seaborn | BSD (classifier: BSD License); full text not in `License` field | |
| statsmodels | BSD-3-Clause | |
| matplotlib | matplotlib license (PSF-derivative, classifier: Python Software Foundation License) | permissive, BSD-like, standard for this package |
| openpyxl | MIT | |
| jupyterlab (dev) | BSD (Project Jupyter, classifier: BSD License) | |
| pytest (dev) | MIT | |
| ruff (dev) | MIT | |

**Flagged items:** none. Every direct and dev dependency resolved to a
permissive license (BSD/MIT/Apache-2.0/PSF-family). No GPL-family,
noncommercial-only, or genuinely missing/unresolvable license was
found. All are compatible with an eventual MIT project license, should
the project owner choose one.

---

## 7. Project software license status

**Updated by explicit user decision during the Phase 9 acceptance
review (this pass).** The project owner chose the MIT License and the
copyright-holder name "Nathan Yang." A standard, canonical-text
`LICENSE` file was created at the repository root:

```
MIT License

Copyright (c) 2026 Nathan Yang
```

`pyproject.toml` records `license = "MIT"` (PEP 639 SPDX expression)
and `license-files = ["LICENSE"]`.

**Scope limitation (important, and explicitly required by the project owner's
decision): the MIT License covers only this project's own original
code and original documentation.** It does **not**:

- relicense, or grant any right to, third-party data or source material
  redistributed or referenced in this repository — including content
  from the U.S. Department of the Treasury, the Federal Reserve Bank of
  New York, the Commodity Futures Trading Commission, or the Federal
  Reserve Bank of Philadelphia, each of which remains governed
  exclusively by its own publisher's terms as documented in Sections
  1-5 above;
- grant any right to use any of those institutions' names, logos, or
  trademarks; and
- imply endorsement by any of those institutions of this project or its
  results. `README.md`'s License section and the dashboard footer both
  state this limitation explicitly.

For a beginner: before this decision, the repository had no license
file, which meant default copyright applied (**all rights reserved** —
public GitHub visibility is not the same as legal permission to reuse).
MIT is short, widely understood, and permits essentially any reuse
(including commercial) of the project's own code, with only a
requirement to keep the copyright/license notice — and, as above, it
says nothing at all about the third-party data the project analyzes.
It is compatible with every dependency license found in Section 6
above.
