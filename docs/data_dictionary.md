# Treasury Fiscal Data — Auctions API: Verified Data Dictionary

Everything in this document was confirmed by directly querying the
live API on 2026-09-10 (via `curl`, working around a couple of
tooling quirks noted below) -- nothing here is copied from the
project plan or guessed from field names.

This dictionary covers only the Phase 1 Treasury auctions API. For the
three Phase 4 market/macro sources' own field/schema documentation,
see `artifacts/treasury_rates_data_quality.md`,
`artifacts/cftc_positioning_data_quality.md`, and
`artifacts/rtdsm_macro_vintage_data_quality.md`.

## Endpoint

- Base URL: `https://api.fiscaldata.treasury.gov/services/api/fiscal_service`
- Path: `/v1/accounting/od/auctions_query`
- Full example:
  `https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?filter=auction_date:gte:2010-01-01,auction_date:lte:2026-09-10&sort=auction_date&page[size]=10000`
- No authentication or API key required (confirmed: a plain unauthenticated request succeeds).
- **Tooling note for anyone using `curl` directly:** the query
  parameters `page[number]` and `page[size]` contain literal `[` `]`
  characters. `curl` treats `[`/`]` in a URL as its own globbing
  syntax by default and will fail with "bad range in URL" (exit code
  3) unless you pass `-g` / `--globoff`. This is a curl quirk, not
  something about the API.

## Pagination

- Parameters: `page[number]` (1-indexed) and `page[size]`.
- **Verified maximum `page[size]` is exactly 10000.** Requesting more
  (tested 10001, 12000, 15000) returns HTTP 400 with body
  `{"error":"Invalid Query Param","message":"Invalid query parameter:
  Limit '<n>' is invalid. Expected an integer between 1 and 10000. ..."}`.
- Requesting a `page[number]` beyond the last page returns HTTP 400
  with a similar `{"error": "Invalid Query Param", ...}` body, **not**
  an empty `data` array. Pagination code must stop based on
  `meta.total-pages`, not by probing past the end.
- Response includes `links.next` / `links.prev` (null at the
  boundaries) for reference, but this project's client walks pages by
  incrementing `page[number]` up to `meta.total-pages` directly.

## Filtering

- Parameter: `filter=field:operator:value`, multiple conditions
  comma-separated (`filter=a:gte:2010-01-01,b:lte:2020-01-01`).
- Verified operators: `gte`, `lte`, `in` (e.g.
  `filter=security_type:in:(Note,Bond)`).
- Date format: `YYYY-MM-DD` (verified working).

## Sorting

- Parameter: `sort=field` (ascending) or `sort=-field` (descending).
  Verified with `sort=auction_date` and `sort=-auction_date`.

## A verified quirk with the `fields=` parameter -- avoided in this project's code

The documentation implies `fields=` simply selects a subset of
columns to return. Empirically, it does **not** behave that way:
requesting `fields=security_type` alone returned only 3 rows total
(`Bill`, `Bond`, `Note`) with `meta.total-count` reported as 3 --
i.e. the server appears to apply something like a `SELECT DISTINCT`
over just the requested field(s), collapsing the whole dataset down
to its distinct value combinations, rather than projecting columns
of every row. This project's downloader **never uses `fields=`** and
always requests full rows, specifically to avoid this behavior
silently corrupting a real pull.

## Response structure

Top-level keys: `data` (list of row objects), `meta`, `links`.

`meta` contains: `count` (rows in this response), `labels` (field ->
human-readable name), `dataTypes` (field -> API's self-reported type),
`dataFormats` (field -> display format string), `total-count` (rows
matching the whole query), `total-pages`.

**The API's own `dataTypes` metadata is internally inconsistent** for
some numeric-looking fields: e.g. `high_price`, `low_price`,
`avg_med_price`, and `price_per100` are reported as `STRING` while the
structurally identical `high_yield`/`low_yield`/`avg_med_yield` are
reported as `NUMBER`, even though both hold plain decimal strings like
`"99.759"`. This project's numeric-parsing field list
(`schema.NUMERIC_FIELDS`) was built by hand-checking real values, not
by trusting `dataTypes` wholesale.

## Missing-value convention (important)

Missing values are represented as the **literal JSON string `"null"`**
(four characters, a string), not a JSON `null`. A field like
`comp_accepted` for an old or not-yet-settled auction literally holds
the string `"null"`. Any code that does `float(x)` without checking
for this first will raise; code that checks `if not x` would also
wrongly treat `"null"` the string as falsy-but-not-missing in some
contexts. `normalize.py` handles this explicitly (see
`_null_to_na`) and keeps missing values distinct from real zeros.

## HTTP error format

Errors return normal HTTP status codes (observed: 400 for bad query
parameters) with a JSON body `{"error": "<short label>", "message":
"<human-readable detail>"}`. No rate-limit headers were observed on
successful responses; the documentation states the API "does not
require a user account or registration."

## Timezone conventions for this project's own pipeline (added by the Phase 2 acceptance review)

The API itself only ever returns plain calendar dates (no time
component) for `auction_date`, `announcemt_date`, etc. -- no change
needed there. What *is* timezone-sensitive is this project's own
pipeline: `retrieval_timestamp_utc` is a full UTC instant (unchanged,
correct); the default `--end-date` for `treasury_auction_stress.data.cli`
(the `auction_date` query upper bound) is now derived in
**America/New_York**, not UTC, because the Treasury auction market
itself operates on Eastern time (verified directly from official
announcement PDFs, which state closing times in ET). An earlier
version of this pipeline used a UTC calendar date for that default,
which disagrees with the Eastern calendar date for several hours
around every UTC midnight. See `docs/point_in_time_rules.md` and
`treasury_auction_stress.data.time_utils` for the full rule and
the Phase 2 acceptance review for the investigation.

## Total dataset size (as of 2026-09-10)

`total-count` for all security types, all history: **11109** rows.
For `auction_date` between 2010-01-01 and 2026-09-10: **5740** rows
(fits in a single page at the 10000 max page size).

## Nominal coupon identification (verified against live 2010+ data)

Cross-tabulating `security_type` x `inflation_index_security` x
`floating_rate` over all 5740 2010+ records gives exactly:

| security_type | inflation_index_security | floating_rate | count | what it is |
|---|---|---|---|---|
| Bill | No | No | 4107 | Treasury Bills (out of scope) |
| Note | No | No | 1003 | **nominal coupon notes (2/3/5/7/10-year)** |
| Note | No | Yes | 154 | 2-year FRN (out of scope) |
| Note | Yes | No | 156 | TIPS notes (out of scope) |
| Bond | No | No | 278 | **nominal coupon bonds (20/30-year)** |
| Bond | Yes | No | 42 | TIPS bonds (out of scope) |

So: **nominal coupon universe = `security_type in {"Note", "Bond"}` AND
`inflation_index_security == "No"` AND `floating_rate == "No"`.**
Filtering on `original_security_term` within that subset yields exactly
the seven expected values: `2-Year`, `3-Year`, `5-Year`, `7-Year`,
`10-Year`, `20-Year`, `30-Year` -- no fragile string-guessing needed.

## Reopening identification

The `reopening` field is a direct `"Yes"`/`"No"` string -- no need to
infer it from CUSIP reuse or dated_date comparisons. Verified: over
2010+ nominal coupons, `reopening` takes only these two values.

## Candidate primary key

`(cusip, auction_date)` was verified unique across all 5740 records
retrieved for 2010-01-01 through 2026-09-10 (zero duplicate keys).
Note that `cusip` alone is **not** unique -- a reopened security keeps
the same CUSIP as its original issuance, so the same CUSIP legitimately
appears on multiple rows (once per auction event, each at a different
`auction_date`).

## Target-relevant field coverage (2010+ nominal coupon subset)

Of 1281 nominal-coupon rows retrieved for 2010-01-01 through
2026-09-10, exactly 1 lacked result fields (`total_accepted`,
`bid_to_cover_ratio`, `primary_dealer_accepted`, etc.) -- and that one
row is the 30-year bond auctioned on 2026-09-10 itself (today, as of
retrieval), whose results simply haven't been published yet. Every
other row has complete coverage of every candidate target field listed
in `docs/target_specification.md`. See
`artifacts/auction_data_quality.md` for the generated, current numbers.

## Announcement date timing (verified against official Treasury sources, 2026-09-10)

`announcemt_date` is a calendar date (`YYYY-MM-DD`, no time
component) in the Fiscal Data API itself -- confirmed directly from
the live schema (`meta.dataFormats.announcemt_date == "YYYY-MM-DD"`).
This section documents what was checked to determine whether an exact
intraday release time could safely be attached to it anyway.

- **31 CFR 356.10** (Uniform Offering Circular, Subpart B, "What is
  the purpose of an auction announcement?"), verified verbatim
  against the **official U.S. Government Publishing Office edition of
  the Code of Federal Regulations** on govinfo.gov -- specifically
  *Code of Federal Regulations, Title 31, Volume 2* (2024 annual
  edition), Part 356:
  `https://www.govinfo.gov/content/pkg/CFR-2024-title31-vol2/xml/CFR-2024-title31-vol2-part356.xml`
  (also available as a single-section PDF at
  `https://www.govinfo.gov/content/pkg/CFR-2024-title31-vol2/pdf/CFR-2024-title31-vol2-sec356-10.pdf`).
  Full text of § 356.10, quoted verbatim from that official XML:
  > "By issuing an auction announcement, we provide public notice of
  > the sale of bills, notes, and bonds. The auction announcement
  > lists the specifics of each auction, e.g., offering amount, term
  > and type of security, CUSIP number, and issue and maturity dates.
  > The auction announcement and this part, including the Appendices,
  > specify the terms and conditions of sale. If anything in the
  > auction announcement differs from this part, the auction
  > announcement will control. If you intend to bid, you should read
  > the applicable auction announcement along with this part."

  This regulation requires specific *content* and that notice precede
  bidding; it does **not** specify a clock time by which the
  announcement must be released.

  (eCFR.gov -- the other official, continuously-updated source named
  by the project owner -- was also attempted, but its site blocked automated
  fetches with a bot-protection challenge page rather than serving
  the regulation text. GovInfo.gov, the U.S. Government Publishing
  Office's own official repository, was used instead and is
  authoritative on its own; no non-government secondary source is
  relied on for this citation.)
- **Real announcement press releases** (`treasurydirect.gov`, sampled
  via web search across 2018-2026) carry their own "embargoed
  until"/"for release at" line, and that time is **not constant**:
  observed values include 08:30 A.M., 10:30 A.M., and 11:00 A.M.
  Eastern across different releases and years. This was found by
  sampling actual press-release documents, not inferred.
- **TreasuryDirect's own scheduling pages**
  (`treasurydirect.gov/auctions/general-auction-timing/`,
  `treasurydirect.gov/auctions/how-auctions-work/`) describe the
  auction calendar only in terms of dates and days-of-week ("usually
  announced in the second half of the month," etc.), with an explicit
  caveat that Treasury borrowing needs, financing-policy decisions, or
  debt-limit-related Congressional action "could alter or delay the
  pattern." The one concrete clock time found on these pages --
  "Updated every Friday by 10:45 AM, Eastern U.S. time" -- describes
  when the *list of upcoming auctions* is refreshed, not when an
  individual auction's announcement itself is released.

**Conclusion: it is not safe to treat `announcemt_date` as an exact
announcement-time timestamp.** No single release time is documented or
observed consistently across the 2010-2026 sample window. This
project does not invent one. `announcemt_date` is kept as a
date-level event field, and any later phase needing an
"announcement-time" prediction cutoff must use the conservative,
date-level rule documented in `docs/point_in_time_rules.md` ("the
announcement-time cutoff is date-level, not an exact timestamp")
rather than assuming a specific hour of the day.

## Phase 2 finding: SOMA/FIMA add-ons and the correct share denominator

**This section was corrected during the Phase 2 acceptance review** --
see the Phase 2 acceptance review for the full investigation.
The original Phase 2 draft excluded both `soma_accepted` and
`fima_noncomp_accepted` from the share denominator; that was wrong for
FIMA, as shown below.

Phase 2 discovered, by exact-dollar reconciliation over the full
1281-row settled nominal-coupon sample (2010-01-01 through this
session's retrieval date), that `total_accepted` is **not** simply the
sum of the primary-dealer/direct-bidder/indirect-bidder/noncompetitive
amounts:

```
total_accepted == comp_accepted + noncomp_accepted
                   + fima_noncomp_accepted + soma_accepted
comp_accepted   == primary_dealer_accepted + direct_bidder_accepted
                    + indirect_bidder_accepted
```

verified with **zero exceptions** across every settled row. Both
`soma_accepted` and `fima_noncomp_accepted` were not in Phase 1's
`NUMERIC_FIELDS` list (so they were left as unparsed raw strings)
until Phase 2 added them -- see `schema.py`'s `NUMERIC_FIELDS` and its
inline comment.

The acceptance review went further and read Treasury's own auction-
results press releases directly (not just the API's field names) for
two specific auctions -- see the "special auction" section below.
Those releases show a "Subtotal" line, `Competitive + Noncompetitive +
FIMA (Noncompetitive)`, with a footnoted Bid-to-Cover Ratio computed
from that Subtotal, and a separate "SOMA" line added only afterward to
reach "Total." **Treasury's own methodology excludes only SOMA, not
FIMA.** `soma_accepted` (the Fed rolling its own maturing holdings into
the new issue) is the Fed's own portfolio operation, not third-party
demand; `fima_noncomp_accepted` (foreign and international monetary
authorities' noncompetitive bids) is real, allocated demand from a
distinct bidder category, submitted through the New York Fed.

SOMA alone averages ~6-8% of `total_accepted` per tenor but exceeds
40% for some individual auctions. Using `total_accepted` (which
includes SOMA) as a bidder-category share denominator would
materially understate every category's true share of the actually-
competed portion of the auction. See `docs/target_specification.md`
and `treasury_auction_stress.features.targets` for the corrected
formula (denominator = `public_accepted_amount = comp_accepted +
noncomp_accepted + fima_noncomp_accepted`) and
`artifacts/auction_data_profile.md` for the full quantitative writeup.

**Bid-to-cover reconciliation:** Treasury's own precomputed
`bid_to_cover_ratio` field reconciles **exactly** (not just
approximately) against a from-scratch calculation using this corrected
formula:

```
bid_to_cover_ratio == round(
    (comp_tendered + noncomp_accepted + fima_noncomp_accepted) / public_accepted_amount,
    2,
)
```

Verified over the main modeling sample (n=1279, excluding the two
special auctions discussed below): **100% match exactly** (max
absolute difference 0.0000). The earlier, FIMA-excluding formula gave
"93.3% within 0.01, 99.8% within 0.02" and mischaracterized the
residual as rounding noise -- it was not; it was this formula error.
Using `total_accepted`/`total_tendered` directly (i.e. including SOMA
on both sides) does **not** reconcile nearly as well (max diff up to
0.65) -- confirming Treasury's published ratio is a "market demand"
figure that includes FIMA but excludes the Fed's own rollover.

## Phase 2 acceptance review finding: a verified special auction category

Two settled auctions have `offering_amt` of exactly $25,000,000 (vs.
typical tens-of-billions for their tenors): 2019-06-21 (CUSIP
`9128286T2`, 10-year reopening) and 2021-12-02 (CUSIP `912810TC2`,
20-year reopening). The acceptance review fetched Treasury's own
official announcement and results press releases directly (URLs and
full quoted text in the Phase 2 acceptance review) and
confirmed both are **restricted, primary-dealer-only reopenings**:
Treasury's own announcement text states "Customer bids will not be
accepted. Noncompetitive tenders, including FIMA tenders, will not be
accepted," with bids submitted by phone/email directly to the New York
Fed rather than through the standard electronic auction system. This
is a verified, resolved finding, not an inference from the offering
size alone.

`noncomp_tenders_accepted` (a plain "Yes"/"No" API field, always
populated) is `"No"` for exactly these two rows and no others across
the entire 2010-2026 nominal-coupon sample -- verified by
cross-tabulation, not assumed -- and is now used as the authoritative
classification signal (see
`treasury_auction_stress.features.eligibility`). It was added to
`schema.py`'s `CATEGORICAL_IDENTITY_FIELDS` during this review.

## Core fields this project relies on

See `src/treasury_auction_stress/data/schema.py` for the authoritative,
machine-readable list (`DATE_FIELDS`, `NUMERIC_FIELDS`,
`CATEGORICAL_IDENTITY_FIELDS`). Notable naming quirk: the announcement
date field is spelled **`announcemt_date`** (missing the second "n" in
"announcement") -- verified directly against the live API, not a typo
in this project's code.

| Field | Verified API type | Role |
|---|---|---|
| `cusip` | STRING | Security identifier; half of candidate primary key |
| `auction_date` | DATE | Other half of candidate primary key |
| `announcemt_date` | DATE | Announcement-time forecast cutoff anchor -- **date-level only, no verified exact release time; see "Announcement date timing" below** |
| `issue_date` | DATE | Settlement/issue date |
| `maturity_date` | DATE | Maturity date |
| `record_date` | DATE | Treasury's own record-date stamp (not the same as `issue_date` for older records) |
| `security_type` | STRING | `Note`/`Bond`/`Bill` (used for nominal-coupon filter) |
| `security_term` | STRING | Current remaining term description (e.g. "9-Year 10-Month" for a reopened 10-year) |
| `original_security_term` | STRING | Term at original issuance -- **this is the canonical tenor label** used for `tenor` |
| `reopening` | STRING | `Yes`/`No` -- direct reopening flag |
| `inflation_index_security` | STRING | `Yes`/`No` -- used to exclude TIPS |
| `floating_rate` | STRING | `Yes`/`No` -- used to exclude FRNs |
| `offering_amt` | CURRENCY0 (numeric string) | Announced offering size |
| `total_accepted` | NUMBER (numeric string) | Total amount accepted (auction-result field) |
| `total_tendered` | NUMBER (numeric string) | Total amount bid (auction-result field) |
| `bid_to_cover_ratio` | NUMBER (numeric string) | Treasury's own precomputed bid-to-cover |
| `primary_dealer_accepted` / `_tendered` | NUMBER | Dealer take-down numerator/denominator inputs |
| `direct_bidder_accepted` / `_tendered` | NUMBER | Direct-bidder share inputs |
| `indirect_bidder_accepted` / `_tendered` | NUMBER | Indirect-bidder share inputs |
| `noncomp_accepted` | NUMBER | Noncompetitive bid amount |
| `comp_accepted` / `comp_tendered` | NUMBER | Competitive bid amounts |
| `soma_accepted` / `soma_tendered` | NUMBER | Federal Reserve (SOMA) add-on, awarded outside competitive bidding -- the Fed's own portfolio operation, **excluded** from bidder-category share denominators, see "Phase 2 finding" above |
| `fima_noncomp_accepted` / `_tendered` | NUMBER | Foreign/International Monetary Authorities noncompetitive add-on -- real third-party demand, **included** in bidder-category share denominators (corrected during the Phase 2 acceptance review; see above) |
| `treas_retail_accepted` | NUMBER | TreasuryDirect retail-investor amount (a subset of `noncomp_accepted`, not a separate add-on) |
| `noncomp_tenders_accepted` | STRING | `Yes`/`No` -- `"No"` identifies the verified special, restricted primary-dealer-only auction category (2 rows in the 2010-2026 sample), added to `CATEGORICAL_IDENTITY_FIELDS` during the Phase 2 acceptance review; see `treasury_auction_stress.features.eligibility` |
| `high_yield` / `low_yield` / `avg_med_yield` | NUMBER | Auction yield statistics |
| `allocation_pctage` | NUMBER | Allocation percentage at the stop |

All auction-result fields above are, per
`docs/point_in_time_rules.md`, only known **after** the auction closes
and must never be used as a pre-auction feature in later phases.

## Phase 3 — NY Fed Primary Dealer Statistics: verified data dictionary

Everything below was confirmed by directly querying the live API
(`https://markets.newyorkfed.org/api/pd/...`) on 2026-09-10/11 and by
reading the Federal Reserve Board's own FR 2004 form-family
documentation. Full detail and primary-source citations live in
`src/treasury_auction_stress/data/dealer_stats_schema.py`'s module
docstring and `artifacts/primary_dealer_data_quality.md`; this section
is the summary entry point.

### Endpoint

- List active series: `GET https://markets.newyorkfed.org/api/pd/list/timeseries.json`
- Fetch one series' full history: `GET https://markets.newyorkfed.org/api/pd/get/{keyid}.json`
  -- no date-range parameter; always returns the complete history NY
  Fed has under that `keyid`.
- No authentication required (verified).

### Response shape

`{"pd": {"timeseries": [{"asofdate": "YYYY-MM-DD", "keyid": "...", "value": "..."}, ...]}}`.
The list endpoint's entries additionally carry `seriesbreak` (the
API's own internal current-code-set tag, `"SBN2024"` for every entry
as of this session) and a free-text `description`.

### Missing-value convention — a third, distinct token

The literal string **`"*"`** represents a missing/withheld weekly
value. This is **not** the same as the Fiscal Data auctions API's
`"null"` string (see above) -- two different official sources, two
different missing-value conventions, both handled explicitly and
never conflated in this project's code
(`dealer_stats_schema.MISSING_VALUE_TOKEN` vs. `schema.API_NULL_TOKEN`).

### Units and valuation basis

"Millions of dollars" throughout, per the FR 2004 form family's own
documentation: FR 2004A (positions) at **market value**; FR 2004B
(transactions) and FR 2004C (financing/fails) at **principal value**
(or fair/market value of pledged securities when only securities, not
cash, changed hands).

### Observation timing

Every selected series' `asofdate` is a **Wednesday**, exactly 7
calendar days apart, with **zero gaps**, from 2013-04-03 through the
most recent release as of this session -- verified directly, not
assumed. This matches the FR 2004 form family's own "as of Wednesday"
definition.

### Publication timing (see `docs/point_in_time_rules.md` for the full rule)

NY Fed's own page: "Data are updated on Thursdays at approximately
4:15 p.m. with the previous week's statistics." No source found
confirms an exact historical release minute or holiday-shift rule for
every week, so this project computes a conservative
`publication_date` and a `publication_safe_available_date` (one
further day of buffer) rather than inventing precision it cannot
verify.

### Historical coverage and verified schema regimes

- This project's selected keyids' history begins **2013-04-03** (the
  April 2013 FR 2004 redesign) -- **not** 1998, even though the NY Fed
  website's own historical data tool claims coverage back to
  1998-01-28; that earlier history is not exposed under these current
  keyids in this API and is not spliced in.
- `PDPOSGSC-G11L21` (11-21y) and `PDPOSGSC-G21` (>21y) Treasury coupon
  position buckets only exist from **2022-01-05** -- verified directly
  (701 weeks for every other selected series vs. exactly 244 for these
  two). No currently-listed keyid covers a combined "more than 11
  years" bucket before this date.
- FR 2004C's repo/reverse-repo counterparty-segment reporting was
  revised **2022-01-05** (Federal Reserve FEDS Note, "Insights from
  revised Form FR2004 into primary dealer securities financing and MBS
  activity," 2022-08-05): tri-party ex-GCF, GCF, cleared bilateral,
  uncleared bilateral, and sponsored repo became separately reported
  for the first time. This project's selected repo/reverse-repo
  totals (`PDSORA-UTSETTOT`, `PDSIRRA-UTSETTOT`) remain one continuous
  keyid across this date, but the methodology underlying the total
  changed -- flagged via
  `dealer_stats_schema.COUNTERPARTY_GRANULARITY_BREAK_2022_01_05`.

### One verified, deliberately-excluded, likely-mislabeled series

`PDTRGST-TOT`'s own API `description` claims to be a TIPS-only
transactions total, but its naming pattern (parallel to
`PDPOSGST-TOT`, the verified excl-TIPS position total) and its value
level (~$873B average vs. ~$23B for the unambiguous TIPS total
`PDTIPSTOT`, vs. ~$847B for the correctly-labeled excl-TIPS total
`PDGSWOEXTTOT`) together indicate its own description field is a
copy-paste artifact, not ground truth. This project uses
`PDGSWOEXTTOT` instead and never fetches `PDTRGST-TOT` for either
interpretation.

### Selected series (16)

See `dealer_stats_schema.SELECTED_SERIES` for the authoritative list
with per-series economic rationale (reproduced in
`artifacts/phase_3_primary_dealer_integration.md`): 9 Treasury (excl.
TIPS) net-position series by maturity bucket, 1 aggregate transaction-
volume series, 4 financing series (repo, reverse repo, securities
borrowed, securities lent), and 2 settlement-fails series. TIPS and
FRN dealer-statistics series exist in the API but are out of scope,
matching this project's nominal-coupon-only universe
(`docs/project_plan.md`).
