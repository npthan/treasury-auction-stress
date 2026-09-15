# Point-in-Time Rules

## Why this document exists

The single most common way a quantitative research project quietly
produces fake-looking accuracy is **data leakage**: letting a model
see, directly or indirectly, information that would not actually have
been available at the moment it needed to make a prediction. Treasury
auction research is unusually exposed to this because many natural
"features" (yields, positioning data, macro releases) are published on
a delay, revised after the fact, or observed on a different date than
they are released. This document defines the vocabulary and rules used
throughout the project to avoid that. It describes the eventual
target state; Phase 1 does not yet build the feature matrix these
rules govern, but every field ingested from Phase 1 onward must be
labeled with enough timing information to support them later.

## Four dates that are easy to confuse

- **Observation date** — the date the underlying real-world event or
  measurement pertains to. Example: the Tuesday a CFTC position
  snapshot describes, or the auction date of a Treasury security.
- **Publication date** — the date the data describing that event was
  actually made public. Example: CFTC's Commitments of Traders report
  for a Tuesday position is not published until the following Friday.
  A macro data point (e.g. a GDP estimate) is often revised weeks or
  months after its observation date, and each vintage has its own
  publication date.
- **Retrieval date** — the date *this project* actually downloaded or
  recorded the data. This can lag the publication date (e.g. if a
  pipeline is run a day late), and every raw file this project
  produces must record it in a metadata sidecar so it is auditable
  later.
- **Prediction cutoff** — the exact point in time a forecast is
  allowed to "stand at." Only information with a publication date at
  or before the cutoff may be used as a feature for a prediction made
  at that cutoff.

These four dates are **not interchangeable**. A model that uses
observation date instead of publication date to decide what data was
"available" is implicitly assuming instant, error-free publication,
which is false for nearly every real-world data source and is a
common, subtle source of leakage.

## Timezone conventions (added by the Phase 2 acceptance review)

Every one of the four dates above can also be ambiguous about *which
timezone's calendar day* it refers to, and this project's own pipeline
originally conflated two different notions of "today" without saying
so. This was found and fixed during the Phase 2 acceptance review (see
the Phase 2 acceptance review for the investigation) and the
resulting conventions, implemented in
`treasury_auction_stress.data.time_utils`, are now load-bearing rules:

- **Retrieval timestamps are stored in UTC, with explicit timezone
  info** (`RawArtifactMetadata.retrieval_timestamp_utc`, an ISO 8601
  string with a `+00:00` offset). This is unchanged and correct — it
  is an operational fact about when this project's own pipeline ran,
  and UTC is the unambiguous, DST-free choice for that.
- **Retrieval-date bucketing (cache/raw-artifact filenames) also uses
  UTC** (`time_utils.utc_today_date`) — same reasoning: it answers
  "what day did this pipeline run," not a market question.
- **The default upper bound of an `auction_date` query range uses
  America/New_York** (`time_utils.market_today_date`), because the
  U.S. Treasury auction market itself operates on Eastern time —
  verified directly from official announcement PDFs, which state
  competitive/noncompetitive closing times in ET (e.g. "11:00 AM ET,"
  "1:00 PM ET"; see the Phase 2 acceptance review for the
  citations). **This was the actual bug**: the pipeline previously
  used `datetime.now(UTC).date()` for this default, which disagrees
  with the America/New_York calendar date for several hours around
  every UTC midnight (which falls in the evening, U.S. Eastern time).
  A pipeline run in that window would silently request one extra day
  of "through today" auction data, using a UTC date that had already
  rolled over while it was still the previous evening in the Treasury
  market's own timezone.
- **Never describe a calendar date as "today" in this project's code,
  CLI help text, or generated reports without naming the timezone.**
  `treasury_auction_stress.data.cli` and `download.py` now say "UTC
  calendar date" or "America/New_York calendar date" explicitly
  wherever a default date is surfaced.
- **Pending-vs-completed status is decided solely by whether the API
  populated an auction's result fields** (`results_available` in
  `normalize.py`) — **never** by comparing `auction_date` against
  "today" in any timezone. This means the UTC/Eastern disagreement
  above can change which auctions are *requested* in a given pull, but
  it can never make an upcoming, not-yet-settled auction look
  completed — there is no date comparison in that decision at all.
  `tests/test_time_utils.py` has controlled, deterministic tests for
  the exact boundary-disagreement scenario (a fixed instant shortly
  after UTC midnight, still the previous evening in America/New_York).

## Planned prediction cutoffs

1. **Announcement-time forecast** — cutoff is the Treasury's public
   auction announcement (which fixes tenor, offering amount, auction
   date, etc., but happens *before* the auction).
2. **Pre-auction forecast** — cutoff is a documented point shortly
   before the auction itself, such as the previous business day's
   market close. The exact cutoff must be written down and used
   consistently, not chosen after seeing results.

### The announcement-time cutoff is date-level, not an exact timestamp

The Fiscal Data API's `announcemt_date` field (see
`docs/data_dictionary.md` for the exact field name and its
"announcement" typo) is a **calendar date**, not a timestamp — the
API itself stores it with format `YYYY-MM-DD` and no time component.

This project investigated, using only official Treasury sources,
whether a single reliable intraday release time could be attached to
that date anyway (e.g. "announcements are always public by 11:00 AM
Eastern"), so that the announcement-time cutoff could be made more
precise than a full calendar day. The answer, backed by primary
sources, is **no**:

- **31 CFR 356.10** (the Uniform Offering Circular's own definition of
  the purpose of an auction announcement) requires only that Treasury
  give "public notice" and disclose specific terms (offering amount,
  security type/term, CUSIP, issue and maturity dates) — it contains
  **no requirement on the clock time** of release. Verified verbatim
  on 2026-09-10 against the official Government Publishing Office
  (GovInfo.gov) annual Code of Federal Regulations, Title 31, Volume
  2 (2024 edition) — see `docs/data_dictionary.md` for the exact
  citation and full quoted text. (The eCFR site was also attempted as
  an alternate official source but blocked automated access with a
  bot-protection challenge; GovInfo's official annual edition was used
  instead.)
- Real Treasury offering-announcement press releases (sampled across
  2018-2026 from `treasurydirect.gov`) carry an "embargoed until" /
  "for release at" time that is **not standardized**: observed values
  include 08:30 A.M., 10:30 A.M., and 11:00 A.M. Eastern across
  different releases and years. This is a directly observed historical
  exception to "one fixed announcement time," not a hypothetical one.
- TreasuryDirect's own scheduling guidance (`general-auction-timing`,
  `how-auctions-work`) describes the auction calendar only in terms of
  dates and days-of-week, using hedged language ("usually," "generally,"
  "tentative") and explicitly warns that "Treasury borrowing
  requirements, financing policy decisions, and the timing of
  Congressional action on the debt limit could alter or delay the
  pattern" — so even the *date* is not contractually guaranteed, let
  alone an intraday instant.

**Conservative rule adopted by this project:** `announcemt_date` is
used only as a **date-level** signal of "this auction's terms are
public knowledge no later than the end of this calendar date
(Eastern)." No later phase may treat it as available at a specific
hour/minute on that date, and no later phase may invent or assume an
intraday timestamp for it. Concretely:

- Any feature or join that needs to sequence the announcement against
  another same-day, intraday-timestamped event (e.g. an intraday
  market data point) must treat the announcement as available only as
  of the **next** full trading/business day, unless a specific,
  separately-documented source proves same-day availability before
  that event's own timestamp.
- An "announcement-time forecast" cutoff, when implemented in a later
  phase, is defined as "public information as of the end of
  `announcemt_date`," not "as of `announcemt_date` at market open" or
  any other specific hour.
- This rule is deliberately more conservative than the true, unknown
  release time in most cases (real releases mostly happen mid-morning
  Eastern, well before that day's market close) — the cost of that
  conservatism is a strictly less leaky, if slightly less precise,
  cutoff.
   consistently, not chosen after seeing results.

## Rules that follow from this (target state — enforced starting in later phases)

- **Auction-result fields cannot be features.** Bid-to-cover, dealer
  allotment, and every other outcome-side field is only known *after*
  the auction closes, so none of them may appear on the right-hand
  side of a pre-auction model — even indirectly, such as through a
  feature engineered from them.
- **Primary Dealer Statistics must be joined by publication timing,
  not observation week.** The NY Fed publishes weekly dealer position
  data on a lag; a forecast made on a given date may only use the most
  recent dealer-statistics release whose *publication* date is at or
  before the cutoff, not the release whose *observation* week happens
  to be most recent.
- **CFTC positions must use release date, not just the Tuesday
  observation date.** Commitments of Traders reports describing a
  Tuesday's positions are released the following Friday; a model
  cutoff that falls between Tuesday and Friday must not see that
  report yet.
- **Macroeconomic values must use the vintage available at forecast
  time.** Series like GDP or employment are revised after initial
  release. ALFRED/FRED point-in-time (a.k.a. "real-time") data exists
  specifically to let researchers reconstruct what a value looked like
  *as of* a given past date, rather than using today's fully-revised
  number.
- **Rolling statistics must exclude the target auction itself.** A
  rolling average "of the last N auctions" computed for auction *i*
  must be computed over auctions strictly before *i* — including the
  auction being predicted in its own rolling window is a leakage bug,
  not a feature.
- **Scalers, encoders, imputers, thresholds, feature selection, and
  residualization must be fit using training data only**, and refit
  independently inside every chronological evaluation fold. Fitting
  any of these on the full dataset (including future folds) leaks
  future distributional information into the past.

## Phase 2: this rule implemented for the Dealer Absorption Surprise target

`treasury_auction_stress.features.dealer_absorption` is the first
concrete implementation of the residualization rule above.
`DealerAbsorptionSurpriseModel` separates `fit` (learns tenor/reopening
group means and two linear coefficients from a training-fold dataframe
only) from `transform` (applies the already-frozen parameters to any
dataframe). `StressThreshold` does the same for the provisional
stress-event percentile cutoff. `tests/test_dealer_absorption.py`
proves directly that appending future rows to a `transform` call never
changes an already-fitted model's output for historical rows, and that
passing extreme values into `StressThreshold.transform` never moves an
already-fitted `threshold_`. See `docs/target_specification.md` for the
formula and `artifacts/auction_data_profile.md` for a full-history,
walk-forward (expanding-origin, annual-fold) construction used purely
for descriptive analysis in this phase — not a chronological
evaluation fold to be reused by a later phase's model evaluation
without being re-derived under that phase's own protocol.

## Phase 3: the Primary Dealer Statistics publication rule, implemented

This document's "Primary Dealer Statistics must be joined by
publication timing, not observation week" rule (above) is now
concretely implemented in
`treasury_auction_stress.data.dealer_stats_normalize` and
`treasury_auction_stress.features.dealer_join`. See
`artifacts/primary_dealer_data_quality.md` and
`artifacts/phase_3_primary_dealer_integration.md` for the full
investigation; summarized here because it is a load-bearing,
project-wide rule, following the same "verify, don't assume" and
"date-level, not intraday" discipline established above for Treasury
auction announcements.

**Verified source facts** (not assumed): NY Fed's own page states
"Data are updated on Thursdays at approximately 4:15 p.m. with the
previous week's statistics." The underlying FR 2004 filings (Federal
Reserve Board's own form documentation) collect positions/transactions/
financing-and-fails "as of Wednesday," reported "the next business
day." Empirically, every one of this project's selected series has an
observation exactly every 7 days on a Wednesday, with zero gaps, from
2013-04-03 onward -- verified directly against the live API, not
assumed.

**No source found states an exact historical release minute for every
week, nor an explicit holiday-shift rule** -- so, per this document's
standing policy, a conservative, disclosed, date-level rule is used
instead of inventing one:

1. `publication_date` = the Wednesday observation date + 1 calendar
   day (the nominal Thursday), advanced past any U.S. federal holiday
   (Federal Reserve's own holiday schedule, verified 2026-09-10) to the
   next business day. This is a **disclosed heuristic** for the
   holiday case specifically -- no primary source confirms an exact
   historical holiday-shifted release date, and this project does not
   claim one.
2. `publication_safe_available_date` = `publication_date` + 1
   additional calendar day. Every join uses **this** column, never
   `observation_date` or the raw `publication_date` directly. The
   extra day exists because the intraday release time (~4:15pm ET) is
   stated but not independently verified for every historical week --
   a cutoff landing on the same calendar date as `publication_date` is
   genuinely ambiguous, and requiring strict advancement to the next
   day resolves it the conservative way automatically (falls back to
   the prior week's release), exactly matching this document's
   "ambiguous same-day boundary -> fall back to the previous confirmed
   release" rule stated above for Treasury announcements.

**A concrete, verified finding this design decision protects against**:
the overwhelming majority of nominal-coupon Treasury auction
announcements fall on a **Thursday** -- the same weekday NY Fed
publishes Primary Dealer Statistics (e.g. 185/190 for 2-Year auctions,
verified directly from the processed auction table). This makes the
"ambiguous same-day" case the *typical* case for the announcement-date
cutoff, not a rare edge case -- see
`artifacts/phase_3_primary_dealer_integration.md` for the full count
(861 of 1281 auctions).

**The pre-auction cutoff, defined for the first time in Phase 3**: "the
previous business day's market close" (per this document's own
"Planned prediction cutoffs" section, above) is computed as the
closest calendar day before `auction_date` that is not a weekend or
U.S. federal holiday. This uses the Federal Reserve holiday calendar,
not a SIFMA bond-market holiday calendar -- Good Friday is a disclosed
divergence (SIFMA recommends bond markets closed; it is not a Federal
Reserve holiday) with no practical effect on this project's weekly-
cadence dealer join (it would only matter if it moved the cutoff
across a Wednesday/Thursday boundary).

**The join itself** (`treasury_auction_stress.features.dealer_join.as_of_join`)
uses `pandas.merge_asof(..., direction="backward")` keyed on
`publication_safe_available_date`, which is structurally incapable of
selecting a later-published observation or backfilling from the
future. No auction is ever dropped for lacking dealer coverage; every
missing value carries an explicit, specific reason (no coverage yet
before 2013-04-03, a series' own regime not yet started, or the source
itself reporting a missing `"*"` value for the nearest available
release) -- see `artifacts/phase_3_primary_dealer_integration.md`.

## Phase 4: market and macro sources, three independent publication-timing rules

Phase 4 adds three more sources to the announcement/pre-auction cutoff
join, each with its own verified (or, where unverified, explicitly
conservative) publication-timing rule -- **never FRED/ALFRED, for
anything, including gap-filling or cross-validation** (see
`docs/data_source_governance.md`). See
the Phase 4 integration record for the implementation
record and the Phase 4 acceptance review for the acceptance
review that corrected the timing bugs described below.

**The shared "safe availability" rule**
(`time_utils.next_full_business_day_after`): data published/available
on date *D* is never usable until the next **full U.S. business day**
after *D* -- never merely `D + 1 calendar day`, which silently
produces a **Saturday** availability date for a Friday publication.
The acceptance review found this exact bug in both the Treasury-rates
and CFTC modules (each had independently reimplemented a slightly
different, subtly wrong version of the same rule); both now call the
one shared, tested function instead of re-deriving it.

**Treasury daily par yield curve**
(`treasury_auction_stress.features.treasury_rates_join`): Treasury's
own methodology page states inputs are collected "at or near 3:30 PM"
ET and "usually available ... by 6:00 PM Eastern Time each trading
day," with a stated possible delay and no exact historical intraday
timestamp for any date. `publication_safe_available_date` = the next
full business day after `rate_date` -- so a same-day rate is never
used for a same-day cutoff, and a Friday rate is never available on a
Saturday.

**CFTC Traders in Financial Futures (TFF), Futures Only**
(`treasury_auction_stress.data.cftc_release_calendar`,
`treasury_auction_stress.features.cftc_join`): CFTC's own materials
document a standard rule (Tuesday observation, Friday publication,
3:30 PM ET) plus **individual per-report overrides** for 31 reports
across 3 known disruptions (the 2018-2019 and 2025 lapses in
appropriations, and the 2023 ION Cleared Derivatives cyber incident) --
each report gets its own actual (or, where two official CFTC
announcements conflict, the more conservative of the two) publication
date, sourced from CFTC's own press releases and its Historical
Special Announcements page, never a single bulk end-of-window date.
The acceptance review found the original 2023 window was 3 weeks too
short (stopped at 2023-02-21 instead of the actually-affected
2023-03-14), which would have joined those weeks' auctions to a report
that had not yet been published -- fixed by extending the override
table to all 7 affected reports. CFTC positioning history is also no
longer truncated at the auction sample's own 2010-01-01 start: the
full retained source history (from 2006-06-13) is used for the as-of
join, so early-2010 auctions are never artificially unmatched at the
sample boundary.

**Philadelphia Fed RTDSM**
(`treasury_auction_stress.data.rtdsm_normalize`,
`treasury_auction_stress.features.rtdsm_join`): a "vintage" is a whole
column, not a per-observation release. RUC's quarterly vintage has a
**documented exact collection day** (the 15th of the middle month of
the quarter, stated explicitly by the Philadelphia Fed). Every
monthly-vintage variable (ROUTPUT, PCPI, IPT, EMPLOY, HSTARTS) now has
its **own** nominal-publication-day rule cited to *that variable's
own* releasing institution (BEA, BLS, Federal Reserve Board, Census/HUD
respectively) -- never one variable's evidence borrowed for another.
The acceptance review found the original implementation used a single
"day 18" rule (Federal Reserve industrial-production evidence) for all
four monthly-vintage variables, which was **unsafe** for ROUTPUT (BEA
GDP actually releases in the 23rd-29th range) -- fixed per-variable,
each clamped to the true length of its own vintage month. Quarterly
CPI was also replaced with monthly-vintage PCPI (documented decision,
the Phase 4 acceptance review) for materially better
point-in-time granularity against weekly Treasury auctions. The join
itself is two-step: `merge_asof(..., direction="backward")` against a
small vintage index finds which vintage was available as of the
cutoff, then a plain merge looks up that vintage's own latest
non-null observation -- never today's fully-revised figure, and never
forward-filled across a genuine source-reported gap (e.g. RUC's real
October 2025 government-shutdown-adjacent gap, explicitly flagged via
`{mnemonic}_source_gap_detected` rather than silently presented as an
on-time reading).

All three joins reuse the same NaT-robust `merge_asof` pattern and the
same `announcement_cutoff_date`/`pre_auction_cutoff_date` columns
(`treasury_auction_stress.features.auction_cutoffs`) as Phase 3's
dealer join. `treasury_auction_stress.features.phase4d_source_joins`
builds one join table per source (never a combined feature matrix,
deferred to Phase 5) and asserts, for every table, that no
post-auction-result column (yields, prices, accepted/tendered amounts,
allocation percentages, the coupon rate itself) is present, and that
CFTC source-report availability is never conflated with a tenor's
direct-futures-contract availability (two independent axes -- see
`treasury_auction_stress.features.cftc_join`'s module docstring).

## Phase 5: the combined feature matrix, a whitelist assembly discipline, and one corrected Phase 3 defect

Phase 5 (`treasury_auction_stress.features.feature_matrix`) is the
first phase to combine all four source families (NY Fed Primary Dealer
Statistics, Treasury rates, CFTC TFF positioning, RTDSM macro vintages)
plus a new auction-structure feature family
(`treasury_auction_stress.features.auction_candidate_features`) into
two point-in-time predictor matrices -- one per prediction cutoff
defined above. It does not introduce a new timing rule for any source;
every join reuses the exact Phase 3/4 as-of-join mechanism and safe-
availability columns described in this document.

**Assembly discipline**: the two matrices are built by *explicitly
selecting* the literal predictor names declared in
`configs/phase_5_features.yml` (via
`treasury_auction_stress.features.feature_manifest`) from each
source's join table -- never by taking "every column except a
forbidden list." A blacklist can miss a renamed field; a whitelist
cannot accidentally include one that was never asked for.
`feature_matrix.validate_predictor_matrix` additionally re-checks the
forbidden-field list as a second, independent line of defense, and
`leakage_audit.forbidden_field_scan` adds a case-insensitive
substring/pattern scan so a renamed-but-equivalent result field would
still be caught.

**Auction-structure features are cutoff-independent, but their
same-day-announcement handling is a new, explicit rule**: trailing/
previous-auction features (e.g. `previous_same_tenor_offering_amt`,
`trailing_nominal_coupon_supply_8`) must never treat one auction as
preceding another auction announced on the exact same calendar date --
a real, verified scenario in this project's own data (35 auctions
share their own tenor's announcement date with another same-tenor
auction; 401 distinct announcement dates cover more than one auction).
`auction_candidate_features.py` enforces this via
`pandas.merge_asof(..., direction="backward", allow_exact_matches=False)`
for single-lookback features and via a per-distinct-date aggregation
step before any rolling window for trailing-sum features -- see that
module's docstring for the full mechanism.

**One previously-disclosed Phase 3 limitation was corrected here**: the
Phase 4 acceptance review found (but explicitly left out of scope) that
`dealer_stats_normalize.py`'s `publication_safe_available_date` used a
plain `+ 1 calendar day` added to `publication_date` -- safe in the
ordinary Thursday-publication case, but capable of landing on a
Saturday whenever a federal holiday shifts `publication_date` itself
onto a Friday. Phase 5 fixed this by calling the same shared
`next_full_business_day_after` function Treasury rates and CFTC
already use. Verified directly: re-running the full auction join
against the pre-fix and post-fix dealer feature tables selects the
identical dealer release for every one of this project's 1,281 real
auctions, under both cutoffs -- because every cutoff this project ever
computes is itself always a business day, so no cutoff could ever fall
in the narrow (weekend-only) window the defect could have affected. See
`artifacts/phase_3_primary_dealer_integration.md`'s "Phase 5 correction"
section for the full analysis.

**Cross-cutoff monotonicity, new in Phase 5**: because the pre-auction
cutoff is always later than (or equal to, in a degenerate case) the
announcement cutoff for the same auction, the pre-auction view's
selected release identity for every source must never be earlier than
the announcement view's own selection. `leakage_audit.
assert_cross_cutoff_monotonicity` proves this directly against every
one of this project's real auctions, for all four source families.

**No new full-sample statistic is fit anywhere in Phase 5.** Every
transform is either a fixed arithmetic/logarithmic/calendar
computation, or a source-local trailing/rolling computation that uses
only strictly preceding observations (the same `shift(1)`-before-
`rolling` discipline already established in Phase 3/4). Category
encoding, standardization, imputation, winsorization, feature
selection, and stress-threshold estimation all remain explicitly
deferred to Phase 6's own chronological training folds -- see
`docs/target_specification.md`'s Dealer Absorption Surprise section,
unchanged by Phase 5.

## Chronological validation

Because of all of the above, model evaluation must use **chronological
rolling-origin validation**: folds are built by walking forward through
time, always training on an earlier block of auctions and testing on a
later one, and never shuffling auctions randomly across folds. Random
shuffling would let a model trained partly on 2023 data "predict" a
2015 auction, which is not a real forecasting scenario and would
overstate accuracy for reasons unrelated to any of the leakage issues
above.

## Phase 6: chronological validation implemented, and a new auction-RESULT availability rule

Phase 6 (`treasury_auction_stress.evaluation`) is the first phase to
actually train and evaluate predictive models, implementing the
chronological-validation rule stated just above as an annual,
expanding-window, rolling-origin backtest -- see
`configs/phase_6_evaluation.yml` for the frozen protocol and
`artifacts/phase_6_evaluation_protocol.md` for the generated record.

**A new timing concept, not needed by any earlier phase**: every prior
phase's publication-timing rules govern when a *predictor* (dealer
stats, rates, CFTC, RTDSM) becomes safely available. Phase 6 is the
first phase that also needs to know when an auction's own *outcome*
(the thing being predicted) becomes safely available, so that an
earlier auction's real result is never used to train a fold whose test
year has not yet started, and so a training pool never accidentally
includes an auction whose own result would not really have been public
knowledge yet. Treasury's Fiscal Data API does not publish a verified
result-release timestamp (only `auction_date` and `announcemt_date`
exist on the schema), so, per this document's standing "verify, don't
assume; when unverifiable, use a disclosed conservative proxy" policy,
Phase 6 uses `result_safe_available_date = next_full_business_day_after
(auction_date)` -- the same shared safe-availability rule every other
source in this project already uses, applied here to the auction's own
result for the first time. This is disclosed as a conservative proxy,
not a verified release time: real Treasury results are published the
same day the auction closes, so this rule can only make a training row
look *less* available than it really was, never more.

**Fold construction rule**: for a test year *Y*, the fold's fit origin
is the *minimum* `announcement_cutoff_date` over year *Y*'s own test
auctions, computed once and reused identically for both the
announcement and pre-auction cutoff views (so the two views are always
compared on an identical training-eligible key set). A training
candidate must have `auction_date` in a year strictly before *Y* AND
`result_safe_available_date <= fit_origin` -- `auction_date < Y` alone
is proven, in this project's own real data, not to be sufficient:
`treasury_auction_stress.evaluation.timing`'s module docstring and
`artifacts/phase_6_evaluation_protocol.md`'s fold calendar both document
this; `treasury_auction_stress.evaluation.dealer_absorption_audit`
separately found, checking the full 8-auction lookback window at both
prediction cutoffs, 34 real (auction, lookback-slot, cutoff) violations
across 18 distinct auctions where a same-tenor auction's result would
not yet have been safely available at another auction's own forecast
cutoff (see that report's Dealer Absorption Surprise section) --
concrete evidence, not merely a hypothetical, for why this extra
condition matters. Every model is fit exactly once per fold and
frozen for that entire test year, per this document's general
chronological-validation rule above.

**Every learned preprocessing step (imputation, scaling, one-hot
encoding, hyperparameter selection) is fit strictly inside each fold's
own training pool**, per this document's standing "scalers, encoders,
imputers, thresholds ... must be fit using training data only, and
refit independently inside every chronological evaluation fold" rule --
`treasury_auction_stress.models.linear_models` implements this
directly, and `tests/test_phase6_linear_models.py` proves inner
hyperparameter selection never touches an outer fold's own test rows.

**Dealer Absorption Surprise remains explicitly deferred to Phase 7**:
`dealer_absorption.add_regime_feature`'s `shift(1)`-based trailing
mean, sorted by `auction_date`, is not automatically equivalent to
"publicly available by this auction's own forecast cutoff" -- see
`artifacts/phase_6_evaluation_protocol.md` for the full reasoning and the
real, verified violation this phase found.

## What Phase 1 is responsible for now

Phase 1 does not compute any cutoff-aware feature. Its responsibility
is narrower but foundational: every raw field retrieved must be
traceable to (a) its observation date as reported by the source, and
(b) a retrieval timestamp recorded by this project's own pipeline. That
is what makes it possible, in a later phase, to reconstruct "what was
knowable as of date X" without re-downloading history under different
assumptions.
