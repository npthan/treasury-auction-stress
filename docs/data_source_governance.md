# Data Source Governance

This document explains, in one place, how this project decides which
external data sources it is allowed to use, and records the specific
decision to exclude FRED/ALFRED from Phase 4. **This is documentation
of a project decision, not a legal opinion.**

## Why first-party and research-authorized sources are preferred

Every source this project has ingested so far (Treasury Fiscal Data,
NY Fed Primary Dealer Statistics, and, from Phase 4A onward, Treasury's
own daily par yield curve, CFTC's Traders in Financial Futures report,
and the Philadelphia Fed's Real-Time Data Set for Macroeconomists) is
published directly by the government body or Federal Reserve Bank that
originates the underlying data, or (RTDSM) by an institution that
explicitly states a research/forecasting purpose for its own
redistribution of primary-agency data. This matters for two independent
reasons this project treats separately:

1. **Analytical correctness.** A first-party source lets this project
   verify, from primary documentation, exactly what a field means, how
   it is timed, and how its definition has changed over time --
   exactly the point-in-time discipline `docs/point_in_time_rules.md`
   depends on. A downstream aggregator can introduce its own timing,
   revision, or definitional conventions that are not the originating
   agency's.
2. **Source-governance risk.** Separately from whether a source's data
   is analytically trustworthy, this project must consider whether the
   source's own terms of use permit the way this project intends to
   use the data -- including, eventually, machine-learning model
   training. These are genuinely different questions: a source can be
   analytically excellent and still carry terms that do not clearly
   permit this project's intended use, or vice versa.

## Why FRED/ALFRED is excluded from this implementation

**FRED and ALFRED are deliberately excluded from Phase 4 and from every
future phase until this decision is explicitly revisited.** No FRED or
ALFRED data, and no `FRED_API_KEY`, is used anywhere in this
repository's code.

The reason is a source-governance decision, not a data-quality
judgment and **not a claim that using FRED would be illegal**: the
current FRED Services terms of use appeared, on review, to not clearly
and unambiguously permit this project's intended future use (training
machine-learning models on the retrieved data). Rather than proceed on
an uncertain reading of those terms, or attempt to characterize them
definitively (this document is not a legal opinion and does not
attempt one), this project simply does not use FRED/ALFRED until a
written clarification from the Federal Reserve Bank of St. Louis (FRED's
publisher) or from the project owner, obtained with the project owner's explicit
involvement, resolves the ambiguity one way or the other.

Concretely, this means:

- No code in this repository calls `api.stlouisfed.org` or any other
  FRED/ALFRED endpoint.
- No code in this repository reads the `FRED_API_KEY` environment
  variable, under any circumstance, even if a value is present in the
  environment.
- `configs/sources.yml`'s `alfred_fred_point_in_time` entry is retained,
  marked `authorized_in_current_phase: false` and
  `permitted_project_use: "NONE"`, purely so a future session has a
  record of what was considered and explicitly excluded, rather than
  silently omitted (which could look like it was never evaluated).
- `configs/sources.yml`'s superseded `cftc_commitments_of_traders`
  placeholder entry (a Phase-0-era generic "Commitments of Traders"
  description) is likewise retained, unauthorized, now superseded by
  the more specific `cftc_tff_futures_only` entry Phase 4B actually
  uses.

## The Philadelphia Fed's RTDSM is a separate, independently-justified source -- not a workaround

Phase 4C uses the Federal Reserve Bank of Philadelphia's **Real-Time
Data Set for Macroeconomists (RTDSM)** for point-in-time macroeconomic
vintages, in place of the FRED/ALFRED source originally planned for
this concept. This is deliberately **not** framed as "the same data
via a different door": RTDSM is evaluated and authorized independently,
on its own terms, because:

- It is published directly by a Federal Reserve Bank as a first-party
  research dataset, not redistributed from another agency's API by a
  third party.
- The Philadelphia Fed's own documentation states RTDSM is a
  collection of historical vintages/snapshots explicitly intended for
  macroeconomic **research**, **verifying empirical results**,
  **policy analysis**, and **forecasting** -- see
  `configs/sources.yml`'s `philadelphia_fed_rtdsm` entry for the exact
  citation and the terms-review date. That stated purpose lines up
  directly with this project's own stated research purpose
  (`docs/project_plan.md`).
- No credential is required.
- If RTDSM's own terms were later found to carry the same ambiguity
  problem this document describes for FRED, this project would treat
  that exactly the same way: stop, document, and seek clarification --
  RTDSM is not being used *because* it lacks FRED's problem without
  having actually been checked; it was checked, and its stated purpose
  is the reason it is authorized.

RTDSM's own recommended academic citation is Croushore, Dean and Tom
Stark, "A Real-Time Data Set for Macroeconomists," *Journal of
Econometrics*, Vol. 105, November 2001, pp. 111-130 -- cited here and
in `artifacts/rtdsm_macro_vintage_data_quality.md` per the Philadelphia
Fed's own attribution request. Using RTDSM does not imply Federal
Reserve System or Federal Reserve Board endorsement of this project or
its conclusions.

## No FRED-derived data exists in this repository

As of Phase 4, this statement is verifiable directly: no file under
`data/raw/`, `data/interim/`, or `data/processed/` originates from a
FRED or ALFRED request (none of this project's downloader modules ever
constructs a `stlouisfed.org` URL), and a repository-wide search for
`FRED_API_KEY`/`fred.stlouisfed.org`/`api.stlouisfed.org` in this
project's own source code (excluding documentation that explains this
exclusion) returns no matches. This is re-checked as part of every
phase's final verification gate.

## The difference between source access permission and analytical correctness

These are independent axes, and this project evaluates them
separately for every source:

- A source can be **freely accessible with no restrictive terms** and
  still be **analytically wrong to use naively** -- e.g. joining a
  weekly release by its observation date instead of its publication
  date, which is exactly the leakage bug class `docs/point_in_time_rules.md`
  exists to prevent.
- A source can be **analytically excellent** (well-documented,
  point-in-time reconstructible, stable) and still carry **terms that
  do not clearly permit** a specific downstream use -- which is
  exactly the FRED situation this document describes.

Passing one test does not exempt a source from the other. Every source
in `configs/sources.yml` is expected to carry evidence for both.

## Terms can change; recheck before public release

Every `terms_reviewed_date` recorded in `configs/sources.yml` is the
date this project last actually read that source's terms of use --
not an assumption that the terms are static. Before this project's
results are shared publicly, distributed, or used to train a model
that will itself be distributed, every source's terms should be
re-read, not assumed unchanged from their `terms_reviewed_date`. This
document, and the per-source entries in `configs/sources.yml`, are
working notes for an active research project, not a durable legal
clearance.
