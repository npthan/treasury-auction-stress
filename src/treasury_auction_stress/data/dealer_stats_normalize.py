"""Turn raw NY Fed Primary Dealer Statistics API responses into a
tidy, release-aware internal representation.

Never mutates or looks at the raw JSON file on disk directly -- reads
the payload built by `dealer_stats_raw_store.build_raw_payload` and
returns a long-format ("tidy": one row per series-observation) pandas
DataFrame, with the publication-timing columns
`docs/point_in_time_rules.md` requires attached at normalization time,
not deferred to the join step -- so a later phase can never
accidentally join on `observation_date` and forget publication timing.

## The publication rule, implemented (see docs/point_in_time_rules.md)

The New York Fed states its own cadence: "Data are updated on
Thursdays at approximately 4:15 p.m. with the previous week's
statistics." No source found by this project confirms an exact
historical release *minute* for every week, nor an explicit holiday-
shift rule, so -- per this project's standing "do not invent an
unverified timestamp, use a conservative date-level rule" policy --
two dates are computed for every observation:

- `publication_date`: the calendar day this project assumes NY Fed
  made the observation public -- the Wednesday `asofdate` plus one
  calendar day (Thursday), further advanced past any U.S. federal
  holiday (`time_utils.next_business_day`) on the theory that a Fed
  holiday delays both the dealer's Thursday filing and NY Fed's
  Thursday publication by at least one business day. This is a
  disclosed **heuristic** for the holiday case specifically (this
  project found no primary source confirming exact historical
  holiday-shifted release dates) -- see
  `artifacts/primary_dealer_data_quality.md`.
- `publication_safe_available_date`: the next **full U.S. business day**
  after `publication_date`
  (`treasury_auction_stress.data.time_utils.next_full_business_day_after`)
  -- this is the column every join actually uses. The extra business
  day exists because the *intraday* release time (~4:15pm ET) is
  stated but not independently verified for every historical week, so
  a prediction cutoff that lands on the same calendar date as
  `publication_date` is genuinely ambiguous (was the cutoff before or
  after 4:15pm ET?) -- exactly the "ambiguous same-day boundary -> fall
  back to the previous confirmed release" case
  `docs/point_in_time_rules.md` calls for. Requiring the cutoff's date
  to be strictly *after* `publication_date` (i.e. on or after
  `publication_safe_available_date`) resolves that ambiguity the
  conservative way automatically: such a cutoff will simply match the
  prior week's release instead.

  **Phase 5 correction** (a previously-disclosed Phase 3 limitation,
  flagged in the Phase 4 acceptance review section 16 and
  fixed here): this used to add a plain `+ 1 calendar day` to
  `publication_date`, which is safe for an ordinary Thursday
  publication (Thursday + 1 = Friday, a business day) but silently
  produces a **Saturday** `publication_safe_available_date` whenever a
  federal holiday shifts `publication_date` itself onto a Friday (e.g.
  a Thursday-holiday week, such as Thanksgiving) -- the exact same bug
  class the Phase 4 acceptance review already found and fixed in
  `treasury_rates_normalize.py` and `cftc_release_calendar.py`. It now
  calls the same shared, tested `next_full_business_day_after` function
  those modules use, rather than re-deriving a subtly different
  calendar-day rule here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from treasury_auction_stress.data.dealer_stats_schema import (
    CATEGORY_POSITION,
    COUNTERPARTY_GRANULARITY_BREAK_2022_01_05,
    DISCONTINUED_COMPONENT_STABLE_NAMES,
    LEGACY_COMPONENT_STABLE_NAMES,
    LEGACY_EXTENSION_BY_KEYID,
    LEGACY_PERIOD_KEY,
    MISSING_VALUE_TOKEN,
    REGIME_DISCONTINUED_LONG_BUCKET,
    REGIME_LEGACY_SBP2013,
    SELECTED_KEYIDS,
    SERIES_BY_KEYID,
    UNITS,
    VALUATION_MARKET,
)
from treasury_auction_stress.data.time_utils import (
    next_business_day,
    next_full_business_day_after,
    us_federal_holidays,
)

# Sentinel used only internally to make the (keyid, period) merge key
# well-defined for the common "current/plain fetch" case, where the
# real `period` value is `None` -- pandas merges handle a consistent
# sentinel far more predictably than a column of `None`/NaN.
_CURRENT_PERIOD_SENTINEL = "CURRENT"

_KEYID_COL = "original_series_code"


def pages_to_records(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every fetched series' `pd.timeseries` list into one list
    of `{keyid, asofdate, value, period}` records, plus the source URL
    each series was fetched from. `period` is `None` for a plain fetch
    or a schema-period key (e.g. `"SBP2013"`) for a legacy,
    period-scoped fetch -- see `dealer_stats_client.FetchedSeries`.
    """
    records: list[dict[str, Any]] = []
    for series in raw_payload.get("series", []):
        keyid = series["keyid"]
        url = series["url"]
        period = series.get("period")
        for obs in series.get("body", {}).get("pd", {}).get("timeseries", []):
            records.append(
                {
                    "keyid": obs.get("keyid", keyid),
                    "asofdate": obs.get("asofdate"),
                    "value": obs.get("value"),
                    "source_url": url,
                    "period": period,
                }
            )
    return records


def to_raw_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """The rawest useful representation: one row per (series, week),
    every value column a plain string exactly as the API sent it
    (including the literal `"*"` missing-value token). No parsing.
    """
    return pd.DataFrame.from_records(records)


def check_schema_drift(list_payload: dict[str, Any] | None) -> dict[str, list[str]]:
    """Compare the live `/list/timeseries.json` response's keyids
    against `dealer_stats_schema.SELECTED_KEYIDS`. Returns
    `missing_keyids` (selected but no longer listed as active -- a
    genuine, actionable drift signal) and does **not** report
    `new_keyids` as drift, since the live list legitimately contains
    hundreds of series this project deliberately never selected.
    """
    if list_payload is None:
        return {"missing_keyids": [], "checked": False}
    live_keyids = {
        entry["keyid"] for entry in list_payload.get("pd", {}).get("timeseries", [])
    }
    missing = sorted(set(SELECTED_KEYIDS) - live_keyids)
    return {"missing_keyids": missing, "checked": True}


def _null_to_na(series: pd.Series) -> pd.Series:
    """Replace this API's literal `"*"` missing-value token with a real
    missing value -- NOT the same token as the Fiscal Data auctions
    API's `"null"` string (see `dealer_stats_schema.MISSING_VALUE_TOKEN`).
    """
    return series.mask(series == MISSING_VALUE_TOKEN)


def compute_publication_dates(observation_date: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return `(publication_date, publication_safe_available_date)` for
    a series of Wednesday observation dates. See module docstring for
    the rule.
    """
    obs = pd.to_datetime(observation_date)
    if obs.empty:
        empty = pd.Series([], dtype="datetime64[ns]", index=obs.index)
        return empty, empty
    holidays = us_federal_holidays(
        start=(obs.min() - pd.Timedelta(days=7)).isoformat(),
        end=(obs.max() + pd.Timedelta(days=14)).isoformat(),
    )
    # Nominal Thursday, unless that Thursday is itself a holiday.
    nominal_thursday = obs + pd.Timedelta(days=1)
    holiday_set = set(pd.DatetimeIndex(holidays).normalize())
    publication_date = nominal_thursday.where(
        ~nominal_thursday.isin(holiday_set),
        nominal_thursday.map(lambda d: next_business_day(d - pd.Timedelta(days=1), holidays)),
    )
    publication_safe_available_date = publication_date.map(
        lambda d: next_full_business_day_after(d, holidays)
    )
    return publication_date, publication_safe_available_date


def _resolve_metadata(keyid: str, period_key: str) -> dict[str, Any] | None:
    """Metadata for one (keyid, period) pair -- see
    `dealer_stats_schema.py`'s historical-extension section for the
    three cases this distinguishes. Returns `None` for a genuinely
    unrecognized combination (never guessed or defaulted).
    """
    if period_key == _CURRENT_PERIOD_SENTINEL:
        if keyid in SERIES_BY_KEYID:
            s = SERIES_BY_KEYID[keyid]
            return {
                "stable_series_name": s.stable_name,
                "category": s.category,
                "maturity_bucket": s.maturity_bucket,
                "valuation_basis": s.valuation_basis,
                "historical_regime_id": s.regime_id,
            }
        if keyid in DISCONTINUED_COMPONENT_STABLE_NAMES:
            return {
                "stable_series_name": DISCONTINUED_COMPONENT_STABLE_NAMES[keyid],
                "category": CATEGORY_POSITION,
                "maturity_bucket": "coupons_gt_11y_combined_discontinued_2022_01_05",
                "valuation_basis": VALUATION_MARKET,
                "historical_regime_id": REGIME_DISCONTINUED_LONG_BUCKET,
            }
        return None
    if period_key == LEGACY_PERIOD_KEY:
        if keyid in LEGACY_EXTENSION_BY_KEYID:
            ext = LEGACY_EXTENSION_BY_KEYID[keyid]
            canonical = SERIES_BY_KEYID[ext.canonical_keyid]
            return {
                "stable_series_name": ext.canonical_stable_name,
                "category": canonical.category,
                "maturity_bucket": canonical.maturity_bucket,
                "valuation_basis": canonical.valuation_basis,
                "historical_regime_id": REGIME_LEGACY_SBP2013,
            }
        if keyid in LEGACY_COMPONENT_STABLE_NAMES:
            return {
                "stable_series_name": LEGACY_COMPONENT_STABLE_NAMES[keyid],
                "category": CATEGORY_POSITION,
                "maturity_bucket": None,
                "valuation_basis": VALUATION_MARKET,
                "historical_regime_id": REGIME_LEGACY_SBP2013,
            }
        return None
    return None


def normalize_dealer_stats(raw_payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse raw records into the tidy, release-aware representation.

    Every field this project promised to preserve
    (docs/point_in_time_rules.md's Primary Dealer Statistics section)
    is a column here: original series code, stable internal name,
    value, units, observation date, publication availability date(s),
    historical regime identifier, source URL, and missingness
    information. `value_raw` keeps the untouched original string
    alongside the parsed `value`, mirroring
    `treasury_auction_stress.data.normalize`'s `<field>_raw` convention.
    """
    records = pages_to_records(raw_payload)
    anomalies: dict[str, Any] = {"unselected_keyids": [], "unparseable_values": {}}
    if not records:
        return pd.DataFrame(), anomalies

    raw_df = to_raw_dataframe(records)

    raw_df["observation_date"] = pd.to_datetime(raw_df["asofdate"], format="%Y-%m-%d", errors="coerce")
    invalid_dates = int(raw_df["observation_date"].isna().sum())
    if invalid_dates:
        anomalies["unparseable_values"]["asofdate"] = invalid_dates

    raw_df["value_raw"] = raw_df["value"]
    cleaned_value = _null_to_na(raw_df["value"])
    raw_df["value"] = pd.to_numeric(cleaned_value, errors="coerce").astype("Float64")
    unparseable_numeric = int((raw_df["value"].isna() & cleaned_value.notna()).sum())
    if unparseable_numeric:
        anomalies["unparseable_values"]["value"] = unparseable_numeric

    raw_df["is_missing"] = raw_df["value"].isna()
    raw_df["missing_reason"] = pd.array([pd.NA] * len(raw_df), dtype="string")
    raw_df.loc[raw_df["value_raw"] == MISSING_VALUE_TOKEN, "missing_reason"] = (
        f"source reported the literal '{MISSING_VALUE_TOKEN}' token for this week "
        "(withheld/not available at the source) -- not imputed or forward-filled"
    )

    # Metadata is a pure function of (keyid, period): a plain fetch
    # (period is None -> the sentinel) resolves against the current
    # selected series or the one discontinued-but-fetchable middle
    # bucket; a legacy, period-scoped fetch (period == LEGACY_PERIOD_KEY)
    # resolves either onto a canonical column it directly extends
    # (`LEGACY_EXTENSION_BY_KEYID`) or its own legacy-only component
    # name (`LEGACY_COMPONENT_STABLE_NAMES`). Anything else is a
    # genuinely unrecognized keyid/period pair, reported, never guessed.
    period_key = raw_df["period"].fillna(_CURRENT_PERIOD_SENTINEL)
    resolved = [_resolve_metadata(k, p) for k, p in zip(raw_df["keyid"], period_key, strict=True)]
    unresolved_mask = pd.Series([r is None for r in resolved], index=raw_df.index)
    if unresolved_mask.any():
        anomalies["unselected_keyids"] = sorted(set(raw_df.loc[unresolved_mask, "keyid"]))
        raw_df = raw_df.loc[~unresolved_mask].reset_index(drop=True)
        resolved = [r for r, bad in zip(resolved, unresolved_mask, strict=True) if not bad]

    meta_df = pd.DataFrame(resolved, index=raw_df.index)
    raw_df = pd.concat([raw_df, meta_df], axis=1)
    raw_df["units"] = UNITS
    raw_df["has_counterparty_granularity_break_2022_01_05"] = raw_df["keyid"].isin(
        COUNTERPARTY_GRANULARITY_BREAK_2022_01_05
    )

    pub_date, pub_safe_date = compute_publication_dates(raw_df["observation_date"])
    raw_df["publication_date"] = pub_date
    raw_df["publication_safe_available_date"] = pub_safe_date

    raw_df = raw_df.rename(columns={"keyid": _KEYID_COL})
    ordered_cols = [
        _KEYID_COL,
        "stable_series_name",
        "category",
        "maturity_bucket",
        "value",
        "value_raw",
        "units",
        "valuation_basis",
        "observation_date",
        "publication_date",
        "publication_safe_available_date",
        "historical_regime_id",
        "has_counterparty_granularity_break_2022_01_05",
        "source_url",
        "is_missing",
        "missing_reason",
    ]
    out = raw_df[ordered_cols].sort_values([_KEYID_COL, "observation_date"]).reset_index(drop=True)
    return out, anomalies


def pivot_wide(long_df: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
    """Pivot the tidy long-format table to one row per `observation_date`,
    one column per `stable_series_name` -- the shape
    `treasury_auction_stress.features.dealer_candidate_features` needs
    for rolling/diff computations and the as-of join (one shared
    `publication_safe_available_date` per observation week, since NY
    Fed publishes every series together in a single weekly release --
    verified: every selected series shares the same observation-date
    calendar, see artifacts/primary_dealer_data_quality.md).

    A week absent for a given series (e.g. before its regime start)
    becomes NaN in that column, indistinguishable at this layer from a
    week present but flagged `is_missing` (the long-format table is the
    source of truth for *why* a value is absent; see
    `treasury_auction_stress.features.dealer_join.classify_missing_reason`
    for how the join layer tells the two apart).
    """
    wide = long_df.pivot_table(
        index="observation_date",
        columns="stable_series_name",
        values=value_col,
        aggfunc="first",
    )
    wide.columns.name = None
    pub_dates = (
        long_df.groupby("observation_date")[["publication_date", "publication_safe_available_date"]]
        .first()
        .reindex(wide.index)
    )
    return wide.join(pub_dates).reset_index()
