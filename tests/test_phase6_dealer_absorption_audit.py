from __future__ import annotations

import pandas as pd

from treasury_auction_stress.evaluation.dealer_absorption_audit import (
    find_same_tenor_availability_violations,
    summarize_regime_feature_risk,
)
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)


def _sample(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal sample and compute BOTH real cutoff columns via
    the actual production function -- never hand-computed cutoff dates,
    so these tests exercise the same cutoff logic Phase 6 itself uses.
    """
    df = pd.DataFrame(rows)
    df["cusip"] = df["cusip"]
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    df["announcemt_date"] = pd.to_datetime(df["announcemt_date"])
    return add_cutoff_dates(df)


def _row(cusip, tenor, auction_date, announcemt_date):
    return {"cusip": cusip, "tenor": tenor, "auction_date": auction_date, "announcemt_date": announcemt_date}


def test_finds_the_real_verified_2013_case_at_both_cutoffs():
    """Reproduces the actual, verified real-data case documented in the
    module docstring: two 7-Year auctions announced 2013-08-22, one
    calendar day apart. One business day apart is tight enough that
    the violation persists at BOTH the announcement and pre-auction
    cutoffs (verified against the real 2010-2026 sample)."""
    sample = _sample(
        [
            _row("912828RE2", "7-Year", "2013-08-28", "2013-08-22"),
            _row("912828VV9", "7-Year", "2013-08-29", "2013-08-22"),
        ]
    )
    violations = find_same_tenor_availability_violations(sample)
    assert len(violations) == 2  # one per cutoff view
    assert set(violations["cutoff_view_column"]) == {ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL}
    assert (violations["cusip"] == "912828VV9").all()
    assert (violations["previous_same_tenor_cusip"] == "912828RE2").all()
    assert (violations["lag"] == 1).all()
    assert (violations["gap_days"] < 0).all()


def test_finds_a_violation_at_a_lag_greater_than_one():
    """The broader (up to lag=8) window finds violations the original
    single-predecessor (lag=1 only) check would miss entirely: three
    same-tenor auctions in rapid succession, where the SECOND-most-
    recent (lag=2) predecessor is also not yet safely available."""
    sample = _sample(
        [
            _row("A", "7-Year", "2019-03-26", "2019-03-19"),
            _row("B", "7-Year", "2019-03-27", "2019-03-20"),
            _row("C", "7-Year", "2019-03-28", "2019-03-21"),
        ]
    )
    violations = find_same_tenor_availability_violations(sample)
    # C has both a lag=1 (vs B) and a lag=2 (vs A) violation at the announcement cutoff.
    c_ann = violations[
        (violations["cusip"] == "C") & (violations["cutoff_view_column"] == ANNOUNCEMENT_CUTOFF_COL)
    ]
    assert set(c_ann["lag"]) == {1, 2}


def test_no_violation_when_same_tenor_auctions_are_well_spaced():
    sample = _sample(
        [
            _row("A", "10-Year", "2020-01-15", "2020-01-08"),
            _row("B", "10-Year", "2020-02-15", "2020-02-08"),
            _row("C", "10-Year", "2020-03-15", "2020-03-08"),
        ]
    )
    violations = find_same_tenor_availability_violations(sample)
    assert violations.empty


def test_no_violation_for_a_tenors_first_ever_auction():
    sample = _sample([_row("A", "20-Year", "2020-05-20", "2020-05-14")])
    violations = find_same_tenor_availability_violations(sample)
    assert violations.empty


def test_window_parameter_limits_how_far_back_is_checked():
    """A violation only reachable at lag=2 must disappear if window=1
    (matching a hypothetically narrower regime feature) -- proves the
    window bound is actually respected, not a decorative parameter."""
    sample = _sample(
        [
            _row("A", "7-Year", "2019-03-26", "2019-03-19"),
            _row("B", "7-Year", "2019-03-27", "2019-03-20"),
            _row("C", "7-Year", "2019-03-28", "2019-03-21"),
        ]
    )
    narrow = find_same_tenor_availability_violations(sample, window=1)
    assert set(narrow["lag"]) == {1}
    wide = find_same_tenor_availability_violations(sample, window=8)
    assert set(wide["lag"]) == {1, 2}


def test_summarize_regime_feature_risk_reports_the_violation():
    sample = _sample(
        [
            _row("912828RE2", "7-Year", "2013-08-28", "2013-08-22"),
            _row("912828VV9", "7-Year", "2013-08-29", "2013-08-22"),
        ]
    )
    summary = summarize_regime_feature_risk(sample)
    assert summary["n_total_violations"] == 2
    assert summary["n_distinct_auctions_affected"] == 1
    assert summary["example_violation"]["cusip"] == "912828VV9"
    assert set(summary["n_violations_by_cutoff"].keys()) == {ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL}


def test_summarize_regime_feature_risk_on_safe_data_reports_zero():
    sample = _sample(
        [
            _row("A", "10-Year", "2020-01-15", "2020-01-08"),
            _row("B", "10-Year", "2020-02-15", "2020-02-08"),
        ]
    )
    summary = summarize_regime_feature_risk(sample)
    assert summary["n_total_violations"] == 0
    assert summary["example_violation"] is None


def test_real_data_full_window_both_cutoffs_matches_known_counts():
    """Pinned regression against the real, full Phase 5 sample -- if
    Phase 5's data or the safe-availability rule ever changes, this
    test should be re-verified, not silently "fixed" by updating the
    numbers to whatever the code now produces."""
    import pytest

    processed = __import__("pathlib").Path("data/processed")
    if not (processed / "feature_matrix_announcement.parquet").exists():
        pytest.skip("processed Phase 5 tables not present in this environment")
    ann = pd.read_parquet(processed / "feature_matrix_announcement.parquet")
    summary = summarize_regime_feature_risk(ann)
    assert summary["n_total_violations"] == 34
    assert summary["n_distinct_auctions_affected"] == 18
    assert summary["n_violations_by_cutoff"][ANNOUNCEMENT_CUTOFF_COL] == 19
    assert summary["n_violations_by_cutoff"][PRE_AUCTION_CUTOFF_COL] == 15
