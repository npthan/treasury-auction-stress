from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.evaluation.timing import (
    RESULT_SAFE_AVAILABLE_DATE_COL,
    Fold,
    add_result_safe_available_date,
    annual_fold_years,
    assert_fold_is_leakage_safe,
    build_fold,
    build_inner_holdout,
)
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL
from treasury_auction_stress.features.feature_matrix import AUCTION_KEY_COL


def _row(cusip, tenor, auction_date, announcemt_date, target=0.3, is_reopening=False):
    return {
        "cusip": cusip,
        "tenor": tenor,
        "auction_date": pd.Timestamp(auction_date),
        "announcemt_date": pd.Timestamp(announcemt_date),
        ANNOUNCEMENT_CUTOFF_COL: pd.Timestamp(announcemt_date),
        "is_reopening": is_reopening,
        "primary_dealer_share": target,
        AUCTION_KEY_COL: f"{cusip}_{pd.Timestamp(auction_date).date().isoformat()}",
    }


def _basic_pool() -> pd.DataFrame:
    rows = []
    # Five years of quarterly same-tenor auctions, 2010-2014, then one
    # test-year auction in 2015.
    for year in range(2010, 2015):
        for q, month in enumerate([1, 4, 7, 10]):
            rows.append(_row(f"A{year}{q}", "10-Year", f"{year}-{month:02d}-15", f"{year}-{month:02d}-01"))
    rows.append(_row("A2015", "10-Year", "2015-01-15", "2015-01-02"))
    df = pd.DataFrame(rows)
    return add_result_safe_available_date(df)


def test_add_result_safe_available_date_is_next_business_day():
    df = pd.DataFrame({"auction_date": [pd.Timestamp("2026-09-10")]})  # a Thursday
    out = add_result_safe_available_date(df)
    assert out[RESULT_SAFE_AVAILABLE_DATE_COL].iloc[0] == pd.Timestamp("2026-09-11")


def test_add_result_safe_available_date_skips_weekend():
    # 2026-09-11 is a Friday -> next business day is Monday 2026-09-14.
    df = pd.DataFrame({"auction_date": [pd.Timestamp("2026-09-11")]})
    out = add_result_safe_available_date(df)
    assert out[RESULT_SAFE_AVAILABLE_DATE_COL].iloc[0] == pd.Timestamp("2026-09-14")


def test_build_fold_basic_expanding_window():
    pool = _basic_pool()
    fold = build_fold(pool, test_year=2015)
    assert len(fold.test) == 1
    assert fold.test["auction_date"].iloc[0] == pd.Timestamp("2015-01-15")
    assert len(fold.train) == 20  # all of 2010-2014
    assert fold.fit_origin == pd.Timestamp("2015-01-02")
    assert_fold_is_leakage_safe(fold)


def test_build_fold_train_test_disjoint_and_no_shuffle_needed():
    pool = _basic_pool()
    fold = build_fold(pool, test_year=2013)
    train_keys = set(fold.train[AUCTION_KEY_COL])
    test_keys = set(fold.test[AUCTION_KEY_COL])
    assert train_keys.isdisjoint(test_keys)
    assert len(fold.train) == 12  # 2010, 2011, 2012 -> 3*4


def test_build_fold_raises_on_year_with_no_rows():
    pool = _basic_pool()
    with pytest.raises(ValueError):
        build_fold(pool, test_year=1999)


def test_build_fold_is_invariant_to_input_row_order():
    pool = _basic_pool()
    shuffled = pool.sample(frac=1.0, random_state=7).reset_index(drop=True)
    fold_a = build_fold(pool, test_year=2014)
    fold_b = build_fold(shuffled, test_year=2014)
    assert sorted(fold_a.train[AUCTION_KEY_COL]) == sorted(fold_b.train[AUCTION_KEY_COL])
    assert sorted(fold_a.test[AUCTION_KEY_COL]) == sorted(fold_b.test[AUCTION_KEY_COL])
    assert fold_a.fit_origin == fold_b.fit_origin


def test_same_announcement_date_across_year_boundary_excludes_the_earlier_sibling():
    """Modeled directly on a real, verified case in this project's data
    (two same-tenor auctions announced on the identical date, auctioned
    a few days apart -- see
    treasury_auction_stress.evaluation.dealer_absorption_audit): two
    7-Year auctions share announcement date 2013-12-30. One auctions on
    2013-12-31 (year Y), the other on 2014-01-02 (year Y+1). Building
    the 2014 fold must NOT let the Dec-31 sibling's result train the
    forecast for the Jan-2 sibling, because both share a fit origin
    (2013-12-30) that precedes either auction's own settlement.
    """
    rows = [
        _row("REOPEN", "7-Year", "2013-12-31", "2013-12-30", target=0.5),
        _row("NEWISSUE", "7-Year", "2014-01-02", "2013-12-30", target=0.5),
        # Ordinary, safely-available prior history so the fold isn't empty.
        _row("EARLIER1", "7-Year", "2013-06-01", "2013-05-15", target=0.3),
        _row("EARLIER2", "7-Year", "2013-03-01", "2013-02-15", target=0.3),
    ]
    pool = add_result_safe_available_date(pd.DataFrame(rows))
    fold = build_fold(pool, test_year=2014)

    assert "REOPEN" not in " ".join(fold.train[AUCTION_KEY_COL])
    assert "NEWISSUE" in " ".join(fold.test[AUCTION_KEY_COL])
    assert_fold_is_leakage_safe(fold)


def test_assert_fold_is_leakage_safe_rejects_a_deliberately_leaky_fold():
    """Negative control: a fold whose training pool contains a row with
    a safe-availability date AFTER the fit origin must be rejected."""
    pool = _basic_pool()
    fold = build_fold(pool, test_year=2013)

    poisoned_row = pool.iloc[[-1]].copy()  # the 2015 row: far future safe date
    poisoned_train = pd.concat([fold.train, poisoned_row], ignore_index=True)
    leaky_fold = Fold(
        test_year=fold.test_year,
        fit_origin=fold.fit_origin,
        train=poisoned_train,
        test=fold.test,
        n_train_dropped_for_year_boundary=fold.n_train_dropped_for_year_boundary,
    )
    with pytest.raises(AssertionError):
        assert_fold_is_leakage_safe(leaky_fold)


def test_assert_fold_is_leakage_safe_rejects_train_test_overlap():
    pool = _basic_pool()
    fold = build_fold(pool, test_year=2013)
    leaky_fold = Fold(
        test_year=fold.test_year,
        fit_origin=fold.fit_origin,
        train=pd.concat([fold.train, fold.test], ignore_index=True),
        test=fold.test,
        n_train_dropped_for_year_boundary=fold.n_train_dropped_for_year_boundary,
    )
    with pytest.raises(AssertionError):
        assert_fold_is_leakage_safe(leaky_fold)


def test_annual_fold_years():
    pool = _basic_pool()
    assert annual_fold_years(pool) == list(range(2010, 2016))


def _dense_pool(start_year=2010, end_year=2015) -> pd.DataFrame:
    """~18 auctions/year across 3 tenors -- enough to clear
    MIN_INNER_CV_TRAIN_ROWS once a few years accumulate."""
    rows = []
    for year in range(start_year, end_year):
        for tenor in ("2-Year", "5-Year", "10-Year"):
            for month in range(1, 13):
                rows.append(_row(f"A{year}{tenor}{month}", tenor, f"{year}-{month:02d}-15", f"{year}-{month:02d}-01"))
    return add_result_safe_available_date(pd.DataFrame(rows))


def test_build_inner_holdout_usable_with_enough_history():
    pool = _dense_pool(2010, 2016)
    fold = build_fold(pool, test_year=2015)
    holdout = build_inner_holdout(fold.train)
    assert holdout.usable
    assert holdout.inner_val["auction_date"].dt.year.nunique() == 1
    assert holdout.inner_val["auction_date"].dt.year.iloc[0] == 2014
    assert holdout.inner_train["auction_date"].dt.year.max() == 2013


def test_build_inner_holdout_unusable_with_one_year_of_history():
    rows = [_row(f"A{i}", "10-Year", f"2014-{m:02d}-15", f"2014-{m:02d}-01") for i, m in enumerate([1, 4, 7, 10])]
    pool = add_result_safe_available_date(pd.DataFrame(rows))
    holdout = build_inner_holdout(pool)
    assert not holdout.usable
    assert "year" in holdout.reason_if_unusable


def test_result_safe_available_date_exactly_equal_to_fit_origin_is_included():
    """Boundary equality case: a training candidate whose
    result_safe_available_date lands EXACTLY on the fold's fit origin
    must be INCLUDED (`<=`, not `<`)."""
    rows = [
        # Wednesday 2013-12-25 is a federal holiday (Christmas); a
        # Tuesday 2013-12-24 auction's next full business day is
        # Thursday 2013-12-26.
        _row("BOUNDARY", "10-Year", "2013-12-24", "2013-11-01", target=0.4),
        _row("TESTROW", "10-Year", "2014-01-15", "2013-12-26", target=0.5),
    ]
    pool = add_result_safe_available_date(pd.DataFrame(rows))
    fold = build_fold(pool, test_year=2014)
    assert fold.fit_origin == pd.Timestamp("2013-12-26")
    assert "BOUNDARY" in " ".join(fold.train[AUCTION_KEY_COL])


def test_result_safe_available_date_one_day_after_fit_origin_is_excluded():
    rows = [
        _row("TOOLATE", "10-Year", "2013-12-24", "2013-11-01", target=0.4),  # safe date 2013-12-26
        _row("TESTROW", "10-Year", "2014-01-15", "2013-12-25", target=0.5),  # fit_origin 2013-12-25
    ]
    pool = add_result_safe_available_date(pd.DataFrame(rows))
    fold = build_fold(pool, test_year=2014)
    assert fold.fit_origin == pd.Timestamp("2013-12-25")
    assert "TOOLATE" not in " ".join(fold.train[AUCTION_KEY_COL])


def test_future_row_appended_to_pool_does_not_change_an_earlier_folds_train_or_test():
    pool = _basic_pool()
    fold_before = build_fold(pool, test_year=2013)

    future_row = pd.DataFrame([_row("FUTURE", "10-Year", "2099-06-01", "2099-05-25")])
    future_row = add_result_safe_available_date(future_row)
    extended_pool = pd.concat([pool, future_row], ignore_index=True)
    fold_after = build_fold(extended_pool, test_year=2013)

    assert sorted(fold_before.train[AUCTION_KEY_COL]) == sorted(fold_after.train[AUCTION_KEY_COL])
    assert sorted(fold_before.test[AUCTION_KEY_COL]) == sorted(fold_after.test[AUCTION_KEY_COL])
    assert fold_before.fit_origin == fold_after.fit_origin


def test_training_row_count_never_exceeds_pool_minus_test_year():
    pool = _basic_pool()
    fold = build_fold(pool, test_year=2015)
    prior_year_rows = pool.loc[pool["auction_date"].dt.year < 2015]
    assert len(fold.train) <= len(prior_year_rows)
    assert len(fold.train) + fold.n_train_dropped_for_year_boundary == len(prior_year_rows)


def test_build_inner_holdout_unusable_with_too_few_rows():
    rows = [_row("A", "10-Year", "2013-01-15", "2013-01-01"), _row("B", "10-Year", "2014-01-15", "2014-01-01")]
    pool = add_result_safe_available_date(pd.DataFrame(rows))
    holdout = build_inner_holdout(pool)
    assert not holdout.usable
    assert "row" in holdout.reason_if_unusable
