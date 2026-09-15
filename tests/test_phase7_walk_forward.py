"""Phase 7: correctness tests for
`treasury_auction_stress.evaluation.walk_forward.walk_forward_transform`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.evaluation.timing import add_result_safe_available_date
from treasury_auction_stress.evaluation.walk_forward import walk_forward_transform
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL


def _pool(rows):
    df = pd.DataFrame(rows)
    df[ANNOUNCEMENT_CUTOFF_COL] = df["auction_date"]
    return add_result_safe_available_date(df, date_col="auction_date")


def _mean_fit(train_df):
    return float(train_df["value"].mean())


def _residual_transform(model_mean, test_df):
    out = test_df.copy()
    out["residual"] = out["value"] - model_mean
    out["fitted_on_mean"] = model_mean
    return out


def test_walk_forward_excludes_the_earliest_year_entirely():
    rows = [
        {"auction_key": "a2010", "auction_date": pd.Timestamp("2010-01-01"), "value": 10.0},
        {"auction_key": "a2011", "auction_date": pd.Timestamp("2011-01-01"), "value": 20.0},
        {"auction_key": "a2012", "auction_date": pd.Timestamp("2012-01-01"), "value": 30.0},
    ]
    pool = _pool(rows)
    result = walk_forward_transform(pool, fit_fn=_mean_fit, transform_fn=_residual_transform)

    assert set(result["auction_key"]) == {"a2011", "a2012"}
    # 2011's model was fit only on 2010 (mean=10.0) -> residual 10.0.
    r2011 = result.loc[result["auction_key"] == "a2011"].iloc[0]
    assert r2011["fitted_on_mean"] == pytest.approx(10.0)
    assert r2011["residual"] == pytest.approx(10.0)
    # 2012's model was fit on 2010+2011 (mean=15.0) -> residual 15.0.
    r2012 = result.loc[result["auction_key"] == "a2012"].iloc[0]
    assert r2012["fitted_on_mean"] == pytest.approx(15.0)
    assert r2012["residual"] == pytest.approx(15.0)


def test_walk_forward_never_uses_a_years_own_rows_to_fit_its_own_model():
    """A deliberately poisoned later year (huge outlier value) must
    never change an earlier year's own fitted mean or residual --
    proof that `fit_fn` only ever sees strictly-prior years.
    """
    rows = [
        {"auction_key": "a2010", "auction_date": pd.Timestamp("2010-01-01"), "value": 10.0},
        {"auction_key": "a2011", "auction_date": pd.Timestamp("2011-01-01"), "value": 20.0},
        {"auction_key": "a2012", "auction_date": pd.Timestamp("2012-01-01"), "value": 999_999.0},
    ]
    pool = _pool(rows)
    result = walk_forward_transform(pool, fit_fn=_mean_fit, transform_fn=_residual_transform)
    r2011 = result.loc[result["auction_key"] == "a2011"].iloc[0]
    assert r2011["fitted_on_mean"] == pytest.approx(10.0)


def test_walk_forward_returns_empty_frame_for_single_year_pool():
    rows = [{"auction_key": "a2010", "auction_date": pd.Timestamp("2010-01-01"), "value": 10.0}]
    pool = _pool(rows)
    result = walk_forward_transform(pool, fit_fn=_mean_fit, transform_fn=_residual_transform)
    assert result.empty


def test_walk_forward_respects_availability_rule_not_just_year_boundary():
    """A December auction announced just before a January test year's
    own earliest announcement must still be excluded from that year's
    training fold if its own result is not yet safely available --
    exactly the same rule `build_fold` enforces for outer folds.
    """
    rows = [
        {"auction_key": "dec2010", "auction_date": pd.Timestamp("2010-12-30"), "value": 5.0},
        {"auction_key": "jan2011", "auction_date": pd.Timestamp("2011-01-03"), "value": 20.0},
        {"auction_key": "later2012", "auction_date": pd.Timestamp("2012-06-01"), "value": 30.0},
    ]
    pool = _pool(rows)
    # Force jan2011's announcement cutoff to be BEFORE dec2010's own
    # safe-availability date, so it cannot use dec2010 in its own
    # training fold at all -- the fold for test_year=2011 then has an
    # EMPTY training pool and must be skipped, matching the outer-fold
    # discipline this function reuses.
    pool.loc[pool["auction_key"] == "jan2011", ANNOUNCEMENT_CUTOFF_COL] = pd.Timestamp("2010-12-29")
    result = walk_forward_transform(pool, fit_fn=_mean_fit, transform_fn=_residual_transform)
    assert "jan2011" not in set(result["auction_key"])
