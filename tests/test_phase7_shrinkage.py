"""Phase 7: correctness tests for
`treasury_auction_stress.models.shrinkage.ShrinkageTenorReopeningBaseline`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.models.shrinkage import ShrinkageTenorReopeningBaseline


def _rows(tenor, is_reopening, values):
    return [
        {"tenor": tenor, "is_reopening": is_reopening, "primary_dealer_share": v}
        for v in values
    ]


def test_identical_group_means_shrink_almost_entirely_to_global_mean():
    """Zero true between-group variance -> the shrinkage weight on any
    single group's own (noisy) mean should be tiny.
    """
    rng = np.random.default_rng(0)
    rows = []
    for tenor in ("2-Year", "5-Year", "10-Year"):
        for reopening in (True, False):
            values = 0.30 + rng.normal(0, 0.01, size=20)  # same true mean everywhere
            rows.extend(_rows(tenor, reopening, values))
    df = pd.DataFrame(rows)

    model = ShrinkageTenorReopeningBaseline().fit(df)
    for weight in model.group_weight_.values():
        assert weight < 0.25


def test_distinct_group_means_with_low_noise_shrink_very_little():
    rows = []
    for i, tenor in enumerate(("2-Year", "5-Year", "10-Year")):
        for reopening in (True, False):
            base = 0.10 * i + (0.05 if reopening else 0.0)
            values = [base] * 30  # zero within-group noise, large between-group spread
            rows.extend(_rows(tenor, reopening, values))
    df = pd.DataFrame(rows)

    model = ShrinkageTenorReopeningBaseline().fit(df)
    for (tenor, reopening), weight in model.group_weight_.items():
        assert weight > 0.90


def test_sparse_group_gets_less_weight_on_its_own_mean_than_a_well_populated_one():
    rng = np.random.default_rng(1)
    rows = []
    # 5-Year, reopening=True: only 2 observations (sparse).
    rows.extend(_rows("5-Year", True, [0.80, 0.20]))
    # 5-Year, reopening=False: 50 observations (well populated), same true mean.
    rows.extend(_rows("5-Year", False, 0.50 + rng.normal(0, 0.05, size=50)))
    # A second tenor so between-group variance is estimable.
    rows.extend(_rows("10-Year", True, 0.20 + rng.normal(0, 0.05, size=50)))
    rows.extend(_rows("10-Year", False, 0.20 + rng.normal(0, 0.05, size=50)))
    df = pd.DataFrame(rows)

    model = ShrinkageTenorReopeningBaseline().fit(df)
    sparse_weight = model.group_weight_[("5-Year", True)]
    populated_weight = model.group_weight_[("5-Year", False)]
    assert sparse_weight < populated_weight


def test_single_tenor_present_needs_no_shrinkage():
    rows = _rows("2-Year", True, [0.30, 0.35, 0.40]) + _rows("2-Year", False, [0.32, 0.28])
    df = pd.DataFrame(rows)
    model = ShrinkageTenorReopeningBaseline().fit(df)
    assert model.tenor_weight_["2-Year"] == pytest.approx(1.0)
    assert model.tenor_shrunk_["2-Year"] == pytest.approx(df["primary_dealer_share"].mean())


def test_predict_falls_back_for_unseen_tenor_and_unseen_group():
    rows = _rows("2-Year", True, [0.30, 0.35, 0.40, 0.31, 0.29]) + _rows("2-Year", False, [0.32, 0.28, 0.31, 0.33, 0.30])
    df = pd.DataFrame(rows)
    model = ShrinkageTenorReopeningBaseline().fit(df)

    test_df = pd.DataFrame(
        [
            {"tenor": "30-Year", "is_reopening": True},   # unseen tenor entirely
            {"tenor": "2-Year", "is_reopening": True},     # seen group
        ]
    )
    predictions = model.predict(test_df)
    assert predictions.iloc[0] == pytest.approx(model.global_mean_)
    assert predictions.iloc[1] == pytest.approx(model.group_shrunk_[("2-Year", True)])


def test_describe_shrinkage_reports_own_mean_prior_and_weight():
    rows = _rows("2-Year", True, [0.30, 0.35, 0.40]) + _rows("5-Year", False, [0.10, 0.12, 0.11])
    df = pd.DataFrame(rows)
    model = ShrinkageTenorReopeningBaseline().fit(df)
    table = model.describe_shrinkage()
    assert set(table.columns) == {
        "tenor",
        "is_reopening",
        "n",
        "shrinkage_weight_on_own_mean",
        "wider_pool_prior_tenor_level",
        "shrunk_estimate",
    }
    assert len(table) == 2


def test_raises_on_all_missing_target():
    df = pd.DataFrame({"tenor": ["2-Year"], "is_reopening": [True], "primary_dealer_share": [float("nan")]})
    with pytest.raises(ValueError):
        ShrinkageTenorReopeningBaseline().fit(df)
