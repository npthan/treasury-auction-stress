"""Phase 7: correctness tests for
`treasury_auction_stress.models.logistic_classifier`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.models.logistic_classifier import (
    FALLBACK_C,
    fit_predict_logistic_classifier,
)

FEATURE_COLS = ["tenor", "x1", "x2"]
LABEL_COL = "label"


def _synthetic(n: int, *, n_years: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenors = ["2-Year", "5-Year"]
    rows = []
    for i in range(n):
        x1 = rng.normal(0, 1)
        x2 = rng.normal(0, 1)
        label = int(x1 + x2 > 0.5)
        year = 2010 + (i % n_years)
        auction_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=i)
        rows.append(
            {
                "tenor": tenors[i % 2],
                "x1": x1,
                "x2": x2,
                LABEL_COL: label,
                "auction_date": auction_date,
                "announcement_cutoff_date": auction_date,
                "result_safe_available_date": auction_date + pd.Timedelta(days=1),
            }
        )
    return pd.DataFrame(rows)


def test_fit_predict_uses_inner_cv_with_enough_history(monkeypatch):
    import treasury_auction_stress.models.logistic_classifier as module

    class _FakeHoldout:
        usable = True

        def __init__(self, inner_train, inner_val):
            self.inner_train = inner_train
            self.inner_val = inner_val
            self.reason_if_unusable = None

    train_df = _synthetic(200, seed=1)
    inner_train, inner_val = train_df.iloc[:150], train_df.iloc[150:]
    monkeypatch.setattr(module, "build_inner_holdout", lambda df: _FakeHoldout(inner_train, inner_val))

    test_df = _synthetic(20, seed=2)
    result = fit_predict_logistic_classifier(train_df, test_df, feature_cols=FEATURE_COLS, label_col=LABEL_COL)

    assert result.used_inner_cv is True
    assert 0.0 <= result.predicted_proba.min() and result.predicted_proba.max() <= 1.0
    assert result.decision_threshold == pytest.approx(result.train_prevalence)
    assert set(result.predicted_label.unique()) <= {True, False}


def test_fit_predict_falls_back_with_insufficient_history():
    train_df = _synthetic(30, n_years=1, seed=3)
    test_df = _synthetic(5, n_years=1, seed=4)
    result = fit_predict_logistic_classifier(train_df, test_df, feature_cols=FEATURE_COLS, label_col=LABEL_COL)
    assert result.used_inner_cv is False
    assert result.selected_c == FALLBACK_C


def test_rows_with_missing_label_are_excluded_from_fitting():
    train_df = _synthetic(50, seed=5)
    train_df.loc[train_df.index[:10], LABEL_COL] = np.nan
    test_df = _synthetic(5, seed=6)
    result = fit_predict_logistic_classifier(train_df, test_df, feature_cols=FEATURE_COLS, label_col=LABEL_COL)
    assert result.n_train_rows == 40


def test_raises_with_fewer_than_two_classes():
    train_df = _synthetic(30, seed=7)
    train_df[LABEL_COL] = 0
    test_df = _synthetic(5, seed=8)
    with pytest.raises(ValueError):
        fit_predict_logistic_classifier(train_df, test_df, feature_cols=FEATURE_COLS, label_col=LABEL_COL)


def test_raises_when_no_rows_have_a_label():
    train_df = _synthetic(30, seed=9)
    train_df[LABEL_COL] = np.nan
    test_df = _synthetic(5, seed=10)
    with pytest.raises(ValueError):
        fit_predict_logistic_classifier(train_df, test_df, feature_cols=FEATURE_COLS, label_col=LABEL_COL)
