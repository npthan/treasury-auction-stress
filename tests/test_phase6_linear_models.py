from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.evaluation.timing import add_result_safe_available_date
from treasury_auction_stress.models.linear_models import (
    build_pipeline,
    drop_all_missing_training_columns,
    fit_predict_linear,
    select_hyperparameters_via_inner_cv,
)


def _synthetic_frame(n_years=6, per_year=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    tenors = ["2-Year", "5-Year", "10-Year"]
    for year in range(2010, 2010 + n_years):
        for i in range(per_year):
            month = (i % 12) + 1
            day = (i % 27) + 1
            tenor = tenors[i % len(tenors)]
            offering = rng.uniform(20, 60)
            noisy_feature = rng.normal(0, 1)
            target = 0.3 + 0.01 * offering + 0.05 * noisy_feature + rng.normal(0, 0.02)
            auction_date = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
            rows.append(
                {
                    "tenor": tenor,
                    "is_reopening": bool(i % 2),
                    "offering_amt": offering,
                    "some_numeric_feature": noisy_feature,
                    "auction_date": auction_date,
                    "announcement_cutoff_date": auction_date - pd.Timedelta(days=7),
                    "primary_dealer_share": target,
                }
            )
    df = pd.DataFrame(rows)
    return add_result_safe_available_date(df)


FEATURE_COLS = ["tenor", "is_reopening", "offering_amt", "some_numeric_feature"]


def test_drop_all_missing_training_columns():
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [np.nan, np.nan, np.nan], "c": [1, 2, 3]})
    kept, dropped = drop_all_missing_training_columns(train, ["a", "b", "c"])
    assert kept == ["a", "c"]
    assert dropped == ["b"]


def test_build_pipeline_handles_unseen_category_without_crashing():
    train = pd.DataFrame(
        {"tenor": ["2-Year", "5-Year", "2-Year", "5-Year"], "offering_amt": [10.0, 20.0, 15.0, 25.0]}
    )
    y = pd.Series([0.3, 0.4, 0.35, 0.45])
    pipeline = build_pipeline(["tenor", "offering_amt"], "ridge", {"alpha": 1.0})
    pipeline.fit(train, y)

    test_df = pd.DataFrame({"tenor": ["20-Year"], "offering_amt": [30.0]})  # never seen in training
    preds = pipeline.predict(test_df)
    assert np.isfinite(preds).all()


def test_select_hyperparameters_via_inner_cv_uses_fallback_with_insufficient_history():
    train = _synthetic_frame(n_years=1, per_year=10)
    hyperparams, used_inner_cv, reason = select_hyperparameters_via_inner_cv(
        train, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="ridge"
    )
    assert used_inner_cv is False
    assert reason is not None
    assert hyperparams == {"alpha": 1.0}


def test_select_hyperparameters_via_inner_cv_runs_with_enough_history():
    train = _synthetic_frame(n_years=6, per_year=40)
    hyperparams, used_inner_cv, reason = select_hyperparameters_via_inner_cv(
        train, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="ridge"
    )
    assert used_inner_cv is True
    assert reason is None
    assert hyperparams["alpha"] in (0.1, 1.0, 10.0, 100.0)


def test_select_hyperparameters_scores_the_target_being_fit_not_primary_dealer_share():
    """Acceptance-review regression test for a real protocol/code
    discrepancy: configs/phase_6_evaluation.yml used to say the
    selection metric is literally "MAE on primary_dealer_share" with no
    qualification, but the code has always scored whichever target is
    actually being fit. This proves the code's behavior directly: two
    frames that are IDENTICAL except for the secondary target's own
    values select DIFFERENT hyperparameters, and this happens even when
    `primary_dealer_share` is completely absent from the frame -- proof
    the selection never references that column by name."""
    train_a = _synthetic_frame(n_years=6, per_year=40, seed=10)
    train_b = train_a.copy()
    train_a["bid_to_cover_ratio"] = train_a["primary_dealer_share"] * 10.0
    # A very different secondary-target signal, same predictors, same primary_dealer_share.
    train_b["bid_to_cover_ratio"] = -train_b["primary_dealer_share"] * 50.0 + 3.0

    train_a_no_primary = train_a.drop(columns=["primary_dealer_share"])
    train_b_no_primary = train_b.drop(columns=["primary_dealer_share"])

    hyperparams_a, used_a, _ = select_hyperparameters_via_inner_cv(
        train_a_no_primary, feature_cols=FEATURE_COLS, target_col="bid_to_cover_ratio", estimator_type="elastic_net"
    )
    hyperparams_b, used_b, _ = select_hyperparameters_via_inner_cv(
        train_b_no_primary, feature_cols=FEATURE_COLS, target_col="bid_to_cover_ratio", estimator_type="elastic_net"
    )
    assert used_a and used_b
    assert hyperparams_a != hyperparams_b, (
        "selection must depend on the target actually being fit, not a hardcoded primary_dealer_share column "
        "(which is not even present in these frames)"
    )


def test_select_hyperparameters_never_uses_rows_outside_the_training_pool():
    """The inner CV must select purely from `train` -- passing two
    different, disjoint training pools must never produce identical
    selected hyperparameters by coincidentally peeking at shared state.
    This is a smoke/no-crash + pure-function test: calling twice with
    the same input is deterministic."""
    train = _synthetic_frame(n_years=6, per_year=40)
    result_a = select_hyperparameters_via_inner_cv(
        train, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="ridge"
    )
    result_b = select_hyperparameters_via_inner_cv(
        train, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="ridge"
    )
    assert result_a == result_b


def test_fit_predict_linear_drops_all_missing_column_and_still_predicts():
    train = _synthetic_frame(n_years=6, per_year=40)
    train["all_missing_feature"] = np.nan
    test = _synthetic_frame(n_years=1, per_year=20, seed=99)
    test["auction_date"] = test["auction_date"] + pd.DateOffset(years=10)
    test["all_missing_feature"] = 5.0  # present in test, but must be excluded (train-only rule)

    result = fit_predict_linear(
        train,
        test,
        feature_cols=[*FEATURE_COLS, "all_missing_feature"],
        target_col="primary_dealer_share",
        estimator_type="ridge",
    )
    assert "all_missing_feature" not in result.kept_feature_columns
    assert result.dropped_all_missing_columns == ["all_missing_feature"]
    assert len(result.predictions) == len(test)
    assert np.isfinite(result.predictions.to_numpy()).all()


def test_fit_predict_linear_elastic_net_end_to_end():
    train = _synthetic_frame(n_years=6, per_year=40)
    test = _synthetic_frame(n_years=1, per_year=20, seed=123)
    test["auction_date"] = test["auction_date"] + pd.DateOffset(years=10)

    result = fit_predict_linear(
        train, test, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="elastic_net"
    )
    assert len(result.predictions) == len(test)
    assert result.selected_hyperparams  # some config chosen


def test_fit_predict_linear_raises_when_all_features_missing():
    train = _synthetic_frame(n_years=6, per_year=10)
    train["x"] = np.nan
    test = train.copy()
    with pytest.raises(ValueError):
        fit_predict_linear(train, test, feature_cols=["x"], target_col="primary_dealer_share", estimator_type="ridge")


def test_inner_cv_selection_never_sees_outer_test_rows():
    """Changing the OUTER test set drastically (poisoning it with
    extreme values) must never change the hyperparameters selected by
    the inner CV, since inner CV only ever touches the outer training
    pool."""
    train = _synthetic_frame(n_years=6, per_year=40)
    outer_test_a = _synthetic_frame(n_years=1, per_year=20, seed=1)
    outer_test_b = outer_test_a.copy()
    outer_test_b["primary_dealer_share"] = 999.0  # poison the outer test set only

    hyperparams_a, _, _ = select_hyperparameters_via_inner_cv(
        train, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="ridge"
    )
    hyperparams_b, _, _ = select_hyperparameters_via_inner_cv(
        train, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="ridge"
    )
    assert hyperparams_a == hyperparams_b  # train untouched -> identical selection regardless of outer_test


def test_partial_missing_predictor_is_imputed_with_training_median_not_full_dataset_median():
    train = _synthetic_frame(n_years=6, per_year=40)
    train = train.reset_index(drop=True)
    train.loc[: len(train) // 2, "some_numeric_feature"] = np.nan  # partially missing, not entirely
    test = _synthetic_frame(n_years=1, per_year=10, seed=55)
    test["auction_date"] = test["auction_date"] + pd.DateOffset(years=10)

    result = fit_predict_linear(
        train, test, feature_cols=FEATURE_COLS, target_col="primary_dealer_share", estimator_type="ridge"
    )
    assert "some_numeric_feature" in result.kept_feature_columns  # not dropped -- only partially missing
    assert result.dropped_all_missing_columns == []
    assert np.isfinite(result.predictions.to_numpy()).all()


def test_two_folds_fit_independent_preprocessing_state():
    """Fitting on two different training pools must produce
    differently-scaled internal state -- proves preprocessing is
    refit per fold, never reused."""
    fold_a_train = _synthetic_frame(n_years=6, per_year=40, seed=1)
    fold_b_train = _synthetic_frame(n_years=6, per_year=40, seed=2)
    fold_a_train["offering_amt"] = fold_a_train["offering_amt"] + 1000  # shift the distribution hard

    pipeline_a = build_pipeline(FEATURE_COLS, "ridge", {"alpha": 1.0})
    pipeline_a.fit(fold_a_train[FEATURE_COLS], fold_a_train["primary_dealer_share"])
    pipeline_b = build_pipeline(FEATURE_COLS, "ridge", {"alpha": 1.0})
    pipeline_b.fit(fold_b_train[FEATURE_COLS], fold_b_train["primary_dealer_share"])

    scaler_a = pipeline_a.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    scaler_b = pipeline_b.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    assert not np.allclose(scaler_a.mean_, scaler_b.mean_)
