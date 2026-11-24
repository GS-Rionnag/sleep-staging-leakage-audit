from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from sleepaudit.evaluate import (
    expected_calibration_error,
    per_subject_metrics,
    pooled_metrics,
    run_oof,
)
from sleepaudit.models import MODELS, ModelSpec
from sleepaudit.splits import make_folds
from synthetic import synthetic_table


class SpyScaler(BaseEstimator, TransformerMixin):
    """Identity transformer that records the row ids (column 0) it was fitted on."""

    fits: ClassVar[list] = []

    def fit(self, X, y=None):
        type(self).fits.append(set(np.asarray(X)[:, 0].astype(int).tolist()))
        return self

    def transform(self, X):
        return np.asarray(X)


def spy_spec():
    return ModelSpec(
        name="spy",
        columns=lambda df: ["bp_rowid", "bp_delta", "bp_theta"],
        factory=lambda seed: Pipeline(
            [("scaler", SpyScaler()), ("clf", DummyClassifier(strategy="prior"))]
        ),
        has_scaler=True,
    )


@pytest.fixture
def table():
    t = synthetic_table(n_subjects=6, n_per_subject=40)
    t["bp_rowid"] = np.arange(len(t), dtype=float)
    return t


def test_spy_scaler_fit_rows(table):
    folds = make_folds(table, "subject", 3, 0)
    SpyScaler.fits = []
    run_oof(spy_spec(), table, folds, seed=0)
    assert len(SpyScaler.fits) == len(folds)
    for seen, fold in zip(SpyScaler.fits, folds, strict=True):
        assert seen == set(fold.train.tolist())
        assert seen & set(fold.test.tolist()) == set()

    SpyScaler.fits = []
    run_oof(spy_spec(), table, folds, seed=0, prefit_scaler=True)
    assert len(SpyScaler.fits) == 1
    assert SpyScaler.fits[0] == set(range(len(table)))


def test_prefit_rejected_without_scaler(table):
    folds = make_folds(table, "subject", 3, 0)
    with pytest.raises(ValueError, match="scaler"):
        run_oof(MODELS["majority"], table, folds, seed=0, prefit_scaler=True)


def test_oof_covers_all_rows(table):
    for arm in ("subject", "night", "epoch"):
        oof = run_oof(MODELS["bandpower_lr"], table, make_folds(table, arm, 3, 0), seed=0)
        assert len(oof) == len(table)
        assert oof["row"].is_unique
        assert sorted(oof["row"]) == list(range(len(table)))
        assert list(oof.columns) == ["row", "y_pred"] + [f"proba_{i}" for i in range(5)]


def test_oof_block_arm_still_predicts_every_row(table):
    folds = make_folds(table, "block", 3, 0, block_epochs=10, buffer_epochs=2)
    oof = run_oof(MODELS["bandpower_lr"], table, folds, seed=0)
    assert len(oof) == len(table) and oof["row"].is_unique


def _with_truth(table, oof):
    truth = table[["subject", "label"]].rename(columns={"label": "y_true"})
    return oof.merge(truth, left_on="row", right_index=True)


def test_majority_kappa_zero(table):
    oof = run_oof(MODELS["majority"], table, make_folds(table, "subject", 3, 0), seed=0)
    per = per_subject_metrics(_with_truth(table, oof))
    assert len(per) == table["subject"].nunique()
    np.testing.assert_allclose(per["kappa"].to_numpy(), 0.0, atol=1e-12)


def _oof_frame(y_true, y_pred, subject=0):
    df = pd.DataFrame(np.eye(5)[np.asarray(y_pred)], columns=[f"proba_{i}" for i in range(5)])
    df.insert(0, "y_pred", y_pred)
    df.insert(0, "y_true", y_true)
    df.insert(0, "subject", subject)
    return df


def test_kappa_nan_when_degenerate():
    per = per_subject_metrics(_oof_frame([0] * 20, [0] * 20))
    assert np.isnan(per.loc[0, "kappa"])
    assert per.loc[0, "accuracy"] == 1.0
    assert per.loc[0, "n_epochs"] == 20


def test_kappa_defined_when_only_truth_is_constant():
    per = per_subject_metrics(_oof_frame([0] * 20, [0] * 10 + [2] * 10))
    assert per.loc[0, "kappa"] == pytest.approx(0.0, abs=1e-12)


def test_per_subject_columns_and_one_row_each():
    a = _oof_frame([0, 1, 2, 3, 4, 2], [0, 1, 2, 3, 3, 2], subject=3)
    b = _oof_frame([0, 0, 1, 1], [0, 1, 1, 1], subject=7)
    per = per_subject_metrics(pd.concat([a, b], ignore_index=True))
    assert list(per.columns) == ["subject", "kappa", "accuracy", "macro_f1", "ece", "n_epochs"]
    assert per["subject"].tolist() == [3, 7]
    assert per["n_epochs"].tolist() == [6, 4]
    assert per.loc[0, "accuracy"] == pytest.approx(5 / 6)
    assert per.loc[0, "kappa"] < 1.0


def test_perfect_prediction_kappa_one():
    per = per_subject_metrics(_oof_frame([0, 1, 2, 3, 4, 2], [0, 1, 2, 3, 4, 2]))
    assert per.loc[0, "kappa"] == pytest.approx(1.0)
    assert per.loc[0, "macro_f1"] == pytest.approx(1.0)


def test_ece_perfect_is_zero():
    y = np.array([0, 1, 2, 3, 4, 2, 1])
    assert expected_calibration_error(y, np.eye(5)[y]) == 0.0


def test_ece_confident_and_wrong_is_one():
    y = np.array([0, 0, 0, 0])
    assert expected_calibration_error(y, np.eye(5)[[1, 1, 1, 1]]) == pytest.approx(1.0)


def test_ece_calibrated_half_right_is_zero():
    # confidence 0.5 on class 0 for every row; class 0 is right for half of them.
    proba = np.tile([0.5, 0.5, 0.0, 0.0, 0.0], (4, 1))
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(y, proba) == pytest.approx(0.0)


def test_pooled_metrics_keys_and_values():
    out = pooled_metrics(_oof_frame([0, 1, 2, 3, 4, 4], [0, 1, 2, 3, 4, 0]))
    assert set(out) == {
        "accuracy",
        "kappa",
        "macro_f1",
        "f1_W",
        "f1_N1",
        "f1_N2",
        "f1_N3",
        "f1_REM",
    }
    assert out["accuracy"] == pytest.approx(5 / 6)
    assert out["f1_N1"] == 1.0
    assert 0 < out["kappa"] < 1
