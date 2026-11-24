import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sleepaudit.evaluate import run_oof
from sleepaudit.models import MODELS, ModelSpec
from sleepaudit.splits import make_folds
from synthetic import synthetic_table


def test_registry_has_the_three_task7_models():
    assert {"majority", "bandpower_lr", "bandpower_ctx_lr"} <= set(MODELS)
    assert all(isinstance(m, ModelSpec) for m in MODELS.values())


def test_column_selection_by_prefix():
    table = synthetic_table()
    cols = MODELS["bandpower_lr"].columns(table)
    assert len(cols) == 5 and all(c.startswith("bp_") for c in cols)
    ctx_cols = MODELS["bandpower_ctx_lr"].columns(table)
    assert len(ctx_cols) == 25
    assert sum(c.startswith("bpctx_") for c in ctx_cols) == 20


def test_factories_use_fixed_hyperparameters():
    lr = MODELS["bandpower_lr"].factory(0)
    assert isinstance(lr, Pipeline)
    assert isinstance(lr.named_steps["scaler"], StandardScaler)
    clf = lr.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight == "balanced" and clf.max_iter == 1000
    assert MODELS["bandpower_lr"].has_scaler and MODELS["bandpower_ctx_lr"].has_scaler
    maj = MODELS["majority"].factory(0)
    assert isinstance(maj, DummyClassifier) and maj.strategy == "most_frequent"
    assert not MODELS["majority"].has_scaler


def test_bandpower_lr_learns():
    table = synthetic_table()
    folds = make_folds(table, "subject", 3, 0)
    oof = run_oof(MODELS["bandpower_lr"], table, folds, seed=0)
    acc = (oof.sort_values("row")["y_pred"].to_numpy() == table["label"].to_numpy()).mean()
    assert acc > 0.9


def test_models_are_deterministic_for_a_seed():
    table = synthetic_table()
    folds = make_folds(table, "subject", 3, 0)
    a = run_oof(MODELS["bandpower_ctx_lr"], table, folds, seed=1)
    b = run_oof(MODELS["bandpower_ctx_lr"], table, folds, seed=1)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("name", ["majority", "bandpower_lr", "bandpower_ctx_lr"])
def test_every_model_returns_five_probability_columns(name):
    table = synthetic_table()
    oof = run_oof(MODELS[name], table, make_folds(table, "subject", 3, 0), seed=0)
    proba = oof[[f"proba_{i}" for i in range(5)]].to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
