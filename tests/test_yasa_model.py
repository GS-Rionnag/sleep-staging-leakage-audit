from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleepaudit import features
from sleepaudit.config import data_dir, load_config
from sleepaudit.evaluate import per_subject_metrics, run_oof
from sleepaudit.features import build_feature_table
from sleepaudit.labels import recording_labels
from sleepaudit.models import MODELS
from sleepaudit.records import select_recordings
from sleepaudit.splits import ARMS, make_folds
from sleepaudit.yasa_model import YASA_TO_CODE, PrecomputedClassifier, yasa_features
from synthetic import synthetic_table

needs_psg = pytest.mark.needs_psg
CI_CONFIG = Path("configs/ci.yaml")
STAGES = ["W", "N1", "N2", "N3", "REM"]


def test_yasa_code_map():
    # YASA 0.8.1 labels Hypnogram.proba columns WAKE, N1, N2, N3, REM.
    for stage in ["WAKE", "N1", "N2", "N3", "REM"]:
        assert YASA_TO_CODE[stage] in range(5)
    assert [YASA_TO_CODE[s] for s in ["WAKE", "N1", "N2", "N3", "REM"]] == [0, 1, 2, 3, 4]
    assert all(v in range(5) for v in YASA_TO_CODE.values())
    assert YASA_TO_CODE["W"] == 0 and YASA_TO_CODE["R"] == 4


def test_precomputed_ignores_training():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(5), 30)
    clf = PrecomputedClassifier().fit(rng.normal(size=(10, 5)), rng.integers(0, 5, 10))
    assert clf.classes_.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(clf.predict_proba(proba), proba)
    np.testing.assert_array_equal(clf.predict(proba), proba.argmax(axis=1))
    # Training on one class only changes nothing.
    clf2 = PrecomputedClassifier().fit(rng.normal(size=(10, 5)), np.zeros(10, dtype=int))
    assert clf2.classes_.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(clf2.predict(proba), proba.argmax(axis=1))


def test_pretrained_split_invariant():
    table = synthetic_table(n_subjects=6, n_per_subject=100, yasa=True)
    spec = MODELS["yasa_pretrained"]
    assert not spec.has_scaler
    kappas = {}
    for arm in ARMS:
        folds = make_folds(table, arm, 3, seed=0)
        oof = run_oof(spec, table, folds, seed=0)
        truth = table[["subject", "label"]].rename(columns={"label": "y_true"})
        per = per_subject_metrics(oof.merge(truth, left_on="row", right_index=True))
        kappas[arm] = per["kappa"].to_numpy()
    for arm in ARMS:
        np.testing.assert_array_equal(kappas[arm], kappas["epoch"])  # exact equality


def test_pretrained_uses_only_proba_columns():
    table = synthetic_table(yasa=True)
    cols = MODELS["yasa_pretrained"].columns(table)
    assert cols == [f"yasa_proba_{s}" for s in STAGES]


def test_lgbm_columns_exclude_probabilities():
    table = synthetic_table(yasa=True)
    cols = MODELS["yasa_lgbm"].columns(table)
    assert cols == ["yasa_feat_a", "yasa_feat_b"]
    assert not any(c.startswith("yasa_proba_") for c in cols)


def test_lgbm_uses_fixed_hyperparameters_and_seed():
    est = MODELS["yasa_lgbm"].factory(7)
    params = est.get_params()
    assert params["n_estimators"] == 500 and params["max_depth"] == 5
    assert params["num_leaves"] == 90 and params["colsample_bytree"] == 0.6
    assert params["random_state"] == 7
    assert params["class_weight"] == {0: 1.0, 1: 2.2, 2: 1.0, 3: 1.2, 4: 1.4}
    assert not MODELS["yasa_lgbm"].has_scaler


def test_lgbm_runs_out_of_fold():
    table = synthetic_table(n_subjects=4, n_per_subject=60, yasa=True)
    spec = replace(MODELS["yasa_lgbm"], factory=lambda seed: _small(MODELS["yasa_lgbm"], seed))
    oof = run_oof(spec, table, make_folds(table, "subject", 2, 0), seed=0)
    assert len(oof) == len(table) and oof["row"].is_unique


def _small(spec, seed):
    return spec.factory(seed).set_params(n_estimators=5)


def test_feature_params_track_yasa_use():
    cfg = load_config(CI_CONFIG)
    no_yasa = replace(cfg, models=["majority", "bandpower_lr"])
    assert features.feature_params(cfg) != features.feature_params(no_yasa)
    assert features.feature_params(no_yasa)["extractors"] == []


@needs_psg
def test_yasa_rows_align():
    root = data_dir() / "sleep-cassette"
    rec = next(r for r in select_recordings([0]) if r.stem == "SC4001E")
    labels = recording_labels(rec, root)
    kept = labels["epoch_no"].to_numpy()
    out = yasa_features(rec, root, kept)
    assert len(out) == len(kept)
    proba = out[[f"yasa_proba_{s}" for s in STAGES]]
    assert not proba.isna().any().any()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert all(c.startswith("yasa_") for c in out.columns)
    assert not out.drop(columns=list(proba.columns)).isna().all(axis=None)
    # Row alignment: YASA's time_norm is the relative position inside the cropped segment.
    expected_pos = (kept - kept.min()) / (kept.max() - kept.min())
    np.testing.assert_allclose(out["yasa_time_norm"].to_numpy(), expected_pos, atol=1e-6)
    # The pretrained control agrees with the hypnogram far above chance (~0.85 observed).
    pred = proba.to_numpy().argmax(axis=1)
    assert (pred == labels["label"].to_numpy()).mean() > 0.7


@needs_psg
def test_feature_table_has_yasa_columns(tmp_path):
    cfg = replace(load_config(CI_CONFIG), subset=[0], results_dir=tmp_path)
    table = build_feature_table(cfg)
    assert len([c for c in table.columns if c.startswith("yasa_proba_")]) == 5
    assert len([c for c in table.columns if c.startswith("yasa_")]) > 100
    assert not table.filter(regex="^yasa_proba_").isna().any().any()
    root = data_dir() / "sleep-cassette"
    expect = pd.concat([recording_labels(r, root) for r in select_recordings([0])])
    assert table["epoch_no"].tolist() == expect["epoch_no"].tolist()


@needs_psg
def test_yasa_epochs_start_on_the_label_grid():
    """YASA epoch i of the crop must be original-grid epoch lo + i: the per-epoch EEG std from YASA
    matches our own signal epochs at lag 0 and clearly worse at lag +-1 (a crop off by one epoch)."""
    from sleepaudit.features import load_signal_epochs

    root = data_dir() / "sleep-cassette"
    rec = next(r for r in select_recordings([0]) if r.stem == "SC4001E")
    kept = recording_labels(rec, root)["epoch_no"].to_numpy()
    ours = np.log(load_signal_epochs(rec, root, kept).std(axis=1))
    theirs = np.log(yasa_features(rec, root, kept)["yasa_eeg_std"].to_numpy())
    lag0 = np.corrcoef(ours, theirs)[0, 1]
    lag1 = max(np.corrcoef(ours[1:], theirs[:-1])[0, 1], np.corrcoef(ours[:-1], theirs[1:])[0, 1])
    assert lag0 > 0.99 and lag0 - lag1 > 0.05, (lag0, lag1)
