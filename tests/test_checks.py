"""Runtime pipeline-failure checks (the pre-registered criteria, applied to the run's own data)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleepaudit.checks import (
    PipelineFailure,
    check_epoch_counts,
    check_honest_folds,
    check_split_invariant,
)
from sleepaudit.config import load_config
from sleepaudit.evaluate import run_evaluation
from sleepaudit.splits import Fold, make_folds
from synthetic import synthetic_table

CI = Path(__file__).resolve().parents[1] / "configs" / "ci.yaml"


def _labels(counts: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame({"label": np.concatenate([[k] * n for k, n in counts.items()])})


SC20 = {0: 8285, 1: 2804, 2: 17799, 3: 5703, 4: 7717}


def test_epoch_counts_match_published_sc20():
    assert check_epoch_counts(_labels(SC20), "sc20") is True
    assert check_epoch_counts(_labels({**SC20, 0: 8286}), "sc20") is True  # W within 1
    assert check_epoch_counts(_labels({**SC20, 0: 8287}), "sc20") is False
    assert check_epoch_counts(_labels({**SC20, 1: 2805}), "sc20") is False  # sleep stages exact


def test_epoch_counts_not_checked_for_fixtures():
    assert check_epoch_counts(_labels(SC20), [0, 1, 2, 3]) is None


def test_honest_fold_sharing_a_subject_raises():
    table = synthetic_table(n_subjects=4, n_per_subject=10)
    folds = make_folds(table, "subject", 2, 0)
    check_honest_folds(table, folds)  # clean folds pass
    bad = [Fold(np.append(folds[0].train, folds[0].test[:1]), folds[0].test), *folds[1:]]
    with pytest.raises(PipelineFailure, match="subject"):
        check_honest_folds(table, bad)


def _ps(values_by_arm: dict[str, list[float]], model="yasa_pretrained") -> pd.DataFrame:
    rows = [
        {"model": model, "arm": arm, "seed": seed, "subject": s, "kappa": v}
        for arm, vals in values_by_arm.items()
        for seed in (0, 1)
        for s, v in enumerate(vals)
    ]
    return pd.DataFrame(rows)


def test_split_invariance_check():
    same = _ps({"epoch": [0.5, 0.6], "subject": [0.5, 0.6], "night": [0.5, 0.6]})
    assert check_split_invariant(same) is True
    diff = _ps({"epoch": [0.5, 0.6], "subject": [0.5, 0.61]})
    assert check_split_invariant(diff) is False
    assert check_split_invariant(_ps({"epoch": [0.5]}, model="bandpower_lr")) is None


def test_run_evaluation_records_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    table = synthetic_table(n_subjects=6, n_per_subject=60, yasa=True)
    cfg = replace(
        load_config(CI),
        k=3,
        n_boot=100,
        models=["majority", "yasa_pretrained"],
        results_dir=tmp_path / "res",
    )
    run_evaluation(cfg, table=table, allow_without_prereg=True)
    checks = json.loads((tmp_path / "res" / "meta.json").read_text())["pipeline_checks"]
    assert checks == {
        "honest_folds_subject_disjoint": True,
        "yasa_pretrained_split_invariant": True,
        "epoch_counts_match_published": None,
    }
