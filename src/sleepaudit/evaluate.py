"""Out-of-fold predictions and per-subject / pooled metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from sleepaudit.models import ModelSpec
from sleepaudit.splits import Fold

LABELS = [0, 1, 2, 3, 4]
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
PROBA_COLS = [f"proba_{i}" for i in LABELS]


def run_oof(
    spec: ModelSpec,
    table: pd.DataFrame,
    folds: list[Fold],
    seed: int,
    prefit_scaler: bool = False,
) -> pd.DataFrame:
    """Out-of-fold predictions: columns ``row, y_pred, proba_0..proba_4``.

    ``prefit_scaler=True`` is the pipeline-leak condition (arm ``subject_prefit``): the scaler is
    fitted on all rows before the loop and the pipeline's own scaler becomes a pass-through.
    """
    if prefit_scaler and not spec.has_scaler:
        raise ValueError(f"{spec.name}: prefit_scaler needs a model with a scaler step")
    X = table[spec.columns(table)].to_numpy(dtype=float)
    y = table["label"].to_numpy()
    if prefit_scaler:
        X = clone(spec.factory(seed).named_steps["scaler"]).fit(X).transform(X)
    parts = []
    for fold in folds:
        est = spec.factory(seed)
        if prefit_scaler:
            est.set_params(scaler="passthrough")
        est.fit(X[fold.train], y[fold.train])
        proba = np.zeros((len(fold.test), len(LABELS)))
        proba[:, np.asarray(est.classes_, dtype=int)] = est.predict_proba(X[fold.test])
        part = pd.DataFrame(proba, columns=PROBA_COLS)
        part.insert(0, "y_pred", est.predict(X[fold.test]).astype(int))
        part.insert(0, "row", fold.test)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values("row", ignore_index=True)


def _kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa over all five labels; NaN when undefined (no chance-corrected range)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = cohen_kappa_score(y_true, y_pred, labels=LABELS)
    return float(value) if np.isfinite(value) else float("nan")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over the stages present in truth or prediction (absent stages are not scored 0)."""
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """ECE on the max-probability (confidence), equal-width bins over (0, 1]."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    bins = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def per_subject_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: kappa (NaN if undefined), accuracy, macro_f1, ece, n_epochs."""
    rows = []
    for subject, g in oof.groupby("subject", sort=True):
        t, p = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        rows.append(
            {
                "subject": subject,
                "kappa": _kappa(t, p),
                "accuracy": float(accuracy_score(t, p)),
                "macro_f1": _macro_f1(t, p),
                "ece": expected_calibration_error(t, g[PROBA_COLS].to_numpy()),
                "n_epochs": len(g),
            }
        )
    return pd.DataFrame(
        rows, columns=["subject", "kappa", "accuracy", "macro_f1", "ece", "n_epochs"]
    )


def pooled_metrics(oof: pd.DataFrame) -> dict[str, float]:
    """Metrics over all epochs pooled: accuracy, kappa, macro-F1 and per-class F1."""
    t, p = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()
    per_class = f1_score(t, p, labels=LABELS, average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(t, p)),
        "kappa": _kappa(t, p),
        "macro_f1": _macro_f1(t, p),
    }
    out.update({f"f1_{n}": float(v) for n, v in zip(STAGE_NAMES, per_class, strict=True)})
    return out
