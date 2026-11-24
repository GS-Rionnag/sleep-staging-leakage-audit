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


def _terciles(values: pd.Series) -> pd.Series:
    """low / mid / high thirds by rank (ties broken by order), robust to fewer than 3 values."""
    n = len(values)
    third = np.ceil(values.rank(method="first").to_numpy() * 3 / n).astype(int) - 1
    return pd.Series(np.array(["low", "mid", "high"])[np.clip(third, 0, 2)], index=values.index)


def _night_table(table: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Per-recording kappa per model and arm (seeds pooled) joined with night-level summaries."""
    from sleepaudit.fragmentation import night_summary

    rows = [
        {"model": m, "arm": a, "recording": r, "kappa": _kappa(g["y_true"].to_numpy(), g["y_pred"].to_numpy())}
        for (m, a, r), g in oof.groupby(["model", "arm", "recording"], observed=True)
    ]  # fmt: skip
    nights = night_summary(table)
    for col in ("transition_share", "n1_share"):
        nights[f"tercile_{col}"] = _terciles(nights[col])
    return pd.DataFrame(rows).merge(nights, on="recording", how="left")


def run_evaluation(cfg, table: pd.DataFrame | None = None, allow_without_prereg: bool = False):
    """Out-of-fold predictions for every model x arm x seed, then the results tables.

    Writes ``per_subject.csv``, ``oof.parquet``, ``pooled.csv``, ``strata.csv``, ``nights.csv`` and
    ``meta.json`` under ``cfg.results_dir``. ``subject_prefit`` is skipped (and logged) for models
    without a scaler. ``table`` may be passed in (tests); otherwise it is built from the cache.
    """
    from datetime import UTC, datetime

    from sleepaudit.checks import (
        HONEST_ARMS,
        PipelineFailure,
        check_epoch_counts,
        check_honest_folds,
        check_split_invariant,
    )
    from sleepaudit.fragmentation import with_transition_distance
    from sleepaudit.models import MODELS
    from sleepaudit.runmeta import check_prereg, write_meta
    from sleepaudit.splits import make_folds

    check_prereg(cfg, allow_without_prereg)
    started = datetime.now(UTC)
    if table is None:
        from sleepaudit.features import build_feature_table

        table = build_feature_table(cfg)
    table = with_transition_distance(table.reset_index(drop=True))
    counts_ok = check_epoch_counts(table, cfg.subset)
    if counts_ok is False:
        raise PipelineFailure(f"{cfg.subset}: kept-epoch counts differ from the published totals")
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    honest_checked = False

    fold_cache: dict[tuple[str, int], list[Fold]] = {}
    per_subject, pooled, oofs = [], [], []
    for model in cfg.models:
        spec = MODELS[model]
        for arm in cfg.arms:
            if arm == "subject_prefit" and not spec.has_scaler:
                print(f"skip {model} x subject_prefit: no scaler step, identical to subject")
                continue
            for seed in cfg.seeds:
                key = (arm, seed)
                if key not in fold_cache:
                    fold_cache[key] = make_folds(
                        table, arm, cfg.k, seed, cfg.block_epochs, cfg.buffer_epochs
                    )
                    if arm in HONEST_ARMS:
                        check_honest_folds(table, fold_cache[key])  # raises before training
                        honest_checked = True
                oof = run_oof(
                    spec, table, fold_cache[key], seed, prefit_scaler=arm == "subject_prefit"
                )
                rows = oof["row"].to_numpy()
                full = oof.drop(columns="row").copy()
                full.insert(0, "y_true", table["label"].to_numpy()[rows])
                for col in ("trans_dist", "stratum", "epoch_no", "subject", "recording"):
                    full.insert(0, col, table[col].to_numpy()[rows])
                tag = {"model": model, "arm": arm, "seed": seed}
                m = per_subject_metrics(full)
                per_subject.append(m.assign(dataset=cfg.name, **tag))
                pooled.append({**tag, **pooled_metrics(full)})
                oofs.append(full.assign(**tag))
                print(f"{model} {arm} seed {seed}: mean per-subject kappa {m['kappa'].mean():.3f}")

    out = cfg.results_dir
    ps = pd.concat(per_subject, ignore_index=True)
    ps = ps[
        ["dataset", "model", "arm", "seed", "subject", "kappa", "accuracy", "macro_f1", "ece", "n_epochs"]
    ]  # fmt: skip
    ps.to_csv(out / "per_subject.csv", index=False)
    pd.DataFrame(pooled).to_csv(out / "pooled.csv", index=False)
    oof_all = pd.concat(oofs, ignore_index=True)
    strata = (
        oof_all.assign(correct=oof_all["y_true"] == oof_all["y_pred"])
        .groupby(["model", "arm", "stratum"])["correct"]
        .agg(accuracy="mean", n_epochs="size")
        .reset_index()
    )
    strata.to_csv(out / "strata.csv", index=False)
    _night_table(table, oof_all).to_csv(out / "nights.csv", index=False)
    checks = {
        "honest_folds_subject_disjoint": True if honest_checked else None,
        "yasa_pretrained_split_invariant": check_split_invariant(ps),
        "epoch_counts_match_published": counts_ok,
    }
    print(f"pipeline checks: {checks}")
    compact = oof_all.astype(
        {
            **{c: "float32" for c in PROBA_COLS + ["trans_dist"]},
            **{c: "category" for c in ("model", "arm", "stratum", "recording")},
            **{c: "int8" for c in ("y_true", "y_pred", "seed")},
        }
    )
    compact.to_parquet(out / "oof.parquet", index=False)
    write_meta(
        cfg,
        out / "meta.json",
        started=started,
        finished=datetime.now(UTC),
        extra={"allow_without_prereg": allow_without_prereg, "pipeline_checks": checks},
    )
    if checks["yasa_pretrained_split_invariant"] is False:
        raise PipelineFailure("yasa_pretrained per-subject kappa differs between arms or seeds")
