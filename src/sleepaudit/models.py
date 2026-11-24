"""Model registry: fixed hyperparameters, no tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sleepaudit.yasa_model import PrecomputedClassifier, lgbm_factory


@dataclass(frozen=True)
class ModelSpec:
    name: str
    columns: Callable[[pd.DataFrame], list[str]]  # feature columns of an epoch table
    factory: Callable[[int], BaseEstimator]  # seed -> unfitted estimator
    has_scaler: bool  # True if the estimator is a Pipeline with a step named "scaler"


def by_prefix(*prefixes: str, exclude: tuple[str, ...] = ()) -> Callable[[pd.DataFrame], list[str]]:
    """Select columns that start with any of ``prefixes`` and none of ``exclude``."""

    def select(df: pd.DataFrame) -> list[str]:
        return [
            c
            for c in df.columns
            if c.startswith(prefixes) and not (exclude and c.startswith(exclude))
        ]

    return select


def _logreg(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


MODELS: dict[str, ModelSpec] = {
    "majority": ModelSpec(
        "majority", by_prefix("bp_"), lambda seed: DummyClassifier(strategy="most_frequent"), False
    ),
    "bandpower_lr": ModelSpec("bandpower_lr", by_prefix("bp_"), _logreg, True),
    "bandpower_ctx_lr": ModelSpec("bandpower_ctx_lr", by_prefix("bp_", "bpctx_"), _logreg, True),
    "yasa_lgbm": ModelSpec(
        "yasa_lgbm", by_prefix("yasa_", exclude=("yasa_proba_",)), lgbm_factory, False
    ),
    "yasa_pretrained": ModelSpec(
        "yasa_pretrained", by_prefix("yasa_proba_"), lambda seed: PrecomputedClassifier(), False
    ),
}
