"""Pre-registered pipeline-failure criteria, checked on each run's own folds and results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sleepaudit.splits import Fold

# Published trimmed-epoch counts (AttnSleep / DeepSleepNet-Lite): sleep stages exact, W and the
# total within the tolerance.
PUBLISHED_COUNTS = {
    "sc20": {"stages": {1: 2804, 2: 17799, 3: 5703, 4: 7717}, "W": 8285, "total": 42308, "tol": 1},
    "sc78": {"stages": {1: 21522, 2: 69132, 3: 13039, 4: 25835}, "W": 65951, "total": 195479, "tol": 10},
}  # fmt: skip
HONEST_ARMS = ("subject", "subject_prefit")
CONTROL_MODEL = "yasa_pretrained"


class PipelineFailure(RuntimeError):
    """A pre-registered pipeline-failure criterion fired."""


def check_epoch_counts(table: pd.DataFrame, subset) -> bool | None:
    """True if the kept epochs match the published counts; None for explicit-subject fixtures."""
    if isinstance(subset, list):
        return None
    ref = PUBLISHED_COUNTS[subset]
    counts = table["label"].value_counts()
    stages_ok = all(int(counts.get(k, 0)) == n for k, n in ref["stages"].items())
    w_ok = abs(int(counts.get(0, 0)) - ref["W"]) <= ref["tol"]
    total_ok = abs(len(table) - ref["total"]) <= ref["tol"]
    return bool(stages_ok and w_ok and total_ok)


def check_honest_folds(index: pd.DataFrame, folds: list[Fold]) -> None:
    """Raise PipelineFailure if any fold has a subject on both its train and test side."""
    subject = index["subject"].to_numpy()
    for i, fold in enumerate(folds):
        shared = np.intersect1d(subject[fold.train], subject[fold.test])
        if shared.size:
            raise PipelineFailure(f"honest fold {i} shares subject(s) {shared.tolist()}")


def check_split_invariant(per_subject: pd.DataFrame, model: str = CONTROL_MODEL) -> bool | None:
    """True if ``model``'s per-subject kappa is identical in every arm and seed; None if not run."""
    sel = per_subject[per_subject["model"] == model]
    if sel.empty:
        return None
    wide = sel.pivot_table(index="subject", columns=["arm", "seed"], values="kappa", dropna=False)
    values = wide.to_numpy()
    first = values[:, :1]
    same = (values == first) | (np.isnan(values) & np.isnan(first))
    return bool(same.all())
