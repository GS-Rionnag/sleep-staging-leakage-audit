"""Cross-validation splits for the four arms. This is the only module that makes folds."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ARMS = ("epoch", "block", "night", "subject", "subject_prefit")


class Fold(NamedTuple):
    train: np.ndarray  # positional row indices
    test: np.ndarray


def _group_ids(index: pd.DataFrame, arm: str, block_epochs: int) -> np.ndarray:
    recording = index["recording"].to_numpy()
    if arm == "block":
        keys = pd.MultiIndex.from_arrays([recording, index["epoch_no"].to_numpy() // block_epochs])
        return pd.factorize(keys)[0]
    column = "recording" if arm == "night" else "subject"
    return pd.factorize(index[column].to_numpy())[0]


def _drop_buffer(index: pd.DataFrame, folds: list[Fold], buffer_epochs: int) -> list[Fold]:
    """Remove from each train set the rows of a test recording within ``buffer_epochs`` of a test row."""
    recording = index["recording"].to_numpy()
    epoch = index["epoch_no"].to_numpy()
    out = []
    for fold in folds:
        keep = np.ones(len(fold.train), dtype=bool)
        train_rec = recording[fold.train]
        train_epoch = epoch[fold.train]
        for rec in np.unique(recording[fold.test]):
            test_epochs = np.sort(epoch[fold.test][recording[fold.test] == rec])
            in_rec = np.flatnonzero(train_rec == rec)
            if not len(in_rec):
                continue
            e = train_epoch[in_rec]
            pos = np.searchsorted(test_epochs, e)
            left = np.abs(e - test_epochs[np.clip(pos - 1, 0, len(test_epochs) - 1)])
            right = np.abs(e - test_epochs[np.clip(pos, 0, len(test_epochs) - 1)])
            keep[in_rec[np.minimum(left, right) <= buffer_epochs]] = False
        out.append(Fold(fold.train[keep], fold.test))
    return out


def make_folds(
    index: pd.DataFrame,
    arm: str,
    k: int,
    seed: int,
    block_epochs: int = 60,
    buffer_epochs: int = 8,
) -> list[Fold]:
    """K folds of ``index`` rows for one arm.

    ``epoch``: shuffled rows (neighbouring epochs and subjects leak).
    ``block``: contiguous ``block_epochs`` blocks per recording as groups, then train rows within
        ``buffer_epochs`` of a test row of the same recording are dropped (distance <= buffer).
    ``night``: whole recordings held out (the subject's other night may be in train).
    ``subject`` / ``subject_prefit``: whole subjects held out (identical folds).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    if arm == "epoch":
        n_groups, unit = len(index), "epoch"
    else:
        groups = _group_ids(index, arm, block_epochs)
        n_groups = int(groups.max()) + 1 if len(groups) else 0
        unit = {"block": "block", "night": "night"}.get(arm, "subject")
    if k > n_groups:
        raise ValueError(f"k={k} exceeds the number of {unit} groups ({n_groups}) for arm {arm!r}")

    rows = np.arange(len(index))
    if arm == "epoch":
        splitter = KFold(k, shuffle=True, random_state=seed)
        folds = [Fold(tr, te) for tr, te in splitter.split(rows)]
    else:
        splitter = GroupKFold(k, shuffle=True, random_state=seed)
        folds = [Fold(tr, te) for tr, te in splitter.split(rows, groups=groups)]
    if arm == "block":
        folds = _drop_buffer(index, folds, buffer_epochs)
    return folds
