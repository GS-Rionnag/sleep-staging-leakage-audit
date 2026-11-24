"""YASA features, the retrained LightGBM model and the pretrained-YASA control.

Verified against yasa 0.8.1: ``SleepStaging.get_features()`` returns a DataFrame (one row per
30-s epoch, 116 columns: ``eeg_*``, ``eog_*``, ``time_hour``, ``time_norm``) and
``predict().proba`` has columns ``WAKE, N1, N2, N3, REM``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from sleepaudit.records import Recording

EEG = "EEG Fpz-Cz"
EOG = "EOG horizontal"
EPOCH_SECONDS = 30
STAGES = ["W", "N1", "N2", "N3", "REM"]
YASA_TO_CODE = {"WAKE": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}
LGBM_CLASS_WEIGHT = {0: 1.0, 1: 2.2, 2: 1.0, 3: 1.2, 4: 1.4}  # W, N1, N2, N3, REM


def yasa_version() -> str:
    import yasa

    return yasa.__version__


def yasa_features(rec: Recording, root: Path, epoch_nos: np.ndarray) -> pd.DataFrame:
    """YASA features and pretrained probabilities for the given original-grid epochs.

    The raw PSG is cropped to the trimmed segment [min(epoch_nos), max(epoch_nos)+1) * 30 s
    (unfiltered, not z-scored); YASA's own filtering and its +-7.5 min smoothing run on that
    segment. Rows are those at ``epoch_nos - min(epoch_nos)``.
    """
    import mne
    import yasa

    epoch_nos = np.asarray(epoch_nos, dtype=int)
    lo, hi = int(epoch_nos.min()), int(epoch_nos.max()) + 1
    raw = mne.io.read_raw_edf(
        root / rec.psg_file, include=[EEG, EOG], preload=True, verbose="error"
    )
    raw.crop(tmin=lo * EPOCH_SECONDS, tmax=hi * EPOCH_SECONDS, include_tmax=False)
    sls = yasa.SleepStaging(raw, eeg_name=EEG, eog_name=EOG)
    feats = sls.get_features()
    proba = sls.predict().proba
    if len(feats) != hi - lo or len(proba) != hi - lo:
        raise ValueError(
            f"{rec.stem}: YASA returned {len(feats)} feature rows and {len(proba)} probability "
            f"rows for {hi - lo} epochs"
        )
    rows = epoch_nos - lo
    out = feats.iloc[rows].reset_index(drop=True).add_prefix("yasa_")
    probs = proba.iloc[rows].reset_index(drop=True)
    probs.columns = [f"yasa_proba_{STAGES[YASA_TO_CODE[c]]}" for c in probs.columns]
    return pd.concat([out, probs[[f"yasa_proba_{s}" for s in STAGES]]], axis=1)


class PrecomputedClassifier(ClassifierMixin, BaseEstimator):
    """Pass-through for probabilities computed elsewhere: ``fit`` does nothing."""

    def fit(self, X, y=None):
        self.classes_ = np.arange(len(STAGES))
        return self

    def predict_proba(self, X):
        return np.asarray(X, dtype=float)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def lgbm_factory(seed: int):
    """LightGBM with the fixed Global-Constraints settings (no tuning)."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        num_leaves=90,
        colsample_bytree=0.6,
        class_weight=LGBM_CLASS_WEIGHT,
        random_state=seed,
        verbose=-1,
    )
