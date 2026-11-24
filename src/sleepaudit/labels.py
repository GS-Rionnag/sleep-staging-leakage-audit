"""Hypnogram annotations -> 30-s epoch labels, with clipping and wake trimming."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from sleepaudit.download import edf_duration_seconds
from sleepaudit.records import Recording

EPOCH_SECONDS = 30.0
LABEL_MAP: dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}


def epoch_labels(
    onsets: np.ndarray,
    durations: np.ndarray,
    descriptions: Sequence[str],
    psg_seconds: float,
) -> np.ndarray:
    """Per-epoch labels on the PSG's 30-s grid; -1 where unknown, moved or uncovered."""
    n = math.floor(psg_seconds / EPOCH_SECONDS)
    out = np.full(n, -1, dtype=np.int8)
    for onset, duration, desc in zip(onsets, durations, descriptions, strict=True):
        code = LABEL_MAP.get(str(desc))
        if code is None:
            raise ValueError(f"unknown annotation description: {desc!r}")
        start = max(0, math.ceil(onset / EPOCH_SECONDS))
        stop = min(n, math.floor((onset + duration) / EPOCH_SECONDS))
        if stop > start:
            out[start:stop] = code
    return out


def trim(labels: np.ndarray, edge: int = 60, name: str = "") -> np.ndarray:
    """Original-grid indices kept after dropping -1 epochs and trimming wake.

    DeepSleepNet order: drop unknown epochs first, then keep from `edge` epochs before the first
    non-wake epoch to `edge` after the last, counted on the remaining sequence, clipped to bounds.
    """
    labels = np.asarray(labels)
    valid = np.flatnonzero(labels != -1)
    sleep = np.flatnonzero(labels[valid] != 0)
    if sleep.size == 0:
        raise ValueError(f"{name}: no sleep epochs")
    lo = max(0, int(sleep[0]) - edge)
    hi = min(valid.size - 1, int(sleep[-1]) + edge)
    return valid[lo : hi + 1]


def _psg_seconds(rec: Recording, root: Path) -> float:
    for candidate in (root / f"{rec.psg_file}.header", root / rec.psg_file):
        if candidate.exists():
            with candidate.open("rb") as fh:
                return edf_duration_seconds(fh.read(256))
    raise FileNotFoundError(f"no PSG header or file for {rec.stem} in {root}")


def recording_labels(rec: Recording, root: Path, edge: int = 60) -> pd.DataFrame:
    """Kept epochs of one recording: recording, subject, night, epoch_no, label."""
    import mne

    ann = mne.read_annotations(root / rec.hyp_file)
    labels = epoch_labels(ann.onset, ann.duration, list(ann.description), _psg_seconds(rec, root))
    kept = trim(labels, edge=edge, name=rec.stem)
    return pd.DataFrame(
        {
            "recording": rec.stem,
            "subject": rec.subject,
            "night": rec.night,
            "epoch_no": kept.astype(int),
            "label": labels[kept].astype(np.int8),
        }
    )
