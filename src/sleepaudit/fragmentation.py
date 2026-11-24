"""Distance to the nearest stage transition, strata and night-level fragmentation summaries."""

from __future__ import annotations

from pathlib import Path

import mne.datasets.sleep_physionet as _sp
import numpy as np
import pandas as pd

STRATA = {"0-1": (0, 1), "2-3": (2, 3), "4": (4, 4), "stable": (5, np.inf)}
EPOCH_MIN = 0.5


def transition_distance(labels: np.ndarray) -> np.ndarray:
    """Distance in epochs to the nearest index i with ``labels[i] != labels[i-1]``.

    The epoch where the change lands has distance 0 and its predecessor 1. A sequence with no
    transition gets ``inf`` everywhere. Run it per recording on the kept-epoch sequence.
    """
    labels = np.asarray(labels)
    n = len(labels)
    out = np.full(n, np.inf)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if n == 0 or len(change) == 0:
        return out
    pos = np.arange(n)
    right = np.searchsorted(change, pos, side="left")
    nxt = np.where(right < len(change), change[np.minimum(right, len(change) - 1)], -(10**9))
    prv = np.where(right > 0, change[np.maximum(right - 1, 0)], -(10**9))
    d_next = np.where(right < len(change), np.abs(nxt - pos), np.inf)
    d_prev = np.where(right > 0, np.abs(pos - prv), np.inf)
    return np.minimum(d_next, d_prev)


def stratum(dist: np.ndarray) -> np.ndarray:
    """Map transition distances to the stratum names in ``STRATA``."""
    dist = np.asarray(dist, dtype=float)
    out = np.empty(len(dist), dtype=object)
    for name, (lo, hi) in STRATA.items():
        out[(dist >= lo) & (dist <= hi)] = name
    return out.astype(str)


def with_transition_distance(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``table`` with ``trans_dist`` (per recording, in ``epoch_no`` order) and ``stratum``.

    Uses the true ``label`` column; row order of the input is preserved.
    """
    out = table.copy()
    out["trans_dist"] = np.inf
    for _, g in out.groupby("recording", sort=False):
        order = g.sort_values("epoch_no").index
        out.loc[order, "trans_dist"] = transition_distance(out.loc[order, "label"].to_numpy())
    out["stratum"] = stratum(out["trans_dist"].to_numpy())
    return out


def _age_table() -> pd.DataFrame:
    path = Path(_sp.__file__).parent / "age_records.csv"
    ages = pd.read_csv(path)
    ages = ages[ages["record type"] == "PSG"]
    return pd.DataFrame({"recording": ages["fname"].str[:7], "age": ages["age"]}).drop_duplicates(
        "recording"
    )


def night_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per recording: transition_share, n1_share, n_awakenings, waso_min, age."""
    rows = []
    for rec, g in table.sort_values(["recording", "epoch_no"]).groupby("recording", sort=True):
        lab = g["label"].to_numpy()
        n = len(lab)
        sleep = np.flatnonzero(lab != 0)
        awakenings, waso = 0, 0.0
        if len(sleep):
            inner = lab[sleep[0] : sleep[-1] + 1] == 0
            waso = float(inner.sum()) * EPOCH_MIN
            awakenings = int(np.sum(inner[1:] & ~inner[:-1]) + (inner[0] if len(inner) else 0))
        rows.append(
            {
                "recording": rec,
                "subject": int(g["subject"].iloc[0]),
                "night": int(g["night"].iloc[0]),
                "n_epochs": n,
                "transition_share": float(np.sum(lab[1:] != lab[:-1]) / n),
                "n1_share": float(np.mean(lab == 1)),
                "n_awakenings": awakenings,
                "waso_min": waso,
            }
        )
    out = pd.DataFrame(rows)
    return out.merge(_age_table(), on="recording", how="left")
