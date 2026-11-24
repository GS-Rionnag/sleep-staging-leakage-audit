"""Band-power and context features with a per-recording parquet cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

import sleepaudit
from sleepaudit.config import Config, data_dir
from sleepaudit.labels import recording_labels
from sleepaudit.records import Recording, load_manifest, select_recordings

CHANNEL = "EEG Fpz-Cz"
SFREQ = 100.0
EPOCH_SAMPLES = 3000
CONTEXT_WIDTH = 2
# MNE sleep-staging tutorial bands.
BANDS = {
    "delta": (0.5, 4.5),
    "theta": (4.5, 8.5),
    "alpha": (8.5, 11.5),
    "sigma": (11.5, 15.5),
    "beta": (15.5, 30.0),
}


def band_power(epochs: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Relative band power per epoch from a Welch PSD; shape (n, len(BANDS)), rows sum to 1."""
    freqs, psd = welch(epochs, fs=sfreq, nperseg=min(256, epochs.shape[-1]), axis=-1)
    absolute = np.stack(
        [psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in BANDS.values()], axis=1
    )
    total = absolute.sum(axis=1, keepdims=True)
    return absolute / np.where(total > 0, total, 1.0)


def context_features(df: pd.DataFrame, cols: list[str], width: int = CONTEXT_WIDTH) -> pd.DataFrame:
    """Neighbouring-epoch copies of ``cols`` within each recording, edge-padded.

    Neighbours follow ``epoch_no`` order inside a recording, never across recordings.
    Output rows are aligned to ``df``'s row order.
    """
    offsets = [o for o in range(-width, width + 1) if o != 0]
    out = {f"bpctx_{c}_{o:+d}": np.empty(len(df)) for c in cols for o in offsets}
    for rows in df.groupby("recording", sort=False).indices.values():
        rows = np.asarray(rows)
        order = rows[np.argsort(df["epoch_no"].to_numpy()[rows], kind="stable")]
        n = len(order)
        for c in cols:
            values = df[c].to_numpy(dtype=float)[order]
            for o in offsets:
                shifted = values[np.clip(np.arange(n) + o, 0, n - 1)]
                out[f"bpctx_{c}_{o:+d}"][order] = shifted
    return pd.DataFrame(out, index=df.index)


def load_signal_epochs(rec: Recording, root: Path, epoch_nos: np.ndarray) -> np.ndarray:
    """(n, 3000) float32 epochs of ``CHANNEL`` at the given original-grid epoch indices."""
    import mne

    raw = mne.io.read_raw_edf(root / rec.psg_file, include=[CHANNEL], preload=True, verbose="error")
    if raw.info["sfreq"] != SFREQ:
        raise ValueError(f"{rec.stem}: expected {SFREQ} Hz, got {raw.info['sfreq']}")
    data = raw.get_data()[0]
    epoch_nos = np.asarray(epoch_nos, dtype=int)
    idx = epoch_nos[:, None] * EPOCH_SAMPLES + np.arange(EPOCH_SAMPLES)[None, :]
    return data[idx].astype(np.float32)


def feature_params(cfg: Config) -> dict:
    """Every parameter that changes feature values; part of the cache key."""
    return {
        "trim_epochs": cfg.trim_epochs,
        "bands": BANDS,
        "channel": CHANNEL,
        "sfreq": SFREQ,
        "context_width": CONTEXT_WIDTH,
        "extractors": [],
    }


def cache_key(rec: Recording, sha256: str, params: dict) -> str:
    """Hash of the recording, its PSG checksum, feature parameters and package version."""
    payload = json.dumps(
        {
            "stem": rec.stem,
            "psg_sha256": sha256,
            "params": params,
            "version": sleepaudit.__version__,
        },
        sort_keys=True,
        default=str,
    )
    return f"{rec.stem}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _recording_features(rec: Recording, root: Path, cfg: Config) -> pd.DataFrame:
    table = recording_labels(rec, root, edge=cfg.trim_epochs).reset_index(drop=True)
    epochs = load_signal_epochs(rec, root, table["epoch_no"].to_numpy())
    bp_cols = [f"bp_{name}" for name in BANDS]
    table[bp_cols] = band_power(epochs)
    ctx = context_features(table, bp_cols)
    return pd.concat([table, ctx], axis=1)


def build_feature_table(cfg: Config) -> pd.DataFrame:
    """Epoch table with ``bp_*`` and ``bpctx_*`` columns; per-recording parquet cache."""
    root = data_dir() / "sleep-cassette"
    cache_dir = data_dir() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    sha = dict(zip(manifest["filename"], manifest["sha256"], strict=True))
    params = feature_params(cfg)
    parts = []
    for rec in select_recordings(cfg.subset, manifest):
        path = cache_dir / f"{cache_key(rec, sha[rec.psg_file], params)}.parquet"
        if path.exists():
            part = pd.read_parquet(path)
        else:
            part = _recording_features(rec, root, cfg)
            tmp = path.with_suffix(".parquet.part")
            part.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cfg.results_dir / "features.parquet", index=False)
    return table
