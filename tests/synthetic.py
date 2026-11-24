"""Synthetic epoch tables shared by the model and evaluation tests (no signal data needed)."""

import numpy as np
import pandas as pd

BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
STAGES = ["W", "N1", "N2", "N3", "REM"]


def synthetic_table(n_subjects=6, n_per_subject=100, seed=0, ctx=True, yasa=False):
    """Separable synthetic epoch table: each stage has its own band-power centre."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(n_subjects):
        label = rng.integers(0, 5, n_per_subject)
        feats = np.eye(5)[label] * 3.0 + rng.normal(0, 0.5, (n_per_subject, 5))
        df = pd.DataFrame(feats, columns=[f"bp_{b}" for b in BANDS])
        df.insert(0, "label", label.astype(np.int8))
        df.insert(0, "epoch_no", np.arange(n_per_subject))
        df.insert(0, "night", 1)
        df.insert(0, "subject", s)
        df.insert(0, "recording", f"SC4{s:02d}1E")
        if ctx:
            for b in BANDS:
                for off in (-2, -1, 1, 2):
                    df[f"bpctx_bp_{b}_{off:+d}"] = df[f"bp_{b}"].shift(-off).bfill().ffill()
        if yasa:
            proba = rng.dirichlet(np.ones(5), n_per_subject)
            for i, st in enumerate(STAGES):
                df[f"yasa_proba_{st}"] = proba[:, i]
            df["yasa_feat_a"] = rng.normal(size=n_per_subject)
            df["yasa_feat_b"] = rng.normal(size=n_per_subject)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
