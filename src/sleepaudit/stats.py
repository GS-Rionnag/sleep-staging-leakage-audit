"""Paired subject bootstrap: subjects are the unit of evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MIN_WILCOXON = 6


@dataclass
class BootResult:
    mean_diff: float
    lo: float
    hi: float
    frac_honest_lower: float  # share of subjects whose honest score is below the leaky one
    wilcoxon_p: float
    n_subjects: int
    n_dropped: int


def _per_subject(df: pd.DataFrame, metric: str) -> pd.Series:
    """Average over seeds within subject (NaN seeds are skipped; all-NaN stays NaN)."""
    return df.groupby("subject")[metric].mean()


def paired_bootstrap(
    honest: pd.DataFrame,
    leaky: pd.DataFrame,
    metric: str = "kappa",
    n_boot: int = 10000,
    seed: int = 0,
) -> BootResult:
    """Bootstrap of the mean paired difference ``honest - leaky`` over subjects.

    Seeds are averaged within subject first, then the two arms are inner-joined on subject.
    Subjects with a NaN in either arm are dropped pairwise and counted in ``n_dropped``.
    """
    pair = pd.concat(
        [_per_subject(honest, metric).rename("h"), _per_subject(leaky, metric).rename("l")],
        axis=1,
        join="inner",
    )
    n_joined = len(pair)
    pair = pair.dropna()
    if pair.empty:
        raise ValueError(f"no subject has a defined {metric} in both arms")
    d = (pair["h"] - pair["l"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if len(d) >= MIN_WILCOXON and np.any(d != 0):
        p = float(stats.wilcoxon(d).pvalue)
    else:
        p = float("nan")
    return BootResult(
        mean_diff=float(d.mean()),
        lo=float(lo),
        hi=float(hi),
        frac_honest_lower=float((d < 0).mean()),
        wilcoxon_p=p,
        n_subjects=len(d),
        n_dropped=n_joined - len(d),
    )
