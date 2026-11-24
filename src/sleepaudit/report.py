"""Summary table (paired bootstrap) and figures from ``per_subject.csv``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sleepaudit.stats import paired_bootstrap

HONEST = "subject"
LEAKY_ARMS = ("epoch", "block", "night", "subject_prefit")
BASELINE = "majority"
SUMMARY_COLUMNS = [
    "model", "honest_arm", "leaky_arm", "metric", "mean_diff", "lo", "hi",
    "frac_honest_lower", "wilcoxon_p", "n_subjects", "n_dropped",
]  # fmt: skip


def build_summary(per_subject: pd.DataFrame, n_boot: int, metric: str = "kappa") -> pd.DataFrame:
    """One row per model x comparison (``subject`` against each leaky arm that was run)."""
    rows = []
    for model in per_subject["model"].unique():
        sub = per_subject[per_subject["model"] == model]
        honest = sub[sub["arm"] == HONEST]
        if honest.empty:
            continue
        for leaky_arm in LEAKY_ARMS:
            leaky = sub[sub["arm"] == leaky_arm]
            if leaky.empty:
                continue
            res = paired_bootstrap(honest, leaky, metric=metric, n_boot=n_boot)
            rows.append(
                {"model": model, "honest_arm": HONEST, "leaky_arm": leaky_arm, "metric": metric}
                | res.__dict__
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), *np.percentile(means, [2.5, 97.5])


def _arm_scores(per_subject: pd.DataFrame, model: str, arm: str) -> pd.Series:
    sel = per_subject[(per_subject["model"] == model) & (per_subject["arm"] == arm)]
    return sel.groupby("subject")["kappa"].mean()


def slopegraph(per_subject: pd.DataFrame, model: str, leaky_arm: str, path: Path) -> None:
    """Per-subject kappa, leaky arm vs ``subject``: faint lines, bold mean +/- CI, majority line."""
    pair = pd.concat(
        [_arm_scores(per_subject, model, leaky_arm), _arm_scores(per_subject, model, HONEST)],
        axis=1,
        join="inner",
    ).dropna()
    fig, ax = plt.subplots(figsize=(4.5, 5))
    for a, b in pair.to_numpy():
        ax.plot([0, 1], [a, b], color="0.6", alpha=0.35, lw=0.8)
    means = []
    for x, col in enumerate(pair.columns):
        mean, lo, hi = _mean_ci(pair[col].to_numpy())
        means.append(mean)
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color="C3", marker="o", lw=2.5,
                    capsize=4, zorder=5)  # fmt: skip
    ax.plot([0, 1], means, color="C3", lw=2.5, zorder=4)
    base = _arm_scores(per_subject, BASELINE, HONEST)
    if model != BASELINE and not base.empty:
        ax.axhline(base.mean(), ls="--", color="k", lw=1, label="majority baseline")
        ax.legend(loc="lower left", frameon=False)
    ax.set_xticks([0, 1], [leaky_arm, HONEST])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("per-subject Cohen's kappa")
    ax.set_title(f"{model}: {leaky_arm} vs {HONEST} (n={len(pair)} subjects)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def forest(summary: pd.DataFrame, path: Path) -> None:
    """Paired gaps (honest - leaky) with 95% intervals per model and comparison."""
    labels = [f"{r.model}: {HONEST} - {r.leaky_arm}" for r in summary.itertuples()]
    y = np.arange(len(summary))[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(summary) + 1)))
    ax.errorbar(summary["mean_diff"], y,
                xerr=[summary["mean_diff"] - summary["lo"], summary["hi"] - summary["mean_diff"]],
                fmt="o", color="C0", capsize=3)  # fmt: skip
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("paired difference in per-subject kappa (honest - leaky), 95% interval")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_report(results_dir: Path, n_boot: int = 10000) -> None:
    """Write ``summary.csv`` and the figures from ``per_subject.csv`` in ``results_dir``."""
    results_dir = Path(results_dir)
    per_subject = pd.read_csv(results_dir / "per_subject.csv")
    summary = build_summary(per_subject, n_boot)
    summary.to_csv(results_dir / "summary.csv", index=False)
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    arms = set(per_subject["arm"])
    leaky_arm = "epoch" if "epoch" in arms else next((a for a in LEAKY_ARMS if a in arms), None)
    if leaky_arm is not None and HONEST in arms:
        for model in per_subject["model"].unique():
            slopegraph(per_subject, model, leaky_arm, figures / f"slopegraph_{model}.png")
    forest(summary, figures / "forest.png")
