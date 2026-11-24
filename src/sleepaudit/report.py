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


TRAINED_MODELS = ("bandpower_lr", "bandpower_ctx_lr", "yasa_lgbm")
H2_STEPS = (("block", "epoch"), ("night", "block"), ("subject", "night"))
H3_REFERENCE = "bandpower_lr"
CONTRAST_COLUMNS = [
    "hypothesis", "model", "contrast", "metric", "mean_diff", "lo", "hi",
    "frac_negative", "wilcoxon_p", "n_subjects", "n_dropped",
]  # fmt: skip


def _subject_means(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.groupby("subject")[metric].mean()


def _contrast_row(hyp, model, contrast, metric, a: pd.Series, b: pd.Series, n_boot) -> dict:
    """Paired subject bootstrap of ``a - b`` (both indexed by subject)."""
    res = paired_bootstrap(
        a.rename(metric).reset_index(), b.rename(metric).reset_index(), metric=metric, n_boot=n_boot
    )
    out = {"hypothesis": hyp, "model": model, "contrast": contrast, "metric": metric}
    out |= {k: v for k, v in res.__dict__.items() if k != "frac_honest_lower"}
    return out | {"frac_negative": res.frac_honest_lower}


def build_contrasts(
    per_subject: pd.DataFrame, oof: pd.DataFrame | None, n_boot: int, metric: str = "kappa"
) -> pd.DataFrame:
    """Pre-registered contrasts for H2-H4, each a paired subject bootstrap (seeds averaged first).

    H2: adjacent arms ``block - epoch``, ``night - block``, ``subject - night`` per trained model.
    H3: per-subject gap ``subject - epoch`` of a context/smoothed model minus that of
        ``bandpower_lr`` (negative = larger inflation than band power alone).
    H4: per-subject accuracy gain ``epoch - subject`` in the ``stable`` stratum minus the gain in
        the ``0-1`` stratum (positive = the leaky gain concentrates in stable epochs). Needs ``oof``.
    """
    rows = []
    models = [m for m in TRAINED_MODELS if m in set(per_subject["model"])]
    arm_means = {
        (m, a): _subject_means(g, metric)
        for (m, a), g in per_subject[per_subject["model"].isin(models)].groupby(["model", "arm"])
    }
    for m in models:
        for honest, leaky in H2_STEPS:
            if (m, honest) in arm_means and (m, leaky) in arm_means:
                rows.append(_contrast_row("H2", m, f"{honest} - {leaky}", metric,
                                          arm_means[m, honest], arm_means[m, leaky], n_boot))  # fmt: skip

    def gap(m):
        return arm_means[m, HONEST] - arm_means[m, "epoch"]

    has_gap = {m for m in models if (m, HONEST) in arm_means and (m, "epoch") in arm_means}
    if H3_REFERENCE in has_gap:
        for m in sorted(has_gap - {H3_REFERENCE}, key=TRAINED_MODELS.index):
            rows.append(_contrast_row("H3", m, f"gap({HONEST} - epoch) minus {H3_REFERENCE}'s",
                                      metric, gap(m), gap(H3_REFERENCE), n_boot))  # fmt: skip

    if oof is not None:
        sel = oof[
            oof["model"].isin(models)
            & oof["arm"].isin(["epoch", HONEST])
            & oof["stratum"].isin(["stable", "0-1"])
        ]
        acc = (
            sel.assign(correct=(sel["y_true"] == sel["y_pred"]).astype(float))
            .groupby(["model", "arm", "stratum", "seed", "subject"], observed=True)["correct"]
            .mean()
            .groupby(["model", "arm", "stratum", "subject"], observed=True)
            .mean()
        )
        for m in models:
            try:
                gain = {s: acc[m, "epoch", s] - acc[m, HONEST, s] for s in ("stable", "0-1")}
            except KeyError:
                continue
            rows.append(_contrast_row("H4", m, "gain(epoch - subject) in stable minus in 0-1",
                                      "accuracy", gain["stable"], gain["0-1"], n_boot))  # fmt: skip
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS)


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
    oof_path = results_dir / "oof.parquet"
    oof = (
        pd.read_parquet(oof_path, columns=["model", "arm", "seed", "subject", "stratum", "y_true", "y_pred"])
        if oof_path.exists()
        else None
    )  # fmt: skip
    build_contrasts(per_subject, oof, n_boot).to_csv(results_dir / "contrasts.csv", index=False)
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    arms = set(per_subject["arm"])
    leaky_arm = "epoch" if "epoch" in arms else next((a for a in LEAKY_ARMS if a in arms), None)
    if leaky_arm is not None and HONEST in arms:
        for model in per_subject["model"].unique():
            slopegraph(per_subject, model, leaky_arm, figures / f"slopegraph_{model}.png")
    forest(summary, figures / "forest.png")
