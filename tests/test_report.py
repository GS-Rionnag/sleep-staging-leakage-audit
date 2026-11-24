import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleepaudit.__main__ import main
from sleepaudit.config import load_config
from sleepaudit.evaluate import run_evaluation
from sleepaudit.report import render_report
from sleepaudit.runmeta import check_prereg, write_meta
from synthetic import synthetic_table

SUMMARY_COLS = [
    "model", "honest_arm", "leaky_arm", "metric", "mean_diff", "lo", "hi",
    "frac_honest_lower", "wilcoxon_p", "n_subjects", "n_dropped",
]  # fmt: skip
META_KEYS = [
    "commit", "dirty", "uv_lock_sha256", "manifest_sha256", "packages", "config", "seeds",
    "started", "finished", "host", "platform", "cpu_count", "gpu",
]  # fmt: skip


def _per_subject(models=("majority", "bandpower_lr"), arms=("epoch", "subject", "block")):
    rng = np.random.default_rng(0)
    rows = []
    for model in models:
        for arm in arms:
            for seed in (0, 1):
                for subject in range(12):
                    base = 0.0 if model == "majority" else 0.7
                    gap = -0.1 if arm == "subject" and model != "majority" else 0.0
                    rows.append(
                        ("syn", model, arm, seed, subject, base + gap + rng.normal(0, 0.02), 0.8, 0.7, 0.1, 100)
                    )  # fmt: skip
    cols = [
        "dataset",
        "model",
        "arm",
        "seed",
        "subject",
        "kappa",
        "accuracy",
        "macro_f1",
        "ece",
        "n_epochs",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_summary_columns(tmp_path):
    _per_subject().to_csv(tmp_path / "per_subject.csv", index=False)
    render_report(tmp_path, n_boot=200)
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert list(summary.columns) == SUMMARY_COLS
    # one row per model x comparison that exists: epoch and block against subject
    assert len(summary) == 4
    row = summary[(summary.model == "bandpower_lr") & (summary.leaky_arm == "epoch")].iloc[0]
    assert row.honest_arm == "subject" and row.mean_diff == pytest.approx(-0.1, abs=0.03)
    assert (tmp_path / "figures" / "forest.png").stat().st_size > 0
    for model in ("majority", "bandpower_lr"):
        assert (tmp_path / "figures" / f"slopegraph_{model}.png").stat().st_size > 0


def test_meta_fields(tmp_path):
    cfg = load_config(Path("configs/ci.yaml"))
    meta = write_meta(cfg, tmp_path / "meta.json")
    assert set(META_KEYS) <= set(meta)
    assert isinstance(meta["dirty"], bool)
    assert meta["seeds"] == [0] and isinstance(meta["packages"], dict) and meta["packages"]
    assert json.loads((tmp_path / "meta.json").read_text())["commit"] == meta["commit"]


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )  # fmt: skip


def test_prereg_gate(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    monkeypatch.chdir(tmp_path)
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "sc78.yaml"
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--config", str(cfg_path)])
    assert exc.value.code not in (0, None) and "prereg-v1" in str(exc.value.code)
    with pytest.raises(SystemExit) as exc:
        main(["all", "--config", str(cfg_path)])  # `all` must stop before downloading anything
    assert "prereg-v1" in str(exc.value.code)


def test_prereg_gate_tag_must_be_in_history(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a").write_text("x")
    _git(tmp_path, "add", "a")
    _git(tmp_path, "commit", "-q", "-m", "one")
    _git(tmp_path, "tag", "prereg-v1")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "sc78.yaml")
    check_prereg(cfg, allow=False)  # passes: tag exists and is an ancestor of HEAD
    _git(tmp_path, "checkout", "-q", "--orphan", "other")
    (tmp_path / "b").write_text("y")
    _git(tmp_path, "add", "b")
    _git(tmp_path, "commit", "-q", "-m", "unrelated")
    with pytest.raises(SystemExit):
        check_prereg(cfg, allow=False)  # tag exists but is not in the current history


def test_prereg_gate_fixture_and_escape_hatch(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1] / "configs"
    check_prereg(load_config(root / "ci.yaml"), allow=False)  # explicit-subject fixture: no gate
    check_prereg(load_config(root / "sc20.yaml"), allow=True)  # escape hatch


def test_run_evaluation_writes_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no git repo here: meta must still be written
    table = synthetic_table(n_subjects=6, n_per_subject=60, yasa=False)
    cfg = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs" / "ci.yaml"),
        k=3,
        n_boot=100,
        models=["majority", "bandpower_lr"],
        arms=["epoch", "block", "night", "subject", "subject_prefit"],
        results_dir=tmp_path / "res",
    )
    run_evaluation(cfg, table=table, allow_without_prereg=True)
    res = tmp_path / "res"
    for name in (
        "per_subject.csv",
        "oof.parquet",
        "pooled.csv",
        "strata.csv",
        "nights.csv",
        "meta.json",
    ):
        assert (res / name).exists(), name  # fmt: skip
    ps = pd.read_csv(res / "per_subject.csv")
    assert set(ps.arm[ps.model == "majority"]) == {"epoch", "block", "night", "subject"}
    assert "subject_prefit" in set(ps.arm[ps.model == "bandpower_lr"])
    oof = pd.read_parquet(res / "oof.parquet")
    for (_, _, _), g in oof.groupby(["model", "arm", "seed"]):
        assert len(g) == len(table) and g["epoch_no"].notna().all()
    assert {"trans_dist", "stratum", "y_true", "proba_4"} <= set(oof.columns)
    nights = pd.read_csv(res / "nights.csv")
    assert {"kappa", "model", "arm", "tercile_transition_share", "tercile_n1_share"} <= set(
        nights.columns
    )
    assert set(nights.tercile_n1_share) <= {"low", "mid", "high"}  # fmt: skip
    meta = json.loads((res / "meta.json").read_text())
    assert meta["allow_without_prereg"] is True
    render_report(res, n_boot=100)
    assert (
        len(pd.read_csv(res / "summary.csv")) == 4 + 3
    )  # bandpower_lr: 4 comparisons; majority has no subject_prefit


CONTRAST_COLS = [
    "hypothesis", "model", "contrast", "metric", "mean_diff", "lo", "hi",
    "frac_negative", "wilcoxon_p", "n_subjects", "n_dropped",
]  # fmt: skip
ARM_KAPPA = {
    "bandpower_lr": {"epoch": 0.80, "block": 0.76, "night": 0.72, "subject": 0.70},
    "bandpower_ctx_lr": {"epoch": 0.85, "block": 0.80, "night": 0.70, "subject": 0.65},
}


def _ps_arms(arm_kappa=ARM_KAPPA, n_subjects=10):
    rows = [
        ("syn", m, arm, seed, s, k, 0.8, 0.7, 0.1, 100)
        for m, arms in arm_kappa.items()
        for arm, k in arms.items()
        for seed in (0, 1)
        for s in range(n_subjects)
    ]
    cols = ["dataset", "model", "arm", "seed", "subject", "kappa", "accuracy", "macro_f1", "ece",
            "n_epochs"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def _row(c, hyp, model, contrast):
    sel = c[(c.hypothesis == hyp) & (c.model == model) & (c.contrast == contrast)]
    assert len(sel) == 1, (hyp, model, contrast, c[["hypothesis", "model", "contrast"]])
    return sel.iloc[0]


def test_contrasts_h2_adjacent_arms():
    from sleepaudit.report import build_contrasts

    c = build_contrasts(_ps_arms(), oof=None, n_boot=200)
    assert list(c.columns) == CONTRAST_COLS
    for model, k in ARM_KAPPA.items():
        for honest, leaky in (("block", "epoch"), ("night", "block"), ("subject", "night")):
            r = _row(c, "H2", model, f"{honest} - {leaky}")
            assert r.mean_diff == pytest.approx(k[honest] - k[leaky])
            assert r.lo == pytest.approx(r.mean_diff) and r.hi == pytest.approx(r.mean_diff)


def test_contrasts_h3_gap_versus_bandpower_lr():
    from sleepaudit.report import build_contrasts

    c = build_contrasts(_ps_arms(), oof=None, n_boot=200)
    r = _row(c, "H3", "bandpower_ctx_lr", "gap(subject - epoch) minus bandpower_lr's")
    assert r.mean_diff == pytest.approx((0.65 - 0.85) - (0.70 - 0.80))  # -0.1: larger gap
    assert r.frac_negative == 1.0
    assert set(c[c.hypothesis == "H3"].model) == {"bandpower_ctx_lr"}  # yasa_lgbm absent here


def _oof_strata(n_subjects=8, n=40):
    """epoch arm right on every stable epoch, subject arm right on half; both right on 0-1."""
    rows = []
    for arm in ("epoch", "subject"):
        for seed in (0, 1):
            for s in range(n_subjects):
                for i in range(n):
                    stratum = "stable" if i % 2 else "0-1"
                    right = stratum == "0-1" or arm == "epoch" or i % 4 == 1
                    rows.append(("bandpower_lr", arm, seed, s, stratum, 2, 2 if right else 1))
    return pd.DataFrame(rows, columns=["model", "arm", "seed", "subject", "stratum", "y_true",
                                       "y_pred"])  # fmt: skip


def test_contrasts_h4_stable_versus_transition_gain():
    from sleepaudit.report import build_contrasts

    c = build_contrasts(_ps_arms(), oof=_oof_strata(), n_boot=200)
    r = _row(c, "H4", "bandpower_lr", "gain(epoch - subject) in stable minus in 0-1")
    assert r.metric == "accuracy"
    assert r.mean_diff == pytest.approx(0.5)  # stable gain 0.5, 0-1 gain 0
    assert r.lo == pytest.approx(0.5)


def test_render_report_writes_contrasts(tmp_path):
    _ps_arms().to_csv(tmp_path / "per_subject.csv", index=False)
    render_report(tmp_path, n_boot=100)
    c = pd.read_csv(tmp_path / "contrasts.csv")
    assert set(c.hypothesis) == {"H2", "H3"}  # no oof.parquet, so no H4 rows
