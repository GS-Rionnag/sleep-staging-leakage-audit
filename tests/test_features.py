from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleepaudit import features
from sleepaudit.config import data_dir, load_config
from sleepaudit.features import BANDS, band_power, build_feature_table, cache_key, context_features
from sleepaudit.labels import recording_labels
from sleepaudit.records import Recording, select_recordings

needs_psg = pytest.mark.needs_psg

CI_CONFIG = Path("configs/ci.yaml")
REC = Recording("SC4001E", 0, 1, "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf")


def test_band_power_alpha_sine():
    t = np.arange(3000) / 100.0
    epochs = np.stack([np.sin(2 * np.pi * 10 * t + p) for p in (0.0, 1.0, 2.0)])
    out = band_power(epochs)
    assert out.shape == (3, 5)
    assert (out.argmax(axis=1) == list(BANDS).index("alpha")).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_context_does_not_cross_recordings():
    df = pd.DataFrame(
        {
            "recording": ["A"] * 3 + ["B"] * 3,
            "epoch_no": [0, 1, 2, 0, 1, 2],
            "bp_delta": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    out = context_features(df, ["bp_delta"], width=2)
    assert out.shape == (6, 4)
    assert list(out.columns) == [
        "bpctx_bp_delta_-2",
        "bpctx_bp_delta_-1",
        "bpctx_bp_delta_+1",
        "bpctx_bp_delta_+2",
    ]
    # B's first row: all earlier context is edge-padded with B's own first epoch.
    assert out.iloc[3]["bpctx_bp_delta_-1"] == 10.0
    assert out.iloc[3]["bpctx_bp_delta_-2"] == 10.0
    # A's last row: later context is padded with A's own last epoch.
    assert out.iloc[2]["bpctx_bp_delta_+1"] == 3.0
    # Interior neighbours are real neighbours.
    assert out.iloc[1]["bpctx_bp_delta_-1"] == 1.0
    assert out.iloc[1]["bpctx_bp_delta_+1"] == 3.0


def test_cache_key_tracks_label_and_feature_code(tmp_path, monkeypatch):
    """Editing labels.py or the feature code must invalidate cached features (line endings ignored)."""
    cfg = load_config(CI_CONFIG)
    names = {p.name for p in features.FEATURE_SOURCES}
    assert {"labels.py", "features.py", "yasa_model.py"} <= names
    src = tmp_path / "labels.py"
    src.write_bytes(b"x = 1\n")
    monkeypatch.setattr(features, "FEATURE_SOURCES", (src,))
    before = cache_key(REC, "abc", features.feature_params(cfg))
    src.write_bytes(b"x = 1\r\n")  # a CRLF checkout of the same code: same key
    assert cache_key(REC, "abc", features.feature_params(cfg)) == before
    src.write_bytes(b"x = 2\n")
    assert cache_key(REC, "abc", features.feature_params(cfg)) != before


def test_context_uses_epoch_no_order_not_row_order():
    df = pd.DataFrame({"recording": ["A"] * 3, "epoch_no": [2, 0, 1], "bp_delta": [3.0, 1.0, 2.0]})
    out = context_features(df, ["bp_delta"], width=1)
    # Row 1 holds epoch 0 (value 1): its +1 neighbour is epoch 1 (value 2).
    assert out.iloc[1]["bpctx_bp_delta_+1"] == 2.0
    assert out.iloc[0]["bpctx_bp_delta_-1"] == 2.0


def test_cache_key_changes_with_params(monkeypatch):
    base = {"trim_epochs": 60, "bands": BANDS}
    key = cache_key(REC, "abc", base)
    assert key == cache_key(REC, "abc", dict(base))
    assert key != cache_key(REC, "abc", {**base, "trim_epochs": 30})
    assert key != cache_key(REC, "abc", {**base, "bands": {**BANDS, "delta": (0.5, 4.0)}})
    assert key != cache_key(REC, "def", base)


def test_feature_params_follow_module_bands(monkeypatch):
    cfg = load_config(CI_CONFIG)
    before = features.feature_params(cfg)
    monkeypatch.setattr(features, "BANDS", {**BANDS, "delta": (0.5, 4.0)})
    assert features.feature_params(cfg) != before


@needs_psg
def test_feature_table_rows_match_labels(tmp_path):
    cfg = load_config(CI_CONFIG)
    cfg = type(cfg)(**{**cfg.__dict__, "subset": [0], "results_dir": tmp_path})
    table = build_feature_table(cfg)
    root = data_dir() / "sleep-cassette"
    expect = pd.concat([recording_labels(r, root) for r in select_recordings([0])])
    assert len(table) == len(expect)
    assert table["epoch_no"].tolist() == expect["epoch_no"].tolist()
    assert (tmp_path / "features.parquet").exists()
    assert table.filter(regex="^bp_").shape[1] == 5
    assert table.filter(regex="^bpctx_").shape[1] == 20


@needs_psg
def test_stale_cache_not_reused(tmp_path):
    """A changed preprocessing parameter must produce a new cache file, not reuse the old one."""
    cfg = load_config(CI_CONFIG)
    cfg = type(cfg)(**{**cfg.__dict__, "subset": [0], "results_dir": tmp_path})
    t1 = build_feature_table(cfg)
    cfg2 = type(cfg)(**{**cfg.__dict__, "trim_epochs": 30})
    t2 = build_feature_table(cfg2)
    assert len(t2) < len(t1)  # a stale 60-epoch cache would give the same length
    root = data_dir() / "sleep-cassette"
    expect = pd.concat([recording_labels(r, root, edge=30) for r in select_recordings([0])])
    assert len(t2) == len(expect)


@needs_psg
def test_features_cli_writes_table(tmp_path):
    from sleepaudit.__main__ import main

    raw = CI_CONFIG.read_text(encoding="utf-8").replace("subset: [0, 1, 2, 3]", "subset: [0]")
    raw = raw.replace("results/ci", tmp_path.as_posix())
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(raw, encoding="utf-8")
    assert main(["features", "--config", str(cfg_path)]) == 0
    assert (tmp_path / "features.parquet").exists()
