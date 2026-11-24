from pathlib import Path

import pytest

from sleepaudit.config import data_dir, load_config


def test_load_ci_config():
    cfg = load_config(Path("configs/ci.yaml"))
    assert cfg.subset == [0, 1, 2, 3]
    assert cfg.k == 2
    assert "subject" in cfg.arms


def test_full_configs_pin_k():
    sc78 = load_config(Path("configs/sc78.yaml"))
    sc20 = load_config(Path("configs/sc20.yaml"))
    assert sc78.k == 10
    assert sc20.k == 20
    assert sc78.seeds == [0, 1, 2, 3, 4]
    assert sc20.seeds == [0, 1, 2, 3, 4]


def test_unknown_arm_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nsubset: sc20\nk: 2\nseeds: [0]\narms: [random]\n"
        "models: [majority]\nresults_dir: results/bad\n"
    )
    with pytest.raises(ValueError, match="random"):
        load_config(bad)


def test_data_dir_env(tmp_path, monkeypatch):
    target = tmp_path / "dir with space"
    monkeypatch.setenv("SLEEPAUDIT_DATA", str(target))
    assert data_dir() == target
    assert target.exists()
