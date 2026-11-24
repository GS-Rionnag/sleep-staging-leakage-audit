import numpy as np
import pandas as pd
import pytest

from sleepaudit.config import data_dir
from sleepaudit.labels import epoch_labels, recording_labels, trim
from sleepaudit.records import select_recordings

W, S1, S2, S3, S4, R, Q, M = (
    "Sleep stage W",
    "Sleep stage 1",
    "Sleep stage 2",
    "Sleep stage 3",
    "Sleep stage 4",
    "Sleep stage R",
    "Sleep stage ?",
    "Movement time",
)


def test_epoch_labels_merges_n4_and_clips():
    out = epoch_labels(
        np.array([0.0, 60.0, 120.0]),
        np.array([60.0, 60.0, 480.0]),
        [W, S4, Q],
        psg_seconds=150,
    )
    assert out.dtype == np.int8
    assert out.tolist() == [0, 0, 3, 3, -1]


def test_epoch_labels_nonzero_onset():
    out = epoch_labels(np.array([60.0]), np.array([60.0]), [S2], psg_seconds=120)
    assert out.tolist() == [-1, -1, 2, 2]


def test_epoch_labels_all_stage_codes():
    onsets = np.arange(0, 240, 30.0)
    out = epoch_labels(onsets, np.full(8, 30.0), [W, S1, S2, S3, S4, R, Q, M], 240)
    assert out.tolist() == [0, 1, 2, 3, 3, 4, -1, -1]


def test_trim_drops_unknown_before_trimming():
    labels = np.array([0] * 100 + [-1] * 10 + [0] * 30 + [2] * 20 + [0] * 100)
    kept = trim(labels, edge=60)
    valid = np.flatnonzero(labels != -1)
    assert kept.tolist() == valid[70:210].tolist()  # 130 valid epochs precede first N2; keep 60
    assert (labels[kept] != -1).all()
    first_sleep = np.flatnonzero(labels[kept] != 0)[0]
    assert first_sleep == 60


def test_trim_clips_to_bounds():
    labels = np.array([0] * 10 + [2] * 5 + [0] * 10)
    assert trim(labels, edge=60).tolist() == list(range(25))


@pytest.mark.parametrize("labels", [[0] * 100, [-1] * 50 + [0] * 5, []])
def test_trim_all_wake_raises(labels):
    with pytest.raises(ValueError, match="SC4999E"):
        trim(np.array(labels), name="SC4999E")


def _counts(subset: str) -> tuple[dict[str, int], int]:
    root = data_dir() / "sleep-cassette"
    frames = [recording_labels(rec, root) for rec in select_recordings(subset)]
    labels = pd.concat(frames)["label"]
    counts = labels.value_counts()
    names = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    return {names[k]: int(counts.get(k, 0)) for k in names}, len(labels)


@pytest.mark.needs_labels
def test_sc20_counts():
    counts, total = _counts("sc20")
    assert {k: counts[k] for k in ("N1", "N2", "N3", "REM")} == {
        "N1": 2804, "N2": 17799, "N3": 5703, "REM": 7717,
    }
    assert abs(counts["W"] - 8285) <= 1
    assert abs(total - 42308) <= 1


@pytest.mark.needs_labels
def test_sc78_counts():
    counts, total = _counts("sc78")
    assert {k: counts[k] for k in ("N1", "N2", "N3", "REM")} == {
        "N1": 21522, "N2": 69132, "N3": 13039, "REM": 25835,
    }
    assert abs(counts["W"] - 65951) <= 10
    assert abs(total - 195479) <= 10


@pytest.mark.needs_labels
def test_tutorial_recordings():
    root = data_dir() / "sleep-cassette"
    recs = {r.stem: r for r in select_recordings("sc78")}
    assert len(recording_labels(recs["SC4001E"], root)) == 841
    assert len(recording_labels(recs["SC4011E"], root)) == 1103
