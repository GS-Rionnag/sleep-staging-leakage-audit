import numpy as np
import pandas as pd
import pytest

from sleepaudit.fragmentation import (
    STRATA,
    night_summary,
    stratum,
    transition_distance,
    with_transition_distance,
)


def test_transition_distance():
    d = transition_distance(np.array([0, 0, 0, 2, 2, 2, 2, 2]))
    assert d.tolist() == [3, 2, 1, 0, 1, 2, 3, 4]


def test_transition_distance_no_transition():
    d = transition_distance(np.array([2, 2, 2]))
    assert np.isinf(d).all()


def test_transition_distance_empty():
    assert len(transition_distance(np.array([], dtype=int))) == 0


def test_strata_bins():
    out = stratum(np.array([0, 1, 2, 3, 4, 5, 9, np.inf]))
    assert out.tolist() == ["0-1", "0-1", "2-3", "2-3", "4", "stable", "stable", "stable"]
    assert set(STRATA) == {"0-1", "2-3", "4", "stable"}


def test_with_transition_distance_is_per_recording():
    # recording A ends in label 1, B starts with label 1: no transition may be created at the join
    df = pd.DataFrame(
        {
            "recording": ["A"] * 3 + ["B"] * 3,
            "epoch_no": [0, 1, 2, 0, 1, 2],
            "label": [0, 0, 1, 1, 1, 1],
        }
    )
    out = with_transition_distance(df.sample(frac=1, random_state=0))
    out = out.sort_values(["recording", "epoch_no"])
    assert out["trans_dist"].tolist()[:3] == [2, 1, 0]
    assert np.isinf(out["trans_dist"].to_numpy()[3:]).all()


def _night(labels, recording="SC4001E"):
    return pd.DataFrame(
        {
            "recording": recording,
            "subject": 0,
            "night": 1,
            "epoch_no": np.arange(len(labels)),
            "label": labels,
        }
    )


def test_awakenings_count():
    # N2 W W N2 N2 W N2 N2 -> two W runs inside sleep
    s = night_summary(_night([2, 0, 0, 2, 2, 0, 2, 2]))
    assert s.loc[0, "n_awakenings"] == 2
    assert s.loc[0, "waso_min"] == pytest.approx(3 * 0.5)


def test_leading_trailing_wake_not_awakening():
    s = night_summary(_night([0, 0, 2, 2, 0, 0]))
    assert s.loc[0, "n_awakenings"] == 0 and s.loc[0, "waso_min"] == 0


def test_transition_and_n1_share():
    s = night_summary(_night([2, 2, 1, 1]))  # transitions at index 2 only (index 0 has no predecessor)
    assert s.loc[0, "n1_share"] == 0.5
    assert s.loc[0, "transition_share"] == 0.25


def test_age_joined_and_missing_ok():
    s = night_summary(_night([2, 2, 2]))
    assert "age" in s.columns and s.loc[0, "age"] > 0  # SC4001E is in MNE's age table
    s2 = night_summary(_night([2, 2, 2], recording="SC4999E"))
    assert np.isnan(s2.loc[0, "age"])
