import numpy as np
import pandas as pd
import pytest

from sleepaudit.config import VALID_ARMS
from sleepaudit.splits import ARMS, Fold, make_folds

N_EPOCHS = 200
BUFFER = 8


def _index(subjects: list[int], one_night: tuple[int, ...] = (13,), n_epochs: int = N_EPOCHS):
    rows = []
    for s in subjects:
        for night in (1,) if s in one_night else (1, 2):
            rec = f"SC4{s:02d}{night}E"
            for e in range(n_epochs):
                rows.append((rec, s, night, e))
    return pd.DataFrame(rows, columns=["recording", "subject", "night", "epoch_no"])


@pytest.fixture(scope="module")
def idx():
    """Subjects 0-5 with two nights each, plus subject 13 with one night."""
    return _index([0, 1, 2, 3, 4, 5, 13])


K = {"epoch": 5, "block": 5, "night": 4, "subject": 3, "subject_prefit": 3}


def _folds(idx, arm, seed=0, **kw):
    return make_folds(idx, arm, K[arm], seed, **kw)


def test_arms_match_config():
    assert ARMS == VALID_ARMS == ("epoch", "block", "night", "subject", "subject_prefit")


def test_unknown_arm_raises(idx):
    with pytest.raises(ValueError, match="arm"):
        make_folds(idx, "bogus", 3, 0)


@pytest.mark.parametrize("arm", ARMS)
def test_folds_are_named_tuples_of_row_indices(idx, arm):
    folds = _folds(idx, arm)
    assert len(folds) == K[arm]
    for f in folds:
        assert isinstance(f, Fold)
        assert f.train.dtype.kind == "i" and f.test.dtype.kind == "i"
        assert len(f.test) > 0 and len(f.train) > 0
        assert not set(f.train) & set(f.test)  # a row is never both trained on and tested


def test_subject_arm_no_overlap(idx):
    for seed in range(5):
        for f in _folds(idx, "subject", seed):
            assert (
                set(idx["subject"].to_numpy()[f.train]) & set(idx["subject"].to_numpy()[f.test])
                == set()
            )


def test_subject_prefit_folds_identical_to_subject(idx):
    a = _folds(idx, "subject", 2)
    b = _folds(idx, "subject_prefit", 2)
    for fa, fb in zip(a, b, strict=True):
        np.testing.assert_array_equal(fa.train, fb.train)
        np.testing.assert_array_equal(fa.test, fb.test)


def test_night_arm_shares_subjects_not_recordings(idx):
    folds = _folds(idx, "night")
    rec = idx["recording"].to_numpy()
    sub = idx["subject"].to_numpy()
    shared = []
    for f in folds:
        assert set(rec[f.train]) & set(rec[f.test]) == set()
        shared.append(bool(set(sub[f.train]) & set(sub[f.test])))
    assert any(shared)


def test_epoch_arm_leaks(idx):
    sub = idx["subject"].to_numpy()
    folds = _folds(idx, "epoch")
    assert any(set(sub[f.train]) & set(sub[f.test]) for f in folds)
    # epoch-wise: recordings are shared too, and so are neighbouring epochs
    rec = idx["recording"].to_numpy()
    assert any(set(rec[f.train]) & set(rec[f.test]) for f in folds)


@pytest.mark.parametrize("arm", ARMS)
def test_every_row_tested_once(idx, arm):
    for seed in (0, 1):
        tested = np.concatenate([f.test for f in _folds(idx, arm, seed)])
        assert sorted(tested.tolist()) == list(range(len(idx)))


def test_block_buffer(idx):
    rec = idx["recording"].to_numpy()
    epoch = idx["epoch_no"].to_numpy()
    for seed in range(3):
        for f in _folds(idx, "block", seed, buffer_epochs=BUFFER):
            for r in np.unique(rec[f.test]):
                test_e = epoch[f.test][rec[f.test] == r]
                train_e = epoch[f.train][rec[f.train] == r]
                if not len(train_e):
                    continue
                gap = np.abs(train_e[:, None] - test_e[None, :]).min(axis=1)
                assert gap.min() > BUFFER, f"{r}: train row {gap.min()} epochs from a test row"


def test_block_arm_tests_whole_blocks_and_drops_only_neighbours(idx):
    rec = idx["recording"].to_numpy()
    epoch = idx["epoch_no"].to_numpy()
    folds = _folds(idx, "block", 0, block_epochs=60, buffer_epochs=BUFFER)
    key = [f"{r}:{e // 60}" for r, e in zip(rec, epoch, strict=True)]
    key = np.asarray(key)
    for f in folds:
        # A block is entirely in test or entirely out of it.
        assert set(key[f.train]) & set(key[f.test]) == set()
        for k in set(key[f.test]):
            assert (key == k).sum() == (key[f.test] == k).sum()
        # Buffering removes rows from train only.
        assert len(f.train) + len(f.test) < len(idx)
        # Rows of recordings with no test rows in this fold are untouched.
        untouched = ~np.isin(rec, np.unique(rec[f.test]))
        assert set(np.flatnonzero(untouched)) <= set(f.train)


def test_block_buffer_zero_keeps_all_other_blocks(idx):
    for f in _folds(idx, "block", 0, buffer_epochs=0):
        assert len(f.train) + len(f.test) == len(idx)


def test_same_seed_same_folds(idx):
    for arm in ARMS:
        a, b = _folds(idx, arm, 3), _folds(idx, arm, 3)
        for fa, fb in zip(a, b, strict=True):
            np.testing.assert_array_equal(fa.train, fb.train)
            np.testing.assert_array_equal(fa.test, fb.test)


def test_different_seed_different_folds(idx):
    for arm in ARMS:
        a, b = _folds(idx, arm, 0), _folds(idx, arm, 1)
        assert any(not np.array_equal(fa.test, fb.test) for fa, fb in zip(a, b, strict=True)), arm


@pytest.mark.parametrize("arm", ["night", "subject", "subject_prefit"])
def test_one_night_subject_ok(idx, arm):
    sub = idx["subject"].to_numpy()
    for seed in range(5):
        folds = _folds(idx, arm, seed)
        hits = [bool((sub[f.test] == 13).any()) for f in folds]
        assert sum(hits) == 1, f"{arm} seed {seed}: subject 13 tested in {sum(hits)} folds"


def test_k_too_large_raises(idx):
    with pytest.raises(ValueError, match="subject"):
        make_folds(idx, "subject", 20, 0)
    with pytest.raises(ValueError, match="subject"):
        make_folds(idx, "subject_prefit", 20, 0)


def test_k_too_large_raises_for_every_arm(idx):
    n_rec = idx["recording"].nunique()
    n_blocks = len(set(zip(idx["recording"], idx["epoch_no"] // 60, strict=True)))
    assert n_rec == 13 and n_blocks == 52
    with pytest.raises(ValueError, match="night"):
        make_folds(idx, "night", n_rec + 1, 0)
    with pytest.raises(ValueError, match="block"):
        make_folds(idx, "block", n_blocks + 1, 0)
    with pytest.raises(ValueError, match="epoch"):
        make_folds(idx, "epoch", len(idx) + 1, 0)
    # exactly as many folds as groups is allowed
    assert len(make_folds(idx, "subject", 7, 0)) == 7


def test_folds_not_contiguous_ids():
    idx20 = _index(list(range(20)), one_night=(13,))
    folds = make_folds(idx20, "subject", 4, 0)
    ranges_contiguous = []
    for f in folds:
        ids = np.unique(idx20["subject"].to_numpy()[f.test])
        ranges_contiguous.append(bool(ids.max() - ids.min() + 1 == len(ids)))
    assert not all(ranges_contiguous)
    assert (
        len(set(map(tuple, (np.unique(idx20["subject"].to_numpy()[f.test]) for f in folds)))) == 4
    )


def test_input_index_is_not_mutated(idx):
    before = idx.copy()
    _folds(idx, "block")
    pd.testing.assert_frame_equal(idx, before)


def test_unsorted_row_order_is_fine(idx):
    shuffled = idx.sample(frac=1.0, random_state=0).reset_index(drop=True)
    sub = shuffled["subject"].to_numpy()
    for f in make_folds(shuffled, "subject", 3, 0):
        assert set(sub[f.train]) & set(sub[f.test]) == set()
