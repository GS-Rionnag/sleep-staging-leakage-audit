import numpy as np
import pandas as pd

from sleepaudit.stats import paired_bootstrap


def _frame(values, seeds=(0,), subjects=None):
    subjects = list(range(len(values))) if subjects is None else subjects
    return pd.DataFrame(
        [(s, seed, v) for s, v in zip(subjects, values, strict=True) for seed in seeds],
        columns=["subject", "seed", "kappa"],
    )


def test_bootstrap_known_shift():
    leaky = np.linspace(0.5, 0.9, 30)
    res = paired_bootstrap(_frame(leaky - 0.1), _frame(leaky), n_boot=500)
    assert abs(res.mean_diff + 0.1) < 1e-12
    assert abs(res.lo + 0.1) < 1e-12 and abs(res.hi + 0.1) < 1e-12
    assert res.frac_honest_lower == 1.0
    assert res.n_subjects == 30 and res.n_dropped == 0


def test_bootstrap_drops_nan_pairwise():
    leaky = np.linspace(0.5, 0.9, 30)
    leaky_nan = leaky.copy()
    leaky_nan[3] = np.nan
    res = paired_bootstrap(_frame(leaky - 0.1), _frame(leaky_nan), n_boot=200)
    assert res.n_dropped == 1 and res.n_subjects == 29


def test_bootstrap_drops_nan_in_honest_too():
    honest = np.linspace(0.4, 0.8, 30)
    honest[[0, 1]] = np.nan
    res = paired_bootstrap(_frame(honest), _frame(np.full(30, 0.7)), n_boot=200)
    assert res.n_dropped == 2 and res.n_subjects == 28


def test_bootstrap_seed_averaging():
    honest = _frame([0.5, 0.7], seeds=(0, 1))  # seed 0 -> 0.5, seed 1 -> 0.7 (rows interleaved)
    honest["kappa"] = [0.5, 0.7, 0.5, 0.7]
    leaky = _frame([0.9, 0.9, 0.9, 0.9][:2], seeds=(0, 1))
    leaky["kappa"] = 0.9
    res = paired_bootstrap(honest, leaky, n_boot=100)
    assert abs(res.mean_diff - (0.6 - 0.9)) < 1e-12


def test_bootstrap_reproducible_and_seed_dependent():
    rng = np.random.default_rng(1)
    leaky = rng.uniform(0.5, 0.9, 25)
    honest = leaky - rng.uniform(0, 0.2, 25)
    a = paired_bootstrap(_frame(honest), _frame(leaky), n_boot=300, seed=3)
    b = paired_bootstrap(_frame(honest), _frame(leaky), n_boot=300, seed=3)
    c = paired_bootstrap(_frame(honest), _frame(leaky), n_boot=300, seed=4)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert (a.lo, a.hi) != (c.lo, c.hi)
    assert a.lo < a.mean_diff < a.hi
    assert 0 <= a.wilcoxon_p <= 1


def test_bootstrap_resamples_subjects_not_rows():
    # two seeds per subject must not inflate the interval: same result as one seed with the mean
    rng = np.random.default_rng(2)
    base = rng.uniform(0.4, 0.8, 20)
    one = paired_bootstrap(_frame(base), _frame(base + 0.1), n_boot=300)
    two = paired_bootstrap(_frame(base, seeds=(0, 1)), _frame(base + 0.1, seeds=(0, 1)), n_boot=300)
    assert (one.lo, one.hi) == (two.lo, two.hi)


def test_wilcoxon_nan_with_few_subjects():
    res = paired_bootstrap(_frame([0.1, 0.2, 0.3]), _frame([0.2, 0.3, 0.5]), n_boot=50)
    assert np.isnan(res.wilcoxon_p) and res.n_subjects == 3


def test_bootstrap_all_dropped_raises():
    import pytest

    with pytest.raises(ValueError):
        paired_bootstrap(_frame([np.nan]), _frame([0.5]), n_boot=10)
