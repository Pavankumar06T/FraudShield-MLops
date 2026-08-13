"""Invariants for environment reproducibility.

These exist because a divergence already happened: the same configuration
and the same data produced a 651-tree model on Colab and a 969-tree model
locally. The cause was thread count, not pandas -- and nothing in either run
record said so, which is the part worth fixing permanently.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.training.train import (
    LGB_HYPERPARAMETERS,
    XGB_HYPERPARAMETERS,
    environment_params,
    positive_proba,
    train_lightgbm,
    train_model,
    trees_used,
)


def imbalanced(n: int = 12_000, seed: int = 21):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(signal * 1.6 - 3.6)))).astype(int))
    X = pd.DataFrame(
        {"a": signal, "b": rng.normal(size=n), "c": rng.normal(size=n)}
    )
    cut = int(n * 0.85)
    return X[:cut], y[:cut], X[cut:], y[cut:]


# --------------------------------------------------------------------------
# What the run record must carry
# --------------------------------------------------------------------------


def test_environment_params_capture_everything_that_moves_a_model():
    params = environment_params()
    for key in (
        "env.python",
        "env.platform",
        "env.cpu_count",
        "env.n_jobs_requested",
        "env.n_jobs_effective",
        "env.xgboost",
        "env.lightgbm",
        "env.pandas",
        "env.numpy",
        "env.sklearn",
        "env.pyarrow",
    ):
        assert key in params, key
        assert params[key] not in (None, "")


def test_pandas_version_is_recorded():
    """It was the first suspect for a divergence it did not cause. Cheap to
    log, expensive to reconstruct after the fact."""
    assert environment_params()["env.pandas"] == pd.__version__


def test_effective_thread_count_resolves_minus_one():
    """-1 in the record is useless: it says 'all cores' without saying how
    many, which is exactly the number needed to explain a divergence."""
    detected = os.cpu_count() or 1
    assert environment_params(-1)["env.n_jobs_effective"] == detected
    assert environment_params(2)["env.n_jobs_effective"] == 2
    assert environment_params(2)["env.n_jobs_requested"] == 2


def test_unpinned_default_is_recorded_as_such():
    params = environment_params()
    assert params["env.n_jobs_requested"] == XGB_HYPERPARAMETERS["n_jobs"]


# --------------------------------------------------------------------------
# Thread determinism -- the measured behaviour of each library
# --------------------------------------------------------------------------


def test_lightgbm_is_thread_deterministic():
    """Measured, not assumed: LightGBM gave bit-identical probabilities at
    1 and 4 threads on the real data."""
    Xf, yf, Xs, ys = imbalanced()
    one = train_lightgbm(Xf, yf, Xs, ys, n_jobs=1)
    two = train_lightgbm(Xf, yf, Xs, ys, n_jobs=2)
    np.testing.assert_array_equal(positive_proba(one, Xs), positive_proba(two, Xs))
    assert trees_used(one) == trees_used(two)


def test_xgboost_is_reproducible_at_a_pinned_thread_count():
    """Pinning is what makes a run repeatable. Two fits at the same n_jobs
    must agree exactly; without that guarantee nothing downstream can be
    compared at all."""
    Xf, yf, Xs, ys = imbalanced()
    first = train_model(Xf, yf, Xs, ys, n_jobs=1)
    second = train_model(Xf, yf, Xs, ys, n_jobs=1)
    np.testing.assert_array_equal(
        positive_proba(first, Xs), positive_proba(second, Xs)
    )
    assert trees_used(first) == trees_used(second)


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >= 2 cores")
def test_xgboost_thread_count_changes_the_model():
    """The finding, pinned as a regression test rather than a comment.

    If a future XGBoost makes hist thread-deterministic this test fails --
    which is the right outcome: it means the n_jobs pin can be relaxed, and
    someone should notice rather than carry the constraint forever.
    """
    Xf, yf, Xs, ys = imbalanced()
    one = positive_proba(train_model(Xf, yf, Xs, ys, n_jobs=1), Xs)
    two = positive_proba(train_model(Xf, yf, Xs, ys, n_jobs=2), Xs)
    assert not np.array_equal(one, two), (
        "XGBoost hist now appears thread-deterministic -- if so the n_jobs pin "
        "is no longer needed for reproducibility"
    )


def test_n_jobs_override_reaches_both_libraries():
    """A knob that silently does nothing is worse than no knob."""
    Xf, yf, Xs, ys = imbalanced(n=3_000)
    assert train_model(Xf, yf, Xs, ys, n_jobs=1).n_jobs == 1
    assert train_lightgbm(Xf, yf, Xs, ys, n_jobs=1).n_jobs == 1


def test_override_does_not_mutate_the_promoted_config():
    """n_jobs is an environment knob; it must not leak into the config the
    promotion was decided on."""
    before_xgb = dict(XGB_HYPERPARAMETERS)
    before_lgb = dict(LGB_HYPERPARAMETERS)
    Xf, yf, Xs, ys = imbalanced(n=3_000)
    train_model(Xf, yf, Xs, ys, n_jobs=1)
    train_lightgbm(Xf, yf, Xs, ys, n_jobs=1)
    assert XGB_HYPERPARAMETERS == before_xgb
    assert LGB_HYPERPARAMETERS == before_lgb
