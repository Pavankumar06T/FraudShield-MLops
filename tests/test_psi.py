"""Invariants for the PSI implementation.

These pin the behaviours that decide whether a sweep reproduces Phase 0.
Every one of them was chosen because getting it wrong changes the reported
numbers rather than raising an error -- a PSI implementation fails quietly
or not at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift.psi import (
    DEFAULT_EPSILON,
    psi_categorical,
    psi_contributions,
    psi_numeric,
    reference_edges,
)
from src.drift.report import (
    EXCLUDED_COLUMNS,
    classify,
    compute_psi_report,
    decompose_drift,
    is_categorical_feature,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --------------------------------------------------------------------------
# The formula itself
# --------------------------------------------------------------------------


def test_matches_hand_computed_worked_example():
    """Five buckets whose PSI was computed by hand as 0.2082.

    Reference 35/30/20/10/5 shifting to 20/25/25/20/10. Built as explicit
    counts so the quantile edges land exactly on the intended boundaries.
    """
    edges_pct = [0.35, 0.30, 0.20, 0.10, 0.05]
    current_pct = [0.20, 0.25, 0.25, 0.20, 0.10]
    expected = sum(
        (c - r) * np.log(c / r) for r, c in zip(edges_pct, current_pct)
    )
    assert expected == pytest.approx(0.2082, abs=5e-4)

    scale = 100_000
    reference = pd.Series(
        np.repeat(np.arange(5, dtype=float), [int(p * scale) for p in edges_pct])
    )
    current = pd.Series(
        np.repeat(np.arange(5, dtype=float), [int(p * scale) for p in current_pct])
    )
    assert psi_categorical(reference, current) == pytest.approx(expected, abs=1e-9)


def test_identical_distributions_score_zero(rng):
    numeric = pd.Series(rng.normal(size=20_000))
    categorical = pd.Series(rng.choice(list("abcde"), size=20_000))
    assert psi_numeric(numeric, numeric) == pytest.approx(0.0, abs=1e-12)
    assert psi_categorical(categorical, categorical) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_never_negative(rng):
    """Both factors share a sign, so opposing bucket moves cannot cancel."""
    for _ in range(20):
        a = pd.Series(rng.normal(0, 1, 5_000))
        b = pd.Series(rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 2), 5_000))
        assert psi_numeric(a, b) >= 0.0


def test_contributions_sum_to_the_scalar(rng):
    a = pd.Series(rng.normal(0, 1, 20_000))
    b = pd.Series(rng.normal(0.5, 1, 20_000))
    contributions = psi_contributions(a, b)
    assert contributions["contribution"].sum() == pytest.approx(
        psi_numeric(a, b), abs=1e-9
    )


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_outer_edges_are_infinite_so_unseen_tails_still_bin(rng):
    """A value larger than anything in training must land in the top bucket.

    With finite outer edges it would become NaN and be miscounted as
    missing -- drift in the tail would read as a coverage change.
    """
    edges = reference_edges(pd.Series(rng.uniform(0, 100, 10_000)))
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    codes = pd.cut(pd.Series([-1e12, 1e12]), bins=edges, labels=False)
    assert codes.isna().sum() == 0


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([5.0] * 100, id="constant"),
        pytest.param([np.nan] * 100, id="all-missing"),
    ],
)
def test_unbinnable_reference_returns_nan(values):
    series = pd.Series(values)
    assert reference_edges(series) is None
    assert np.isnan(psi_numeric(series, series))


def test_quantile_edges_are_frozen_to_the_reference(rng):
    """Edges come from the reference only.

    Recomputing them on the current window would make every window uniform
    across its own quantiles, so PSI would read ~0 forever and the monitor
    would never fire.
    """
    reference = pd.Series(rng.normal(0, 1, 20_000))
    shifted = pd.Series(rng.normal(5, 1, 20_000))
    assert psi_numeric(reference, shifted) > 1.0


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------


def test_missing_forms_its_own_bucket_numeric(rng):
    populated = pd.Series(rng.normal(size=20_000))
    half_missing = populated.copy()
    half_missing[:10_000] = np.nan
    assert psi_numeric(populated, half_missing) > 1.0


def test_missing_forms_its_own_bucket_categorical():
    reference = pd.Series(["x"] * 9_000 + ["y"] * 1_000)
    current = pd.Series(["x"] * 5_000 + ["y"] * 1_000 + [None] * 4_000)
    assert psi_categorical(reference, current) > 1.0


def test_dropna_isolates_missingness_from_value_drift(rng):
    """The decomposition's whole purpose, on a case with a known answer.

    Coverage jumps from 24% to 90% while the populated values are drawn
    from the same distribution: psi_all must be large, psi_non_null small.
    """
    n = 30_000
    reference = pd.DataFrame(
        {"f": np.where(rng.random(n) < 0.76, np.nan, rng.normal(size=n))}
    )
    current = pd.DataFrame(
        {"f": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n))}
    )
    row = decompose_drift(reference, current, ["f"]).loc["f"]
    assert row["psi_all"] > 0.20
    assert row["psi_non_null"] < 0.10
    assert row["na_rate_reference"] == pytest.approx(0.76, abs=0.02)
    assert row["na_rate_current"] == pytest.approx(0.10, abs=0.02)
    assert row["psi_from_missingness"] == pytest.approx(
        row["psi_all"] - row["psi_non_null"]
    )


# --------------------------------------------------------------------------
# Bucket universe and epsilon -- the two choices that move the numbers
# --------------------------------------------------------------------------


def test_level_absent_from_reference_is_counted():
    """The id_31 case: a browser version that did not exist in training.

    Reindexing onto reference buckets instead of the union would drop this
    level entirely and report the feature as perfectly stable.
    """
    reference = pd.Series(["chrome"] * 8_000 + ["firefox"] * 2_000)
    current = pd.Series(["chrome"] * 6_000 + ["firefox"] * 2_000 + ["brave"] * 2_000)
    assert psi_categorical(reference, current) > 1.0


def test_smaller_epsilon_reports_more_drift():
    """Epsilon is part of the metric definition, not an implementation detail.

    For a newly-appearing level it spans more than a full point across two
    orders of magnitude, so it is the first thing to check when a
    reproduction disagrees.
    """
    reference = pd.Series(["chrome"] * 8_000 + ["firefox"] * 2_000)
    current = pd.Series(["chrome"] * 6_000 + ["firefox"] * 2_000 + ["brave"] * 2_000)
    tight = psi_categorical(reference, current, epsilon=1e-8)
    default = psi_categorical(reference, current, epsilon=DEFAULT_EPSILON)
    loose = psi_categorical(reference, current, epsilon=1e-4)
    assert tight > default > loose
    assert tight - loose > 1.0


# --------------------------------------------------------------------------
# Dtype dispatch
# --------------------------------------------------------------------------


def test_bool_is_routed_to_the_categorical_path():
    assert is_categorical_feature(pd.Series([True, False]))
    assert is_categorical_feature(pd.Series(["a", "b"]))
    assert not is_categorical_feature(pd.Series([1.0, 2.0]))
    assert not is_categorical_feature(pd.Series([1, 2]))


def test_balanced_bool_scores_the_same_on_either_path():
    """At moderate imbalance the paths agree, which is why the bug is subtle."""
    reference = pd.Series([True] * 7_000 + [False] * 3_000)
    current = pd.Series([True] * 3_000 + [False] * 7_000)
    assert psi_categorical(reference, current) == pytest.approx(0.6778, abs=1e-3)
    assert psi_numeric(reference, current) == pytest.approx(
        psi_categorical(reference, current), abs=1e-9
    )


def test_lopsided_bool_is_lost_by_the_numeric_path():
    """Past ~90/10 every quantile lands on the majority value.

    np.unique then collapses the edges below three and the numeric path
    returns NaN -- so without the bool routing the most lopsided binary
    features vanish from the report exactly when they swing hardest.
    """
    reference = pd.Series([True] * 9_500 + [False] * 500)
    current = pd.Series([True] * 500 + [False] * 9_500)
    assert np.isnan(psi_numeric(reference, current))
    assert psi_categorical(reference, current) == pytest.approx(5.30, abs=0.01)


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "band"),
    [
        (0.0, "stable"),
        (0.099, "stable"),
        (0.10, "moderate"),
        (0.20, "moderate"),
        (0.201, "major"),
        (float("nan"), "unmeasurable"),
    ],
)
def test_band_edges(value, band):
    assert classify(value) == band


def test_report_excludes_key_label_and_split_axis(rng):
    n = 5_000
    frame = pd.DataFrame(
        {
            "TransactionID": range(n),
            "isFraud": rng.integers(0, 2, n),
            "TransactionDT": rng.integers(0, 10**6, n),
            "feature": rng.normal(size=n),
        }
    )
    psi = compute_psi_report(frame, frame)
    assert set(psi.index) == {"feature"}
    assert EXCLUDED_COLUMNS == {"TransactionID", "isFraud", "TransactionDT"}


def test_report_is_sorted_with_unmeasurable_last(rng):
    n = 10_000
    reference = pd.DataFrame(
        {
            "stable": rng.normal(size=n),
            "drifted": rng.normal(0, 1, n),
            "constant": np.ones(n),
        }
    )
    current = pd.DataFrame(
        {
            "stable": rng.normal(size=n),
            "drifted": rng.normal(2, 1, n),
            "constant": np.ones(n),
        }
    )
    psi = compute_psi_report(reference, current)

    assert psi.index[0] == "drifted"
    assert psi.index[-1] == "constant"
    assert np.isnan(psi["constant"])
    ranked = psi.dropna().tolist()
    assert ranked == sorted(ranked, reverse=True)
    assert psi["stable"] < 0.10 < psi["drifted"]
