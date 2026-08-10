"""Population Stability Index.

PSI answers one question, deliberately a narrow one: has the *shape* of a
feature's distribution moved between a reference window and a current one?
It needs no labels, which is the whole reason it is the retraining trigger
here -- fraud ground truth arrives weeks later via chargebacks, so accuracy
cannot be measured in real time, but PSI can.

    PSI = sum over buckets of  (current% - reference%) * ln(current% / reference%)

Both factors share a sign, so every term is non-negative and drift in
opposite directions across buckets can never cancel out.

Conventional bands, from credit-scorecard practice:

    PSI < 0.10          stable
    0.10 <= PSI <= 0.20 moderate, monitor
    PSI > 0.20          major, retrain

Two implementation choices materially change the numbers, so they are
stated here rather than buried:

1. **Bucket universe is the union of reference and current**, not the
   reference alone. A category that appears only in the current window --
   a browser version that did not exist during training -- is drift, and
   the most important kind. Reindexing onto reference buckets would drop
   it and report stability. This is why id_31 scores as high as it does.

2. **Zero shares are floored at ``epsilon``, not dropped.** ln(x/0) is
   infinite, so a bucket present in one window and absent from the other
   needs a floor to stay finite. The floor value is therefore part of the
   metric definition: a smaller epsilon reports *more* drift for
   newly-appearing buckets. If a reproduction disagrees with a previously
   computed figure, epsilon is the first thing to check.

Bin edges are derived from the reference and must be frozen alongside the
model that was trained on it. Recomputing quantile edges on live data makes
every window uniform across its own quantiles, so PSI reads near zero
forever and the monitor never fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Quantile bins for numeric features. 10 is the scorecard convention.
DEFAULT_BINS: int = 10

#: Floor applied to zero bucket shares before taking the log ratio.
#: Part of the metric definition -- see module docstring.
DEFAULT_EPSILON: float = 1e-6

#: Bucket code for missing values on the numeric path. Negative so it can
#: never collide with a pd.cut bin index, which starts at 0.
NA_CODE: int = -1

#: Bucket label for missing values on the categorical path. Distinctive
#: enough that collision with a real IEEE-CIS category is implausible.
NA_LABEL: str = "__MISSING__"


def _psi_from_shares(
    reference_share: pd.Series,
    current_share: pd.Series,
    epsilon: float,
) -> float:
    """Sum the PSI contributions over the union of both bucket sets.

    Alignment happens here, once, for both the numeric and categorical
    paths. ``sort=False`` on the union avoids ordering a mixed-type
    categorical index; PSI is a sum, so bucket order is irrelevant.
    """
    buckets = reference_share.index.union(current_share.index, sort=False)
    ref = reference_share.reindex(buckets, fill_value=0.0).to_numpy(dtype=float)
    cur = current_share.reindex(buckets, fill_value=0.0).to_numpy(dtype=float)

    ref = np.clip(ref, epsilon, None)
    cur = np.clip(cur, epsilon, None)

    return float(np.sum((cur - ref) * np.log(cur / ref)))


def reference_edges(
    reference: pd.Series,
    bins: int = DEFAULT_BINS,
) -> np.ndarray | None:
    """Quantile bin edges derived from ``reference``, or None if unusable.

    Missing values are excluded from the quantile calculation -- they get
    their own bucket rather than influencing where the cuts fall.

    ``np.unique`` collapses duplicate edges, which is what makes this safe
    on the many near-constant V-columns in IEEE-CIS: a feature that is 95%
    a single value produces identical quantiles that would otherwise become
    empty bins. Fewer than 3 surviving edges means fewer than 2 real
    buckets, where PSI is not meaningful, and None is returned.

    The outer edges are widened to +/-inf so values beyond the reference
    range -- an amount larger than anything seen in training -- still land
    in the end buckets instead of falling out as NaN and being miscounted
    as missing.
    """
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None

    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return None

    edges = edges.astype(float, copy=True)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bucket_codes(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Assign each value to a bin index, with missing mapped to NA_CODE."""
    numeric = pd.to_numeric(values, errors="coerce")
    codes = pd.cut(numeric, bins=edges, labels=False, right=True)
    return pd.Series(codes).fillna(NA_CODE).astype("int64")


def psi_numeric(
    reference,
    current,
    bins: int = DEFAULT_BINS,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI for a numeric feature, using quantile buckets from the reference.

    Missing values form their own bucket, so a feature that simply stops
    being populated registers as drift -- which it is.

    Returns NaN when the reference cannot yield at least two distinct
    buckets (constant, near-constant, or entirely missing).
    """
    reference = pd.Series(reference)
    current = pd.Series(current)

    edges = reference_edges(reference, bins=bins)
    if edges is None:
        return float("nan")

    reference_share = _bucket_codes(reference, edges).value_counts(normalize=True)
    current_share = _bucket_codes(current, edges).value_counts(normalize=True)
    if reference_share.empty or current_share.empty:
        return float("nan")

    return _psi_from_shares(reference_share, current_share, epsilon)


def psi_categorical(
    reference,
    current,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI for a categorical feature, one bucket per observed level.

    Missing is a level like any other (``dropna=False``), and levels that
    appear only in the current window are counted -- see the module
    docstring on why that asymmetry is deliberate.

    Returns NaN when either side is empty.
    """
    reference = pd.Series(reference).fillna(NA_LABEL)
    current = pd.Series(current).fillna(NA_LABEL)

    if reference.empty or current.empty:
        return float("nan")

    reference_share = reference.value_counts(normalize=True, dropna=False)
    current_share = current.value_counts(normalize=True, dropna=False)

    return _psi_from_shares(reference_share, current_share, epsilon)


def psi_contributions(
    reference,
    current,
    bins: int = DEFAULT_BINS,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    """Per-bucket PSI breakdown for a numeric feature.

    The scalar PSI is the trigger; this is the explanation. A total of 0.21
    built almost entirely from one collapsing bucket is a different finding
    from the same total spread evenly, and only this tells them apart.
    """
    reference = pd.Series(reference)
    current = pd.Series(current)

    edges = reference_edges(reference, bins=bins)
    if edges is None:
        return pd.DataFrame(
            columns=["bucket", "reference_pct", "current_pct", "contribution"]
        )

    reference_share = _bucket_codes(reference, edges).value_counts(normalize=True)
    current_share = _bucket_codes(current, edges).value_counts(normalize=True)

    buckets = reference_share.index.union(current_share.index, sort=False)
    ref = np.clip(reference_share.reindex(buckets, fill_value=0.0).to_numpy(float), epsilon, None)
    cur = np.clip(current_share.reindex(buckets, fill_value=0.0).to_numpy(float), epsilon, None)

    frame = pd.DataFrame(
        {
            "bucket": [
                "missing"
                if code == NA_CODE
                else f"({edges[code]:.4g}, {edges[code + 1]:.4g}]"
                for code in buckets
            ],
            "reference_pct": ref,
            "current_pct": cur,
            "contribution": (cur - ref) * np.log(cur / ref),
        }
    )
    return frame.sort_values("contribution", ascending=False, ignore_index=True)
