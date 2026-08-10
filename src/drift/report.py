"""Whole-dataset PSI sweep and missingness decomposition.

Two questions, two outputs:

``psi_report.csv``
    One PSI per feature, train vs stream, sorted worst-first. This is the
    ranked list of what moved.

``psi_decomposition.csv``
    For the features that moved: how much of that movement is a feature
    becoming *absent* rather than its values changing. A column whose NA
    rate jumps from 76% to 24% will post a large PSI while its populated
    values are untouched. That is a pipeline or coverage change, not
    fraudster behaviour, and it calls for a different response than genuine
    value drift -- so the two are separated before anyone reads the ranking.

Run directly to regenerate both:

    python -m src.drift.report
"""

from __future__ import annotations

import sys

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.common.config import (
    REPORTS_DIR,
    STREAM_PARQUET,
    TRAIN_PARQUET,
    ensure_dirs,
    split_stats,
)
from src.drift.psi import DEFAULT_BINS, DEFAULT_EPSILON, psi_categorical, psi_numeric

#: Not features. TransactionID is a key, isFraud is the label, and
#: TransactionDT is the axis the split was cut on -- it drifts perfectly by
#: construction and would sit at the top of every report saying nothing.
EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {"TransactionID", "isFraud", "TransactionDT"}
)

#: Band edges. Moderate is inclusive at both ends, so exactly 0.20 is not
#: yet "major".
STABLE_BELOW: float = 0.10
MAJOR_ABOVE: float = 0.20

#: What the Phase 0 sweep produced. Used only to flag a divergent port --
#: a reproduction that misses these has an implementation bug, not a new
#: finding about the data.
PHASE0_BANDS: dict[str, int] = {"stable": 278, "moderate": 4, "major": 27}
PHASE0_TOP: dict[str, float] = {
    "id_31": 1.525,
    "id_13": 0.566,
    "M8": 0.530,
    "M9": 0.530,
}

REPORT_PATH = REPORTS_DIR / "psi_report.csv"
DECOMPOSITION_PATH = REPORTS_DIR / "psi_decomposition.csv"


def is_categorical_feature(series: pd.Series) -> bool:
    """Decide which PSI path a column takes.

    Booleans are routed to the categorical path despite being numeric to
    pandas. For a roughly balanced boolean the two paths agree exactly --
    quantile interpolation places a single interior edge between the two
    values, recovering the same two buckets. They diverge once the minority
    class is rarer than the first decile: past about 90/10 every quantile
    lands on the majority value, np.unique collapses the edges below three,
    and the numeric path reports NaN. Measured on a 95/5 split it returns
    NaN where the categorical path returns 5.30.

    So the routing is not cosmetic -- without it, the most lopsided binary
    features go missing from the report exactly when they swing hardest.
    The IEEE-CIS M-columns land here.
    """
    return is_bool_dtype(series) or not is_numeric_dtype(series)


def psi_for_column(
    reference: pd.Series,
    current: pd.Series,
    bins: int = DEFAULT_BINS,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute PSI for one column, dispatching on the reference dtype."""
    if is_categorical_feature(reference):
        return psi_categorical(reference, current, epsilon=epsilon)
    return psi_numeric(reference, current, bins=bins, epsilon=epsilon)


def feature_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    exclude: frozenset[str] = EXCLUDED_COLUMNS,
) -> list[str]:
    """Columns scoreable in both frames, in reference order."""
    shared = set(current.columns)
    return [
        column
        for column in reference.columns
        if column not in exclude and column in shared
    ]


def compute_psi_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    exclude: frozenset[str] = EXCLUDED_COLUMNS,
    bins: int = DEFAULT_BINS,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.Series:
    """PSI for every feature, sorted descending.

    NaN entries are features the reference could not bucket -- constant,
    near-constant, or entirely missing. They sort last and are excluded
    from the band counts rather than being reported as stable, because
    "unmeasurable" and "did not move" are different claims.
    """
    scores = {
        column: psi_for_column(
            reference[column], current[column], bins=bins, epsilon=epsilon
        )
        for column in feature_columns(reference, current, exclude)
    }
    return pd.Series(scores, name="psi", dtype=float).sort_values(
        ascending=False, na_position="last"
    )


def classify(psi: float) -> str:
    """Map a PSI value to its band."""
    if pd.isna(psi):
        return "unmeasurable"
    if psi < STABLE_BELOW:
        return "stable"
    if psi <= MAJOR_ABOVE:
        return "moderate"
    return "major"


def band_counts(psi: pd.Series) -> dict[str, int]:
    """Count features per band, including unmeasurable ones."""
    counts = psi.map(classify).value_counts().to_dict()
    return {
        band: int(counts.get(band, 0))
        for band in ("stable", "moderate", "major", "unmeasurable")
    }


def decompose_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    bins: int = DEFAULT_BINS,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    """Split each feature's PSI into missingness-driven and value-driven parts.

    ``psi_all`` scores every row, with missing as its own bucket.
    ``psi_non_null`` scores only populated rows. Bin edges are identical
    between the two -- they are derived from non-null reference values
    either way -- so the difference isolates the missing bucket's
    contribution.

    Reading the output:

    * high ``psi_all``, near-zero ``psi_non_null``, large ``na_rate_delta``
      -- coverage changed, the values did not. Investigate the join or the
      upstream feed, do not retrain on it.
    * ``psi_non_null`` close to ``psi_all`` -- the populated values really
      moved. This is the drift worth retraining for.
    """
    records = []
    for column in features:
        reference_column = reference[column]
        current_column = current[column]
        categorical = is_categorical_feature(reference_column)

        def score(left: pd.Series, right: pd.Series) -> float:
            if categorical:
                return psi_categorical(left, right, epsilon=epsilon)
            return psi_numeric(left, right, bins=bins, epsilon=epsilon)

        psi_all = score(reference_column, current_column)
        psi_non_null = score(reference_column.dropna(), current_column.dropna())
        reference_na = float(reference_column.isna().mean())
        current_na = float(current_column.isna().mean())

        records.append(
            {
                "feature": column,
                "kind": "categorical" if categorical else "numeric",
                "psi_all": psi_all,
                "psi_non_null": psi_non_null,
                "psi_from_missingness": psi_all - psi_non_null,
                "na_rate_reference": reference_na,
                "na_rate_current": current_na,
                "na_rate_delta": current_na - reference_na,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "kind",
                "psi_all",
                "psi_non_null",
                "psi_from_missingness",
                "na_rate_reference",
                "na_rate_current",
                "na_rate_delta",
            ],
            index=pd.Index([], name="feature"),
        )

    frame = pd.DataFrame.from_records(records).set_index("feature")
    return frame.sort_values("psi_all", ascending=False, na_position="last")


def _format_summary(psi: pd.Series) -> str:
    """Render band counts and the top of the ranking against Phase 0."""
    counts = band_counts(psi)
    lines = [
        "",
        f"PSI sweep: {len(psi)} features scored "
        f"({counts['unmeasurable']} unmeasurable)",
        "",
        f"  {'band':<14}{'found':>7}{'phase 0':>9}",
    ]
    matched = True
    for band in ("stable", "moderate", "major"):
        expected = PHASE0_BANDS[band]
        found = counts[band]
        flag = "" if found == expected else "   <-- differs"
        matched &= found == expected
        lines.append(f"  {band:<14}{found:>7}{expected:>9}{flag}")

    lines += ["", "  top 10 by PSI:"]
    for rank, (feature, value) in enumerate(psi.head(10).items(), start=1):
        expected = PHASE0_TOP.get(feature)
        note = f"   (phase 0 ~{expected:.3f})" if expected is not None else ""
        lines.append(f"    {rank:>2}. {feature:<18}{value:>8.3f}{note}")

    lines.append("")
    if matched:
        lines.append("  band counts reproduce Phase 0.")
    else:
        lines.append(
            "  band counts DIFFER from Phase 0 -- treat this as an implementation\n"
            "  bug, not a finding. Check, in order: epsilon (src/drift/psi.py),\n"
            "  bool columns routing to the categorical path, and whether the\n"
            "  bucket universe is the union of both frames or reference-only."
        )
    return "\n".join(lines)


def main() -> int:
    """Sweep train vs stream and write both reports."""
    ensure_dirs()
    stats = split_stats()

    print(f"loading  {TRAIN_PARQUET}")
    train = pd.read_parquet(TRAIN_PARQUET)
    stats["train"].assert_matches(train)

    print(f"loading  {STREAM_PARQUET}")
    stream = pd.read_parquet(STREAM_PARQUET)
    stats["stream"].assert_matches(stream)

    psi = compute_psi_report(train, stream)
    psi.rename_axis("feature").to_frame().assign(band=psi.map(classify)).to_csv(
        REPORT_PATH
    )
    print(_format_summary(psi))
    print(f"\nwrote    {REPORT_PATH}")

    drifting = psi[psi >= STABLE_BELOW].index.tolist()
    decomposition = decompose_drift(train, stream, drifting)
    decomposition.to_csv(DECOMPOSITION_PATH)
    print(
        f"wrote    {DECOMPOSITION_PATH} "
        f"({len(drifting)} features at or above {STABLE_BELOW:.2f})"
    )

    if not decomposition.empty:
        missingness_driven = decomposition[
            (decomposition["psi_non_null"] < STABLE_BELOW)
            & (decomposition["psi_all"] >= MAJOR_ABOVE)
        ]
        if not missingness_driven.empty:
            print(
                f"\n  {len(missingness_driven)} of the major drifters are "
                "missingness-driven -- their populated values are stable:"
            )
            for feature, row in missingness_driven.head(10).iterrows():
                print(
                    f"    {feature:<18} psi {row['psi_all']:.3f} -> "
                    f"{row['psi_non_null']:.3f} on populated rows, "
                    f"NA {row['na_rate_reference']:.1%} -> {row['na_rate_current']:.1%}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
