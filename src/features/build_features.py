"""Turn merged IEEE-CIS rows into a matrix gradient boosting can consume.

Three things happen here and nothing else: identifier columns are dropped,
the label is separated, and non-numeric columns are ordinal-encoded because
XGBoost cannot accept raw strings.

**Missing values are left missing.** No imputation, no fill, no "missing"
category. XGBoost learns a default direction per split from the data, which
is strictly better than a value we invent -- and in this dataset missingness
is signal in its own right: the drift decomposition found features whose
entire PSI came from coverage changing, with their populated values
untouched. Filling those would erase the finding.

Encoding keeps three states distinct, and the distinction is the point:

    0 .. n-1        a category seen while fitting
    UNKNOWN_CODE    a category that appeared later and was never fitted
    NaN             absent

The middle state is why this module exists in the shape it does. The stream
slice carries browser and device strings that do not occur in train -- that
is the drift we measured -- so applying a fitted encoder must never raise on
an unseen value. It also must not quietly merge unseen values into NaN,
because "a browser we have never seen" and "no browser recorded" are
different facts and the model should be able to split on them separately.

Fit once on train, apply everywhere else. Refitting on the stream would
renumber every category and silently invalidate the model trained against
the old numbering -- classic training-serving skew, and it fails silently
rather than loudly.

    X, y, encoders = build_features(train)          # fit
    X_stream, y_stream, _ = build_features(stream, encoders)   # apply

Run directly to fit encoders on train and report unseen-category coverage
against stream:

    python -m src.features.build_features
    python -m src.features.build_features --sample 50000
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.common.config import (
    MODELS_DIR,
    STREAM_PARQUET,
    TRAIN_PARQUET,
    ensure_dirs,
    split_stats,
)

#: Dropped: a primary key and the axis the temporal split was cut on. Both
#: correlate perfectly with the split by construction and would let the model
#: memorise which window a row came from.
ID_COLUMNS: tuple[str, ...] = ("TransactionID", "TransactionDT")

#: Target column. Absent at serving time, where y comes back as None.
TARGET_COLUMN: str = "isFraud"

#: Code for a category not present when the encoder was fitted. Negative so
#: it can never collide with a fitted code, which start at 0.
UNKNOWN_CODE: float = -1.0

#: Encoded columns and nullable extension columns are stored as float32.
#: Codes are small integers, exact in float32 well past IEEE-CIS's largest
#: cardinality (DeviceInfo, a few thousand levels), and XGBoost converts to
#: float32 internally regardless -- so nothing is lost and the frame halves.
ENCODED_DTYPE = "float32"

ENCODERS_PATH: Path = MODELS_DIR / "encoders.pkl"

#: Bumped when the encoding scheme changes in a way that invalidates
#: previously pickled encoders.
ENCODER_FORMAT_VERSION: int = 1


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Columns needing ordinal encoding, in frame order.

    Selecting on ``not is_numeric_dtype`` rather than ``is_object_dtype``
    is load-bearing under pandas 3.x. Strings read back from parquet arrive
    as the ``str`` dtype, not ``object``, so ``is_object_dtype`` matches
    nothing at all -- an encoder built on it would silently encode zero
    columns and hand XGBoost thirty raw string columns.

    This selector catches ``str``, ``object`` and ``category`` while leaving
    ``bool`` alone (already 0/1) and every numeric dtype untouched.
    """
    return [column for column in frame.columns if not is_numeric_dtype(frame[column])]


def _extension_numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Nullable numeric columns (Int64, Float64, boolean) needing conversion.

    These are numeric to pandas but carry pd.NA rather than np.nan, which
    XGBoost does not accept. Casting to float32 turns pd.NA into np.nan and
    preserves the missingness.
    """
    return [
        column
        for column in frame.columns
        if is_numeric_dtype(frame[column])
        and not isinstance(frame[column].dtype, np.dtype)
    ]


@dataclass
class FeatureEncoders:
    """Fitted ordinal mappings plus the exact column contract to reproduce.

    ``feature_names`` is as important as the mappings. XGBoost binds to
    column position, so a frame with the right columns in the wrong order
    trains and predicts without error while scoring every feature against
    the wrong split thresholds.
    """

    mappings: dict[str, dict[object, int]]
    feature_names: list[str]
    extension_columns: list[str] = field(default_factory=list)
    unknown_code: float = UNKNOWN_CODE
    format_version: int = ENCODER_FORMAT_VERSION
    fitted_rows: int | None = None
    pandas_version: str = ""

    @property
    def categorical_names(self) -> list[str]:
        return list(self.mappings)

    def cardinality(self) -> pd.Series:
        """Levels fitted per categorical column, largest first."""
        counts = {name: len(mapping) for name, mapping in self.mappings.items()}
        return pd.Series(counts, dtype="int64").sort_values(ascending=False)

    def save(self, path: Path = ENCODERS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: Path = ENCODERS_PATH) -> "FeatureEncoders":
        if not path.exists():
            raise FileNotFoundError(
                f"No fitted encoders at {path}. Fit them on train first:\n"
                "    python -m src.features.build_features"
            )
        with path.open("rb") as handle:
            encoders = pickle.load(handle)
        if encoders.format_version != ENCODER_FORMAT_VERSION:
            raise ValueError(
                f"{path}: encoder format v{encoders.format_version}, this code "
                f"expects v{ENCODER_FORMAT_VERSION}. Refit rather than adapt -- "
                "the codes would not line up with the model trained on them."
            )
        return encoders


def _fit_mapping(series: pd.Series) -> dict[object, int]:
    """Build a category -> code mapping from one column.

    Missing values are excluded so NaN never becomes a category. Levels are
    sorted so the numbering is reproducible across runs; the resulting order
    is arbitrary but stable, which is what train/serve consistency needs.
    """
    levels = series.dropna().unique()
    return {level: code for code, level in enumerate(sorted(levels, key=str))}


def _apply_mapping(series: pd.Series, mapping: dict[object, int], unknown: float) -> pd.Series:
    """Map one column to codes, separating unseen values from missing ones.

    ``Series.map`` sends both unseen categories and NaN to NaN, which would
    collapse the two states. They are pulled apart afterwards by asking which
    positions were populated in the input but unmapped in the output.

    The categorical detour is deliberate: mapping a ``category`` column
    returns another ``category`` whose values are the codes, not a numeric
    column, so it needs flattening before it can hold ``unknown``.
    """
    codes = series.map(mapping)
    if isinstance(codes.dtype, pd.CategoricalDtype):
        codes = codes.astype("object")
    codes = pd.to_numeric(codes, errors="coerce").astype(ENCODED_DTYPE)

    unseen = series.notna().to_numpy() & codes.isna().to_numpy()
    if unseen.any():
        codes = codes.mask(unseen, unknown)
    return codes


def build_features(
    df: pd.DataFrame,
    encoders: FeatureEncoders | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, FeatureEncoders]:
    """Prepare ``df`` for gradient boosting.

    Returns ``(X, y, encoders)``. ``y`` is None when the frame carries no
    label, as at serving time.

    With ``encoders=None`` the mappings are fitted from this frame and
    returned. With encoders supplied they are applied unchanged -- no
    refitting, and unseen categories become ``UNKNOWN_CODE`` rather than
    raising.
    """
    fitting = encoders is None

    y: pd.Series | None = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].astype("int8")

    dropped = set(ID_COLUMNS) | {TARGET_COLUMN}
    available = [column for column in df.columns if column not in dropped]

    if fitting:
        categoricals = categorical_columns(df[available])
        mappings = {column: _fit_mapping(df[column]) for column in categoricals}
        extensions = _extension_numeric_columns(df[available])
        feature_names = available
    else:
        mappings = encoders.mappings
        extensions = encoders.extension_columns
        feature_names = encoders.feature_names
        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} column(s) the encoders were fitted on are absent: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. "
                "Filling them would train the model against columns that are not "
                "really there."
            )

    # Built as one dict then constructed once. Assigning column-by-column
    # under copy-on-write would copy the frame on every iteration.
    columns: dict[str, pd.Series] = {}
    for name in feature_names:
        series = df[name]
        if name in mappings:
            columns[name] = _apply_mapping(series, mappings[name], UNKNOWN_CODE)
        elif name in extensions:
            columns[name] = series.astype(ENCODED_DTYPE)
        else:
            columns[name] = series

    X = pd.DataFrame(columns, index=df.index, copy=False)

    leftover = [name for name in X.columns if not is_numeric_dtype(X[name])]
    if leftover:
        raise TypeError(
            f"{len(leftover)} column(s) are still non-numeric after encoding: "
            f"{leftover[:10]}. XGBoost would reject the frame."
        )

    if fitting:
        encoders = FeatureEncoders(
            mappings=mappings,
            feature_names=list(X.columns),
            extension_columns=extensions,
            unknown_code=UNKNOWN_CODE,
            fitted_rows=len(df),
            pandas_version=pd.__version__,
        )
    return X, y, encoders


def unknown_counts(X: pd.DataFrame, encoders: FeatureEncoders) -> pd.DataFrame:
    """Per-column tally of values that were never fitted.

    Run this whenever an encoder meets a new window. A column that suddenly
    reports unseen values is drift arriving through the vocabulary rather
    than through the distribution, and PSI on the raw column will already be
    climbing.
    """
    records = []
    for name in encoders.categorical_names:
        if name not in X.columns:
            continue
        column = X[name]
        unknown = int((column == encoders.unknown_code).sum())
        records.append(
            {
                "feature": name,
                "fitted_levels": len(encoders.mappings[name]),
                "unknown_rows": unknown,
                "unknown_pct": 100.0 * unknown / len(column) if len(column) else 0.0,
                "missing_pct": 100.0 * float(column.isna().mean()),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.set_index("feature").sort_values("unknown_pct", ascending=False)


def read_split(path: Path, sample: int | None = None) -> pd.DataFrame:
    """Read a split, optionally only its first ``sample`` rows.

    Sampling streams row groups and stops early rather than loading the file
    and slicing it, so a smoke test on a small machine never materialises the
    full frame.
    """
    if sample is None:
        return pd.read_parquet(path)

    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    batches, taken = [], 0
    for batch in reader.iter_batches(batch_size=min(sample, 65_536)):
        batches.append(batch)
        taken += batch.num_rows
        if taken >= sample:
            break
    if not batches:
        return pd.read_parquet(path).head(0)
    return pa.Table.from_batches(batches).slice(0, sample).to_pandas()


def _memory_mb(frame: pd.DataFrame) -> float:
    return frame.memory_usage(deep=True).sum() / 1024**2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit ordinal encoders on train and check them against stream."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="read only the first N rows of each split (smoke test on a small machine)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    stats = split_stats()

    print(f"loading  {TRAIN_PARQUET}" + (f"  (first {args.sample:,} rows)" if args.sample else ""))
    train = read_split(TRAIN_PARQUET, args.sample)
    if args.sample is None:
        stats["train"].assert_matches(train)

    X, y, encoders = build_features(train)
    print(
        f"\nfitted on {len(train):,} rows\n"
        f"  X            {X.shape[0]:,} x {X.shape[1]}  ({_memory_mb(X):.0f} MB)\n"
        f"  y            {'absent' if y is None else f'{y.mean() * 100:.3f}% positive'}\n"
        f"  encoded      {len(encoders.categorical_names)} categorical columns\n"
        f"  dropped      {', '.join(ID_COLUMNS)} + {TARGET_COLUMN}\n"
        f"  missing kept {X.isna().to_numpy().mean() * 100:.1f}% of cells are NaN"
    )

    top = encoders.cardinality().head(5)
    print("\n  highest cardinality:")
    for name, levels in top.items():
        print(f"    {name:<16}{levels:>7,} levels")

    if args.sample is not None:
        print(
            f"\nNOT saving encoders: fitted on a {args.sample:,}-row sample, so rare\n"
            "categories are missing from the mappings and every one of them would\n"
            "score as unknown in production. Re-run without --sample to persist."
        )
    else:
        path = encoders.save()
        print(f"\nwrote    {path}")

    print(f"\nloading  {STREAM_PARQUET}" + (f"  (first {args.sample:,} rows)" if args.sample else ""))
    stream = read_split(STREAM_PARQUET, args.sample)
    if args.sample is None:
        stats["stream"].assert_matches(stream)

    X_stream, _, _ = build_features(stream, encoders)
    print(f"  applied without refitting -> {X_stream.shape[0]:,} x {X_stream.shape[1]}")
    assert list(X_stream.columns) == list(X.columns), "column order diverged"

    unknowns = unknown_counts(X_stream, encoders)
    live = unknowns[unknowns["unknown_rows"] > 0] if not unknowns.empty else unknowns
    if live.empty:
        print("\n  no unseen categories in stream.")
    else:
        print(
            f"\n  {len(live)} column(s) carry categories absent from train "
            "-- vocabulary drift:"
        )
        print(
            live.head(12).to_string(
                float_format=lambda v: f"{v:7.3f}",
                formatters={"unknown_rows": lambda v: f"{v:,}"},
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
