"""Invariants for feature building.

The failure modes here are all silent: an encoder that refits, a NaN that
becomes a category, an unseen value that merges into missing, a column
order that drifts. None of them raise -- they produce a model that trains
cleanly and scores wrongly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from src.features.build_features import (
    ID_COLUMNS,
    TARGET_COLUMN,
    UNKNOWN_CODE,
    FeatureEncoders,
    build_features,
    categorical_columns,
    unknown_counts,
)


def frame(n: int = 12, *, extra_browser: bool = False, label: bool = True) -> pd.DataFrame:
    """A frame carrying every dtype the real merged data produces."""
    browsers = ["chrome", "safari", None] * (n // 3)
    if extra_browser:
        browsers = ["brave" if i == 0 else b for i, b in enumerate(browsers)]
    data = {
        "TransactionID": np.arange(n),
        "TransactionDT": np.arange(n) * 100,
        "TransactionAmt": np.linspace(1.0, 100.0, n),
        # the three shapes a categorical arrives in
        "str_col": pd.Series(browsers, dtype="str"),
        "obj_col": pd.Series(["a", "b", None] * (n // 3), dtype=object),
        "cat_col": pd.Series(["x", "y", None] * (n // 3), dtype="category"),
        # numeric variants
        "float_col": pd.Series([1.0, np.nan] * (n // 2)),
        "bool_col": pd.Series([True, False] * (n // 2)),
        "nullable_int": pd.array([1, None] * (n // 2), dtype="Int64"),
    }
    if label:
        data[TARGET_COLUMN] = np.tile([0, 1], n // 2)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Column selection -- the pandas 3.x trap
# --------------------------------------------------------------------------


def test_selector_catches_str_object_and_category():
    """is_object_dtype would match none of these under pandas 3.x.

    Strings from parquet arrive as the ``str`` dtype. A selector built on
    is_object_dtype encodes nothing and hands XGBoost raw strings.
    """
    df = frame()
    assert set(categorical_columns(df)) == {"str_col", "obj_col", "cat_col"}


def test_bool_and_numeric_are_not_encoded():
    df = frame()
    selected = categorical_columns(df)
    assert "bool_col" not in selected
    assert "float_col" not in selected
    assert "nullable_int" not in selected


def test_every_output_column_is_numeric():
    X, _, _ = build_features(frame())
    assert all(is_numeric_dtype(X[c]) for c in X.columns)
    assert all(isinstance(X[c].dtype, np.dtype) for c in X.columns)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_ids_dropped_and_target_separated():
    df = frame()
    X, y, _ = build_features(df)
    for column in (*ID_COLUMNS, TARGET_COLUMN):
        assert column not in X.columns
    assert y is not None
    pd.testing.assert_series_equal(
        y, df[TARGET_COLUMN].astype("int8"), check_names=False
    )


def test_target_absent_yields_none():
    X, y, _ = build_features(frame(label=False))
    assert y is None
    assert len(X.columns) > 0


def test_row_count_and_index_preserved():
    df = frame().set_index(pd.Index(range(100, 112), name="row"))
    X, y, _ = build_features(df)
    assert len(X) == len(df)
    assert X.index.equals(df.index)
    assert y.index.equals(df.index)


# --------------------------------------------------------------------------
# The three states: fitted code / unknown / missing
# --------------------------------------------------------------------------


def test_missing_is_never_imputed():
    """NaN must survive encoding as NaN, not become a level or a fill value."""
    df = frame()
    X, _, encoders = build_features(df)
    for column in encoders.categorical_names:
        assert X[column].isna().sum() == df[column].isna().sum()
        assert not (X[column] == encoders.unknown_code).any()


def test_nan_is_not_fitted_as_a_category():
    _, _, encoders = build_features(frame())
    for mapping in encoders.mappings.values():
        assert not any(pd.isna(level) for level in mapping)


def test_unseen_category_maps_to_unknown_without_raising():
    train = frame()
    _, _, encoders = build_features(train)
    stream = frame(extra_browser=True)
    X, _, _ = build_features(stream, encoders)

    unseen_rows = stream["str_col"] == "brave"
    assert unseen_rows.any()
    assert (X.loc[unseen_rows, "str_col"] == UNKNOWN_CODE).all()


def test_unknown_and_missing_stay_distinct():
    """Merging them would erase the difference between 'a browser we have
    never seen' and 'no browser recorded'."""
    train = frame()
    _, _, encoders = build_features(train)
    stream = frame(extra_browser=True)
    X, _, _ = build_features(stream, encoders)

    column = X["str_col"]
    assert (column == UNKNOWN_CODE).any()
    assert column.isna().any()
    assert not column[column.isna()].eq(UNKNOWN_CODE).any()
    assert column.isna().sum() == stream["str_col"].isna().sum()


@pytest.mark.parametrize("column", ["str_col", "obj_col", "cat_col"])
def test_all_three_dtypes_encode_identically(column):
    """category.map() returns a category, not a number -- it needs flattening
    before it can carry the unknown code.

    The new level is introduced by rebuilding the column rather than by
    .loc assignment, which pandas refuses on a Categorical. That mirrors
    production: train and stream are separate parquet files, so each gets
    its own category set and stream's may contain levels train's does not.
    """
    train = frame()
    _, _, encoders = build_features(train)

    stream = train.copy()
    original = stream[column]
    values = original.astype(object).tolist()
    values[0] = "never_seen_before"
    stream[column] = pd.Series(values, index=stream.index, dtype=original.dtype.name)

    X, _, _ = build_features(stream, encoders)
    assert is_numeric_dtype(X[column])
    assert X.loc[0, column] == UNKNOWN_CODE
    assert X[column].isna().sum() == original.isna().sum()


# --------------------------------------------------------------------------
# Fit vs apply
# --------------------------------------------------------------------------


def test_apply_does_not_refit():
    """Refitting on a new window renumbers every category, silently
    invalidating the model trained on the old numbering."""
    train = frame()
    _, _, encoders = build_features(train)
    before = {k: dict(v) for k, v in encoders.mappings.items()}

    stream = frame(extra_browser=True)
    _, _, returned = build_features(stream, encoders)

    assert returned is encoders
    assert encoders.mappings == before
    assert "brave" not in encoders.mappings["str_col"]


def test_codes_are_stable_across_calls():
    train = frame()
    _, _, encoders = build_features(train)
    first, _, _ = build_features(train, encoders)
    second, _, _ = build_features(train, encoders)
    pd.testing.assert_frame_equal(first, second)


def test_fitting_twice_gives_the_same_numbering():
    """Levels are sorted, so numbering is reproducible run to run."""
    _, _, a = build_features(frame())
    _, _, b = build_features(frame())
    assert a.mappings == b.mappings


def test_column_order_is_preserved_on_apply():
    """XGBoost binds to position; reordered columns score against the wrong
    thresholds without any error."""
    train = frame()
    X_train, _, encoders = build_features(train)

    shuffled = frame(extra_browser=True)[list(reversed(frame().columns))]
    X_stream, _, _ = build_features(shuffled, encoders)

    assert list(X_stream.columns) == list(X_train.columns)
    assert list(X_stream.columns) == encoders.feature_names


def test_missing_fitted_column_raises():
    train = frame()
    _, _, encoders = build_features(train)
    with pytest.raises(KeyError, match="absent"):
        build_features(frame().drop(columns=["str_col"]), encoders)


# --------------------------------------------------------------------------
# Persistence and reporting
# --------------------------------------------------------------------------


def test_encoders_round_trip(tmp_path):
    train = frame()
    X, _, encoders = build_features(train)
    path = encoders.save(tmp_path / "encoders.pkl")

    loaded = FeatureEncoders.load(path)
    assert loaded.mappings == encoders.mappings
    assert loaded.feature_names == encoders.feature_names
    assert loaded.unknown_code == encoders.unknown_code

    reapplied, _, _ = build_features(train, loaded)
    pd.testing.assert_frame_equal(reapplied, X)


def test_load_missing_file_points_at_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_features"):
        FeatureEncoders.load(tmp_path / "nope.pkl")


def test_format_version_mismatch_refuses(tmp_path):
    _, _, encoders = build_features(frame())
    encoders.format_version = 999
    path = encoders.save(tmp_path / "stale.pkl")
    with pytest.raises(ValueError, match="format v999"):
        FeatureEncoders.load(path)


def test_unknown_counts_reports_vocabulary_drift():
    train = frame()
    _, _, encoders = build_features(train)
    X, _, _ = build_features(frame(extra_browser=True), encoders)

    report = unknown_counts(X, encoders)
    assert report.loc["str_col", "unknown_rows"] > 0
    assert report.loc["obj_col", "unknown_rows"] == 0
    assert report.index[0] == "str_col"  # sorted by unknown_pct
