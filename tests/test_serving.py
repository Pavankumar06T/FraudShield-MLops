"""Invariants for the serving path.

Two carry the most weight.

The fast encoder must be *bit-identical* to ``build_features``. It exists
only because it is 429x faster; the moment it stops agreeing it is a
training-serving skew generator and must be deleted rather than patched.

Predictions must be bounded to the validated tree count. Early stopping left
849 rounds in a booster of which 799 are the model, and predicting
unbounded moves some probabilities by 8 percentage points -- silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from src.common.config import VAL_PARQUET
from src.features.build_features import (
    UNKNOWN_CODE,
    FeatureEncoders,
    build_features,
    read_split,
)
from src.serving.encoding import RowEncoder, format_number
from src.training.train import XGB_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not VAL_PARQUET.exists() or not XGB_MODEL_PATH.exists(),
    reason="needs the real val split and a trained model",
)


@pytest.fixture(scope="module")
def encoders() -> FeatureEncoders:
    return FeatureEncoders.load()


@pytest.fixture(scope="module")
def row_encoder(encoders) -> RowEncoder:
    return RowEncoder(encoders)


@pytest.fixture(scope="module")
def sample():
    return read_split(VAL_PARQUET, 300)


@pytest.fixture(scope="module")
def booster():
    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)
    return model


# --------------------------------------------------------------------------
# The fast path must equal the reference path
# --------------------------------------------------------------------------


def test_fast_encoder_matches_build_features_exactly(sample, encoders, row_encoder):
    """Every column of every row. Anything less is not equivalence."""
    reference, _, _ = build_features(sample, encoders)
    assert list(reference.columns) == list(row_encoder.feature_names)

    for position in range(len(sample)):
        fast = row_encoder.encode(sample.iloc[position].to_dict()).values
        want = reference.iloc[position].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(
            np.nan_to_num(fast, nan=-9.99e9),
            np.nan_to_num(want, nan=-9.99e9),
            err_msg=f"row {position} diverged",
        )


def test_fast_encoder_preserves_missingness(sample, encoders, row_encoder):
    """NaN must stay NaN -- it is signal, and XGBoost routes it by a
    learned default rather than by an imputed value."""
    reference, _, _ = build_features(sample, encoders)
    for position in range(0, len(sample), 25):
        fast = row_encoder.encode(sample.iloc[position].to_dict()).values
        want = reference.iloc[position].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(want))


def test_absent_key_and_explicit_null_are_the_same(row_encoder):
    """A client omitting a field and sending it as null both mean absent."""
    omitted = row_encoder.encode({"TransactionAmt": 100.0})
    explicit = row_encoder.encode({"TransactionAmt": 100.0, "id_31": None})
    np.testing.assert_array_equal(
        np.nan_to_num(omitted.values, nan=-1.0), np.nan_to_num(explicit.values, nan=-1.0)
    )


# --------------------------------------------------------------------------
# Unseen categories -- the measured drift, not a hypothetical
# --------------------------------------------------------------------------


def test_unseen_browser_scores_rather_than_raising(row_encoder, encoders):
    """id_31 carries browser strings in the stream window that train never
    saw. A transaction naming one must score."""
    position = row_encoder.feature_names.index("id_31")
    encoded = row_encoder.encode({"id_31": "chrome 65.0"})

    assert encoded.values[position] == pytest.approx(UNKNOWN_CODE)
    assert "id_31" in encoded.unseen
    assert "chrome 65.0" not in encoders.mappings["id_31"]


def test_unseen_is_reported_so_drift_is_visible_live(row_encoder):
    encoded = row_encoder.encode(
        {"id_31": "netscape 4.0", "ProductCD": "W", "TransactionAmt": 50.0}
    )
    assert "id_31" in encoded.unseen
    assert "ProductCD" not in encoded.unseen


def test_unseen_and_missing_stay_distinct(row_encoder):
    position = row_encoder.feature_names.index("id_31")
    unseen = row_encoder.encode({"id_31": "never-seen"}).values[position]
    missing = row_encoder.encode({"TransactionAmt": 1.0}).values[position]
    assert unseen == pytest.approx(UNKNOWN_CODE)
    assert np.isnan(missing)


def test_garbage_in_a_numeric_field_does_not_reject_the_transaction(row_encoder):
    """One malformed field should not cost the whole decision."""
    encoded = row_encoder.encode({"TransactionAmt": "not-a-number", "ProductCD": "W"})
    position = row_encoder.feature_names.index("TransactionAmt")
    assert np.isnan(encoded.values[position])
    assert "TransactionAmt" in encoded.missing


def test_empty_transaction_still_scores(row_encoder):
    encoded = row_encoder.encode({})
    assert len(encoded.values) == len(row_encoder.feature_names)
    assert np.isnan(encoded.values).all()


# --------------------------------------------------------------------------
# Prediction must be bounded to the validated trees
# --------------------------------------------------------------------------


def test_unbounded_prediction_would_differ_materially(sample, encoders, booster):
    """Documents why iteration_range is not optional."""
    X, _, _ = build_features(sample, encoders)
    raw = booster.get_booster()
    names = list(X.columns)
    matrix = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=names)

    validated = raw.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
    unbounded = raw.predict(matrix)

    assert raw.num_boosted_rounds() > booster.best_iteration + 1
    np.testing.assert_allclose(validated, booster.predict_proba(X)[:, 1], atol=1e-6)
    assert np.abs(unbounded - validated).max() > 0.01, (
        "if these now agree, early stopping left no surplus rounds and the "
        "iteration_range guard can be revisited"
    )


def test_contributions_sum_to_the_probability(sample, encoders, booster, row_encoder):
    """One booster call yields probability and explanation; they must be
    consistent by construction, not by coincidence."""
    from src.serving.predictor import sigmoid

    raw = booster.get_booster()
    names = list(row_encoder.feature_names)
    for position in range(0, 40, 7):
        encoded = row_encoder.encode(sample.iloc[position].to_dict())
        matrix = xgb.DMatrix(encoded.values.reshape(1, -1), feature_names=names)
        contributions = raw.predict(
            matrix, pred_contribs=True, iteration_range=(0, booster.best_iteration + 1)
        )[0]

        X, _, _ = build_features(sample.iloc[[position]], encoders)
        expected = booster.predict_proba(X)[0, 1]
        assert sigmoid(contributions.sum()) == pytest.approx(expected, abs=1e-5)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def test_codes_decode_to_browser_strings(row_encoder, encoders):
    for level, code in list(encoders.mappings["id_31"].items())[:5]:
        assert row_encoder.decode("id_31", float(code)) == str(level)


def test_decode_names_the_three_states(row_encoder):
    assert row_encoder.decode("id_31", UNKNOWN_CODE) == "<unseen>"
    assert row_encoder.decode("id_31", float("nan")) == "<missing>"
    assert row_encoder.decode("id_31", 999999) == "<code 999999>"


@pytest.mark.parametrize(
    "value, expected",
    [(31937.39, "31,937.39"), (5236.0, "5,236"), (0.0, "0"), (0.000123, "1.230e-04")],
)
def test_amounts_never_render_in_scientific_notation(value, expected):
    assert format_number(value) == expected


# --------------------------------------------------------------------------
# Sample payloads shipped with the repo
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["transaction_fraud", "transaction_legitimate", "transaction_unseen_browser"]
)
def test_shipped_examples_are_scoreable(name, row_encoder):
    path = Path("examples") / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not generated")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "isFraud" not in payload, "the label must never appear in a request"
    encoded = row_encoder.encode(payload)
    assert len(encoded.values) == len(row_encoder.feature_names)
    assert not np.isnan(encoded.values).all()


def test_unseen_example_really_carries_an_unseen_level(row_encoder):
    path = Path("examples/transaction_unseen_browser.json")
    if not path.exists():
        pytest.skip("example not generated")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "id_31" in row_encoder.encode(payload).unseen
