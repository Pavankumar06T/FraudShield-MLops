"""Invariants for the streaming path.

Two matter above the rest.

**The label cannot reach the model.** ``isFraud`` travels in every message
so the consumer can record whether a decision was right. If it ever reached
the feature vector the model would score near-perfectly and the whole
pipeline would look excellent while being worthless. This is asserted
behaviourally -- flip the label, the prediction must not move by a single
bit -- rather than by checking that some code remembered to pop a key.

**Streaming and HTTP must not diverge.** Both call the same
``predict_one``; a test scores the same transaction through each transport
and requires identical output. Two scoring paths would make the Phase 7
shadow comparison measure the difference between code paths rather than
between models.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.common.config import STREAM_PARQUET, VAL_PARQUET
from src.drift import store
from src.features.build_features import FeatureEncoders, read_split
from src.serving.encoding import RowEncoder
from src.serving.predictor import predict_one
from src.streaming.config import consumer_config, producer_config
from src.streaming.consumer import (
    LABEL_FIELD,
    TRUE_LABEL_IX,
    ensure_schema,
    score_message,
    write_batch,
)
from src.streaming.producer import load_ordered, to_message
from src.training.train import XGB_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not STREAM_PARQUET.exists() or not XGB_MODEL_PATH.exists(),
    reason="needs the real stream split and a trained model",
)


@pytest.fixture(scope="module")
def bundle():
    from src.serving.predictor import load_bundle

    return load_bundle()


@pytest.fixture(scope="module")
def rows():
    return read_split(STREAM_PARQUET, 200)


@pytest.fixture(scope="module")
def encoder():
    return RowEncoder(FeatureEncoders.load())


# --------------------------------------------------------------------------
# The label must never reach the model
# --------------------------------------------------------------------------


def test_label_is_not_a_feature(encoder):
    """build_features dropped it before the feature names were recorded, so
    the encoder has no slot to put it in."""
    assert LABEL_FIELD == "isFraud"
    assert LABEL_FIELD not in encoder.feature_names
    assert "TransactionID" not in encoder.feature_names
    assert "TransactionDT" not in encoder.feature_names


def test_flipping_the_label_changes_nothing_in_the_feature_vector(rows, encoder):
    """The decisive check: not that some code pops the key, but that the
    key is unreachable."""
    for position in range(0, 50, 7):
        payload = to_message(rows.iloc[position])
        payload.pop(LABEL_FIELD, None)

        without = encoder.encode(payload).values
        with_zero = encoder.encode({**payload, LABEL_FIELD: 0}).values
        with_one = encoder.encode({**payload, LABEL_FIELD: 1}).values

        np.testing.assert_array_equal(
            np.nan_to_num(without, nan=-9.99e9), np.nan_to_num(with_zero, nan=-9.99e9)
        )
        np.testing.assert_array_equal(
            np.nan_to_num(with_zero, nan=-9.99e9), np.nan_to_num(with_one, nan=-9.99e9)
        )


def test_flipping_the_label_changes_nothing_in_the_prediction(rows, bundle):
    """End to end through the real scoring path, not just the encoder."""
    for position in range(0, 40, 9):
        payload = to_message(rows.iloc[position])
        payload.pop(LABEL_FIELD, None)

        as_fraud = predict_one(bundle, {**payload, LABEL_FIELD: 1})
        as_clean = predict_one(bundle, {**payload, LABEL_FIELD: 0})
        absent = predict_one(bundle, payload)

        assert as_fraud.fraud_probability == absent.fraud_probability
        assert as_clean.fraud_probability == absent.fraud_probability
        assert as_fraud.decision == absent.decision


def test_a_label_carrying_message_does_not_score_suspiciously_well(rows, bundle):
    """If the label leaked, predictions would track it almost perfectly.

    Correlating predictions against the labels they were told is a direct
    test of leakage that does not depend on knowing how leakage would occur.
    """
    frame = rows[rows[LABEL_FIELD].notna()].head(120)
    if frame[LABEL_FIELD].nunique() < 2:
        pytest.skip("sample has only one class")

    probabilities, labels = [], []
    for _, row in frame.iterrows():
        payload = to_message(row)
        probabilities.append(predict_one(bundle, payload).fraud_probability)
        labels.append(int(row[LABEL_FIELD]))

    correlation = abs(np.corrcoef(probabilities, labels)[0, 1])
    assert correlation < 0.95, (
        f"predictions correlate {correlation:.3f} with the labels handed to the "
        "model -- that is leakage, not skill"
    )


def test_label_is_still_recorded_for_monitoring(rows, bundle):
    """Excluded from the model, kept in the audit trail -- decisions cannot
    be judged without it."""
    row_with_label = rows[rows[LABEL_FIELD].notna()].iloc[0]
    payload = to_message(row_with_label)
    record, _ = score_message(bundle, payload)
    # By position, via the named constant: adding the four shadow columns
    # moved the label from index 10 to 14, and a literal here broke silently.
    assert record[TRUE_LABEL_IX] == int(row_with_label[LABEL_FIELD])


# --------------------------------------------------------------------------
# One scoring path
# --------------------------------------------------------------------------


def test_streaming_and_http_score_identically(rows, bundle):
    """Both transports call predict_one on the same bundle; if they ever
    disagree, the Phase 7 shadow comparison measures code paths."""
    from fastapi.testclient import TestClient

    from src.serving.app import app

    with TestClient(app) as client:
        for position in range(0, 30, 11):
            payload = to_message(rows.iloc[position])
            payload.pop(LABEL_FIELD, None)

            http = client.post("/predict", json=payload).json()
            record, _ = score_message(bundle, payload)

            assert record[3] == pytest.approx(http["fraud_probability"], abs=1e-9)
            assert record[4] == http["decision"]
            assert record[5] == pytest.approx(http["threshold"], abs=1e-9)
            assert record[6] == http["model_version"]
            assert record[7] == http["n_trees"]


def test_consumer_has_no_scoring_code_of_its_own():
    """A structural check: the consumer must delegate, not reimplement."""
    from pathlib import Path

    source = Path("src/streaming/consumer.py").read_text(encoding="utf-8")
    assert "predict_one" in source
    for forbidden in ("pred_contribs", "DMatrix", "iteration_range", "sigmoid("):
        assert forbidden not in source, (
            f"{forbidden!r} in the consumer means a second scoring path exists"
        )


# --------------------------------------------------------------------------
# Replay ordering
# --------------------------------------------------------------------------


def test_replay_is_ordered_by_transaction_time(tmp_path):
    """Shuffled replay would smear gradual drift into a uniform background
    and make every monitor window look the same."""
    n = 500
    frame = pd.DataFrame(
        {"TransactionDT": np.arange(n) * 10.0, "isFraud": 0, "TransactionAmt": 1.0}
    ).sample(frac=1.0, random_state=7)
    path = tmp_path / "stream.parquet"
    frame.to_parquet(path, index=False)

    ordered = load_ordered(path=path)
    times = ordered["TransactionDT"].to_numpy()
    assert np.all(np.diff(times) >= 0), "replay must be non-decreasing in time"
    assert len(ordered) == n


def test_limit_takes_the_earliest_rows_not_a_random_slice(tmp_path):
    n = 300
    frame = pd.DataFrame(
        {"TransactionDT": np.arange(n) * 10.0, "isFraud": 0}
    ).sample(frac=1.0, random_state=2)
    path = tmp_path / "s.parquet"
    frame.to_parquet(path, index=False)

    head = load_ordered(limit=50, path=path)
    assert len(head) == 50
    assert head["TransactionDT"].max() < frame["TransactionDT"].max()
    assert np.all(np.diff(head["TransactionDT"].to_numpy()) >= 0)


# --------------------------------------------------------------------------
# Message shaping
# --------------------------------------------------------------------------


def test_nan_fields_are_omitted_not_serialised(rows):
    """JSON has no NaN literal, and an absent key is already how the encoder
    represents a missing feature.

    Guards a real bug: 399 of these columns are float32, and np.float32 is
    not a subclass of Python float, so an isinstance-based missingness check
    passes them through and json.dumps emits a bare NaN token that strict
    parsers reject.
    """
    for position in range(0, 20, 3):
        row = rows.iloc[position]
        message = to_message(row)

        encoded = json.dumps(message, allow_nan=False)  # raises on NaN/Inf
        assert "NaN" not in encoded
        assert json.loads(encoded) == message

        dropped = [k for k in row.index if k not in message]
        assert dropped, "this dataset is majority-empty; some fields must drop"


def test_float32_nan_is_recognised_as_missing():
    """The exact type that slipped through: float32, not float64."""
    row = pd.Series(
        {"a": np.float32("nan"), "b": np.float64("nan"), "c": np.float32(1.5)}
    )
    message = to_message(row)
    assert set(message) == {"c"}
    json.dumps(message, allow_nan=False)


def test_omitting_nan_scores_the_same_as_sending_null(rows, bundle):
    row = rows.iloc[0]
    message = to_message(row)
    with_nulls = {k: (None if k not in message else message[k]) for k in row.index}
    assert predict_one(bundle, message).fraud_probability == pytest.approx(
        predict_one(bundle, with_nulls).fraud_probability, abs=1e-12
    )


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------


def test_predictions_round_trip_to_sqlite(tmp_path, rows, bundle):
    db = tmp_path / "drift.db"
    ensure_schema(db)

    batch = [score_message(bundle, to_message(rows.iloc[i]))[0] for i in range(5)]
    assert write_batch(batch, db) == 5

    with store.connect(db) as connection:
        stored = connection.execute(
            "SELECT * FROM stream_predictions ORDER BY id"
        ).fetchall()

    assert len(stored) == 5
    first = dict(stored[0])
    assert first["decision"] in {"BLOCK", "ALLOW"}
    assert 0.0 <= first["fraud_probability"] <= 1.0
    assert first["n_trees"] == bundle.n_trees
    assert first["threshold"] == pytest.approx(bundle.threshold)
    assert json.loads(first["top_factors"])
    assert "true_label" in first


def test_audit_table_coexists_with_the_drift_tables(tmp_path, rows, bundle):
    """One store, several writers -- the consumer must not clobber the
    monitor's schema."""
    db = tmp_path / "drift.db"
    ensure_schema(db)
    store.record_unseen({("id_31", "chrome 66.0"): 3}, path=db)
    write_batch([score_message(bundle, to_message(rows.iloc[0]))[0]], db)

    assert store.aggregate_unseen(path=db)[0]["feature"] == "id_31"
    with store.connect(db) as connection:
        tables = {
            r["name"]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"stream_predictions", "unseen_observations", "drift_alerts"} <= tables


# --------------------------------------------------------------------------
# Client configuration
# --------------------------------------------------------------------------


def test_consumer_does_not_autocommit():
    """Offsets are committed after a row is scored and written, so a crash
    replays rather than drops."""
    assert consumer_config()["enable.auto.commit"] is False
    assert consumer_config()["auto.offset.reset"] == "earliest"


def test_producer_does_not_batch_away_the_pacing():
    """A large linger would distort the replay timing the ordering exists
    to preserve."""
    assert producer_config()["linger.ms"] <= 10
