"""FastAPI fraud scoring service.

Loads the promoted model from the MLflow Model Registry once at startup and
serves scored, explained decisions.

**The model comes from the registry, not from disk.** ``models:/<name>@staging``
is the promoted artifact by definition; ``models/baseline_xgb.json`` is
whatever the last training run happened to write, which is not the same
thing and diverges the moment someone experiments. The encoders are pulled
from the *same run* that produced the registered model, so a model can never
be served against a mapping it was not fitted with.

**The threshold comes from baseline_metrics.json, not from 0.5.** With
``scale_pos_weight`` at 29 the probabilities are deliberately uncalibrated,
so 0.5 is an artifact of the class ratio rather than a decision anyone made.
The swept best-F1 point is where the model actually operates -- roughly 0.80
here, flagging 2.7% of traffic.

**Latency.** Measured on the promoted 799-tree model rather than assumed:

    encode one transaction (dict -> numpy)     0.05 ms
    DMatrix + pred_contribs (prob AND SHAP)    3.99 ms p50 / 4.72 ms p95

so per-request SHAP is comfortably affordable -- but only via the booster's
native TreeSHAP. The obvious implementations are not: ``shap.TreeExplainer``
called per request costs 24 ms p50 and 117 ms p95, and running the training
``build_features`` on a single row costs another 22 ms. Naively assembled,
this endpoint would be ~58 ms p50 and over 150 ms at p95. See
docs/serving.md.

**iteration_range is load-bearing.** Early stopping left 849 boosted rounds
in the booster of which only 799 are the model; predicting without bounding
the range silently uses 50 trees that were never validated, and moves some
probabilities by 8 percentage points.

    uvicorn src.serving.app:app --host 0.0.0.0 --port 8080
    curl -s localhost:8080/health | python -m json.tool
"""

from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.drift import store
from src.serving.predictor import (
    DEFAULT_TOP_FACTORS,
    MODEL_ALIAS,
    ModelBundle,
    load_bundle,
    predict_one,
)
from src.training import tracking

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class Factor(BaseModel):
    feature: str
    value: str = Field(description="decoded: browser strings, not ordinal codes")
    contribution: float = Field(description="SHAP value in log-odds space")
    direction: str


class PredictResponse(BaseModel):
    transaction_id: str | None = None
    fraud_probability: float
    decision: str
    threshold: float
    top_factors: list[Factor]
    unseen_categories: list[str] = Field(
        default_factory=list,
        description="fields whose value was absent from training -- vocabulary drift",
    )
    model_version: str
    n_trees: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    model_stage: str
    n_trees: int
    n_features: int
    threshold: float
    threshold_source: str
    run_id: str | None


STATE: dict[str, Any] = {}

#: Unseen (feature, value) pairs buffered in memory, flushed in batches.
#:
#: A category the encoder has never seen is the fastest drift signal there
#: is -- visible on the first request carrying it, where windowed PSI needs
#: a window of traffic to move. But a SQLite write inside the request path
#: would add milliseconds to a 4 ms handler for a signal that does not need
#: per-request durability, so it is counted here and flushed on a size
#: trigger and at shutdown.
UNSEEN_BUFFER: Counter = Counter()
UNSEEN_FLUSH_EVERY: int = 200


def record_unseen(features: list[str], transaction: dict, version: str) -> None:
    """Buffer unseen levels, flushing when the batch is worth a write."""
    if not features:
        return
    for feature in features:
        UNSEEN_BUFFER[(feature, str(transaction.get(feature)))] += 1
    if sum(UNSEEN_BUFFER.values()) >= UNSEEN_FLUSH_EVERY:
        flush_unseen(version)


def flush_unseen(version: str | None = None) -> int:
    """Persist and clear the buffer. Never fails a request."""
    if not UNSEEN_BUFFER:
        return 0
    try:
        written = store.record_unseen(dict(UNSEEN_BUFFER), model_version=version)
    except Exception:
        # Drift telemetry is not worth a 500. The observations are dropped
        # rather than retried: the next unseen category will be recorded,
        # and PSI remains the durable signal.
        UNSEEN_BUFFER.clear()
        return 0
    UNSEEN_BUFFER.clear()
    return written


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once, at startup. Per-request loading would dominate latency."""
    STATE["bundle"] = load_bundle()
    yield
    flush_unseen(STATE.get("bundle").model_version if STATE.get("bundle") else None)
    STATE.clear()


app = FastAPI(
    title="FraudShield",
    description="Fraud scoring with per-decision SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


def bundle() -> ModelBundle:
    loaded = STATE.get("bundle")
    if loaded is None:
        raise HTTPException(503, "model not loaded")
    return loaded


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    active = bundle()
    return HealthResponse(
        status="ok",
        model_name=tracking.REGISTERED_MODEL_NAME,
        model_version=active.model_version,
        model_stage=active.model_stage,
        n_trees=active.n_trees,
        n_features=len(active.feature_names),
        threshold=active.threshold,
        threshold_source=active.threshold_source,
        run_id=active.run_id,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(transaction: dict[str, Any], top_factors: int = DEFAULT_TOP_FACTORS) -> PredictResponse:
    """Score one transaction.

    The body is an open object rather than a strict schema on purpose:
    absent fields are meaningful here. 76% of IEEE-CIS identity columns are
    empty on any given row, and XGBoost routes missing values by a learned
    default -- so requiring all 431 features would reject most real traffic.
    """
    started = time.perf_counter()
    active = bundle()

    if not isinstance(transaction, dict) or not transaction:
        raise HTTPException(422, "body must be a non-empty JSON object")

    prediction = predict_one(active, transaction, top_factors)
    record_unseen(prediction.unseen_categories, transaction, active.model_version)

    identifier = transaction.get("TransactionID")
    return PredictResponse(
        transaction_id=None if identifier is None else str(identifier),
        fraud_probability=round(prediction.fraud_probability, 6),
        decision=prediction.decision,
        threshold=round(prediction.threshold, 6),
        top_factors=[Factor(**f) for f in prediction.top_factors],
        unseen_categories=prediction.unseen_categories,
        model_version=prediction.model_version,
        n_trees=prediction.n_trees,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.get("/drift/unseen")
def unseen_summary() -> dict[str, Any]:
    """Unseen categories seen since startup, buffered plus persisted.

    Exposed so the drift monitor can read the live signal without waiting
    for a flush, and so an operator can see vocabulary drift arriving
    without opening the alert store.
    """
    active = bundle()
    buffered = [
        {"feature": feature, "value": value, "occurrences": count}
        for (feature, value), count in UNSEEN_BUFFER.most_common(20)
    ]
    try:
        persisted = store.aggregate_unseen()
    except Exception:
        persisted = []
    return {
        "model_version": active.model_version,
        "buffered_pending_flush": buffered,
        "buffered_total": sum(UNSEEN_BUFFER.values()),
        "persisted": persisted,
    }


@app.post("/drift/flush")
def flush() -> dict[str, int]:
    """Force the buffer to the alert store. Called before a monitor run."""
    active = bundle()
    return {"written": flush_unseen(active.model_version)}
