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
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common.config import MODELS_DIR
from src.features.build_features import ENCODERS_PATH, FeatureEncoders
from src.serving.encoding import RowEncoder
from src.training import tracking
from src.training.train import DEFAULT_THRESHOLD, METRICS_PATH

#: Registry alias to serve. An alias rather than a stage because MLflow 3
#: deprecates stages; @staging is what survives their removal.
MODEL_ALIAS: str = "staging"

#: How many contributing factors to return per decision.
DEFAULT_TOP_FACTORS: int = 6


def sigmoid(z: float | np.ndarray):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class ModelBundle:
    """Everything needed to score, loaded once."""

    booster: xgb.Booster
    encoder: RowEncoder
    threshold: float
    threshold_source: str
    model_version: str
    model_stage: str
    n_trees: int
    run_id: str | None
    feature_names: tuple[str, ...]

    @property
    def iteration_range(self) -> tuple[int, int]:
        """Bound predictions to the trees the model was validated with."""
        return (0, self.n_trees)


def load_threshold() -> tuple[float, str]:
    """Promoted operating point, or 0.5 with an explicit note if absent."""
    if METRICS_PATH.exists():
        try:
            record = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            block = record["models"]["xgboost"]["val"]["at_best_f1_threshold"]
            return float(block["threshold"]), f"best-F1 from {METRICS_PATH.name}"
        except (KeyError, ValueError, TypeError):
            pass
    return DEFAULT_THRESHOLD, "default 0.5 -- metrics file missing or unreadable"


def _load_encoders_for_run(mlflow, run_id: str | None) -> tuple[FeatureEncoders, str]:
    """Prefer the encoders logged beside the registered model.

    Falling back to the local file is a convenience for development, and it
    is a real risk in production: a local encoders.pkl can be from a
    different training run than the registered model, and the mismatch is
    silent -- the codes simply mean something else.
    """
    if run_id is not None:
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=ENCODERS_PATH.name
            )
            return FeatureEncoders.load(Path(path)), f"mlflow run {run_id[:8]}"
        except Exception:
            pass
    return FeatureEncoders.load(ENCODERS_PATH), f"local {ENCODERS_PATH.name} (FALLBACK)"


def load_bundle(model_name: str | None = None, alias: str = MODEL_ALIAS) -> ModelBundle:
    """Resolve the promoted model and its encoders from the registry."""
    name = model_name or tracking.REGISTERED_MODEL_NAME
    mlflow = tracking.mlflow_module()
    if mlflow is None:
        raise RuntimeError(
            "MLflow is unavailable, so the promoted model cannot be resolved. "
            "Serving from an unversioned local file is not a fallback this "
            "service will make silently."
        )

    mlflow.set_tracking_uri(tracking.tracking_uri())
    from mlflow import MlflowClient

    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)

    model = mlflow.xgboost.load_model(f"models:/{name}@{alias}")
    booster = model.get_booster() if hasattr(model, "get_booster") else model

    # best_iteration survives the round-trip in the saved JSON; without it
    # the booster's full 849 rounds would be used instead of the 799 the
    # model was validated at.
    best = getattr(model, "best_iteration", None)
    n_trees = int(best) + 1 if best is not None else booster.num_boosted_rounds()

    encoders, encoder_source = _load_encoders_for_run(mlflow, version.run_id)
    threshold, threshold_source = load_threshold()

    bundle = ModelBundle(
        booster=booster,
        encoder=RowEncoder(encoders),
        threshold=threshold,
        threshold_source=threshold_source,
        model_version=str(version.version),
        model_stage=version.current_stage or f"@{alias}",
        n_trees=n_trees,
        run_id=version.run_id,
        feature_names=tuple(encoders.feature_names),
    )
    print(
        f"loaded {name} v{version.version} ({bundle.model_stage})\n"
        f"  trees      {n_trees} (booster holds {booster.num_boosted_rounds()})\n"
        f"  features   {len(bundle.feature_names)}\n"
        f"  encoders   {encoder_source}\n"
        f"  threshold  {threshold:.4f} ({threshold_source})"
    )
    return bundle


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once, at startup. Per-request loading would dominate latency."""
    STATE["bundle"] = load_bundle()
    yield
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


def score(active: ModelBundle, transaction: dict, top: int) -> tuple[float, list[Factor], list[str]]:
    """Encode, score and explain in one booster call.

    ``pred_contribs`` returns per-feature SHAP values plus a trailing bias
    term, and their sum is the margin -- so probability and explanation come
    from the same call rather than two, and are guaranteed consistent with
    each other by construction.
    """
    encoded = active.encoder.encode(transaction)
    matrix = xgb.DMatrix(
        encoded.values.reshape(1, -1), feature_names=list(active.feature_names)
    )
    contributions = active.booster.predict(
        matrix, pred_contribs=True, iteration_range=active.iteration_range
    )[0]

    probability = float(sigmoid(contributions.sum()))
    per_feature = contributions[:-1]  # last entry is the bias

    order = np.argsort(-np.abs(per_feature))[:top]
    factors = [
        Factor(
            feature=active.feature_names[i],
            value=active.encoder.decode(
                active.feature_names[i], float(encoded.values[i])
            ),
            contribution=round(float(per_feature[i]), 6),
            direction="toward fraud" if per_feature[i] > 0 else "toward legitimate",
        )
        for i in order
    ]
    return probability, factors, list(encoded.unseen)


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

    probability, factors, unseen = score(active, transaction, top_factors)
    decision = "BLOCK" if probability >= active.threshold else "ALLOW"

    identifier = transaction.get("TransactionID")
    return PredictResponse(
        transaction_id=None if identifier is None else str(identifier),
        fraud_probability=round(probability, 6),
        decision=decision,
        threshold=round(active.threshold, 6),
        top_factors=factors,
        unseen_categories=unseen,
        model_version=active.model_version,
        n_trees=active.n_trees,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )
