"""The prediction path, shared by HTTP serving and stream consumption.

There is exactly one implementation of "score a transaction" in this
project, and it is here. The FastAPI app and the Kafka consumer both call
``predict_one``; neither has a scoring path of its own.

That is structural rather than stylistic. Two code paths that score the same
transaction will eventually disagree -- a threshold read from a different
place, a tree count bounded in one and not the other, an encoder loaded from
a different run -- and the disagreement is invisible because each looks
correct in isolation. Streaming and HTTP scoring the same row differently
would make the Phase 7 shadow comparison meaningless.

Deliberately free of FastAPI: the consumer has no business importing a web
framework to score a row, and keeping this module dependency-light is what
makes sharing it cheap enough that nobody is tempted to write a second one.

Three things this path gets right that a reimplementation would likely miss:

* **iteration_range.** Early stopping left 849 boosted rounds of which 799
  are the model. Unbounded prediction uses 50 unvalidated trees and moves
  some probabilities by 8 percentage points.
* **One booster call.** ``pred_contribs`` returns per-feature SHAP plus a
  bias term whose sum is the margin, so probability and explanation come
  from the same call and agree by construction.
* **The threshold.** Read from the promoted metrics, not hardcoded at 0.5,
  which under ``scale_pos_weight`` is an artifact of the class ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb

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


@dataclass(frozen=True)
class Prediction:
    """One scored transaction, in a form both transports can render."""

    fraud_probability: float
    decision: str
    threshold: float
    model_version: str
    n_trees: int
    top_factors: list[dict] = field(default_factory=list)
    unseen_categories: list[str] = field(default_factory=list)


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

    Falling back to the local file is a convenience for development, and a
    real risk in production: a local encoders.pkl can come from a different
    training run than the registered model, and the mismatch is silent --
    the codes simply mean something else.
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


def predict_one(
    bundle: ModelBundle,
    transaction: dict,
    top_factors: int = DEFAULT_TOP_FACTORS,
) -> Prediction:
    """Score one transaction. The only scoring path in the project.

    The label can never reach the model even if a caller sends it: the
    encoder reads exactly ``bundle.feature_names``, and ``build_features``
    dropped isFraud, TransactionID and TransactionDT before those names were
    recorded. Extra keys in the transaction are not ignored by convention --
    they are unreachable by construction.
    """
    encoded = bundle.encoder.encode(transaction)
    matrix = xgb.DMatrix(
        encoded.values.reshape(1, -1), feature_names=list(bundle.feature_names)
    )
    contributions = bundle.booster.predict(
        matrix, pred_contribs=True, iteration_range=bundle.iteration_range
    )[0]

    probability = float(sigmoid(contributions.sum()))
    per_feature = contributions[:-1]  # the last entry is the bias term

    order = np.argsort(-np.abs(per_feature))[:top_factors]
    factors = [
        {
            "feature": bundle.feature_names[i],
            "value": bundle.encoder.decode(
                bundle.feature_names[i], float(encoded.values[i])
            ),
            "contribution": round(float(per_feature[i]), 6),
            "direction": "toward fraud" if per_feature[i] > 0 else "toward legitimate",
        }
        for i in order
    ]

    return Prediction(
        fraud_probability=probability,
        decision="BLOCK" if probability >= bundle.threshold else "ALLOW",
        threshold=bundle.threshold,
        model_version=bundle.model_version,
        n_trees=bundle.n_trees,
        top_factors=factors,
        unseen_categories=list(encoded.unseen),
    )
