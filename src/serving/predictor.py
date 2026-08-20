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
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.common.config import REPORTS_DIR
from src.features.build_features import ENCODERS_PATH, FeatureEncoders
from src.serving.encoding import RowEncoder
from src.training import tracking
from src.training.train import DEFAULT_THRESHOLD, METRICS_PATH

#: The model that decides. Serving and streaming both resolve this.
CHAMPION_ALIAS: str = "champion"

#: The candidate. Scored in parallel and never allowed to decide anything.
CHALLENGER_ALIAS: str = "challenger"

#: Tried in order when resolving the champion. ``staging`` is here because
#: the registry carried only that alias before champion/challenger existed;
#: a deployment mid-upgrade must not lose its model over a rename.
CHAMPION_FALLBACKS: tuple[str, ...] = (CHAMPION_ALIAS, "staging")

#: Aliases rather than stages because MLflow 3 deprecates stages and will
#: remove them. Stages are still set alongside, for anyone reading the
#: registry UI.
MODEL_ALIAS: str = CHAMPION_ALIAS

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


#: Where each role's swept operating point is recorded. A challenger fitted
#: on a different window has its own best-F1 point, and judging it at the
#: champion's threshold would measure the threshold rather than the model.
THRESHOLD_SOURCES: dict[str, Path] = {
    CHAMPION_ALIAS: METRICS_PATH,
    "staging": METRICS_PATH,
    CHALLENGER_ALIAS: REPORTS_DIR / "retrain_metrics.json",
}


def load_threshold(alias: str = CHAMPION_ALIAS) -> tuple[float, str]:
    """The swept operating point for a role, or 0.5 with an explicit note.

    Not 0.5 by preference: under ``scale_pos_weight`` the probabilities are
    deliberately uncalibrated, so 0.5 is an artifact of the class ratio
    rather than a decision anyone made.
    """
    path = THRESHOLD_SOURCES.get(alias, METRICS_PATH)
    if path.exists():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            block = record["models"]["xgboost"]["val"]["at_best_f1_threshold"]
            return float(block["threshold"]), f"best-F1 from {path.name}"
        except (KeyError, ValueError, TypeError):
            pass
    return DEFAULT_THRESHOLD, f"default 0.5 -- {path.name} missing or unreadable"


class EncoderResolutionError(RuntimeError):
    """A registered model's own encoders could not be resolved."""


#: Escape hatch for development, off by default and deliberately awkward to
#: set. Serving a registered model against another run's encoders is a
#: production incident, not a warning.
ALLOW_FALLBACK_ENV: str = "FRAUDSHIELD_ALLOW_ENCODER_FALLBACK"


def fallback_allowed() -> bool:
    return os.environ.get(ALLOW_FALLBACK_ENV, "").strip().lower() in {"1", "true", "yes"}


def _load_encoders_for_run(
    mlflow,
    run_id: str | None,
    version: str | None = None,
    allow_fallback: bool | None = None,
) -> tuple[FeatureEncoders, str]:
    """The encoders belonging to THIS model version, or an exception.

    Falling back to ``models/encoders.pkl`` is not a degraded mode; it is a
    different model's vocabulary. Measured between two runs of this project,
    five of thirty-one categorical columns assigned different codes to the
    same level -- 970 ``DeviceInfo`` levels, 49 in ``id_31`` -- and 57 of 400
    predictions changed, by up to 0.2502. Every prediction still computes,
    which is exactly why this must raise rather than warn: nothing
    downstream can tell that the codes mean something else.

    The fallback survives only for development, behind an explicit
    environment variable, and says so in the source it reports.
    """
    reason = "the version carries no run id"
    if run_id is not None:
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=ENCODERS_PATH.name
            )
            return FeatureEncoders.load(Path(path)), f"mlflow run {run_id[:8]}"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"

    permitted = fallback_allowed() if allow_fallback is None else allow_fallback
    if not permitted:
        raise EncoderResolutionError(
            f"Model version {version or '?'} (run {run_id or '?'}) has no "
            f"{ENCODERS_PATH.name} logged, so its own encoders cannot be "
            f"resolved: {reason}.\n"
            f"Refusing to serve it against {ENCODERS_PATH}, which belongs to "
            "whichever run wrote it last. A model scored with another run's "
            "encoders produces predictions from a vocabulary it never saw, "
            "and nothing downstream can detect it.\n"
            "Fix by logging the encoders onto that run, or set "
            f"{ALLOW_FALLBACK_ENV}=1 to accept the risk in development."
        )
    return (
        FeatureEncoders.load(ENCODERS_PATH),
        f"local {ENCODERS_PATH.name} (FALLBACK -- {ALLOW_FALLBACK_ENV} is set)",
    )


def resolve_alias(client, name: str, aliases: tuple[str, ...]):
    """First alias in ``aliases`` that resolves, or None."""
    for alias in aliases:
        try:
            return client.get_model_version_by_alias(name, alias), alias
        except Exception:
            continue
    return None, None


def resolve_threshold(client, version, alias: str) -> tuple[float, str]:
    """The operating point belonging to THIS model version.

    Read from the version's own MLflow run, not from a file chosen by role.
    Resolving by role means the threshold follows the alias rather than the
    model: promoting a challenger silently swapped its 0.8032 for the
    baseline's 0.8018, so a model that had been measured at one operating
    point began deciding at another the moment it was promoted.

    Falls back to the role-keyed file only when the run carries no threshold
    metric, and says which happened.
    """
    try:
        metrics = client.get_run(version.run_id).data.metrics
        value = metrics.get("val.at_best_f1_threshold.threshold")
        if value is not None:
            return float(value), f"run {version.run_id[:8]} (v{version.version})"
    except Exception:
        pass
    threshold, source = load_threshold(alias)
    return threshold, f"{source} (FALLBACK -- run carried no threshold metric)"


def load_bundle(model_name: str | None = None, alias: str = MODEL_ALIAS) -> ModelBundle:
    """Resolve a registered model and its encoders.

    ``alias`` may name a single alias or fall through ``CHAMPION_FALLBACKS``
    when it is the champion, so a registry that predates the champion alias
    still serves.
    """
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
    candidates = CHAMPION_FALLBACKS if alias == CHAMPION_ALIAS else (alias,)
    version, alias = resolve_alias(client, name, candidates)
    if version is None:
        raise RuntimeError(
            f"No version of {name!r} carries any of the aliases {candidates}."
        )

    model = mlflow.xgboost.load_model(f"models:/{name}/{version.version}")
    booster = model.get_booster() if hasattr(model, "get_booster") else model

    best = getattr(model, "best_iteration", None)
    n_trees = int(best) + 1 if best is not None else booster.num_boosted_rounds()

    encoders, encoder_source = _load_encoders_for_run(
        mlflow, version.run_id, version=version.version
    )
    threshold, threshold_source = resolve_threshold(client, version, alias)

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


def try_load_bundle(alias: str, model_name: str | None = None) -> ModelBundle | None:
    """Load a bundle if the alias exists, else None.

    Used for the challenger: a registry with no candidate in it is the
    normal state, not an error, and shadow scoring must simply not happen
    rather than fail the consumer.
    """
    try:
        return load_bundle(model_name, alias)
    except Exception as exc:
        print(f"  no {alias} model ({type(exc).__name__}); scoring champion only")
        return None
