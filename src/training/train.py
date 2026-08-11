"""Baseline models: XGBoost, LightGBM, and their probability-averaged ensemble.

This establishes the numbers every later phase argues against. The retrained
model that drift triggers, and the shadow model in the A/B test, are both
judged relative to what is written here -- so the metrics file records not
just the scores but the shape of the data they came from.

All three get the identical metric block so they are directly comparable.
Set expectations for the ensemble: two gradient-boosted tree ensembles on
the same features make highly correlated errors, so averaging them usually
buys very little. The run reports the correlation between the two models'
probabilities alongside the ensemble's score, because a gain of +0.002 at a
0.99 correlation is a different fact from the same gain at 0.85.

**PR-AUC is the headline.** At a 3.4% positive rate the alternatives mislead:

* Accuracy is worthless. Predicting "never fraud" scores 96.6%.
* ROC-AUC is optimistic. Its false-positive-rate axis is normalised by the
  569k negatives, so thousands of false alarms barely move it. A model can
  post 0.90 ROC-AUC while a fraud analyst drowns in review queue.
* PR-AUC lives on the positives. Its no-skill floor is the base rate itself
  -- 0.034, not 0.5 -- so it is reported here alongside that floor and the
  lift over it. The reference XGBoost run's 0.5477 is roughly 16x the floor;
  the same number stated alone would mean nothing.

**Thresholds.** 0.5 is arbitrary here and doubly so with ``scale_pos_weight``
in play: reweighting the positive class inflates predicted probabilities, so
the operating point 0.5 happens to select is an artifact of the class ratio
rather than a decision anyone made. Metrics are reported at 0.5 for
continuity, and at the sweep's best-F1 point for what the model can actually
do.

**No early stopping.** val is the evaluation set, and stopping on it would
tune the models against the same rows the reported metrics come from,
flattering every number in the file. Tree counts are fixed instead. Since
that leaves nothing to reveal an overfit, each model is also scored on the
training rows and the gap is reported.

    python -m src.training.train
    python -m src.training.train --sample 50000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.config import (
    MODELS_DIR,
    REPORTS_DIR,
    TRAIN_PARQUET,
    VAL_PARQUET,
    ensure_dirs,
    split_stats,
)
from src.features.build_features import (
    ENCODERS_PATH,
    FeatureEncoders,
    build_features,
    read_split,
)

XGB_MODEL_PATH: Path = MODELS_DIR / "baseline_xgb.json"
LGB_MODEL_PATH: Path = MODELS_DIR / "baseline_lgb.txt"
METRICS_PATH: Path = REPORTS_DIR / "baseline_metrics.json"

#: The first full XGBoost run on the real splits. Kept as a fixed point so a
#: later run that diverges is visible immediately -- a changed feature
#: pipeline or a re-split would move these, and silently redefining the
#: baseline would make every downstream comparison meaningless.
REFERENCE_BASELINE: dict[str, float] = {"pr_auc": 0.5477, "roc_auc": 0.9031}

#: How far a rerun may drift from the reference before it is called out.
REFERENCE_TOLERANCE: float = 0.01

#: Conventional starting point, not a tuned configuration. Tuning belongs
#: after the drift loop works end to end. `aucpr` matches the headline metric
#: so the training log tracks what is actually being optimised.
XGB_HYPERPARAMETERS: dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": 42,
}

#: Matched to the XGBoost configuration as closely as the two libraries
#: allow, so a difference in score is a difference in algorithm rather than
#: in budget. Two settings need care:
#:
#: * ``num_leaves`` -- LightGBM grows leaf-wise, so ``max_depth`` alone does
#:   not bound tree size the way it does in XGBoost's level-wise growth.
#:   2**6 = 64 makes the capacity comparable; leaving the default 31 would
#:   quietly hand LightGBM a smaller model.
#: * ``subsample_freq`` -- LightGBM ignores ``subsample`` entirely unless
#:   this is >= 1. Without it, row subsampling silently does nothing.
LGB_HYPERPARAMETERS: dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 6,
    "num_leaves": 64,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_weight": 1e-3,
    "objective": "binary",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

DEFAULT_THRESHOLD: float = 0.5

#: Key under which the ensemble is recorded, and the models it averages.
ENSEMBLE_MEMBERS: tuple[str, str] = ("xgboost", "lightgbm")


def scale_pos_weight(y: pd.Series) -> float:
    """Negative-to-positive ratio -- the imbalance lever both libraries share.

    It multiplies the gradient contribution of positive rows, so the model
    stops treating "call everything legitimate" as a good local minimum.
    XGBoost and LightGBM spell it the same way and mean the same thing, so
    both get the identical value computed from the training labels.

    The cost is calibration: predicted probabilities come out inflated and
    can no longer be read as "an 0.8 here means 80% of these are fraud".
    That is fine for ranking, which is what PR-AUC measures and what a review
    queue needs -- but it is exactly why the 0.5 threshold below is reported
    as a formality rather than a recommendation.
    """
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("No positive rows in the training split.")
    return negatives / positives


def threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1 and the confusion matrix at one cut point."""
    predicted = (proba >= threshold).astype(int)
    precision = precision_score(y_true, predicted, zero_division=0)
    recall = recall_score(y_true, predicted, zero_division=0)
    denominator = precision + recall
    f1 = (2 * precision * recall / denominator) if denominator else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "flagged": int(tp + fp),
        "flagged_pct": float(100.0 * (tp + fp) / len(y_true)),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold maximising F1, found from the full precision-recall curve.

    ``precision_recall_curve`` returns one more precision/recall point than
    it does thresholds -- the trailing point is the degenerate
    recall=0, precision=1 corner, which has no threshold and is dropped.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]

    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    if not len(f1):
        return DEFAULT_THRESHOLD
    return float(thresholds[int(np.argmax(f1))])


def evaluate_proba(y: pd.Series | np.ndarray, proba: np.ndarray) -> dict:
    """Score a probability vector against labels, PR-AUC first.

    Takes probabilities rather than a model so the ensemble -- which has no
    model object, only an average -- goes through the identical code path
    and produces a directly comparable block.

    Refuses a degenerate evaluation set. Both cases otherwise fail deep
    inside sklearn or numpy with an error that names neither cause.
    """
    y_true = np.asarray(y)
    if len(y_true) == 0:
        raise ValueError(
            "Evaluation split is empty. Nothing to score -- check that the "
            "parquet file was written, not just created."
        )
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            f"Evaluation split contains only class {classes[0]} across "
            f"{len(y_true):,} rows. PR-AUC and ROC-AUC are undefined without "
            "both classes present -- at a ~3% base rate this usually means the "
            "split is too small or was cut on the wrong axis."
        )

    base_rate = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, proba))

    return {
        "pr_auc": pr_auc,
        "pr_auc_no_skill_floor": base_rate,
        "pr_auc_lift_over_floor": pr_auc / base_rate if base_rate else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "positive_rate": base_rate,
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "at_default_threshold": threshold_metrics(y_true, proba, DEFAULT_THRESHOLD),
        "at_best_f1_threshold": threshold_metrics(
            y_true, proba, best_f1_threshold(y_true, proba)
        ),
    }


def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability from either sklearn wrapper or raw Booster."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        return np.asarray(scores)[:, 1] if np.ndim(scores) == 2 else np.asarray(scores)
    return np.asarray(model.predict(X))  # LightGBM Booster: already P(positive)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score a fitted model on a labelled split."""
    return evaluate_proba(y, positive_proba(model, X))


def ensemble_proba(*probas: np.ndarray) -> np.ndarray:
    """Mean of the member probabilities.

    Averaging in probability space rather than log-odds is the simpler and
    more common choice, and it is what the ensemble is defined as here. Note
    the two are not equivalent: the geometric mean that log-odds averaging
    implies would pull harder toward whichever model is more confident.
    """
    return np.mean(np.column_stack(probas), axis=1)


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Fit the XGBoost baseline with imbalance reweighting."""
    model = xgb.XGBClassifier(**XGB_HYPERPARAMETERS, scale_pos_weight=scale_pos_weight(y))
    model.fit(X, y, verbose=False)
    return model


def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Fit the LightGBM baseline on the identical features and weighting.

    LightGBM handles NaN natively, like XGBoost, so the no-imputation stance
    carries over unchanged. The ordinal codes are passed as plain numerics
    rather than declared via ``categorical_feature`` -- declaring them would
    change the split semantics and the two models would no longer be
    comparable on equal footing.
    """
    model = lgb.LGBMClassifier(**LGB_HYPERPARAMETERS, scale_pos_weight=scale_pos_weight(y))
    model.fit(X, y)
    return model


#: Backwards-compatible alias: train_model was the single-model entry point.
train_model = train_xgboost


def _format_metrics(metrics: dict, title: str) -> str:
    """Render one model's evaluation, leading with the metric that matters."""
    default = metrics["at_default_threshold"]
    best = metrics["at_best_f1_threshold"]
    lines = [
        "",
        f"--- {title} " + "-" * max(0, 58 - len(title)),
        f"  PR-AUC            {metrics['pr_auc']:.4f}",
        f"    no-skill floor  {metrics['pr_auc_no_skill_floor']:.4f}  "
        "(the positive base rate -- what guessing scores)",
        f"    lift over floor {metrics['pr_auc_lift_over_floor']:.1f}x",
        f"  ROC-AUC           {metrics['roc_auc']:.4f}  "
        "(optimistic under imbalance; secondary)",
        "",
        f"  {'':<18}{'f1':>8}{'precision':>11}{'recall':>9}{'flagged':>10}",
    ]
    for label, block in (("at 0.5", default), (f"at {best['threshold']:.4f}", best)):
        lines.append(
            f"  {label:<18}{block['f1']:>8.4f}{block['precision']:>11.4f}"
            f"{block['recall']:>9.4f}{block['flagged']:>10,}"
        )
    lines.append(
        f"\n  best-F1 point: {best['true_positives']:,} caught, "
        f"{best['false_negatives']:,} missed, "
        f"{best['false_positives']:,} false alarms "
        f"({best['flagged_pct']:.2f}% of traffic flagged)"
    )
    return "\n".join(lines)


def _format_comparison(blocks: dict[str, dict]) -> str:
    """Side-by-side table across all three models."""
    lines = [
        "",
        "=" * 66,
        f"  {'model':<12}{'PR-AUC':>9}{'lift':>8}{'ROC-AUC':>10}"
        f"{'best F1':>10}{'train gap':>12}",
        "  " + "-" * 62,
    ]
    for name, block in blocks.items():
        val = block["val"]
        lines.append(
            f"  {name:<12}{val['pr_auc']:>9.4f}"
            f"{val['pr_auc_lift_over_floor']:>7.1f}x"
            f"{val['roc_auc']:>10.4f}"
            f"{val['at_best_f1_threshold']['f1']:>10.4f}"
            f"{block['overfit_gap_pr_auc']:>+12.4f}"
        )
    best = max(blocks, key=lambda name: blocks[name]["val"]["pr_auc"])
    lines += ["  " + "-" * 62, f"  best by PR-AUC: {best}", "=" * 66]
    return "\n".join(lines)


def _model_block(model, X_train, y_train, X_val, y_val, params: dict | None) -> dict:
    """Val metrics, train metrics, and the gap between them, for one model."""
    val = evaluate(model, X_val, y_val)
    train = evaluate(model, X_train, y_train)
    return {
        "val": val,
        "train": train,
        "overfit_gap_pr_auc": float(train["pr_auc"] - val["pr_auc"]),
        "hyperparameters": params,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train XGBoost, LightGBM and their ensemble; record all three."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="read only the first N rows of each split (smoke test on a small machine)",
    )
    args = parser.parse_args(argv)
    sampling = args.sample is not None

    ensure_dirs()
    stats = split_stats()

    suffix = f"  (first {args.sample:,} rows)" if sampling else ""
    print(f"loading  {TRAIN_PARQUET}{suffix}")
    train_frame = read_split(TRAIN_PARQUET, args.sample)
    if not sampling:
        stats["train"].assert_matches(train_frame)

    X_train, y_train, encoders = build_features(train_frame)
    del train_frame
    gc.collect()

    print(f"loading  {VAL_PARQUET}{suffix}")
    val_frame = read_split(VAL_PARQUET, args.sample)
    if not sampling:
        stats["val"].assert_matches(val_frame)

    # Same encoders, no refit. Refitting here would renumber every category
    # and score val against a vocabulary the models were never trained on.
    X_val, y_val, _ = build_features(val_frame, encoders)
    del val_frame
    gc.collect()

    if y_train is None or y_val is None:
        raise ValueError("Both splits must carry the isFraud label to train.")

    weight = scale_pos_weight(y_train)
    print(
        f"\ntrain: {len(X_train):,} rows x {X_train.shape[1]} features, "
        f"{int(y_train.sum()):,} positive ({y_train.mean() * 100:.3f}%)\n"
        f"val:   {len(X_val):,} rows, {int(y_val.sum()):,} positive "
        f"({y_val.mean() * 100:.3f}%)\n"
        f"  scale_pos_weight {weight:.2f}  "
        f"({int(len(y_train) - y_train.sum()):,} neg / {int(y_train.sum()):,} pos), "
        "applied to both models"
    )

    print(f"\n  fitting XGBoost  ({XGB_HYPERPARAMETERS['n_estimators']} trees) ...")
    xgb_model = train_xgboost(X_train, y_train)
    print(f"  fitting LightGBM ({LGB_HYPERPARAMETERS['n_estimators']} trees) ...")
    lgb_model = train_lightgbm(X_train, y_train)

    blocks: dict[str, dict] = {
        "xgboost": _model_block(
            xgb_model, X_train, y_train, X_val, y_val, XGB_HYPERPARAMETERS
        ),
        "lightgbm": _model_block(
            lgb_model, X_train, y_train, X_val, y_val, LGB_HYPERPARAMETERS
        ),
    }

    xgb_val_proba = positive_proba(xgb_model, X_val)
    lgb_val_proba = positive_proba(lgb_model, X_val)
    correlation = float(np.corrcoef(xgb_val_proba, lgb_val_proba)[0, 1])

    ensemble_val = evaluate_proba(y_val, ensemble_proba(xgb_val_proba, lgb_val_proba))
    ensemble_train = evaluate_proba(
        y_train,
        ensemble_proba(
            positive_proba(xgb_model, X_train), positive_proba(lgb_model, X_train)
        ),
    )
    blocks["ensemble"] = {
        "val": ensemble_val,
        "train": ensemble_train,
        "overfit_gap_pr_auc": float(ensemble_train["pr_auc"] - ensemble_val["pr_auc"]),
        "hyperparameters": None,
        "members": list(ENSEMBLE_MEMBERS),
        "method": "mean of member positive-class probabilities",
        "member_probability_correlation": correlation,
    }

    for name, block in blocks.items():
        print(_format_metrics(block["val"], name))
        print(
            f"  train PR-AUC {block['train']['pr_auc']:.4f} vs val "
            f"{block['val']['pr_auc']:.4f}  (gap {block['overfit_gap_pr_auc']:+.4f})"
        )
        if block["overfit_gap_pr_auc"] > 0.25:
            print(
                f"  WARNING: train scores {block['overfit_gap_pr_auc']:.2f} above "
                f"val -- likely too much capacity for {len(X_train):,} rows."
            )

    print(_format_comparison(blocks))

    best_single = max(("xgboost", "lightgbm"), key=lambda n: blocks[n]["val"]["pr_auc"])
    gain = ensemble_val["pr_auc"] - blocks[best_single]["val"]["pr_auc"]
    print(
        f"\n  member probability correlation {correlation:.4f}\n"
        f"  ensemble vs best single ({best_single}): {gain:+.4f} PR-AUC"
    )
    if correlation > 0.95 and abs(gain) < 0.01:
        print(
            "  As expected -- two boosted-tree models on identical features make\n"
            "  nearly the same errors, so averaging has little left to correct."
        )

    observed = blocks["xgboost"]["val"]
    reference_delta = observed["pr_auc"] - REFERENCE_BASELINE["pr_auc"]
    matches = abs(reference_delta) <= REFERENCE_TOLERANCE
    print(
        f"\n  reference XGBoost baseline: PR-AUC {REFERENCE_BASELINE['pr_auc']:.4f} / "
        f"ROC-AUC {REFERENCE_BASELINE['roc_auc']:.4f}\n"
        f"  this run:                   PR-AUC {observed['pr_auc']:.4f} / "
        f"ROC-AUC {observed['roc_auc']:.4f}  ({reference_delta:+.4f})"
    )
    if not matches and not sampling:
        print(
            "  DIVERGED from the reference baseline. Something upstream changed --\n"
            "  the feature pipeline, the split, or the library versions. Do not\n"
            "  redefine the baseline without knowing which."
        )

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampled": sampling,
        "sample_rows": args.sample,
        "data": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "n_features": int(X_train.shape[1]),
            "n_categorical_encoded": len(encoders.categorical_names),
            "train_positive_rate": float(y_train.mean()),
            "train_positives": int(y_train.sum()),
            "val_positive_rate": float(y_val.mean()),
            "val_positives": int(y_val.sum()),
            "scale_pos_weight": float(weight),
        },
        "reference_baseline": {
            "note": "first full XGBoost run on the real splits; fixed point for comparison",
            **REFERENCE_BASELINE,
            "observed_xgboost_pr_auc": observed["pr_auc"],
            "observed_xgboost_roc_auc": observed["roc_auc"],
            "pr_auc_delta": float(reference_delta),
            "within_tolerance": bool(matches),
            "tolerance": REFERENCE_TOLERANCE,
        },
        "models": blocks,
        "best_by_pr_auc": max(blocks, key=lambda n: blocks[n]["val"]["pr_auc"]),
        "environment": {
            "xgboost_version": xgb.__version__,
            "lightgbm_version": lgb.__version__,
            "pandas_version": pd.__version__,
        },
    }

    if sampling:
        print(
            f"\nNOT saving models, encoders or metrics: fitted on a "
            f"{args.sample:,}-row sample.\n"
            "A sampled run is not a baseline -- every later phase compares "
            "against this\nfile, and models trained on a slice would set the "
            "bar in the wrong place.\nRe-run without --sample to persist."
        )
        return 0

    xgb_model.save_model(XGB_MODEL_PATH)
    lgb_model.booster_.save_model(str(LGB_MODEL_PATH))
    encoders.save(ENCODERS_PATH)
    METRICS_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nwrote    {XGB_MODEL_PATH}")
    print(f"wrote    {LGB_MODEL_PATH}")
    print(f"wrote    {ENCODERS_PATH}")
    print(f"wrote    {METRICS_PATH}")

    # A model without its encoders is unusable, and both must round-trip or
    # serving will silently score against the wrong codes.
    probe = X_val.head(1000)
    reloaded_xgb = xgb.XGBClassifier()
    reloaded_xgb.load_model(XGB_MODEL_PATH)
    reloaded_lgb = lgb.Booster(model_file=str(LGB_MODEL_PATH))
    FeatureEncoders.load(ENCODERS_PATH)

    for name, reloaded, original in (
        ("xgboost", reloaded_xgb, xgb_model),
        ("lightgbm", reloaded_lgb, lgb_model),
    ):
        if not np.allclose(
            positive_proba(reloaded, probe), positive_proba(original, probe), atol=1e-6
        ):
            raise RuntimeError(f"Reloaded {name} does not reproduce its predictions.")
    print("verified round-trip: both models reproduce their predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
