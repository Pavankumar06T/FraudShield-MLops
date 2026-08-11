"""Baseline XGBoost model: fit on train, evaluate on val, record the result.

This establishes the number every later phase argues against. The retrained
model that drift triggers, and the shadow model in the A/B test, are both
judged relative to what is written here -- so the metrics file records not
just the scores but the shape of the data they came from.

**PR-AUC is the headline.** At a 3.4% positive rate the alternatives mislead:

* Accuracy is worthless. Predicting "never fraud" scores 96.6%.
* ROC-AUC is optimistic. Its false-positive-rate axis is normalised by the
  569k negatives, so thousands of false alarms barely move it. A model can
  post 0.97 ROC-AUC while a fraud analyst drowns in review queue.
* PR-AUC lives on the positives. Its no-skill floor is the base rate itself
  -- 0.034, not 0.5 -- so it is reported here alongside that floor and the
  lift over it. A PR-AUC of 0.60 against a 0.034 floor is roughly an 18x
  improvement on guessing; the same number means nothing stated alone.

**Thresholds.** 0.5 is arbitrary here and doubly so with ``scale_pos_weight``
in play: reweighting the positive class inflates predicted probabilities, so
the operating point that 0.5 happens to select is an artifact of the class
ratio rather than a decision anyone made. Metrics are reported at 0.5 for
continuity, and at the sweep's best-F1 point for what the model can actually
do. Expect the latter to sit well above 0.5.

**No early stopping.** val is the evaluation set, and stopping on it would
tune the model against the same rows the reported metrics come from,
flattering every number in the file. Tree count is fixed instead.

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

MODEL_PATH: Path = MODELS_DIR / "baseline_xgb.json"
METRICS_PATH: Path = REPORTS_DIR / "baseline_metrics.json"

#: Conventional starting point, not a tuned configuration. Tuning belongs
#: after the drift loop works end to end -- a better baseline would not
#: change what the pipeline needs to do, and would make this file harder to
#: reproduce. `aucpr` matches the headline metric so the training log tracks
#: what is actually being optimised.
HYPERPARAMETERS: dict[str, object] = {
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

DEFAULT_THRESHOLD: float = 0.5


def scale_pos_weight(y: pd.Series) -> float:
    """Negative-to-positive ratio, XGBoost's lever for class imbalance.

    It multiplies the gradient contribution of positive rows, so the model
    stops treating "call everything legitimate" as a good local minimum.

    The cost is calibration: predicted probabilities come out inflated and
    can no longer be read as "an 0.8 here means 80% of these are fraud".
    That is fine for ranking, which is what PR-AUC measures and what a
    review queue needs -- but it is exactly why the 0.5 threshold below is
    reported as a formality rather than a recommendation.
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


def evaluate(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score the model on a labelled split, PR-AUC first.

    Refuses a degenerate evaluation set. Both cases otherwise fail deep
    inside sklearn or numpy -- an empty frame makes predict_proba return
    shape (0, 0), so the positive-class column raises IndexError rather
    than saying the split was empty.
    """
    y_true = y.to_numpy()
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

    proba = model.predict_proba(X)[:, 1]

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


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Fit the baseline classifier with imbalance reweighting."""
    weight = scale_pos_weight(y)
    model = xgb.XGBClassifier(**HYPERPARAMETERS, scale_pos_weight=weight)
    model.fit(X, y, verbose=False)
    return model


def _format_metrics(metrics: dict) -> str:
    """Render the evaluation, leading with the metric that matters."""
    default = metrics["at_default_threshold"]
    best = metrics["at_best_f1_threshold"]
    lines = [
        "",
        f"val: {metrics['rows']:,} rows, {metrics['positives']:,} positive "
        f"({metrics['positive_rate'] * 100:.3f}%)",
        "",
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
    lines += [
        "",
        f"  best-F1 point: {best['true_positives']:,} caught, "
        f"{best['false_negatives']:,} missed, "
        f"{best['false_positives']:,} false alarms "
        f"({best['flagged_pct']:.2f}% of traffic flagged)",
    ]
    if best["threshold"] > DEFAULT_THRESHOLD:
        lines.append(
            f"  0.5 sits below the best operating point, as expected with "
            f"scale_pos_weight inflating probabilities."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the baseline XGBoost model and record its metrics."
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
    # and score val against a vocabulary the model was never trained on.
    X_val, y_val, _ = build_features(val_frame, encoders)
    del val_frame
    gc.collect()

    if y_train is None or y_val is None:
        raise ValueError("Both splits must carry the isFraud label to train.")

    weight = scale_pos_weight(y_train)
    print(
        f"\ntrain: {len(X_train):,} rows x {X_train.shape[1]} features, "
        f"{int(y_train.sum()):,} positive ({y_train.mean() * 100:.3f}%)\n"
        f"  scale_pos_weight {weight:.2f}  "
        f"({int(len(y_train) - y_train.sum()):,} neg / {int(y_train.sum()):,} pos)"
    )

    print(f"  fitting {HYPERPARAMETERS['n_estimators']} trees ...")
    model = train_model(X_train, y_train)

    metrics = evaluate(model, X_val, y_val)
    print(_format_metrics(metrics))

    # Scored on the training rows too, purely as a diagnostic. Tree count is
    # fixed rather than early-stopped -- stopping on val would tune the model
    # against the rows the headline metrics come from -- so nothing else in
    # this run would reveal an overfit. A val PR-AUC near the floor means
    # something quite different depending on whether train sits at 0.9 or
    # also near the floor: the first is too much capacity, the second is a
    # feature problem.
    train_metrics = evaluate(model, X_train, y_train)
    gap = train_metrics["pr_auc"] - metrics["pr_auc"]
    print(
        f"\n  train PR-AUC {train_metrics['pr_auc']:.4f} vs val "
        f"{metrics['pr_auc']:.4f}  (gap {gap:+.4f})"
    )
    if gap > 0.25:
        print(
            f"  WARNING: train scores {gap:.2f} above val. "
            f"{HYPERPARAMETERS['n_estimators']} trees at depth "
            f"{HYPERPARAMETERS['max_depth']} is likely too much capacity for "
            f"{len(X_train):,} rows."
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
        },
        "model": {
            "type": type(model).__name__,
            "scale_pos_weight": float(weight),
            "hyperparameters": HYPERPARAMETERS,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
        },
        "metrics": metrics,
        "train_metrics": train_metrics,
        "overfit_gap_pr_auc": float(gap),
    }

    if sampling:
        print(
            f"\nNOT saving model, encoders or metrics: fitted on a "
            f"{args.sample:,}-row sample.\n"
            "A sampled run is not a baseline -- every later phase compares "
            "against this\nfile, and a model trained on a slice would set the "
            "bar in the wrong place.\nRe-run without --sample to persist."
        )
        return 0

    model.save_model(MODEL_PATH)
    encoders.save(ENCODERS_PATH)
    METRICS_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nwrote    {MODEL_PATH}")
    print(f"wrote    {ENCODERS_PATH}")
    print(f"wrote    {METRICS_PATH}")

    # A model without its encoders is unusable, and the two must round-trip
    # together or serving will silently score against the wrong codes.
    reloaded = xgb.XGBClassifier()
    reloaded.load_model(MODEL_PATH)
    FeatureEncoders.load(ENCODERS_PATH)
    if not np.allclose(
        reloaded.predict_proba(X_val.head(1000))[:, 1],
        model.predict_proba(X_val.head(1000))[:, 1],
    ):
        raise RuntimeError("Reloaded model does not reproduce its own predictions.")
    print("verified round-trip: reloaded model reproduces its predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
