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

**Early stopping runs against a slice carved from train, never against val.**
The fixed-400-tree runs left a +0.25 or larger train/val PR-AUC gap, so tree
count is now chosen rather than assumed: a 2000 ceiling with a 50-round
patience, monitored on average precision rather than logloss, because
logloss rewards calibration this model deliberately gives up when
``scale_pos_weight`` inflates its probabilities.

The carve is **temporal, not random** -- the last ~15% of train by
``TransactionDT``, cut the same way the train/val/stream boundaries were.
A shuffled sample would scatter near-duplicate transactions (the same card,
minutes apart, sharing engineered history) across the boundary, so the
stopping slice would contain near-copies of rows the model just fitted.
Stopping would then fire late, against a target easier than anything the
model will meet in production.

Encoders are fitted on the reduced train only, not on the full split. The
stopping slice is meant to stand in for unseen data, and an encoder that had
already seen its category levels would hide exactly the vocabulary drift
that makes later windows hard.

val is never touched by any of this. It stays an untouched evaluation set,
which is the only reason its numbers mean anything.

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

#: The promoted reference: the first full XGBoost run on the real splits.
#: Kept as a fixed point so a later run that diverges is visible immediately
#: -- a changed feature pipeline, a re-split, or a training-regime change
#: would all move these, and silently redefining the baseline would make
#: every downstream comparison meaningless. Promote a new figure here only
#: deliberately, never as a side effect.
REFERENCE_BASELINE: dict[str, float] = {"pr_auc": 0.5477, "roc_auc": 0.9031}

#: How far a rerun may drift from the reference before it is called out.
REFERENCE_TOLERANCE: float = 0.01

#: Val PR-AUC from the previous fixed-400-tree regime, for all three models.
#: Printed beside the current numbers so the effect of early stopping is
#: directly visible rather than inferred.
FIXED_400_PR_AUC: dict[str, float] = {
    "xgboost": 0.5477,
    "lightgbm": 0.5482,
    "ensemble": 0.5513,
}

#: Ceiling, not a target. Early stopping picks the actual tree count; this
#: only needs to be high enough that it never binds.
N_ESTIMATORS_CEILING: int = 2000

#: Rounds without improvement before stopping. Generous, because PR-AUC on
#: a few thousand positives is noisy round to round and a tight patience
#: stops on noise.
EARLY_STOPPING_ROUNDS: int = 50

#: Tail of train reserved for early stopping, by TransactionDT order.
ES_HOLDOUT_FRACTION: float = 0.15

XGB_HYPERPARAMETERS: dict[str, object] = {
    "n_estimators": N_ESTIMATORS_CEILING,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "tree_method": "hist",
    # aucpr, not logloss: logloss rewards calibrated probabilities, which
    # scale_pos_weight deliberately destroys. Stopping on it would optimise
    # something this model is not trying to be good at.
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
#:
#: Early stopping is a fit-time callback here rather than a constructor
#: argument, which is why it does not appear in this dict. LightGBM 4.x
#: removed ``early_stopping_rounds`` as a fit argument; the callback is the
#: replacement.
LGB_HYPERPARAMETERS: dict[str, object] = {
    "n_estimators": N_ESTIMATORS_CEILING,
    "max_depth": 6,
    "num_leaves": 64,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_weight": 1e-3,
    "objective": "binary",
    # Load-bearing. objective="binary" makes LightGBM monitor binary_logloss
    # by default, and eval_metric in fit APPENDS rather than replaces -- so
    # both metrics get watched. Under scale_pos_weight the probabilities are
    # deliberately uncalibrated, so logloss is best at round 1 and degrades
    # monotonically from there while average precision is still climbing.
    # The early-stopping callback halts when ANY monitored metric stalls, so
    # logloss killed the fit at one tree: PR-AUC 0.3759, zero rows flagged
    # at 0.5. Naming the metric here suppresses binary_logloss entirely;
    # first_metric_only=True on the callback is the second belt.
    "metric": "average_precision",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

#: LightGBM's name for average precision -- the PR-AUC equivalent of
#: XGBoost's "aucpr".
LGB_EVAL_METRIC: str = "average_precision"

DEFAULT_THRESHOLD: float = 0.5

#: Key under which the ensemble is recorded, and the models it averages.
ENSEMBLE_MEMBERS: tuple[str, str] = ("xgboost", "lightgbm")

TIME_COLUMN: str = "TransactionDT"


def temporal_holdout(
    frame: pd.DataFrame,
    fraction: float = ES_HOLDOUT_FRACTION,
    time_column: str = TIME_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Split off the latest ``fraction`` of rows by time, not at random.

    Returns ``(earlier, later, cutoff)``. The cut is on the value of
    ``time_column`` rather than on row position, so it behaves the same
    whether or not the frame arrived sorted -- and matches how the
    train/val/stream boundaries themselves were drawn.

    Ties land in the later part, so the realised fraction can exceed the
    requested one when many transactions share a timestamp. The caller
    reports what it actually got rather than what it asked for.
    """
    if time_column not in frame.columns:
        raise KeyError(
            f"{time_column!r} is required to carve a temporal holdout but is "
            f"not in the frame. Carve before build_features drops it."
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}.")

    times = frame[time_column]
    cutoff = float(times.quantile(1.0 - fraction))
    earlier = frame[times < cutoff]
    later = frame[times >= cutoff]

    if earlier.empty or later.empty:
        raise ValueError(
            f"Temporal cut at {time_column}={cutoff:,.0f} left "
            f"{len(earlier):,} / {len(later):,} rows. The column is likely "
            "constant or near-constant."
        )
    return earlier, later, cutoff


def scale_pos_weight(y: pd.Series) -> float:
    """Negative-to-positive ratio -- the imbalance lever both libraries share.

    It multiplies the gradient contribution of positive rows, so the model
    stops treating "call everything legitimate" as a good local minimum.
    XGBoost and LightGBM spell it the same way and mean the same thing, so
    both get the identical value computed from the training labels.

    Recomputed from whatever rows actually train the model -- after the
    early-stopping slice is carved out, not before. The two differ whenever
    the fraud rate drifts across the split, which in this dataset it does.

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
    """Positive-class probability from either sklearn wrapper or raw Booster.

    Both wrappers predict with their own best iteration once early stopping
    has run, so callers get the stopped model rather than the full ceiling
    without having to ask.
    """
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


def trees_used(model) -> int:
    """How many boosting rounds the stopped model actually predicts with.

    The two libraries index this differently -- XGBoost's ``best_iteration``
    is 0-based, LightGBM's ``best_iteration_`` is 1-based -- and reporting
    one as the other would be off by one in opposite directions.
    """
    best = getattr(model, "best_iteration", None)
    if best is not None:
        return int(best) + 1
    best = getattr(model, "best_iteration_", None)
    if best:
        return int(best)
    return int(getattr(model, "n_estimators", 0))


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    X_stop: pd.DataFrame | None = None,
    y_stop: pd.Series | None = None,
) -> xgb.XGBClassifier:
    """Fit XGBoost, stopping on the carved slice when one is supplied.

    ``early_stopping_rounds`` belongs in the constructor for XGBoost 2.0+;
    passing it to ``fit`` raises. With no stopping slice the ceiling is
    trained out in full, which is only useful for tests.
    """
    params = dict(XGB_HYPERPARAMETERS)
    if X_stop is None:
        params.pop("early_stopping_rounds", None)

    model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight(y))
    if X_stop is None:
        model.fit(X, y, verbose=False)
    else:
        model.fit(X, y, eval_set=[(X_stop, y_stop)], verbose=False)
    return model


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    X_stop: pd.DataFrame | None = None,
    y_stop: pd.Series | None = None,
) -> lgb.LGBMClassifier:
    """Fit LightGBM on the identical features, weighting and stopping rule.

    LightGBM handles NaN natively, like XGBoost, so the no-imputation stance
    carries over unchanged. The ordinal codes are passed as plain numerics
    rather than declared via ``categorical_feature`` -- declaring them would
    change the split semantics and the two models would no longer be
    comparable on equal footing.

    The early-stopping call shape is ``eval_set=[(X, y)]`` plus a
    ``lgb.early_stopping`` callback. What LightGBM 4.x actually removed is
    ``early_stopping_rounds`` as a *fit argument*; ``eval_set`` itself is
    current and is the only form present in stock builds. Do not swap it for
    ``eval_X``/``eval_y`` on the strength of a local deprecation warning --
    that pair exists in some patched builds only, and the substitution
    raises TypeError everywhere else.
    """
    model = lgb.LGBMClassifier(
        **LGB_HYPERPARAMETERS, scale_pos_weight=scale_pos_weight(y)
    )
    if X_stop is None:
        model.fit(X, y)
    else:
        model.fit(
            X,
            y,
            eval_set=[(X_stop, y_stop)],
            eval_metric=LGB_EVAL_METRIC,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False,
                    # Stop on average precision alone. Without this the
                    # callback halts as soon as ANY monitored metric stops
                    # improving -- see the note on "metric" above.
                    first_metric_only=True,
                )
            ],
        )
    return model


def eval_history(model) -> dict[str, list[float]]:
    """Per-round validation scores, normalised across the two libraries.

    LightGBM exposes ``evals_result_`` as an attribute, XGBoost
    ``evals_result()`` as a method, and each nests under its own name for
    the eval set. Returns ``{metric: [score per round]}``.
    """
    if hasattr(model, "evals_result_"):
        results = model.evals_result_
    elif hasattr(model, "evals_result"):
        results = model.evals_result()
    else:
        return {}
    if not results:
        return {}
    first = next(iter(results.values()))
    return {name: list(scores) for name, scores in first.items()}


def format_eval_history(model, rounds: int = 10) -> str:
    """Render the first few boosting rounds, and where each metric peaked.

    Worth printing every run. A model that stops at one tree and a model
    that stops at four hundred look identical in the summary table until
    you can see whether the monitored metric was still climbing when the
    fit was halted.
    """
    history = eval_history(model)
    if not history:
        return "    (no eval history -- fitted without an eval set)"

    names = list(history)
    lines = [
        f"    monitored: {', '.join(names)}",
        "    " + f"{'round':>6}" + "".join(f"{name:>22}" for name in names),
    ]
    for index in range(min(rounds, len(history[names[0]]))):
        lines.append(
            f"    {index + 1:>6}"
            + "".join(f"{history[name][index]:>22.6f}" for name in names)
        )

    for name in names:
        scores = np.asarray(history[name])
        lower_is_better = "loss" in name or "error" in name
        best = int(np.argmin(scores) if lower_is_better else np.argmax(scores)) + 1
        direction = "lower is better" if lower_is_better else "higher is better"
        flag = (
            "   <-- peaks at round 1; this metric would stop the fit immediately"
            if best == 1
            else ""
        )
        lines.append(
            f"    {name} best at round {best} of {len(scores)} ({direction}){flag}"
        )
    return "\n".join(lines)


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
    """Side-by-side across all three models, against the fixed-400 regime."""
    lines = [
        "",
        "=" * 78,
        f"  {'model':<11}{'trees':>7}{'/ ceiling':>11}{'PR-AUC':>9}"
        f"{'was @400':>10}{'delta':>9}{'ROC-AUC':>10}{'train gap':>11}",
        "  " + "-" * 74,
    ]
    for name, block in blocks.items():
        val = block["val"]
        previous = FIXED_400_PR_AUC.get(name)
        used = block.get("n_trees_used")
        ceiling = block.get("n_estimators_ceiling")
        trees = f"{used:,}" if used else "-"
        against = f"{ceiling:,}" if ceiling else "-"
        was = f"{previous:.4f}" if previous is not None else "-"
        delta = f"{val['pr_auc'] - previous:+.4f}" if previous is not None else "-"
        lines.append(
            f"  {name:<11}{trees:>7}{against:>11}{val['pr_auc']:>9.4f}"
            f"{was:>10}{delta:>9}{val['roc_auc']:>10.4f}"
            f"{block['overfit_gap_pr_auc']:>+11.4f}"
        )
    best = max(blocks, key=lambda name: blocks[name]["val"]["pr_auc"])
    lines += ["  " + "-" * 74, f"  best by PR-AUC: {best}", "=" * 78]
    return "\n".join(lines)


def _model_block(
    model, X_train, y_train, X_val, y_val, params: dict | None, stopped: bool
) -> dict:
    """Val metrics, train metrics, the gap, and how many trees survived."""
    val = evaluate(model, X_val, y_val)
    train = evaluate(model, X_train, y_train)
    used = trees_used(model)
    return {
        "val": val,
        "train": train,
        "overfit_gap_pr_auc": float(train["pr_auc"] - val["pr_auc"]),
        "n_trees_used": used,
        "n_estimators_ceiling": N_ESTIMATORS_CEILING,
        "early_stopped": bool(stopped and used < N_ESTIMATORS_CEILING),
        "hit_ceiling": bool(stopped and used >= N_ESTIMATORS_CEILING),
        "hyperparameters": params,
    }


def _split_summary(name: str, y: pd.Series) -> str:
    return (
        f"    {name:<22}{len(y):>9,} rows  {int(y.sum()):>7,} positive  "
        f"{y.mean() * 100:>6.3f}%"
    )


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
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=ES_HOLDOUT_FRACTION,
        metavar="F",
        help=f"tail of train reserved for early stopping (default {ES_HOLDOUT_FRACTION})",
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

    # Carve before build_features, which drops TransactionDT.
    fit_frame, stop_frame, cutoff = temporal_holdout(train_frame, args.holdout_fraction)
    realised = len(stop_frame) / len(train_frame)
    del train_frame
    gc.collect()

    # Encoders see only the rows that train the model. Fitting them on the
    # full split would let the stopping slice's category levels leak in.
    X_fit, y_fit, encoders = build_features(fit_frame)
    X_stop, y_stop, _ = build_features(stop_frame, encoders)
    del fit_frame, stop_frame
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

    if y_fit is None or y_stop is None or y_val is None:
        raise ValueError("Every split must carry the isFraud label to train.")

    weight = scale_pos_weight(y_fit)
    print(
        f"\ntemporal carve at {TIME_COLUMN} >= {cutoff:,.0f}  "
        f"(last {realised * 100:.1f}% of train, requested "
        f"{args.holdout_fraction * 100:.0f}%)\n"
        f"{_split_summary('train (reduced)', y_fit)}\n"
        f"{_split_summary('early-stopping slice', y_stop)}\n"
        f"{_split_summary('val (untouched)', y_val)}\n"
        f"\n  {X_fit.shape[1]} features, scale_pos_weight {weight:.2f} "
        f"recomputed from the reduced train, applied to both models"
    )

    print(
        f"\n  fitting XGBoost  (ceiling {N_ESTIMATORS_CEILING:,}, patience "
        f"{EARLY_STOPPING_ROUNDS}, monitoring aucpr) ..."
    )
    xgb_model = train_xgboost(X_fit, y_fit, X_stop, y_stop)
    print(
        f"  fitting LightGBM (ceiling {N_ESTIMATORS_CEILING:,}, patience "
        f"{EARLY_STOPPING_ROUNDS}, monitoring {LGB_EVAL_METRIC}) ..."
    )
    lgb_model = train_lightgbm(X_fit, y_fit, X_stop, y_stop)

    blocks: dict[str, dict] = {
        "xgboost": _model_block(
            xgb_model, X_fit, y_fit, X_val, y_val, XGB_HYPERPARAMETERS, True
        ),
        "lightgbm": _model_block(
            lgb_model, X_fit, y_fit, X_val, y_val, LGB_HYPERPARAMETERS, True
        ),
    }
    for name, model in (("xgboost", xgb_model), ("lightgbm", lgb_model)):
        block = blocks[name]
        block["stopping_slice"] = evaluate(model, X_stop, y_stop)

    xgb_val_proba = positive_proba(xgb_model, X_val)
    lgb_val_proba = positive_proba(lgb_model, X_val)
    correlation = float(np.corrcoef(xgb_val_proba, lgb_val_proba)[0, 1])

    ensemble_val = evaluate_proba(y_val, ensemble_proba(xgb_val_proba, lgb_val_proba))
    ensemble_train = evaluate_proba(
        y_fit,
        ensemble_proba(
            positive_proba(xgb_model, X_fit), positive_proba(lgb_model, X_fit)
        ),
    )
    blocks["ensemble"] = {
        "val": ensemble_val,
        "train": ensemble_train,
        "overfit_gap_pr_auc": float(ensemble_train["pr_auc"] - ensemble_val["pr_auc"]),
        "n_trees_used": None,
        "n_estimators_ceiling": None,
        "early_stopped": None,
        "hyperparameters": None,
        "members": list(ENSEMBLE_MEMBERS),
        "method": "mean of member positive-class probabilities",
        "member_probability_correlation": correlation,
    }

    fitted = {"xgboost": xgb_model, "lightgbm": lgb_model}
    for name, block in blocks.items():
        print(_format_metrics(block["val"], name))
        if block.get("n_trees_used"):
            ceiling_note = (
                "  HIT THE CEILING -- raise it; the tree count was capped, not chosen."
                if block["hit_ceiling"]
                else ""
            )
            print(
                f"  trees used {block['n_trees_used']:,} of "
                f"{block['n_estimators_ceiling']:,}{ceiling_note}"
            )
            print(f"\n  early-stopping history:\n{format_eval_history(fitted[name])}\n")
        print(
            f"  train PR-AUC {block['train']['pr_auc']:.4f} vs val "
            f"{block['val']['pr_auc']:.4f}  (gap {block['overfit_gap_pr_auc']:+.4f})"
        )
        if block["overfit_gap_pr_auc"] > 0.25:
            print(
                f"  WARNING: train still scores {block['overfit_gap_pr_auc']:.2f} "
                "above val despite early stopping."
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
            "  DIVERGED from the reference. Early stopping changed the training\n"
            "  regime, so a shift here is expected rather than alarming -- but the\n"
            "  constant is NOT updated automatically. Promote the new figure in\n"
            "  REFERENCE_BASELINE only once you have decided it is the better bar."
        )

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampled": sampling,
        "sample_rows": args.sample,
        "data": {
            "train_rows_reduced": int(len(X_fit)),
            "stopping_slice_rows": int(len(X_stop)),
            "val_rows": int(len(X_val)),
            "n_features": int(X_fit.shape[1]),
            "n_categorical_encoded": len(encoders.categorical_names),
            "train_positive_rate": float(y_fit.mean()),
            "train_positives": int(y_fit.sum()),
            "stopping_slice_positive_rate": float(y_stop.mean()),
            "stopping_slice_positives": int(y_stop.sum()),
            "val_positive_rate": float(y_val.mean()),
            "val_positives": int(y_val.sum()),
            "scale_pos_weight": float(weight),
        },
        "early_stopping": {
            "strategy": "temporal tail of train, cut on TransactionDT",
            "requested_fraction": float(args.holdout_fraction),
            "realised_fraction": float(realised),
            "cutoff_transaction_dt": cutoff,
            "n_estimators_ceiling": N_ESTIMATORS_CEILING,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "monitored_metric": {"xgboost": "aucpr", "lightgbm": LGB_EVAL_METRIC},
            "encoders_fitted_on": "reduced train only",
            "val_used_for_stopping": False,
        },
        "reference_baseline": {
            "note": "promoted reference; not updated automatically",
            **REFERENCE_BASELINE,
            "observed_xgboost_pr_auc": observed["pr_auc"],
            "observed_xgboost_roc_auc": observed["roc_auc"],
            "pr_auc_delta": float(reference_delta),
            "within_tolerance": bool(matches),
            "tolerance": REFERENCE_TOLERANCE,
        },
        "previous_fixed_400_pr_auc": FIXED_400_PR_AUC,
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
    lgb_model.booster_.save_model(
        str(LGB_MODEL_PATH), num_iteration=lgb_model.best_iteration_
    )
    encoders.save(ENCODERS_PATH)
    METRICS_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nwrote    {XGB_MODEL_PATH}")
    print(f"wrote    {LGB_MODEL_PATH}")
    print(f"wrote    {ENCODERS_PATH}")
    print(f"wrote    {METRICS_PATH}")

    # With early stopping the saved artifact must predict with the stopped
    # tree count, not the full ceiling. XGBoost carries best_iteration inside
    # the JSON; LightGBM needs num_iteration passed at save time. If either
    # were wrong, serving would quietly use hundreds of trees the model was
    # never validated with.
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
    print(
        f"verified round-trip: both models reproduce their predictions at "
        f"{blocks['xgboost']['n_trees_used']:,} / "
        f"{blocks['lightgbm']['n_trees_used']:,} trees"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
