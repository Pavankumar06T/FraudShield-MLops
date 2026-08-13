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
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import sklearn
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
from src.training import tracking

XGB_MODEL_PATH: Path = MODELS_DIR / "baseline_xgb.json"
LGB_MODEL_PATH: Path = MODELS_DIR / "baseline_lgb.txt"
METRICS_PATH: Path = REPORTS_DIR / "baseline_metrics.json"

#: Ceiling, not a target. Early stopping picks the actual tree count; this
#: only needs to be high enough that it never binds.
N_ESTIMATORS_CEILING: int = 2000

#: Rounds without improvement before stopping. Generous, because PR-AUC on
#: a few thousand positives is noisy round to round and a tight patience
#: stops on noise.
EARLY_STOPPING_ROUNDS: int = 50

#: Tail of train reserved for early stopping, by TransactionDT order.
ES_HOLDOUT_FRACTION: float = 0.15

#: Fixed thread count for both models. Never -1.
#:
#: XGBoost's hist tree method is not thread-deterministic: the subsample and
#: colsample RNG streams are per-thread, so the thread count changes which
#: rows and columns each tree sees. Measured with an identical seed and
#: identical data, only n_jobs differing -- early stopping at 239 / 300 / 140
#: trees for 1 / 2 / 4 threads, predicted probabilities differing by up to
#: 0.44. `-1` resolves to the core count of whatever machine happens to run
#: it, which makes it the one value that cannot be reproduced elsewhere.
#:
#: 2 rather than 1 because it is portable -- Colab, GitHub Actions runners
#: and an 8GB laptop all have at least two cores -- and roughly halves the
#: wall clock against single-threaded.
#:
#: LightGBM measured thread-deterministic (bit-identical at 1 and 4 threads),
#: so its pin is for symmetry rather than necessity: one knob, one place, and
#: no need to remember which library needed it.
#:
#: This matters beyond tidiness. Phase 6 retrains on GitHub Actions runners
#: whose core count differs from any development machine. Unpinned, a
#: drift-triggered retrain would produce a model differing for reasons having
#: nothing to do with the drift, and the Phase 7 shadow A/B would be
#: comparing thread counts as much as models. See docs/reproducibility.md.
N_JOBS: int = 2

#: The promoted reference: XGBoost under the depth4_reg regularization,
#: measured on the real splits (272k reduced train, 98k val). Kept as a
#: fixed point so a later run that diverges is visible immediately -- a
#: changed feature pipeline, a re-split, or a training-regime change would
#: all move it, and silently redefining the baseline would make every
#: downstream comparison meaningless. Promote a new figure here only
#: deliberately, never as a side effect.
#:
#: Measured locally at the pinned thread count, which is what makes it a
#: reference rather than an observation: rerunning this configuration on this
#: machine reproduces these figures exactly, and on any machine reproduces
#: them to within the thread-independent parts.
REFERENCE_BASELINE: dict[str, float | None] = {
    "pr_auc": 0.5248,
    "roc_auc": 0.8917,
    "overfit_gap_pr_auc": 0.2177,
    "n_trees_used": 799,
    "n_jobs": N_JOBS,
    "regime": (
        "depth4_reg, early stopping on a 15% temporal carve of train, "
        f"n_jobs={N_JOBS} pinned"
    ),
}

#: How far a rerun may drift from the reference before it is called out.
REFERENCE_TOLERANCE: float = 0.01

#: Superseded baselines, kept so the numbers are not lost and NOT so they
#: can be compared against. Each entry names the regime that produced it,
#: because a score is only meaningful alongside the data and budget behind
#: it. Nothing in the code reads these for comparison.
BASELINE_HISTORY: tuple[dict, ...] = (
    {
        "label": "fixed-400, full train, no carve",
        "pr_auc": 0.5477,
        "roc_auc": 0.9031,
        "overfit_gap_pr_auc": 0.2488,
        "regime": {
            "train_rows": 319_927,
            "n_estimators": 400,
            "early_stopping": False,
            "temporal_carve": False,
            "max_depth": 6,
        },
        "comparable_to_current": False,
        "why_not": (
            "Trained on all 319,927 rows. The current regime holds back 15% of "
            "train as an early-stopping slice, and that missing 48k rows -- not "
            "the change in tree count or depth -- accounts for the entire "
            "difference in score. See CARVE_COST_FINDING."
        ),
    },
    {
        "label": "depth6 + early stopping, carved train",
        "pr_auc": 0.5291,
        "overfit_gap_pr_auc": 0.2734,
        "n_trees_used": 364,
        "regime": {
            "train_rows": 271_938,
            "n_estimators": N_ESTIMATORS_CEILING,
            "early_stopping": True,
            "temporal_carve": True,
            "max_depth": 6,
            "n_jobs": "unpinned (-1)",
        },
        "comparable_to_current": True,
        "why_not": None,
    },
    {
        "label": "depth4_reg, Colab, unpinned threads",
        "pr_auc": 0.5255,
        "roc_auc": 0.8938,
        "overfit_gap_pr_auc": 0.1932,
        "n_trees_used": 651,
        "regime": {
            "train_rows": 271_938,
            "n_estimators": N_ESTIMATORS_CEILING,
            "early_stopping": True,
            "temporal_carve": True,
            "max_depth": 4,
            "n_jobs": "unpinned (-1), Colab core count unknown",
        },
        "comparable_to_current": False,
        "why_not": (
            "Ran with n_jobs=-1, which resolves to the core count of whatever "
            "machine executes it. XGBoost's hist method is not "
            "thread-deterministic -- its subsample and colsample RNG streams are "
            "per-thread -- so this 651-tree figure is not reproducible even on "
            "Colab: a different runner would give a different tree count. It was "
            "the promoted reference until the thread count was pinned. Kept as "
            "the record of what was measured, not as a target."
        ),
    },
    {
        "label": "depth4_reg, local, unpinned threads (4 cores)",
        "pr_auc": 0.5271,
        "roc_auc": 0.8915,
        "overfit_gap_pr_auc": 0.2343,
        "n_trees_used": 969,
        "regime": {
            "train_rows": 271_938,
            "n_estimators": N_ESTIMATORS_CEILING,
            "early_stopping": True,
            "temporal_carve": True,
            "max_depth": 4,
            "n_jobs": "unpinned (-1), resolved to 4",
        },
        "comparable_to_current": False,
        "why_not": (
            "Same configuration and same data as the current reference, differing "
            "only in thread count: 4 rather than the pinned 2. That alone moved "
            "the model from 799 trees to 969. Not reproducible on a machine with "
            "a different core count."
        ),
    },
)

#: What pinning the thread count actually bought, measured on the real split.
#:
#: The same configuration and data, varying only n_jobs, produced three
#: different XGBoost models. All three score within 0.0023 PR-AUC of each
#: other -- comfortably inside the +/-0.0080 bootstrap standard error on the
#: 98,305-row val split -- so none is better; they are the same model quality
#: reached by different row and column samples.
#:
#: LightGBM was identical to every digit across thread counts, which is the
#: control: it shows the variation is XGBoost's sampling RNG rather than
#: anything about the data or the pipeline.
THREAD_DETERMINISM_FINDING: dict[str, object] = {
    "measured_on": "real splits, 271,938 reduced train / 98,305 val",
    "xgboost_by_n_jobs": {
        "2 (pinned)": {"trees": 799, "pr_auc": 0.5248, "gap": 0.2177},
        "4 (unpinned local)": {"trees": 969, "pr_auc": 0.5271, "gap": 0.2343},
        "unknown (unpinned Colab)": {"trees": 651, "pr_auc": 0.5255, "gap": 0.1932},
    },
    "lightgbm_by_n_jobs": {
        "2 (pinned)": {"trees": 624, "pr_auc": 0.5239, "gap": 0.1932},
        "4 (unpinned local)": {"trees": 624, "pr_auc": 0.5239, "gap": 0.1932},
    },
    "val_pr_auc_bootstrap_std_error": 0.0080,
    "conclusion": (
        "XGBoost's tree count and predictions depend on thread count; LightGBM's "
        "do not. The spread across thread counts (0.0023 PR-AUC) is a quarter of "
        "one standard error, so no thread count produces a better model -- only "
        "a different one. Pinning makes the choice reproducible."
    ),
}

#: The measurement that makes the two regimes non-comparable, from the
#: config comparison on real data.
#:
#: carve_probe -- fixed 400 trees at depth 6 on the *reduced* train, i.e.
#: the old settings with 48k fewer rows -- scored 0.5291. depth6_current,
#: which differs only by early stopping, scored 0.5291 as well. Identical to
#: four decimals. So the whole 0.5477 -> 0.5291 drop is the carve, and early
#: stopping cost nothing at all.
#:
#: This is why the fixed-400 figure lives in history rather than as a bar:
#: any run under the current regime is measured on 15% less training data,
#: and would have to be better in order to look equal.
CARVE_COST_FINDING: dict[str, object] = {
    "measured_on": "real splits, 271,938 reduced train / 98,305 val",
    "full_train_fixed_400_pr_auc": 0.5477,
    "reduced_train_fixed_400_pr_auc": 0.5291,
    "reduced_train_early_stopped_pr_auc": 0.5291,
    "carve_cost_pr_auc": -0.0186,
    "early_stopping_cost_pr_auc": 0.0000,
    "conclusion": (
        "The entire drop from 0.5477 came from the 48k rows the temporal carve "
        "removed. Early stopping cost nothing -- it reached the same score with "
        "364 trees instead of 400."
    ),
}

#: The promoted depth4_reg configuration. Early stopping alone did not close
#: the overfit gap -- it moved from +0.2488 to +0.2734 while using 364 of
#: 2000 trees, so tree count was never the binding constraint and was never
#: going to be the fix. Depth and feature width were.
#:
#: Measured against depth6 on the same carved split: 0.5291 -> 0.5255 val
#: PR-AUC, a loss of 0.0036, for a gap of +0.2734 -> +0.1932, a reduction of
#: 0.0802. Roughly 30% less overfitting for a third of a percent of score.
XGB_HYPERPARAMETERS: dict[str, object] = {
    "n_estimators": N_ESTIMATORS_CEILING,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    # Shallower trees: the single largest contributor to the gap.
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    # Narrower column sampling. With 431 features, most of them anonymised
    # V-columns, 0.8 lets any given tree see almost everything.
    "colsample_bytree": 0.6,
    # A real floor on leaf weight, so a leaf cannot be carved out of a
    # handful of positives. XGBoost's default of 1 is no constraint at a
    # 3.4% base rate.
    "min_child_weight": 10,
    # L2 well above XGBoost's default of 1.0 -- "non-zero" is the default,
    # not a choice.
    "reg_lambda": 10.0,
    "tree_method": "hist",
    # aucpr, not logloss: logloss rewards calibrated probabilities, which
    # scale_pos_weight deliberately destroys. Stopping on it would optimise
    # something this model is not trying to be good at.
    "eval_metric": "aucpr",
    "n_jobs": N_JOBS,
    "random_state": 42,
}

#: Matched to the XGBoost configuration as closely as the two libraries
#: allow, so a difference in score is a difference in algorithm rather than
#: in budget. Two settings need care:
#:
#: * ``num_leaves`` -- LightGBM grows leaf-wise, so ``max_depth`` alone does
#:   not bound tree size the way it does in XGBoost's level-wise growth.
#:   Setting max_depth=4 without touching num_leaves leaves the default 31,
#:   which permits trees far wider than a depth-4 level-wise tree: the
#:   regularization would be nominal. 2**4 = 16 is the matching bound.
#: * ``subsample_freq`` -- LightGBM ignores ``subsample`` entirely unless
#:   this is >= 1. Without it, row subsampling silently does nothing.
#: * ``min_child_weight`` -- both libraries mean minimum summed hessian in a
#:   leaf, so 10 transfers directly. Note LightGBM's default is 1e-3 against
#:   XGBoost's 1, so leaving it alone would have been a far weaker
#:   constraint, not an equal one.
#:
#: Early stopping is a fit-time callback here rather than a constructor
#: argument, which is why it does not appear in this dict. LightGBM 4.x
#: removed ``early_stopping_rounds`` as a fit argument; the callback is the
#: replacement.
LGB_HYPERPARAMETERS: dict[str, object] = {
    "n_estimators": N_ESTIMATORS_CEILING,
    "max_depth": 4,
    "num_leaves": 16,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "min_child_weight": 10,
    "reg_lambda": 10.0,
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
    "n_jobs": N_JOBS,
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
    n_jobs: int | None = None,
) -> xgb.XGBClassifier:
    """Fit XGBoost, stopping on the carved slice when one is supplied.

    ``early_stopping_rounds`` belongs in the constructor for XGBoost 2.0+;
    passing it to ``fit`` raises. With no stopping slice the ceiling is
    trained out in full, which is only useful for tests.

    ``n_jobs`` is a correctness knob here, not only a speed one. XGBoost's
    hist method is not thread-deterministic: with an identical seed and
    identical data, 1, 2 and 4 threads produced early stops at 239, 300 and
    140 trees and predicted probabilities differing by up to 0.44. Pin it to
    reproduce a run on another machine. LightGBM is unaffected -- measured
    bit-identical across thread counts.
    """
    params = dict(XGB_HYPERPARAMETERS)
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
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
    n_jobs: int | None = None,
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
    params = dict(LGB_HYPERPARAMETERS)
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    model = lgb.LGBMClassifier(**params, scale_pos_weight=scale_pos_weight(y))
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
    """Side-by-side across all three models.

    Deliberately carries no column for the superseded fixed-400 figures.
    Those came from a regime with 48k more training rows, so a per-row delta
    against them would read as a regression when it is a difference in
    training data -- exactly the comparison BASELINE_HISTORY exists to
    prevent. The reference line below compares like with like.
    """
    lines = [
        "",
        "=" * 76,
        f"  {'model':<11}{'trees':>7}{'/ ceiling':>11}{'PR-AUC':>9}{'lift':>8}"
        f"{'ROC-AUC':>10}{'best F1':>9}{'train gap':>11}",
        "  " + "-" * 72,
    ]
    for name, block in blocks.items():
        val = block["val"]
        used = block.get("n_trees_used")
        ceiling = block.get("n_estimators_ceiling")
        lines.append(
            f"  {name:<11}{f'{used:,}' if used else '-':>7}"
            f"{f'{ceiling:,}' if ceiling else '-':>11}"
            f"{val['pr_auc']:>9.4f}{val['pr_auc_lift_over_floor']:>7.1f}x"
            f"{val['roc_auc']:>10.4f}{val['at_best_f1_threshold']['f1']:>9.4f}"
            f"{block['overfit_gap_pr_auc']:>+11.4f}"
        )
    best = max(blocks, key=lambda name: blocks[name]["val"]["pr_auc"])
    lines += ["  " + "-" * 72, f"  best by PR-AUC: {best}", "=" * 76]
    return "\n".join(lines)


def _format_reference_check(block: dict) -> tuple[str, bool]:
    """Compare the XGBoost run against the promoted reference.

    Returns the rendered text and whether PR-AUC landed within tolerance.
    ROC-AUC is reported rather than checked while the reference value is
    still pending -- a check against a number nobody recorded would either
    always pass or always fail, and neither means anything.
    """
    val = block["val"]
    expected = float(REFERENCE_BASELINE["pr_auc"])
    delta = val["pr_auc"] - expected
    matches = abs(delta) <= REFERENCE_TOLERANCE

    lines = [
        "",
        f"  reference ({REFERENCE_BASELINE['regime']}):",
        f"    PR-AUC     {expected:.4f}        this run {val['pr_auc']:.4f}  "
        f"({delta:+.4f})  {'MATCH' if matches else 'DIVERGED'}",
    ]

    reference_gap = REFERENCE_BASELINE.get("overfit_gap_pr_auc")
    if reference_gap is not None:
        gap_delta = block["overfit_gap_pr_auc"] - float(reference_gap)
        lines.append(
            f"    train gap  {float(reference_gap):+.4f}       this run "
            f"{block['overfit_gap_pr_auc']:+.4f}  ({gap_delta:+.4f})"
        )

    reference_trees = REFERENCE_BASELINE.get("n_trees_used")
    if reference_trees and block.get("n_trees_used"):
        lines.append(
            f"    trees      {int(reference_trees):,}           this run "
            f"{block['n_trees_used']:,}"
        )

    if REFERENCE_BASELINE.get("roc_auc") is None:
        lines.append(
            f"    ROC-AUC    not yet recorded.  this run {val['roc_auc']:.4f}\n"
            "               If this run is the promoted configuration, set\n"
            f'               REFERENCE_BASELINE["roc_auc"] = {val["roc_auc"]:.4f}'
        )
    else:
        lines.append(
            f"    ROC-AUC    {float(REFERENCE_BASELINE['roc_auc']):.4f}        "
            f"this run {val['roc_auc']:.4f}"
        )

    if not matches:
        lines.append(
            "    Something upstream changed -- the feature pipeline, the split, or\n"
            "    the library versions. The constant is NOT updated automatically."
        )
    return "\n".join(lines), matches


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


#: MLflow flavour per model. The ensemble has none -- it is an average, not
#: an object -- so it logs metrics and its members' run ids instead.
MLFLOW_FLAVOURS: dict[str, str] = {"xgboost": "xgboost", "lightgbm": "lightgbm"}


def environment_params(n_jobs_override: int | None = None) -> dict[str, object]:
    """Everything about the machine that can change the fitted model.

    Logged on every run so a divergence is attributable from the record
    alone rather than reconstructed months later. ``cpu_count`` and the
    effective ``n_jobs`` are here because XGBoost's hist method is not
    thread-deterministic -- two runs identical in every other respect will
    disagree if they used different thread counts, and without these
    parameters there is nothing in the record that says so.

    pandas is here for the opposite reason: it was the first suspect for a
    divergence it turned out not to cause, and the version is cheap to
    record and expensive to reconstruct.
    """
    effective = n_jobs_override if n_jobs_override is not None else XGB_HYPERPARAMETERS["n_jobs"]
    detected = os.cpu_count() or 1
    return {
        "env.python": platform.python_version(),
        "env.platform": platform.platform(),
        "env.cpu_count": detected,
        "env.n_jobs_requested": effective,
        "env.n_jobs_effective": detected if effective in (-1, None) else effective,
        "env.xgboost": xgb.__version__,
        "env.lightgbm": lgb.__version__,
        "env.pandas": pd.__version__,
        "env.numpy": np.__version__,
        "env.sklearn": sklearn.__version__,
        "env.pyarrow": pyarrow.__version__,
    }


def log_training_runs(
    blocks: dict[str, dict],
    fitted: dict[str, object],
    shared_params: dict[str, object],
    artifacts: list[Path],
) -> dict[str, str]:
    """Record one MLflow run per model. Returns {model: run_id} for those logged.

    Siblings rather than parent and children: each run carries the whole data
    description, so it can be reproduced from its own record without needing
    the others to still exist.

    Every call is a no-op when MLflow is unavailable -- see src.training.tracking.
    """
    run_ids: dict[str, str] = {}
    for name, block in blocks.items():
        with tracking.start_run(
            run_name=f"{name}-baseline",
            tags={"model": name, "phase": "baseline", "regime": "depth4_reg"},
        ) as run:
            if run is None:
                continue
            run_ids[name] = run.info.run_id

            params = dict(shared_params)
            params["model"] = name
            for key, value in (block.get("hyperparameters") or {}).items():
                params[f"hp.{key}"] = value
            if block.get("members"):
                params["ensemble.members"] = ",".join(block["members"])
                params["ensemble.method"] = block.get("method")
                # Recorded so the ensemble can be rebuilt from its members
                # rather than merely described.
                for member in block["members"]:
                    if member in run_ids:
                        params[f"ensemble.run_id.{member}"] = run_ids[member]
            tracking.log_params(run, params)

            metrics: dict[str, float] = {}
            for prefix, source in (
                ("val", block["val"]),
                ("train", block["train"]),
                ("stop", block.get("stopping_slice")),
            ):
                if source:
                    metrics.update(
                        {
                            f"{prefix}.{key}": value
                            for key, value in tracking.flatten_metrics(source).items()
                        }
                    )
            metrics["overfit_gap_pr_auc"] = block["overfit_gap_pr_auc"]
            if block.get("n_trees_used"):
                metrics["n_trees_used"] = float(block["n_trees_used"])
            if block.get("member_probability_correlation") is not None:
                metrics["member_probability_correlation"] = float(
                    block["member_probability_correlation"]
                )
            tracking.log_metrics(run, metrics)

            flavour = MLFLOW_FLAVOURS.get(name)
            if flavour and name in fitted:
                tracking.log_model(run, fitted[name], flavour)

            # The encoders travel with every model, including the ensemble.
            # A model without them cannot be served: the ordinal codes it
            # splits on are meaningless against a differently-fitted mapping.
            for artifact in artifacts:
                tracking.log_artifact(run, artifact)
    return run_ids


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
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="skip experiment tracking (it is already skipped when MLflow is absent)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "fit and report but write nothing. For reproducibility experiments: "
            "overwriting models/ while the registry describes a different model "
            "is how the two silently disagree."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "override thread count for both models. XGBoost's hist method is "
            "NOT thread-deterministic -- the same seed and data give different "
            "models at different thread counts -- so pin this to reproduce a run "
            "across machines. LightGBM is unaffected."
        ),
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
    env = environment_params(args.n_jobs)
    print(
        f"  threads {env['env.n_jobs_effective']} of {env['env.cpu_count']} "
        f"(n_jobs={env['env.n_jobs_requested']})"
        # Keyed off the effective value, not off whether the CLI flag was
        # passed: with N_JOBS pinned in the config, a run without the flag is
        # still reproducible, and saying otherwise would be false.
        + (
            "  <-- UNPINNED; XGBoost is not thread-deterministic, so this run\n"
            "      is not reproducible on a machine with a different core count"
            if env["env.n_jobs_requested"] in (-1, None)
            else "  (pinned, reproducible)"
        )
        + f"\n  pandas {pd.__version__}, xgboost {xgb.__version__}, "
        f"lightgbm {lgb.__version__}"
    )

    print(
        f"\n  fitting XGBoost  (ceiling {N_ESTIMATORS_CEILING:,}, patience "
        f"{EARLY_STOPPING_ROUNDS}, monitoring aucpr) ..."
    )
    xgb_model = train_xgboost(X_fit, y_fit, X_stop, y_stop, n_jobs=args.n_jobs)
    print(
        f"  fitting LightGBM (ceiling {N_ESTIMATORS_CEILING:,}, patience "
        f"{EARLY_STOPPING_ROUNDS}, monitoring {LGB_EVAL_METRIC}) ..."
    )
    lgb_model = train_lightgbm(X_fit, y_fit, X_stop, y_stop, n_jobs=args.n_jobs)

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
    reference_delta = observed["pr_auc"] - float(REFERENCE_BASELINE["pr_auc"])
    reference_text, matches = _format_reference_check(blocks["xgboost"])
    print(reference_text)

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
            "observed_xgboost_gap": blocks["xgboost"]["overfit_gap_pr_auc"],
            "pr_auc_delta": float(reference_delta),
            "within_tolerance": bool(matches),
            "tolerance": REFERENCE_TOLERANCE,
        },
        # Superseded figures, recorded so they are not lost. Each names the
        # regime behind it; none is a bar for the current run.
        "baseline_history": list(BASELINE_HISTORY),
        "carve_cost_finding": CARVE_COST_FINDING,
        "thread_determinism_finding": THREAD_DETERMINISM_FINDING,
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

    if args.no_save:
        print(
            "\nNOT saving models, encoders or metrics (--no-save). Nothing on "
            "disk changed,\nso the registered model and its companion artifacts "
            "stay consistent."
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

    if args.no_mlflow:
        print("\nMLflow logging skipped (--no-mlflow).")
        return 0

    print(f"\n{tracking.describe_store()}")
    # Every parameter needed to reproduce the run from its own record: the
    # exact carve boundary, the size and class balance of each slice, and
    # the library versions that produced the numbers.
    shared_params = {
        "split.cutoff_transaction_dt": cutoff,
        "split.holdout_fraction_requested": args.holdout_fraction,
        "split.holdout_fraction_realised": round(realised, 6),
        "split.train_rows_reduced": len(X_fit),
        "split.stopping_slice_rows": len(X_stop),
        "split.val_rows": len(X_val),
        "split.train_fraud_rate_pct": round(float(y_fit.mean()) * 100, 4),
        "split.stopping_slice_fraud_rate_pct": round(float(y_stop.mean()) * 100, 4),
        "split.val_fraud_rate_pct": round(float(y_val.mean()) * 100, 4),
        "split.n_features": X_fit.shape[1],
        "split.n_categorical_encoded": len(encoders.categorical_names),
        "split.scale_pos_weight": round(weight, 6),
        "early_stopping.ceiling": N_ESTIMATORS_CEILING,
        "early_stopping.rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping.encoders_fitted_on": "reduced train only",
        "early_stopping.val_used_for_stopping": False,
        **environment_params(args.n_jobs),
    }
    run_ids = log_training_runs(
        blocks,
        {"xgboost": xgb_model, "lightgbm": lgb_model},
        shared_params,
        artifacts=[ENCODERS_PATH, METRICS_PATH],
    )
    if run_ids:
        for name, run_id in run_ids.items():
            print(f"  logged {name:<9} run {run_id}")
        record["mlflow"] = {
            "tracking_uri": tracking.tracking_uri(),
            "experiment": tracking.EXPERIMENT_NAME,
            "run_ids": run_ids,
        }
        METRICS_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
