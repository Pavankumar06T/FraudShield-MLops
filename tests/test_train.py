"""Invariants for the baseline training run.

Most of these guard claims the module makes in prose -- that the no-skill
PR-AUC floor really is the base rate, that ROC-AUC really does flatter a
model PR-AUC does not, that the swept threshold really does beat 0.5. A
metric that is merely asserted in a docstring is a metric nobody checked.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.training.train import (
    DEFAULT_THRESHOLD,
    LGB_HYPERPARAMETERS,
    XGB_HYPERPARAMETERS,
    best_f1_threshold,
    ensemble_proba,
    evaluate,
    evaluate_proba,
    positive_proba,
    scale_pos_weight,
    threshold_metrics,
    train_lightgbm,
    train_model,
)

BASE_RATE = 0.034


def imbalanced(n: int = 20_000, rate: float = BASE_RATE, seed: int = 0):
    """Labels plus a score that ranks positives above negatives, noisily."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    proba = np.clip(rng.normal(np.where(y == 1, 0.72, 0.28), 0.18), 0.001, 0.999)
    return y, proba


# --------------------------------------------------------------------------
# Class imbalance handling
# --------------------------------------------------------------------------


def test_scale_pos_weight_is_the_negative_positive_ratio():
    y = pd.Series([0] * 966 + [1] * 34)
    assert scale_pos_weight(y) == pytest.approx(966 / 34)


def test_scale_pos_weight_near_the_real_rate():
    """3.4% positives should land near 28x, not near 1."""
    y = pd.Series((np.arange(100_000) % 1000 < 34).astype(int))
    assert scale_pos_weight(y) == pytest.approx(28.4, abs=0.5)


def test_scale_pos_weight_refuses_a_single_class():
    with pytest.raises(ValueError, match="No positive rows"):
        scale_pos_weight(pd.Series([0] * 100))


# --------------------------------------------------------------------------
# PR-AUC and its floor -- the reason the module leads with this metric
# --------------------------------------------------------------------------


def test_random_scores_score_at_the_base_rate():
    """The no-skill floor claim: a useless model's PR-AUC is the base rate."""
    rng = np.random.default_rng(1)
    n = 200_000
    y = (rng.random(n) < BASE_RATE).astype(int)
    noise = rng.random(n)
    assert average_precision_score(y, noise) == pytest.approx(BASE_RATE, abs=0.004)


def test_perfect_ranking_scores_one():
    y = np.array([0] * 900 + [1] * 100)
    assert average_precision_score(y, y.astype(float)) == pytest.approx(1.0)


def test_roc_auc_flatters_where_pr_auc_does_not():
    """A model that ranks well but floods the queue with false positives.

    ROC-AUC normalises false positives by the huge negative pool, so it
    stays high; PR-AUC does not. This is the whole argument for the
    ordering of the printed report.
    """
    y, proba = imbalanced(n=60_000, seed=7)
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    assert roc > 0.9
    assert pr < roc - 0.2


def test_evaluate_reports_floor_and_lift_consistently():
    y, proba = imbalanced()

    class Stub:
        def predict_proba(self, X):
            return np.column_stack([1 - proba, proba])

    metrics = evaluate(Stub(), pd.DataFrame(index=range(len(y))), pd.Series(y))

    assert metrics["pr_auc_no_skill_floor"] == pytest.approx(y.mean())
    assert metrics["positive_rate"] == pytest.approx(y.mean())
    assert metrics["pr_auc_lift_over_floor"] == pytest.approx(
        metrics["pr_auc"] / y.mean()
    )
    assert metrics["pr_auc"] > metrics["pr_auc_no_skill_floor"]
    assert metrics["rows"] == len(y)
    assert metrics["positives"] == int(y.sum())


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


def test_threshold_metrics_match_a_hand_counted_confusion_matrix():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
    out = threshold_metrics(y, proba, 0.5)

    assert (out["true_positives"], out["false_positives"]) == (2, 1)
    assert (out["false_negatives"], out["true_negatives"]) == (1, 1)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)
    assert out["flagged"] == 3


def test_threshold_sweep_beats_the_default():
    """If 0.5 were already optimal the sweep would be pointless."""
    y, proba = imbalanced()
    best = best_f1_threshold(y, proba)
    assert threshold_metrics(y, proba, best)["f1"] >= threshold_metrics(
        y, proba, DEFAULT_THRESHOLD
    )["f1"]


def test_threshold_sweep_returns_an_attainable_cut_point():
    y, proba = imbalanced()
    best = best_f1_threshold(y, proba)
    assert proba.min() <= best <= proba.max()
    assert threshold_metrics(y, proba, best)["flagged"] > 0


def test_extreme_threshold_degenerates_safely():
    """Above every score nothing is flagged; precision is undefined and must
    report 0 rather than raise or emit NaN into the metrics file."""
    y, proba = imbalanced()
    out = threshold_metrics(y, proba, 1.1)
    assert out["flagged"] == 0
    assert out["precision"] == 0.0
    assert out["f1"] == 0.0


# --------------------------------------------------------------------------
# End to end on a small frame
# --------------------------------------------------------------------------


def learnable(n: int = 6_000, seed: int = 3):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(signal * 2 - 3.5)))).astype(int))
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=n),
            "with_gaps": np.where(rng.random(n) < 0.4, np.nan, rng.normal(size=n)),
        }
    )
    return X, y


def test_trained_model_beats_the_no_skill_floor():
    X, y = learnable()
    model = train_model(X, y)
    metrics = evaluate(model, X, y)
    assert metrics["pr_auc"] > 3 * metrics["pr_auc_no_skill_floor"]


def test_model_handles_nan_without_imputation():
    X, y = learnable()
    assert X["with_gaps"].isna().any()
    model = train_model(X, y)
    assert np.isfinite(model.predict_proba(X)[:, 1]).all()


def test_metrics_block_is_json_serialisable():
    """numpy scalars leak through json.dump as TypeError, and the metrics
    file is the artifact every later phase reads."""
    X, y = learnable()
    metrics = evaluate(train_model(X, y), X, y)
    round_tripped = json.loads(json.dumps(metrics))
    assert round_tripped["pr_auc"] == pytest.approx(metrics["pr_auc"])
    assert set(round_tripped) == set(metrics)


def test_model_round_trips_through_disk(tmp_path):
    import xgboost as xgb

    X, y = learnable()
    model = train_model(X, y)
    path = tmp_path / "m.json"
    model.save_model(path)

    reloaded = xgb.XGBClassifier()
    reloaded.load_model(path)
    np.testing.assert_allclose(
        reloaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1], rtol=1e-6
    )


# --------------------------------------------------------------------------
# LightGBM and the ensemble
# --------------------------------------------------------------------------


def test_lightgbm_config_matches_xgboost_where_it_must():
    """Two settings silently diverge if left at LightGBM's defaults.

    num_leaves: LightGBM grows leaf-wise, so max_depth alone does not bound
    tree size. subsample_freq: without it >= 1, `subsample` does nothing.
    """
    assert LGB_HYPERPARAMETERS["num_leaves"] == 2 ** XGB_HYPERPARAMETERS["max_depth"]
    assert LGB_HYPERPARAMETERS["subsample_freq"] >= 1
    for shared in ("n_estimators", "max_depth", "learning_rate", "subsample",
                   "colsample_bytree", "random_state"):
        assert LGB_HYPERPARAMETERS[shared] == XGB_HYPERPARAMETERS[shared]


def test_lightgbm_trains_and_beats_the_floor():
    X, y = learnable()
    metrics = evaluate(train_lightgbm(X, y), X, y)
    assert metrics["pr_auc"] > 3 * metrics["pr_auc_no_skill_floor"]


def test_lightgbm_handles_nan_without_imputation():
    X, y = learnable()
    assert X["with_gaps"].isna().any()
    assert np.isfinite(positive_proba(train_lightgbm(X, y), X)).all()


def test_ensemble_is_the_mean_of_its_members():
    a = np.array([0.1, 0.9, 0.5])
    b = np.array([0.3, 0.7, 0.1])
    np.testing.assert_allclose(ensemble_proba(a, b), [0.2, 0.8, 0.3])


def test_ensemble_of_identical_members_changes_nothing():
    a = np.array([0.1, 0.9, 0.5])
    np.testing.assert_allclose(ensemble_proba(a, a), a)


def test_ensemble_stays_in_probability_space():
    rng = np.random.default_rng(5)
    a, b = rng.random(500), rng.random(500)
    blended = ensemble_proba(a, b)
    assert blended.min() >= 0.0 and blended.max() <= 1.0
    assert np.all(blended >= np.minimum(a, b)) and np.all(blended <= np.maximum(a, b))


def test_evaluate_proba_and_evaluate_agree():
    """The ensemble goes through evaluate_proba while the single models go
    through evaluate; the two must produce identical blocks or the
    comparison table is not comparing like with like."""
    X, y = learnable()
    model = train_model(X, y)
    assert evaluate(model, X, y) == evaluate_proba(y, positive_proba(model, X))


def test_evaluate_proba_rejects_degenerate_splits():
    proba = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="only class"):
        evaluate_proba(np.zeros(50, dtype=int), proba)
    with pytest.raises(ValueError, match="empty"):
        evaluate_proba(np.array([], dtype=int), np.array([]))


def test_all_three_blocks_share_the_same_keys():
    """The comparison table indexes into every block identically."""
    X, y = learnable()
    xgb_model, lgb_model = train_model(X, y), train_lightgbm(X, y)
    blocks = [
        evaluate(xgb_model, X, y),
        evaluate(lgb_model, X, y),
        evaluate_proba(y, ensemble_proba(positive_proba(xgb_model, X),
                                         positive_proba(lgb_model, X))),
    ]
    assert set(blocks[0]) == set(blocks[1]) == set(blocks[2])
    for block in blocks:
        assert set(block["at_best_f1_threshold"]) == set(blocks[0]["at_default_threshold"])


def test_positive_proba_handles_a_raw_lightgbm_booster():
    """Booster.predict already returns P(positive) -- no [:, 1] to slice."""
    X, y = learnable()
    model = train_lightgbm(X, y)
    np.testing.assert_allclose(
        positive_proba(model.booster_, X), positive_proba(model, X), atol=1e-9
    )
