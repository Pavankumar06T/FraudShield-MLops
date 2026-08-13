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
    BASELINE_HISTORY,
    CARVE_COST_FINDING,
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    LGB_EVAL_METRIC,
    REFERENCE_BASELINE,
    _format_reference_check,
    LGB_HYPERPARAMETERS,
    N_ESTIMATORS_CEILING,
    N_JOBS,
    THREAD_DETERMINISM_FINDING,
    XGB_HYPERPARAMETERS,
    best_f1_threshold,
    ensemble_proba,
    eval_history,
    evaluate,
    format_eval_history,
    evaluate_proba,
    positive_proba,
    scale_pos_weight,
    temporal_holdout,
    threshold_metrics,
    train_lightgbm,
    train_model,
    trees_used,
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


def learnable_with_stop(n: int = 6_000, seed: int = 3, fraction: float = 0.15):
    """Same frame, split into fit and early-stopping parts.

    Tests train through the real stopping path rather than the 2000-tree
    ceiling -- both because it is what production does, and because fitting
    the full ceiling on every test would dominate the suite runtime.
    """
    X, y = learnable(n, seed)
    cut = int(len(X) * (1 - fraction))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def test_trained_model_beats_the_no_skill_floor():
    Xf, yf, Xs, ys = learnable_with_stop()
    metrics = evaluate(train_model(Xf, yf, Xs, ys), Xf, yf)
    assert metrics["pr_auc"] > 3 * metrics["pr_auc_no_skill_floor"]


def test_model_handles_nan_without_imputation():
    Xf, yf, Xs, ys = learnable_with_stop()
    assert Xf["with_gaps"].isna().any()
    assert np.isfinite(positive_proba(train_model(Xf, yf, Xs, ys), Xf)).all()


def test_metrics_block_is_json_serialisable():
    """numpy scalars leak through json.dump as TypeError, and the metrics
    file is the artifact every later phase reads."""
    Xf, yf, Xs, ys = learnable_with_stop()
    metrics = evaluate(train_model(Xf, yf, Xs, ys), Xf, yf)
    round_tripped = json.loads(json.dumps(metrics))
    assert round_tripped["pr_auc"] == pytest.approx(metrics["pr_auc"])
    assert set(round_tripped) == set(metrics)


def test_model_round_trips_through_disk(tmp_path):
    """The saved artifact must predict with the stopped tree count, not the
    full ceiling -- otherwise serving quietly uses trees never validated."""
    import xgboost as xgb

    Xf, yf, Xs, ys = learnable_with_stop()
    model = train_model(Xf, yf, Xs, ys)
    path = tmp_path / "m.json"
    model.save_model(path)

    reloaded = xgb.XGBClassifier()
    reloaded.load_model(path)
    assert reloaded.best_iteration == model.best_iteration
    np.testing.assert_allclose(
        reloaded.predict_proba(Xf)[:, 1], model.predict_proba(Xf)[:, 1], rtol=1e-6
    )


def test_lightgbm_round_trips_at_its_stopped_tree_count(tmp_path):
    """LightGBM does not carry best_iteration into the saved file -- it must
    be passed at save time or the artifact holds every tree to the ceiling."""
    import lightgbm as lgb

    Xf, yf, Xs, ys = learnable_with_stop()
    model = train_lightgbm(Xf, yf, Xs, ys)
    path = tmp_path / "m.txt"
    model.booster_.save_model(str(path), num_iteration=model.best_iteration_)

    reloaded = lgb.Booster(model_file=str(path))
    assert reloaded.num_trees() == model.best_iteration_
    np.testing.assert_allclose(
        positive_proba(reloaded, Xf), positive_proba(model, Xf), atol=1e-9
    )


# --------------------------------------------------------------------------
# The temporal carve
# --------------------------------------------------------------------------


def timed(n: int = 1_000, seed: int = 9) -> pd.DataFrame:
    """A frame whose row order deliberately disagrees with its time order."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"TransactionDT": np.arange(n) * 100.0, "v": rng.normal(size=n)}
    )
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_carve_takes_the_latest_rows_not_the_last_positions():
    """The frame is shuffled, so a positional tail would be a random sample.

    This is the property the whole design rests on: every stopping row must
    come after every fitting row in time.
    """
    frame = timed()
    earlier, later, cutoff = temporal_holdout(frame, 0.15)
    assert earlier["TransactionDT"].max() < cutoff <= later["TransactionDT"].min()


def test_carve_partitions_without_overlap_or_loss():
    frame = timed()
    earlier, later, _ = temporal_holdout(frame, 0.15)
    assert len(earlier) + len(later) == len(frame)
    assert not set(earlier.index) & set(later.index)


def test_carve_takes_roughly_the_requested_fraction():
    frame = timed(n=10_000)
    _, later, _ = temporal_holdout(frame, 0.15)
    assert 0.14 <= len(later) / len(frame) <= 0.16


@pytest.mark.parametrize("fraction", [0.05, 0.15, 0.30])
def test_carve_honours_different_fractions(fraction):
    frame = timed(n=10_000)
    _, later, _ = temporal_holdout(frame, fraction)
    assert abs(len(later) / len(frame) - fraction) < 0.01


def test_carve_requires_the_time_column():
    """build_features drops TransactionDT, so carving after it would fail."""
    with pytest.raises(KeyError, match="TransactionDT"):
        temporal_holdout(pd.DataFrame({"v": [1, 2, 3]}))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_carve_rejects_impossible_fractions(fraction):
    with pytest.raises(ValueError, match="fraction"):
        temporal_holdout(timed(), fraction)


def test_carve_rejects_a_constant_time_column():
    frame = pd.DataFrame({"TransactionDT": np.ones(100), "v": np.arange(100.0)})
    with pytest.raises(ValueError, match="constant"):
        temporal_holdout(frame)


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------


def test_early_stopping_chooses_a_tree_count_below_the_ceiling():
    Xf, yf, Xs, ys = learnable_with_stop()
    for model in (train_model(Xf, yf, Xs, ys), train_lightgbm(Xf, yf, Xs, ys)):
        used = trees_used(model)
        assert 0 < used < N_ESTIMATORS_CEILING


def test_lightgbm_fit_accepts_our_exact_early_stopping_call():
    """Call the real LGBMClassifier.fit with the exact keywords we use.

    Nothing here goes through train_lightgbm, and nothing is mocked -- a
    wrapper test can only prove the wrapper is self-consistent, and a mock
    would have happily accepted the eval_X call that crashed the real run.
    If LightGBM's fit signature and ours ever disagree again, this fails in
    a second rather than twenty minutes into a training run.
    """
    import lightgbm as lgb

    Xf, yf, Xs, ys = learnable_with_stop(n=1_200)
    model = lgb.LGBMClassifier(
        **LGB_HYPERPARAMETERS, scale_pos_weight=scale_pos_weight(yf)
    )
    model.fit(
        Xf,
        yf,
        eval_set=[(Xs, ys)],
        eval_metric=LGB_EVAL_METRIC,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        ],
    )
    assert 0 < model.best_iteration_ < N_ESTIMATORS_CEILING
    assert LGB_EVAL_METRIC in model.best_score_["valid_0"]


def test_train_lightgbm_passes_eval_set_and_not_eval_x(monkeypatch):
    """Pin the call shape, because one machine cannot prove portability.

    The environment this was developed on carries a patched LightGBM that
    accepts eval_X as well as eval_set, so the end-to-end test above passed
    locally while the real run raised TypeError. Asserting on the keywords
    themselves is what actually catches the substitution here.
    """
    import lightgbm as lgb

    captured: dict = {}
    original = lgb.LGBMClassifier.fit

    def spy(self, X, y, **kwargs):
        captured.update(kwargs)
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy)

    Xf, yf, Xs, ys = learnable_with_stop(n=1_200)
    train_lightgbm(Xf, yf, Xs, ys)

    assert "eval_set" in captured, "early stopping must be driven by eval_set"
    assert "eval_X" not in captured and "eval_y" not in captured, (
        "eval_X/eval_y exist only in patched builds and raise TypeError on stock "
        "LightGBM"
    )
    # early_stopping_rounds as a fit argument is the part that really was removed
    assert "early_stopping_rounds" not in captured

    (eval_X, eval_y), = captured["eval_set"]
    assert len(eval_X) == len(Xs) and len(eval_y) == len(ys)
    assert captured["eval_metric"] == LGB_EVAL_METRIC
    assert captured["callbacks"]


def test_early_stopping_rounds_really_is_gone_from_fit():
    """Documents the deprecation that is real, as opposed to the one that
    was inferred from a local warning."""
    import inspect

    import lightgbm as lgb

    parameters = inspect.signature(lgb.LGBMClassifier.fit).parameters
    assert "early_stopping_rounds" not in parameters
    assert "eval_set" in parameters
    assert "callbacks" in parameters


def imbalanced_with_stop(n: int = 16_000, seed: int = 12):
    """A frame shaped like the real problem: rare positives, real signal.

    The imbalance is the point. scale_pos_weight inflates probabilities, so
    binary_logloss gets monotonically worse from round 1 while average
    precision is still climbing -- the condition that stopped the real run
    at a single tree.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(signal * 1.6 - 4.0)))).astype(int))
    X = pd.DataFrame(
        {"a": signal, "b": rng.normal(size=n), "c": rng.normal(size=n)}
    )
    cut = int(n * 0.85)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def test_lightgbm_does_not_stop_at_one_tree_under_imbalance():
    """The regression that mattered: PR-AUC 0.3759 and zero rows flagged.

    binary_logloss peaks at round 1 under scale_pos_weight and the callback
    halts when ANY monitored metric stalls, so the fit died immediately. A
    model that stops in single digits here is misconfigured, not trained.
    """
    Xf, yf, Xs, ys = imbalanced_with_stop()
    assert yf.mean() < 0.10 and ys.sum() > 50  # genuinely imbalanced, enough signal

    model = train_lightgbm(Xf, yf, Xs, ys)
    assert model.best_iteration_ > 10, (
        f"stopped at {model.best_iteration_} trees -- binary_logloss is being "
        "monitored again, or first_metric_only was dropped"
    )
    # and it produces a usable ranking rather than an all-negative constant
    assert (positive_proba(model, Xs) >= 0.5).sum() > 0


def test_lightgbm_monitors_average_precision_alone():
    """binary_logloss must be suppressed, not merely deprioritised.

    objective='binary' adds it by default and eval_metric APPENDS rather
    than replaces, so without naming the metric explicitly both are watched.
    """
    assert LGB_HYPERPARAMETERS["metric"] == LGB_EVAL_METRIC

    Xf, yf, Xs, ys = imbalanced_with_stop(n=6_000)
    history = eval_history(train_lightgbm(Xf, yf, Xs, ys))
    assert list(history) == [LGB_EVAL_METRIC], (
        f"monitoring {list(history)}; binary_logloss will stop the fit at round 1"
    )


def test_lightgbm_callback_sets_first_metric_only(monkeypatch):
    """Second belt. Pinned on the keywords, because a build that watches one
    metric anyway would let the flag's removal pass unnoticed."""
    import lightgbm as lgb

    captured: dict = {}
    original = lgb.LGBMClassifier.fit

    def spy(self, X, y, **kwargs):
        captured.update(kwargs)
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy)
    Xf, yf, Xs, ys = learnable_with_stop(n=1_200)
    train_lightgbm(Xf, yf, Xs, ys)

    callbacks = captured["callbacks"]
    assert callbacks
    assert any(getattr(cb, "first_metric_only", False) for cb in callbacks), (
        "no early-stopping callback with first_metric_only=True"
    )


def test_xgboost_monitors_only_aucpr():
    """XGBoost replaces the default metric rather than appending, so it has
    no equivalent failure -- asserted rather than assumed."""
    Xf, yf, Xs, ys = imbalanced_with_stop(n=6_000)
    assert list(eval_history(train_model(Xf, yf, Xs, ys))) == ["aucpr"]


# --------------------------------------------------------------------------
# Eval history
# --------------------------------------------------------------------------


def test_eval_history_normalises_both_libraries():
    """LightGBM exposes an attribute, XGBoost a method, each nested under a
    different eval-set name."""
    Xf, yf, Xs, ys = learnable_with_stop(n=1_200)
    for model in (train_model(Xf, yf, Xs, ys), train_lightgbm(Xf, yf, Xs, ys)):
        history = eval_history(model)
        assert history
        assert all(isinstance(v, list) and v for v in history.values())


def test_eval_history_is_empty_without_an_eval_set():
    X, y = learnable(n=1_200)
    assert eval_history(train_lightgbm(X, y)) == {}
    assert "no eval history" in format_eval_history(train_lightgbm(X, y))


def test_history_flags_a_metric_that_peaks_at_round_one():
    """The tell that diagnosed the bug has to survive in the output."""

    class Stub:
        evals_result_ = {
            "valid_0": {
                "average_precision": [0.1, 0.2, 0.3, 0.4],
                "binary_logloss": [0.20, 0.25, 0.30, 0.35],
            }
        }

    text = format_eval_history(Stub(), rounds=4)
    assert "binary_logloss best at round 1" in text
    assert "would stop the fit immediately" in text
    assert "average_precision best at round 4" in text
    assert "would stop the fit immediately" not in text.split("binary_logloss")[0]


def test_trees_used_handles_both_index_conventions():
    """XGBoost's best_iteration is 0-based, LightGBM's best_iteration_ is
    1-based. Reporting one as the other is off by one, in opposite
    directions."""
    Xf, yf, Xs, ys = learnable_with_stop()

    xgb_model = train_model(Xf, yf, Xs, ys)
    assert trees_used(xgb_model) == xgb_model.best_iteration + 1

    lgb_model = train_lightgbm(Xf, yf, Xs, ys)
    assert trees_used(lgb_model) == lgb_model.best_iteration_


def test_ceiling_is_generous_enough_to_be_a_ceiling():
    assert XGB_HYPERPARAMETERS["n_estimators"] == N_ESTIMATORS_CEILING
    assert LGB_HYPERPARAMETERS["n_estimators"] == N_ESTIMATORS_CEILING


def test_stopping_monitors_pr_auc_not_logloss():
    """logloss rewards calibration that scale_pos_weight deliberately
    destroys, so stopping on it optimises the wrong thing."""
    assert XGB_HYPERPARAMETERS["eval_metric"] == "aucpr"
    assert LGB_EVAL_METRIC == "average_precision"


def test_scale_pos_weight_is_recomputed_from_the_reduced_train():
    """The fraud rate drifts across the carve, so the full-train and
    reduced-train weights differ -- using the former would misweight.

    Modelled on the real shape: train sits at 3.40% and val at 3.93%, so the
    tail of train is richer in positives than the head.
    """
    early = [1 if i % 25 == 0 else 0 for i in range(850)]  # 4.0%
    late = [1 if i % 10 == 0 else 0 for i in range(150)]  # 10.0%
    y_full = pd.Series(early + late)
    y_reduced = y_full.iloc[:850]

    assert y_reduced.mean() < y_full.mean()  # the tail really is richer
    # a rarer positive class needs a larger weight
    assert scale_pos_weight(y_reduced) > scale_pos_weight(y_full)


# --------------------------------------------------------------------------
# LightGBM and the ensemble
# --------------------------------------------------------------------------


def test_lightgbm_config_matches_xgboost_where_it_must():
    """Three settings silently diverge if left at LightGBM's defaults.

    num_leaves: LightGBM grows leaf-wise, so max_depth alone does not bound
    tree size -- at max_depth=4 the default 31 permits trees far wider than
    a depth-4 level-wise tree, making the regularization nominal.
    subsample_freq: without it >= 1, `subsample` does nothing.
    min_child_weight: LightGBM defaults to 1e-3 against XGBoost's 1, so
    leaving it alone is a far weaker constraint rather than an equal one.
    """
    assert LGB_HYPERPARAMETERS["num_leaves"] == 2 ** XGB_HYPERPARAMETERS["max_depth"]
    assert LGB_HYPERPARAMETERS["num_leaves"] == 16
    assert LGB_HYPERPARAMETERS["subsample_freq"] >= 1
    for shared in ("n_estimators", "max_depth", "learning_rate", "subsample",
                   "colsample_bytree", "min_child_weight", "reg_lambda",
                   "random_state"):
        assert LGB_HYPERPARAMETERS[shared] == XGB_HYPERPARAMETERS[shared], shared


# --------------------------------------------------------------------------
# The promoted configuration and the baselines it supersedes
# --------------------------------------------------------------------------


def test_production_config_is_depth4_reg():
    """The promotion, pinned. 0.5255 val PR-AUC at a +0.1932 gap on real
    data -- 0.0036 of score for 0.0802 of gap."""
    assert XGB_HYPERPARAMETERS["max_depth"] == 4
    assert XGB_HYPERPARAMETERS["min_child_weight"] == 10
    assert XGB_HYPERPARAMETERS["subsample"] == pytest.approx(0.8)
    assert XGB_HYPERPARAMETERS["colsample_bytree"] == pytest.approx(0.6)
    assert XGB_HYPERPARAMETERS["reg_lambda"] == pytest.approx(10.0)
    # early stopping retained at the same ceiling and patience
    assert XGB_HYPERPARAMETERS["n_estimators"] == N_ESTIMATORS_CEILING == 2000
    assert XGB_HYPERPARAMETERS["early_stopping_rounds"] == EARLY_STOPPING_ROUNDS == 50


def test_reference_baseline_holds_the_pinned_local_figures():
    """The reference is the pinned-thread local run, not the Colab one."""
    assert REFERENCE_BASELINE["pr_auc"] == pytest.approx(0.5248)
    assert REFERENCE_BASELINE["roc_auc"] == pytest.approx(0.8917)
    assert REFERENCE_BASELINE["overfit_gap_pr_auc"] == pytest.approx(0.2177)
    assert REFERENCE_BASELINE["n_trees_used"] == 799
    assert REFERENCE_BASELINE["n_jobs"] == N_JOBS


def test_reference_records_the_thread_count_it_was_measured_at():
    """Without it the figure is not reproducible, only observed."""
    assert "n_jobs" in REFERENCE_BASELINE["regime"]
    assert REFERENCE_BASELINE["n_jobs"] != -1


def test_unpinned_runs_are_in_history_and_marked_unreproducible():
    """Both the 651 Colab figure and the 969 local one must survive as
    record, and neither may be usable as a bar."""
    by_trees = {e["n_trees_used"]: e for e in BASELINE_HISTORY if "n_trees_used" in e}

    colab = by_trees[651]
    assert colab["pr_auc"] == pytest.approx(0.5255)
    assert colab["comparable_to_current"] is False
    assert "not reproducible" in colab["why_not"]
    assert "unpinned" in colab["regime"]["n_jobs"]

    local = by_trees[969]
    assert local["pr_auc"] == pytest.approx(0.5271)
    assert local["comparable_to_current"] is False
    assert "thread count" in local["why_not"]


def test_thread_finding_shows_lightgbm_unchanged_and_xgboost_moving():
    f = THREAD_DETERMINISM_FINDING
    lgb_runs = list(f["lightgbm_by_n_jobs"].values())
    assert len({r["trees"] for r in lgb_runs}) == 1, "LightGBM must be the control"
    assert len({r["pr_auc"] for r in lgb_runs}) == 1

    xgb_runs = list(f["xgboost_by_n_jobs"].values())
    assert len({r["trees"] for r in xgb_runs}) == len(xgb_runs), "all differ"

    spread = max(r["pr_auc"] for r in xgb_runs) - min(r["pr_auc"] for r in xgb_runs)
    assert spread < f["val_pr_auc_bootstrap_std_error"], (
        "the spread must be inside one standard error -- otherwise thread count "
        "would be choosing a better model, not just a different one"
    )


def test_superseded_baseline_is_kept_and_marked_non_comparable():
    """0.5477 must survive as history, and must be unusable as a bar."""
    entry = next(e for e in BASELINE_HISTORY if e["pr_auc"] == pytest.approx(0.5477))
    assert entry["roc_auc"] == pytest.approx(0.9031)
    assert entry["comparable_to_current"] is False
    assert entry["why_not"]
    assert entry["regime"]["train_rows"] == 319_927
    assert entry["regime"]["temporal_carve"] is False
    assert entry["regime"]["early_stopping"] is False


def test_history_entries_all_name_their_regime():
    for entry in BASELINE_HISTORY:
        assert entry["label"] and entry["regime"]
        assert "train_rows" in entry["regime"]
        if not entry["comparable_to_current"]:
            assert entry["why_not"], f"{entry['label']} needs a reason"


def test_carve_cost_finding_is_internally_consistent():
    """The measurement that makes the two regimes non-comparable.

    carve_probe and depth6_current both scored 0.5291, so the entire drop
    from 0.5477 is the 48k removed rows and early stopping cost nothing.
    """
    f = CARVE_COST_FINDING
    assert f["reduced_train_fixed_400_pr_auc"] == f["reduced_train_early_stopped_pr_auc"]
    assert f["carve_cost_pr_auc"] == pytest.approx(
        f["reduced_train_fixed_400_pr_auc"] - f["full_train_fixed_400_pr_auc"], abs=1e-6
    )
    assert f["early_stopping_cost_pr_auc"] == pytest.approx(
        f["reduced_train_early_stopped_pr_auc"] - f["reduced_train_fixed_400_pr_auc"],
        abs=1e-6,
    )
    assert f["early_stopping_cost_pr_auc"] == pytest.approx(0.0)


def test_nothing_compares_against_the_superseded_figures():
    """The old numbers are recorded, not used. If a run were measured
    against 0.5477 it would look like a regression when it is really 48k
    fewer training rows."""
    assert REFERENCE_BASELINE["pr_auc"] != pytest.approx(0.5477)
    reference = _format_reference_check(
        {
            "val": {"pr_auc": 0.5255, "roc_auc": 0.91,
                    "at_best_f1_threshold": {"f1": 0.5}},
            "overfit_gap_pr_auc": 0.1932,
            "n_trees_used": 651,
        }
    )[0]
    assert "0.5477" not in reference
    assert "0.5255" in reference


def test_reference_check_reports_a_match():
    text, matches = _format_reference_check(
        {
            "val": {"pr_auc": 0.5244, "roc_auc": 0.8912,
                    "at_best_f1_threshold": {"f1": 0.5}},
            "overfit_gap_pr_auc": 0.2180,
            "n_trees_used": 799,
        }
    )
    assert matches
    assert "MATCH" in text
    assert "0.8917" in text and "0.8912" in text  # reference and observed
    assert "not yet recorded" not in text


def test_reference_check_asks_for_a_missing_roc_auc(monkeypatch):
    """The pending-value path stays exercised: a placeholder ROC-AUC would
    make the check either always pass or always fail."""
    monkeypatch.setitem(REFERENCE_BASELINE, "roc_auc", None)
    text, matches = _format_reference_check(
        {
            "val": {"pr_auc": 0.5251, "roc_auc": 0.8874,
                    "at_best_f1_threshold": {"f1": 0.5}},
            "overfit_gap_pr_auc": 0.1940,
            "n_trees_used": 648,
        }
    )
    assert matches  # PR-AUC alone still decides
    assert 'REFERENCE_BASELINE["roc_auc"] = 0.8874' in text


def test_reference_check_flags_a_real_divergence():
    text, matches = _format_reference_check(
        {
            "val": {"pr_auc": 0.4100, "roc_auc": 0.85,
                    "at_best_f1_threshold": {"f1": 0.4}},
            "overfit_gap_pr_auc": 0.30,
            "n_trees_used": 120,
        }
    )
    assert not matches
    assert "DIVERGED" in text
    assert "NOT updated automatically" in text


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
