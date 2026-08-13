"""Invariants for the regularization comparison.

The table is only meaningful if every row differs from the fixed-400
baseline in exactly the way its name claims -- so the merging of overrides
and the carve probe's fidelity to the original settings are what get pinned
here.
"""

from __future__ import annotations

import pytest

from src.training.compare_configs import (
    CONFIGS,
    FIXED_400_GAP,
    FIXED_400_PR_AUC,
    build_params,
    format_table,
    summarise,
)
from src.training.train import N_ESTIMATORS_CEILING, XGB_HYPERPARAMETERS


def test_carve_probe_reproduces_the_pre_promotion_settings():
    """It is a measurement, not a candidate. If it differs from the original
    run in anything but row count, the attribution it exists to provide is
    worthless -- and since the production default is now depth4_reg, every
    one of those settings has to be restated rather than inherited.
    """
    params, stopping = build_params(CONFIGS["carve_probe"])
    assert stopping is False
    assert "early_stopping_rounds" not in params
    assert params["n_estimators"] == 400
    assert params["max_depth"] == 6
    assert params["min_child_weight"] == 1
    assert params["colsample_bytree"] == pytest.approx(0.8)
    assert params["reg_lambda"] == pytest.approx(1.0)
    for shared in ("learning_rate", "subsample", "random_state", "tree_method"):
        assert params[shared] == XGB_HYPERPARAMETERS[shared]


def test_production_config_is_the_promoted_default_unchanged():
    params, stopping = build_params(CONFIGS["production"])
    assert stopping is True
    assert params == XGB_HYPERPARAMETERS


def test_production_is_the_depth4_reg_configuration():
    """The promotion, pinned. These are the settings that scored 0.5255 at a
    +0.1932 gap on real data."""
    params, _ = build_params(CONFIGS["production"])
    assert params["max_depth"] == 4
    assert params["min_child_weight"] == 10
    assert params["subsample"] == pytest.approx(0.8)
    assert params["colsample_bytree"] == pytest.approx(0.6)
    assert params["reg_lambda"] == pytest.approx(10.0)
    assert params["n_estimators"] == N_ESTIMATORS_CEILING
    assert params["early_stopping_rounds"] == 50


def test_legacy_config_is_less_regularized_than_production():
    """depth6_legacy exists so the promotion can be re-derived rather than
    taken on trust; it must really be the looser of the two."""
    legacy, stopping = build_params(CONFIGS["depth6_legacy"])
    production, _ = build_params(CONFIGS["production"])
    assert stopping is True
    assert legacy["max_depth"] > production["max_depth"]
    assert legacy["min_child_weight"] < production["min_child_weight"]
    assert legacy["colsample_bytree"] > production["colsample_bytree"]
    assert legacy["reg_lambda"] < production["reg_lambda"]
    assert legacy["n_estimators"] == N_ESTIMATORS_CEILING


def test_stronger_config_is_strictly_stronger_than_production():
    production, _ = build_params(CONFIGS["production"])
    stronger, _ = build_params(CONFIGS["depth3_strong"])
    assert stronger["max_depth"] < production["max_depth"]
    assert stronger["min_child_weight"] > production["min_child_weight"]
    assert stronger["reg_lambda"] > production["reg_lambda"]
    assert stronger["subsample"] == pytest.approx(0.8)


def test_build_params_does_not_mutate_the_production_config():
    before = dict(XGB_HYPERPARAMETERS)
    build_params(CONFIGS["carve_probe"])
    build_params(CONFIGS["depth3_strong"])
    assert XGB_HYPERPARAMETERS == before


def test_build_params_does_not_mutate_the_config_registry():
    before = {k: dict(v) for k, v in CONFIGS.items()}
    for overrides in CONFIGS.values():
        build_params(overrides)
    assert {k: dict(v) for k, v in CONFIGS.items()} == before


def _row(name, val_pr, train_pr, trees=364):
    return {
        "config": name,
        "n_trees_used": trees,
        "val_pr_auc": val_pr,
        "val_roc_auc": 0.90,
        "val_best_f1": 0.55,
        "train_pr_auc": train_pr,
        "overfit_gap_pr_auc": train_pr - val_pr,
        "delta_vs_fixed_400": val_pr - FIXED_400_PR_AUC,
        "gap_delta_vs_fixed_400": (train_pr - val_pr) - FIXED_400_GAP,
        "hit_ceiling": False,
    }


def test_table_shows_score_and_gap_against_the_bar():
    rows = [_row("carve_probe", 0.5400, 0.7888), _row("depth4_reg", 0.5350, 0.6100)]
    text = format_table(rows)
    assert f"{FIXED_400_PR_AUC:.4f}" in text
    assert "-0.0077" in text  # carve_probe delta vs 0.5477
    assert "the bar" in text
    assert "measurement, not a candidate" in text


def test_summary_attributes_the_drop_between_carve_and_stopping():
    rows = [
        _row("carve_probe", 0.5400, 0.7888),  # carve costs 0.0077
        _row("depth6_legacy", 0.5291, 0.8025),  # total drop 0.0186
    ]
    text = summarise(rows)
    assert "Carve cost: -0.0077" in text
    assert "-0.0186" in text  # total
    assert "-0.0109" in text  # remainder attributable to early stopping


def test_summary_calls_out_early_stopping_costing_nothing():
    """What the real run actually showed: carve_probe and depth6_legacy both
    at 0.5291, so the whole drop is the carve."""
    rows = [
        _row("carve_probe", 0.5291, 0.8146),
        _row("depth6_legacy", 0.5291, 0.8025),
    ]
    text = summarise(rows)
    assert "Carve cost: -0.0186" in text
    assert "Early stopping cost nothing" in text


def test_summary_names_the_tradeoff_when_configs_disagree():
    rows = [
        _row("carve_probe", 0.5400, 0.7888),
        _row("depth6_legacy",0.5450, 0.8184),  # best score, worst gap
        _row("depth4_reg", 0.5300, 0.6100),  # worst score, best gap
    ]
    text = summarise(rows)
    assert "Highest val PR-AUC:  depth6_legacy" in text
    assert "Smallest train gap:  depth4_reg" in text
    assert "Your call." in text
    assert "carve_probe" not in text.split("Highest val PR-AUC")[1]


def test_summary_says_so_when_one_config_wins_both():
    rows = [
        _row("carve_probe", 0.5400, 0.7888),
        _row("depth4_reg", 0.5450, 0.6100),
        _row("depth3_strong", 0.5200, 0.6400),
    ]
    text = summarise(rows)
    assert "no tradeoff to weigh" in text


def test_summary_refuses_to_promote_anything():
    text = summarise([_row("depth4_reg", 0.60, 0.65)])
    assert "REFERENCE_BASELINE is unchanged" in text


def test_summary_suppresses_attribution_on_data_that_is_not_the_bar():
    """Run against a fixture, carve_probe lands nowhere near 0.5477 and the
    attribution arithmetic is meaningless. Say so rather than print it."""
    rows = [_row("carve_probe", 0.0765, 0.9536), _row("depth6_legacy",0.0845, 0.1914)]
    text = summarise(rows)
    assert "Attribution" in text and "suppressed" in text
    assert "Carve cost:" not in text
    assert "Total drop" not in text


def test_summary_still_attributes_within_a_plausible_range():
    rows = [_row("carve_probe", 0.5400, 0.7888), _row("depth6_legacy",0.5291, 0.8025)]
    text = summarise(rows)
    assert "Carve cost: -0.0077" in text
    assert "suppressed" not in text
