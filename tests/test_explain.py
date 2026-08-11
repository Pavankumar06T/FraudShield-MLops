"""Invariants for SHAP explanations.

The one that matters most is additivity. A breakdown whose parts do not sum
to the model's own output is not an explanation -- it is a plausible-looking
picture, and it would be presented to a regulator as if it were evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.features.build_features import UNKNOWN_CODE, build_features
from src.training.explain import (
    MISSING_LABEL,
    UNSEEN_LABEL,
    LabelDecoder,
    check_additivity,
    format_breakdown,
    format_number,
    global_importance,
    shap_values,
    sigmoid,
)


def fitted(n: int = 900, seed: int = 4):
    """A small trained model plus the frame and encoders behind it."""
    rng = np.random.default_rng(seed)
    browsers = rng.choice(["chrome 62", "safari 11", "firefox 70"], n)
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 10,
            "isFraud": (rng.random(n) < 1 / (1 + np.exp(-(signal * 2 - 2.2)))).astype(int),
            "TransactionAmt": np.exp(rng.normal(3.0, 1.0, n)),
            "signal": signal,
            "id_31": pd.Series(browsers, dtype="str"),
            "sparse": np.where(rng.random(n) < 0.5, np.nan, rng.normal(size=n)),
        }
    )
    X, y, encoders = build_features(frame)
    model = xgb.XGBClassifier(n_estimators=40, max_depth=3, tree_method="hist")
    model.fit(X, y)
    return model, X, y, encoders


# --------------------------------------------------------------------------
# Additivity -- the property that makes a breakdown evidence
# --------------------------------------------------------------------------


def test_shap_reconstructs_the_models_own_predictions():
    model, X, _, _ = fitted()
    values, base = shap_values(model, X.head(120))
    proba = model.predict_proba(X.head(120))[:, 1]
    assert check_additivity(values, base, proba) < 1e-4


def test_additivity_check_catches_a_corrupted_decomposition():
    model, X, _, _ = fitted()
    values, base = shap_values(model, X.head(50))
    proba = model.predict_proba(X.head(50))[:, 1]
    values[:, 0] += 1.5
    with pytest.raises(RuntimeError, match="do not sum back"):
        check_additivity(values, base, proba)


def test_shap_values_are_two_dimensional_per_row_and_feature():
    model, X, _, _ = fitted()
    values, base = shap_values(model, X.head(30))
    assert values.shape == (30, X.shape[1])
    assert base.shape == (30,)


def test_sigmoid_matches_the_probability_axis():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert sigmoid(np.array([-20.0]))[0] < 1e-8
    assert sigmoid(np.array([20.0]))[0] > 1 - 1e-8


# --------------------------------------------------------------------------
# Decoding codes back to labels
# --------------------------------------------------------------------------


def test_categorical_codes_decode_to_their_original_strings():
    _, _, _, encoders = fitted()
    decoder = LabelDecoder(encoders)

    assert decoder.is_categorical("id_31")
    for level, code in encoders.mappings["id_31"].items():
        assert decoder.decode("id_31", code) == f'"{level}"'


def test_unseen_and_missing_decode_distinctly():
    """They are different facts and the breakdown must not conflate them."""
    _, _, _, encoders = fitted()
    decoder = LabelDecoder(encoders)

    assert decoder.decode("id_31", UNKNOWN_CODE) == UNSEEN_LABEL
    assert decoder.decode("id_31", np.nan) == MISSING_LABEL
    assert decoder.decode("sparse", np.nan) == MISSING_LABEL


def test_unmapped_code_is_named_rather_than_guessed():
    _, _, _, encoders = fitted()
    decoder = LabelDecoder(encoders)
    assert decoder.decode("id_31", 999) == "<code 999>"


def test_numeric_columns_render_as_numbers():
    _, _, _, encoders = fitted()
    decoder = LabelDecoder(encoders)
    assert not decoder.is_categorical("TransactionAmt")
    assert decoder.decode("TransactionAmt", 1234.5) == "1,234.5"
    assert decoder.decode("signal", -0.5) == "-0.5"


@pytest.mark.parametrize(
    "value, expected",
    [
        (37.86, "37.86"),
        (252.4, "252.4"),
        (5236.0, "5,236"),
        # top of the IEEE-CIS TransactionAmt range -- must not go scientific
        (31937.39, "31,937.39"),
        (1234.5, "1,234.5"),
        (-0.5, "-0.5"),
        (0.0, "0"),
        # too small to read any other way
        (0.000123, "1.230e-04"),
    ],
)
def test_number_formatting_stays_readable_across_the_real_range(value, expected):
    assert format_number(value) == expected


def test_number_formatting_survives_non_finite_values():
    assert format_number(float("inf")) == "inf"
    assert format_number(float("nan")) == "nan"


# --------------------------------------------------------------------------
# Global importance
# --------------------------------------------------------------------------


def test_global_importance_ranks_the_real_driver_first():
    model, X, _, _ = fitted()
    values, _ = shap_values(model, X)
    importance = global_importance(values, list(X.columns))
    assert importance.index[0] == "signal"


def test_global_importance_shares_sum_to_one_hundred():
    model, X, _, _ = fitted()
    values, _ = shap_values(model, X)
    importance = global_importance(values, list(X.columns))
    assert importance["share_pct"].sum() == pytest.approx(100.0)
    assert (importance["mean_abs_shap"] >= 0).all()
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_global_importance_separates_magnitude_from_direction():
    """A feature can matter a lot per-row and be neutral on average."""
    model, X, _, _ = fitted()
    values, _ = shap_values(model, X)
    importance = global_importance(values, list(X.columns))
    assert (importance["mean_abs_shap"] >= importance["mean_shap"].abs() - 1e-12).all()


# --------------------------------------------------------------------------
# Per-prediction rendering
# --------------------------------------------------------------------------


def test_breakdown_reconstructs_the_probability_it_prints():
    model, X, _, encoders = fitted()
    values, base = shap_values(model, X.head(60))
    proba = model.predict_proba(X.head(60))[:, 1]
    position = int(np.argmax(proba))

    decoder = LabelDecoder(encoders)
    text = format_breakdown(X.head(60), values, base, proba, position, decoder, 0.5, 2)

    assert f"P(fraud) {proba[position]:.4f}" in text
    # the margin printed at the bottom must be the true total, not the sum of
    # the rows that happened to be shown
    margin = base[position] + values[position].sum()
    assert f"{margin:>10.4f}" in text

    # with 2 of 4 features itemised, the rest is still accounted for
    assert f"+ {X.shape[1] - 2} other features" in text

    # and showing every feature leaves no remainder line at all
    full = format_breakdown(
        X.head(60), values, base, proba, position, decoder, 0.5, X.shape[1]
    )
    assert "other features" not in full


def test_breakdown_shows_browser_strings_not_codes():
    model, X, _, encoders = fitted()
    values, base = shap_values(model, X.head(40))
    proba = model.predict_proba(X.head(40))[:, 1]

    rendered = "\n".join(
        format_breakdown(
            X.head(40), values, base, proba, i, LabelDecoder(encoders), 0.5, 12
        )
        for i in range(10)
    )
    assert any(f'id_31 = "{level}"' in rendered for level in encoders.mappings["id_31"])


def test_breakdown_decision_follows_the_threshold():
    model, X, _, encoders = fitted()
    values, base = shap_values(model, X.head(60))
    proba = model.predict_proba(X.head(60))[:, 1]
    position = int(np.argmax(proba))
    decoder = LabelDecoder(encoders)

    assert "BLOCK" in format_breakdown(
        X.head(60), values, base, proba, position, decoder, 0.0, 4
    )
    assert "ALLOW" in format_breakdown(
        X.head(60), values, base, proba, position, decoder, 1.01, 4
    )
