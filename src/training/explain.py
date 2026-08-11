"""SHAP explanations for the baseline model.

Two views of the same numbers:

*Global* -- mean |SHAP| per feature, which says what the model relies on
across the population. This is the model-card view.

*Per-prediction* -- the decomposition behind one decision, which is what a
regulator or a disputing customer actually asks for. "The model said so" is
not an answer; "flagged because the browser string was one never seen during
training, and the amount was 12x this card's usual" is.

SHAP values live in log-odds (margin) space and are exactly additive:

    sigmoid(base_value + sum(shap_values)) == predict_proba

That identity is asserted on every run. It is the difference between an
explanation and a plausible-looking picture -- if the parts do not sum to
the whole, the breakdown is decorative.

Encoded categoricals are mapped back through the fitted encoders, so a
contribution reads ``id_31 = "chrome 99"`` rather than ``id_31 = 47``. The
integer is what the model saw; the string is what happened. Three cases the
raw codes cannot express on their own are named explicitly: a level absent
when the encoder was fitted shows as ``<unseen>``, a missing value as
``<missing>``, and a code with no known level as ``<code N>``.

    python -m src.training.explain
    python -m src.training.explain --rows 500 --examples 5 --model lightgbm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.common.config import REPORTS_DIR, VAL_PARQUET, ensure_dirs
from src.features.build_features import FeatureEncoders, build_features, read_split
from src.training.train import (
    DEFAULT_THRESHOLD,
    LGB_MODEL_PATH,
    METRICS_PATH,
    XGB_MODEL_PATH,
    positive_proba,
)

GLOBAL_IMPORTANCE_PATH: Path = REPORTS_DIR / "shap_global_importance.csv"

#: Rows scored for SHAP. TreeExplainer is exact and fast on tree models, but
#: cost grows with rows x trees x depth -- 200 is plenty to rank features
#: stably and keeps a full run interactive.
DEFAULT_ROWS: int = 200

MISSING_LABEL: str = "<missing>"
UNSEEN_LABEL: str = "<unseen>"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def format_number(value: float) -> str:
    """Render a feature value the way an analyst reading a case would.

    A plain ``%g`` is wrong here in both directions: it flips to scientific
    notation at five digits, so a 31,937 transaction amount -- the top of
    the IEEE-CIS range and the single most-inspected number in any fraud
    explanation -- prints as ``3.194e+04``; and at four significant digits
    it renders 1234.5 as ``1,234``, quietly dropping the half.

    So: integers as integers, ordinary magnitudes with thousands separators
    and no trailing zeros, and scientific notation reserved for values too
    small to read any other way (the V-columns reach there).
    """
    if not np.isfinite(value):
        return str(value)
    if float(value) == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 1e-3:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:.3e}"


class LabelDecoder:
    """Maps ordinal codes back to the strings they were fitted from."""

    def __init__(self, encoders: FeatureEncoders):
        self._inverse = {
            column: {code: level for level, code in mapping.items()}
            for column, mapping in encoders.mappings.items()
        }
        self._unknown = encoders.unknown_code

    def is_categorical(self, column: str) -> bool:
        return column in self._inverse

    def decode(self, column: str, value: float) -> str:
        """Render one cell as the analyst should read it, not as stored."""
        if pd.isna(value):
            return MISSING_LABEL
        if not self.is_categorical(column):
            return format_number(value)
        if float(value) == float(self._unknown):
            return UNSEEN_LABEL
        level = self._inverse[column].get(int(value))
        return f'"{level}"' if level is not None else f"<code {int(value)}>"


def load_model(kind: str):
    """Load a persisted baseline model by name."""
    if kind == "xgboost":
        import xgboost as xgb

        if not XGB_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {XGB_MODEL_PATH}. Train one first:\n"
                "    python -m src.training.train"
            )
        model = xgb.XGBClassifier()
        model.load_model(XGB_MODEL_PATH)
        return model

    import lightgbm as lgb

    if not LGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {LGB_MODEL_PATH}. Train one first:\n"
            "    python -m src.training.train"
        )
    return lgb.Booster(model_file=str(LGB_MODEL_PATH))


def operating_threshold() -> tuple[float, str]:
    """The swept best-F1 threshold if a metrics file exists, else 0.5.

    Explaining decisions at 0.5 would misrepresent the system: with
    ``scale_pos_weight`` inflating probabilities, 0.5 is not the point at
    which anything is actually blocked.
    """
    if METRICS_PATH.exists():
        try:
            record = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            models = record.get("models", {})
            block = models.get("xgboost") or next(iter(models.values()))
            return (
                float(block["val"]["at_best_f1_threshold"]["threshold"]),
                "best-F1 from baseline_metrics.json",
            )
        except (KeyError, ValueError, StopIteration):
            pass
    return DEFAULT_THRESHOLD, "default (no metrics file found)"


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeSHAP values and per-row base values, in margin space.

    Binary tree models return a 2-D (rows, features) array here, but some
    SHAP/model combinations return a 3-D array with a trailing class axis.
    The positive-class slice is taken when that happens so callers see one
    shape.
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[..., 1]
        base = base[..., 1] if base.ndim > 1 else base
    if base.ndim == 0:
        base = np.full(len(X), float(base))
    return values, base


def check_additivity(
    values: np.ndarray, base: np.ndarray, proba: np.ndarray, tolerance: float = 1e-4
) -> float:
    """Assert the decomposition reconstructs the model's own output."""
    reconstructed = sigmoid(base + values.sum(axis=1))
    largest = float(np.max(np.abs(reconstructed - proba)))
    if largest > tolerance:
        raise RuntimeError(
            f"SHAP values do not sum back to the model's predictions "
            f"(max error {largest:.2e}). The breakdown cannot be trusted."
        )
    return largest


def global_importance(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature, with the average signed push alongside.

    Magnitude says how much the model leans on a feature; the signed mean
    says which way it usually leans. A feature can be important and roughly
    neutral on average -- that means it matters per-row rather than as a
    blanket risk marker.
    """
    frame = pd.DataFrame(
        {
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "max_abs_shap": np.abs(values).max(axis=0),
        },
        index=pd.Index(columns, name="feature"),
    )
    total = frame["mean_abs_shap"].sum()
    frame["share_pct"] = 100.0 * frame["mean_abs_shap"] / total if total else 0.0
    return frame.sort_values("mean_abs_shap", ascending=False)


def format_global(importance: pd.DataFrame, decoder: LabelDecoder, top: int) -> str:
    lines = [
        "",
        f"Global importance -- mean |SHAP| over the scored rows (top {top})",
        "",
        f"  {'':<4}{'feature':<20}{'mean|SHAP|':>12}{'share':>8}{'avg push':>11}  direction",
        "  " + "-" * 68,
    ]
    for rank, (feature, row) in enumerate(importance.head(top).iterrows(), start=1):
        direction = "toward fraud" if row["mean_shap"] > 0 else "toward legitimate"
        marker = " (cat)" if decoder.is_categorical(feature) else ""
        lines.append(
            f"  {rank:<4}{feature + marker:<20}{row['mean_abs_shap']:>12.4f}"
            f"{row['share_pct']:>7.1f}%{row['mean_shap']:>+11.4f}  {direction}"
        )
    return "\n".join(lines)


def format_breakdown(
    X: pd.DataFrame,
    values: np.ndarray,
    base: np.ndarray,
    proba: np.ndarray,
    position: int,
    decoder: LabelDecoder,
    threshold: float,
    top: int,
) -> str:
    """One transaction's decision, decomposed into named contributions."""
    row_values = values[position]
    order = np.argsort(-np.abs(row_values))
    shown, remainder = order[:top], order[top:]

    decision = "BLOCK" if proba[position] >= threshold else "ALLOW"
    lines = [
        "",
        f"  row {X.index[position]}   P(fraud) {proba[position]:.4f}   ->  {decision}",
        "",
        f"    {'base (population log-odds)':<38}{base[position]:>10.4f}",
    ]
    for index in shown:
        feature = X.columns[index]
        rendered = decoder.decode(feature, X.iloc[position, index])
        label = f"{feature} = {rendered}"
        if len(label) > 36:
            label = label[:33] + "..."
        lines.append(f"    {label:<38}{row_values[index]:>+10.4f}")

    if len(remainder):
        lines.append(
            f"    {f'+ {len(remainder)} other features':<38}"
            f"{row_values[remainder].sum():>+10.4f}"
        )

    margin = base[position] + row_values.sum()
    lines += [
        "    " + "-" * 48,
        f"    {'= margin':<38}{margin:>10.4f}   ->  P(fraud) "
        f"{sigmoid(np.array([margin]))[0]:.4f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHAP explanations for the baseline.")
    parser.add_argument("--model", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, metavar="N")
    parser.add_argument("--examples", type=int, default=3, metavar="N")
    parser.add_argument("--top", type=int, default=12, metavar="N")
    args = parser.parse_args(argv)

    ensure_dirs()
    encoders = FeatureEncoders.load()
    decoder = LabelDecoder(encoders)
    model = load_model(args.model)

    print(f"loading  {VAL_PARQUET}  (first {args.rows:,} rows)")
    frame = read_split(VAL_PARQUET, args.rows)
    X, y, _ = build_features(frame, encoders)

    proba = positive_proba(model, X)
    threshold, threshold_source = operating_threshold()

    print(
        f"\nmodel {args.model}, {len(X):,} rows x {X.shape[1]} features\n"
        f"  threshold {threshold:.4f}  ({threshold_source})\n"
        f"  computing exact TreeSHAP ..."
    )
    values, base = shap_values(model, X)
    error = check_additivity(values, base, proba)
    print(
        f"  additivity verified: base + sum(SHAP) reconstructs every "
        f"prediction to {error:.1e}"
    )

    importance = global_importance(values, list(X.columns))
    print(format_global(importance, decoder, args.top))

    importance.to_csv(GLOBAL_IMPORTANCE_PATH)
    print(f"\nwrote    {GLOBAL_IMPORTANCE_PATH}")

    flagged = np.flatnonzero(proba >= threshold)
    ranked = flagged[np.argsort(-proba[flagged])]
    label = "flagged"
    if not len(ranked):
        ranked = np.argsort(-proba)
        label = "highest-scoring (none cleared the threshold)"

    header = f"Per-prediction breakdown -- {min(args.examples, len(ranked))} {label}"
    print(f"\n{header}\n{'=' * len(header)}")
    if y is not None:
        print(f"  ({len(flagged)} of {len(X)} rows flagged at this threshold)")

    for position in ranked[: args.examples]:
        print(
            format_breakdown(
                X, values, base, proba, int(position), decoder, threshold, args.top
            )
        )
        if y is not None:
            outcome = "fraud" if y.iloc[int(position)] == 1 else "legitimate"
            print(f"    actual: {outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
