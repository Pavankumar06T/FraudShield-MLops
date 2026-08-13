"""Compare XGBoost regularization settings against the fixed-400 baseline.

Early stopping did not close the overfit gap: it moved from +0.2488 to
+0.2734 while val PR-AUC fell 0.5477 -> 0.5291, using 364 of 2000 trees.
Tree count was never the binding constraint, so tree count was never going
to be the fix. Depth and feature width are the remaining levers.

Two questions, and the config list is built to separate them:

*How much did the carve itself cost?* ``carve_probe`` trains the original
fixed-400 depth-6 configuration on the reduced train -- same trees, same
hyperparameters, 48k fewer rows. Its distance from 0.5477 is the price of
the early-stopping holdout alone, with no regularization change mixed in.
Without this row every later comparison confounds "regularization helped"
with "fewer rows hurt".

*Does regularization close the gap, and what does it cost?* The remaining
configs tighten depth, raise the child-weight floor, narrow the column
sample, and add L2. Each is scored on the same untouched val split.

The table is the deliverable. Lower gap at equal PR-AUC is a clear win;
lower gap at lower PR-AUC is a judgement call, which is why both are shown
side by side rather than collapsed into one number.

    python -m src.training.compare_configs
    python -m src.training.compare_configs --configs carve_probe depth4_reg
    python -m src.training.compare_configs --sample 40000
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

from src.common.config import (
    REPORTS_DIR,
    TRAIN_PARQUET,
    VAL_PARQUET,
    ensure_dirs,
    split_stats,
)
from src.features.build_features import build_features, read_split
from src.training.train import (
    ES_HOLDOUT_FRACTION,
    N_ESTIMATORS_CEILING,
    XGB_HYPERPARAMETERS,
    evaluate,
    scale_pos_weight,
    temporal_holdout,
    trees_used,
)

COMPARISON_PATH: Path = REPORTS_DIR / "config_comparison.json"

#: Val PR-AUC of the original fixed-400-tree run on the FULL train split.
#: Every row is measured against this.
FIXED_400_PR_AUC: float = 0.5477
FIXED_400_GAP: float = 0.2488

#: Beyond this, carve_probe cannot plausibly be measuring the same data the
#: bar came from -- dropping 15% of rows does not move PR-AUC by a third.
#: Attribution is suppressed rather than printed as if it meant something.
IMPLAUSIBLE_CARVE_DELTA: float = 0.15

#: Overrides applied on top of XGB_HYPERPARAMETERS. ``early_stopping`` False
#: trains the stated tree count out in full.
CONFIGS: dict[str, dict] = {
    # Not a candidate -- a measurement. Same settings that produced 0.5477,
    # trained on the reduced split, so the drop is attributable to the carve
    # rather than to anything else in the comparison.
    "carve_probe": {
        "early_stopping": False,
        "n_estimators": 400,
        "max_depth": 6,
    },
    # What the current code does, for continuity with the reported numbers.
    "depth6_current": {},
    # Shallower trees, a real child-weight floor so leaves cannot be carved
    # from a handful of positives, narrower column sampling, and L2 well
    # above XGBoost's default of 1.0.
    "depth4_reg": {
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_lambda": 10.0,
    },
    # The same direction, pushed further. Included to show where the curve
    # turns -- if this one loses PR-AUC without closing the gap further,
    # depth4_reg is near the useful limit.
    "depth3_strong": {
        "max_depth": 3,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_lambda": 50.0,
        "reg_alpha": 1.0,
    },
}


def build_params(overrides: dict) -> tuple[dict, bool]:
    """Merge overrides onto the production config; return (params, stopping)."""
    overrides = dict(overrides)
    stopping = bool(overrides.pop("early_stopping", True))
    params = {**XGB_HYPERPARAMETERS, **overrides}
    if not stopping:
        params.pop("early_stopping_rounds", None)
    return params, stopping


def run_config(
    name: str,
    overrides: dict,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_stop: pd.DataFrame,
    y_stop: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Fit one configuration and score it on train and the untouched val."""
    params, stopping = build_params(overrides)
    model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight(y_fit))

    if stopping:
        model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
    else:
        model.fit(X_fit, y_fit, verbose=False)

    val = evaluate(model, X_val, y_val)
    train = evaluate(model, X_fit, y_fit)
    used = trees_used(model)

    return {
        "config": name,
        "overrides": overrides,
        "early_stopping": stopping,
        "n_trees_used": used,
        "n_estimators_ceiling": int(params["n_estimators"]),
        "hit_ceiling": bool(stopping and used >= int(params["n_estimators"])),
        "val_pr_auc": val["pr_auc"],
        "val_roc_auc": val["roc_auc"],
        "val_best_f1": val["at_best_f1_threshold"]["f1"],
        "train_pr_auc": train["pr_auc"],
        "overfit_gap_pr_auc": float(train["pr_auc"] - val["pr_auc"]),
        "delta_vs_fixed_400": float(val["pr_auc"] - FIXED_400_PR_AUC),
        "gap_delta_vs_fixed_400": float(
            (train["pr_auc"] - val["pr_auc"]) - FIXED_400_GAP
        ),
        "val": val,
    }


def format_table(rows: list[dict]) -> str:
    """The deliverable: score and gap side by side, both against fixed-400."""
    lines = [
        "",
        "=" * 92,
        f"  {'config':<16}{'trees':>7}{'val PR-AUC':>12}{'vs .5477':>10}"
        f"{'train gap':>11}{'vs .2488':>10}{'ROC-AUC':>10}{'best F1':>9}",
        "  " + "-" * 88,
    ]
    for row in rows:
        marker = "*" if row["config"] == "carve_probe" else " "
        lines.append(
            f" {marker}{row['config']:<16}{row['n_trees_used']:>7,}"
            f"{row['val_pr_auc']:>12.4f}{row['delta_vs_fixed_400']:>+10.4f}"
            f"{row['overfit_gap_pr_auc']:>+11.4f}"
            f"{row['gap_delta_vs_fixed_400']:>+10.4f}"
            f"{row['val_roc_auc']:>10.4f}{row['val_best_f1']:>9.4f}"
        )
    lines += [
        "  " + "-" * 88,
        f"  fixed-400 on FULL train:      {FIXED_400_PR_AUC:.4f}            "
        f"{FIXED_400_GAP:+.4f}          (the bar)",
        "=" * 92,
        "  * carve_probe is a measurement, not a candidate: identical settings to",
        "    the fixed-400 run, trained on the reduced split. Its gap from 0.5477",
        "    is the cost of the holdout alone.",
    ]
    return "\n".join(lines)


def summarise(rows: list[dict]) -> str:
    """Attribute the PR-AUC loss and name the best gap/score tradeoff."""
    by_name = {row["config"]: row for row in rows}
    lines = [""]

    probe = by_name.get("carve_probe")
    current = by_name.get("depth6_current")
    if probe:
        # Signed the same way as the table's "vs .5477" column throughout:
        # negative is worse than the bar. Reporting a loss as a positive
        # "cost" here would contradict the row directly above it.
        carve_delta = probe["val_pr_auc"] - FIXED_400_PR_AUC
        if abs(carve_delta) > IMPLAUSIBLE_CARVE_DELTA:
            lines += [
                f"  carve_probe scored {probe['val_pr_auc']:.4f} against a "
                f"{FIXED_400_PR_AUC:.4f} bar ({carve_delta:+.4f}).",
                "  Removing 15% of rows cannot move PR-AUC that far. This is almost",
                "  certainly not the dataset the bar was measured on, so every",
                "  'vs .5477' figure above is comparing unlike things. Attribution",
                "  suppressed.",
            ]
            return "\n".join(lines + ["", "  REFERENCE_BASELINE is unchanged."])

        lines.append(
            f"  Carve cost: {carve_delta:+.4f} PR-AUC from removing "
            f"{ES_HOLDOUT_FRACTION * 100:.0f}% of train alone."
        )
        if current:
            total_delta = current["val_pr_auc"] - FIXED_400_PR_AUC
            stopping_delta = total_delta - carve_delta
            lines.append(
                f"  Total drop to depth6_current: {total_delta:+.4f}, of which "
                f"{carve_delta:+.4f} is the carve and {stopping_delta:+.4f} is "
                "early stopping choosing fewer trees."
            )

    candidates = [row for row in rows if row["config"] != "carve_probe"]
    if candidates:
        best_score = max(candidates, key=lambda r: r["val_pr_auc"])
        best_gap = min(candidates, key=lambda r: r["overfit_gap_pr_auc"])
        lines += [
            "",
            f"  Highest val PR-AUC:  {best_score['config']} "
            f"({best_score['val_pr_auc']:.4f}, gap "
            f"{best_score['overfit_gap_pr_auc']:+.4f})",
            f"  Smallest train gap:  {best_gap['config']} "
            f"({best_gap['overfit_gap_pr_auc']:+.4f}, PR-AUC "
            f"{best_gap['val_pr_auc']:.4f})",
        ]
        if best_score["config"] == best_gap["config"]:
            lines.append("  Same config wins both -- no tradeoff to weigh.")
        else:
            lines.append(
                f"  Different configs -- {best_gap['config']} gives up "
                f"{best_score['val_pr_auc'] - best_gap['val_pr_auc']:.4f} PR-AUC to "
                f"close {best_score['overfit_gap_pr_auc'] - best_gap['overfit_gap_pr_auc']:.4f} "
                "of gap. Your call."
            )

    lines += [
        "",
        "  REFERENCE_BASELINE is unchanged. Nothing here is promoted automatically.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare XGBoost regularization settings on the real splits."
    )
    parser.add_argument("--sample", type=int, default=None, metavar="N")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(CONFIGS),
        default=list(CONFIGS),
        metavar="NAME",
        help=f"subset to run (default all: {', '.join(CONFIGS)})",
    )
    parser.add_argument(
        "--holdout-fraction", type=float, default=ES_HOLDOUT_FRACTION, metavar="F"
    )
    args = parser.parse_args(argv)
    sampling = args.sample is not None

    ensure_dirs()
    stats = split_stats()

    print(f"loading  {TRAIN_PARQUET}")
    train_frame = read_split(TRAIN_PARQUET, args.sample)
    if not sampling:
        stats["train"].assert_matches(train_frame)

    fit_frame, stop_frame, cutoff = temporal_holdout(train_frame, args.holdout_fraction)
    del train_frame
    gc.collect()

    X_fit, y_fit, encoders = build_features(fit_frame)
    X_stop, y_stop, _ = build_features(stop_frame, encoders)
    del fit_frame, stop_frame
    gc.collect()

    print(f"loading  {VAL_PARQUET}")
    val_frame = read_split(VAL_PARQUET, args.sample)
    if not sampling:
        stats["val"].assert_matches(val_frame)
    X_val, y_val, _ = build_features(val_frame, encoders)
    del val_frame
    gc.collect()

    print(
        f"\n  reduced train {len(X_fit):,} rows ({y_fit.mean() * 100:.3f}% fraud)\n"
        f"  stopping slice {len(X_stop):,} rows ({y_stop.mean() * 100:.3f}% fraud)\n"
        f"  val            {len(X_val):,} rows ({y_val.mean() * 100:.3f}% fraud), "
        "untouched\n"
        f"  {X_fit.shape[1]} features, carve at TransactionDT >= {cutoff:,.0f}"
    )

    rows = []
    for index, name in enumerate(args.configs, start=1):
        overrides = CONFIGS[name]
        detail = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "production defaults"
        print(f"\n[{index}/{len(args.configs)}] {name}: {detail}")
        row = run_config(
            name, overrides, X_fit, y_fit, X_stop, y_stop, X_val, y_val
        )
        rows.append(row)
        note = "  HIT CEILING" if row["hit_ceiling"] else ""
        print(
            f"    {row['n_trees_used']:,} trees{note}  val PR-AUC "
            f"{row['val_pr_auc']:.4f}  gap {row['overfit_gap_pr_auc']:+.4f}"
        )

    print(format_table(rows))
    print(summarise(rows))

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampled": sampling,
        "sample_rows": args.sample,
        "fixed_400_reference": {"pr_auc": FIXED_400_PR_AUC, "gap": FIXED_400_GAP},
        "data": {
            "reduced_train_rows": int(len(X_fit)),
            "stopping_slice_rows": int(len(X_stop)),
            "val_rows": int(len(X_val)),
            "n_features": int(X_fit.shape[1]),
            "holdout_fraction": float(args.holdout_fraction),
            "cutoff_transaction_dt": cutoff,
        },
        "results": rows,
    }
    if sampling:
        print(
            f"\nNOT writing {COMPARISON_PATH.name}: fitted on a "
            f"{args.sample:,}-row sample, so the numbers are not comparable to "
            "the 0.5477 bar."
        )
        return 0

    COMPARISON_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote    {COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
