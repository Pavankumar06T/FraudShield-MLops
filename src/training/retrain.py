"""Drift-triggered retraining onto a recent window.

Differs from ``train.py`` in exactly one respect that matters: the training
data is a configurable recent window of everything labelled, rather than the
fixed Phase 0 train slice. Everything else -- hyperparameters, the temporal
carve, early stopping, the metric block, MLflow logging -- is imported from
``train.py`` rather than restated, so a retrained model and the baseline are
measured the same way and remain comparable.

**The decision is read, never re-derived.** ``latest_retrain_trigger()``
returns the alert row the monitor wrote, and its PSI, drifting features and
window boundaries are logged as run parameters. Any model in the registry
can therefore be traced back to the specific drift measurement that caused
it, months later, without recomputing anything. Re-deriving the decision
here would let the retrain and the alert disagree about why the retrain
happened -- and the alert is the auditable record.

Only genuine value drift reaches this module. ``retrain_triggered`` is set
solely for drift whose populated values moved; missingness-driven drift gets
``investigate_pipeline`` and never appears in the trigger query. Retraining
on a coverage change would bake a transient upstream join into the model.

**Always Staging, never Production.** A drift-triggered retrain is a
candidate, not a promotion. Phase 7's shadow comparison decides whether it
is better than what is live, and that decision belongs to the A/B, not to
the thing that produced the challenger.

**Threads are asserted, not assumed.** ``n_jobs`` is checked on the fitted
estimators at runtime rather than trusted from the config, because a
retrain that runs on a differently-sized machine would produce a model
differing for reasons unrelated to the drift -- and the shadow comparison
would then measure the core count.

    python -m src.training.retrain --window-days 120
    python -m src.training.retrain --check-only
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import (
    REPORTS_DIR,
    STREAM_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
    ensure_dirs,
)
from src.drift import store
from src.features.build_features import build_features
from src.training import tracking
from src.training.train import (
    EARLY_STOPPING_ROUNDS,
    ES_HOLDOUT_FRACTION,
    LGB_EVAL_METRIC,
    LGB_HYPERPARAMETERS,
    N_ESTIMATORS_CEILING,
    N_JOBS,
    TIME_COLUMN,
    XGB_HYPERPARAMETERS,
    ensemble_proba,
    environment_params,
    evaluate,
    evaluate_proba,
    log_training_runs,
    positive_proba,
    scale_pos_weight,
    temporal_holdout,
    train_lightgbm,
    train_xgboost,
    trees_used,
)

#: Days of labelled history to retrain on. Long enough to hold the drift the
#: monitor saw, short enough that the model is not still fitting behaviour
#: the fraudsters have abandoned.
DEFAULT_WINDOW_DAYS: float = 120.0

#: Slice of the window held back for final evaluation, taken from the end.
#: The early-stopping carve is taken from what remains, so the three parts
#: are disjoint and strictly ordered in time.
EVAL_FRACTION: float = 0.15

SECONDS_PER_DAY: int = 86_400

RETRAIN_REPORT_PATH: Path = REPORTS_DIR / "retrain_metrics.json"

#: Never Production. A challenger is not a promotion.
TARGET_STAGE: str = "Staging"
TARGET_ALIAS: str = "challenger"


class ThreadPinningError(RuntimeError):
    """Raised when a fitted model did not use the pinned thread count."""


def assert_threads_pinned(*models) -> int:
    """Verify the *fitted* estimators used the pinned thread count.

    Checked on the models rather than on the config because that is the
    only thing that proves it: a config value can be overridden by an
    environment variable, a stale import, or a caller passing n_jobs
    explicitly, and every one of those failures is silent.

    XGBoost's hist method is not thread-deterministic -- its subsample and
    colsample RNG streams are per-thread -- so a retrain at the wrong thread
    count produces a model that differs for reasons having nothing to do
    with the drift that triggered it. The Phase 7 shadow comparison would
    then be measuring the runner's core count.
    """
    if XGB_HYPERPARAMETERS.get("n_jobs") != N_JOBS or N_JOBS in (-1, None):
        raise ThreadPinningError(
            f"XGB_HYPERPARAMETERS['n_jobs'] is "
            f"{XGB_HYPERPARAMETERS.get('n_jobs')!r} and N_JOBS is {N_JOBS!r}; "
            "-1 resolves to the host core count and is never reproducible."
        )
    for model in models:
        effective = model.get_params().get("n_jobs")
        if effective != N_JOBS:
            raise ThreadPinningError(
                f"{type(model).__name__} fitted with n_jobs={effective!r}, "
                f"expected {N_JOBS}. The model is not reproducible on another "
                "machine and must not be registered."
            )
    return N_JOBS


@dataclass
class RetrainWindow:
    """The labelled rows a retrain fits on, split three ways in time."""

    fit: pd.DataFrame
    stop: pd.DataFrame
    evaluation: pd.DataFrame
    label: str
    start_dt: float
    end_dt: float
    sources: list[str]

    @property
    def rows(self) -> int:
        return len(self.fit) + len(self.stop) + len(self.evaluation)


def load_recent_window(
    window_days: float = DEFAULT_WINDOW_DAYS,
    eval_fraction: float = EVAL_FRACTION,
    holdout_fraction: float = ES_HOLDOUT_FRACTION,
) -> RetrainWindow:
    """Everything labelled within the last ``window_days``, split in time.

    Train and stream are concatenated because both carry ``isFraud`` and
    both are history by the time a retrain fires; the fixed Phase 0
    boundary was a modelling decision for the baseline, not a property of
    the data.

    The split is three-way and strictly ordered: earliest rows fit, the next
    slice stops, the most recent slice evaluates. Evaluating on the *newest*
    data is the whole point -- a retrain exists because recent behaviour
    changed, so a score against older rows would answer the wrong question.
    The original val slice is deliberately not reused: it sits inside this
    window and would be leakage.
    """
    frames, sources = [], []
    for path in (TRAIN_PARQUET, VAL_PARQUET, STREAM_PARQUET):
        if path.exists():
            frames.append(pd.read_parquet(path))
            sources.append(path.name)
    if not frames:
        raise FileNotFoundError("No labelled splits available to retrain on.")

    everything = pd.concat(frames, ignore_index=True)
    everything = everything.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    times = everything[TIME_COLUMN]
    span = float(times.max() - times.min())
    cutoff = float(times.max()) - min(window_days * SECONDS_PER_DAY, span)
    window = everything[times >= cutoff].reset_index(drop=True)

    if len(window) < 1000:
        raise ValueError(
            f"Window of {window_days:g} days selected only {len(window)} rows; "
            "too few to retrain on."
        )

    remainder, evaluation, _ = temporal_holdout(window, eval_fraction)
    fit, stop, _ = temporal_holdout(remainder, holdout_fraction)

    return RetrainWindow(
        fit=fit,
        stop=stop,
        evaluation=evaluation,
        label=f"last {window_days:g} days",
        start_dt=float(window[TIME_COLUMN].min()),
        end_dt=float(window[TIME_COLUMN].max()),
        sources=sources,
    )


def trigger_params(alert: dict) -> dict[str, object]:
    """Alert fields worth carrying onto the run, so the model is traceable.

    Recorded verbatim from the alert rather than recomputed: the point is
    that the model and the measurement agree about why it exists.
    """
    top = json.loads(alert.get("top_features") or "[]")
    value_features = [f["feature"] for f in top if f.get("type") == "value"]
    return {
        "trigger.alert_id": alert["id"],
        "trigger.created_at": alert["created_at"],
        "trigger.verdict": alert["verdict"],
        "trigger.overall_psi": round(float(alert["overall_psi"]), 6),
        "trigger.overall_psi_feature": alert["overall_psi_feature"],
        "trigger.value_drift_psi": round(float(alert["value_drift_psi"]), 6),
        "trigger.value_drift_features": alert["value_drift_features"],
        "trigger.missingness_psi": round(float(alert["missingness_psi"]), 6),
        "trigger.missingness_features": alert["missingness_features"],
        "trigger.threshold": alert["threshold"],
        "trigger.drifting_features": ",".join(value_features[:20]) or "(none recorded)",
        "trigger.window_label": alert["window_label"],
        "trigger.window_start_dt": alert["window_start_dt"],
        "trigger.window_end_dt": alert["window_end_dt"],
        "trigger.window_rows": alert["window_rows"],
    }


def register_challenger(run_id: str, alert_id: int, metrics: dict) -> dict | None:
    """Register the retrained model as a new Staging version.

    Never Production, and the guard is explicit rather than implied: this
    module produces a challenger, and only the shadow comparison decides
    whether a challenger becomes champion.
    """
    if TARGET_STAGE.lower() == "production":
        raise RuntimeError("retrain must never register straight to Production")

    mlflow = tracking.mlflow_module()
    if mlflow is None:
        print("  MLflow unavailable; model not registered")
        return None

    mlflow.set_tracking_uri(tracking.tracking_uri())
    from mlflow import MlflowClient

    client = MlflowClient()
    name = tracking.REGISTERED_MODEL_NAME

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        version = mlflow.register_model(f"runs:/{run_id}/model", name)
        client.update_model_version(
            name=name,
            version=version.version,
            description=(
                f"Drift-triggered retrain answering alert #{alert_id}.\n"
                f"  eval PR-AUC   {metrics['pr_auc']:.4f}\n"
                f"  eval ROC-AUC  {metrics['roc_auc']:.4f}\n"
                f"  n_jobs        {N_JOBS} (pinned)\n\n"
                "Registered to Staging as a challenger. Promotion to Production "
                "is the shadow A/B's decision, not this run's."
            ),
        )
        client.set_model_version_tag(name, version.version, "trigger_alert_id", str(alert_id))
        client.set_model_version_tag(name, version.version, "origin", "drift_retrain")
        try:
            client.transition_model_version_stage(
                name=name, version=version.version, stage=TARGET_STAGE
            )
        except Exception:
            pass
        client.set_registered_model_alias(name, TARGET_ALIAS, version.version)

    return {"name": name, "version": str(version.version), "stage": TARGET_STAGE,
            "alias": TARGET_ALIAS}


def retrain(
    window_days: float = DEFAULT_WINDOW_DAYS,
    alert: dict | None = None,
    db_path: Path = store.DEFAULT_DB_PATH,
    register: bool = True,
) -> dict:
    """Fit on a recent window, log, and register a Staging challenger."""
    ensure_dirs()

    if alert is None:
        alert = store.latest_retrain_trigger(db_path)
    if alert is None:
        raise RuntimeError(
            "No unresolved retrain-triggering alert. Retraining without a "
            "recorded cause would produce a model nothing can explain."
        )

    print(
        f"answering alert #{alert['id']} ({alert['created_at']})\n"
        f"  verdict          {alert['verdict']}\n"
        f"  value-drift PSI  {alert['value_drift_psi']:.4f} over "
        f"{alert['value_drift_features']} features\n"
        f"  worst feature    {alert['overall_psi_feature']} at "
        f"{alert['overall_psi']:.4f}"
    )

    window = load_recent_window(window_days)
    print(
        f"\nwindow  {window.label}  {window.rows:,} rows from "
        f"{', '.join(window.sources)}\n"
        f"  fit        {len(window.fit):>8,} rows  "
        f"{window.fit['isFraud'].mean() * 100:.3f}% fraud\n"
        f"  stop       {len(window.stop):>8,} rows  "
        f"{window.stop['isFraud'].mean() * 100:.3f}% fraud\n"
        f"  evaluate   {len(window.evaluation):>8,} rows  "
        f"{window.evaluation['isFraud'].mean() * 100:.3f}% fraud  (most recent)"
    )

    X_fit, y_fit, encoders = build_features(window.fit)
    X_stop, y_stop, _ = build_features(window.stop, encoders)
    X_eval, y_eval, _ = build_features(window.evaluation, encoders)
    del window.fit, window.stop
    gc.collect()

    weight = scale_pos_weight(y_fit)
    print(
        f"\n  {X_fit.shape[1]} features, scale_pos_weight {weight:.2f}\n"
        f"  fitting at n_jobs={N_JOBS} (pinned) ..."
    )

    xgb_model = train_xgboost(X_fit, y_fit, X_stop, y_stop)
    lgb_model = train_lightgbm(X_fit, y_fit, X_stop, y_stop)
    assert_threads_pinned(xgb_model, lgb_model)
    print(f"  thread pinning verified on both fitted models: n_jobs={N_JOBS}")

    blocks: dict[str, dict] = {}
    for name, model, params in (
        ("xgboost", xgb_model, XGB_HYPERPARAMETERS),
        ("lightgbm", lgb_model, LGB_HYPERPARAMETERS),
    ):
        val = evaluate(model, X_eval, y_eval)
        train_metrics = evaluate(model, X_fit, y_fit)
        blocks[name] = {
            "val": val,
            "train": train_metrics,
            "overfit_gap_pr_auc": float(train_metrics["pr_auc"] - val["pr_auc"]),
            "n_trees_used": trees_used(model),
            "n_estimators_ceiling": N_ESTIMATORS_CEILING,
            "early_stopped": trees_used(model) < N_ESTIMATORS_CEILING,
            "hit_ceiling": trees_used(model) >= N_ESTIMATORS_CEILING,
            "hyperparameters": params,
            "stopping_slice": evaluate(model, X_stop, y_stop),
        }

    xgb_proba = positive_proba(xgb_model, X_eval)
    lgb_proba = positive_proba(lgb_model, X_eval)
    ensemble_val = evaluate_proba(y_eval, ensemble_proba(xgb_proba, lgb_proba))
    blocks["ensemble"] = {
        "val": ensemble_val,
        "train": evaluate_proba(
            y_fit,
            ensemble_proba(positive_proba(xgb_model, X_fit), positive_proba(lgb_model, X_fit)),
        ),
        "overfit_gap_pr_auc": 0.0,
        "n_trees_used": None,
        "n_estimators_ceiling": None,
        "hyperparameters": None,
        "members": ["xgboost", "lightgbm"],
        "method": "mean of member positive-class probabilities",
        "member_probability_correlation": float(np.corrcoef(xgb_proba, lgb_proba)[0, 1]),
    }
    blocks["ensemble"]["overfit_gap_pr_auc"] = float(
        blocks["ensemble"]["train"]["pr_auc"] - ensemble_val["pr_auc"]
    )

    for name, block in blocks.items():
        val = block["val"]
        trees = f"{block['n_trees_used']:,} trees" if block["n_trees_used"] else "-"
        print(
            f"  {name:<9} PR-AUC {val['pr_auc']:.4f} "
            f"({val['pr_auc_lift_over_floor']:.1f}x floor)  "
            f"ROC-AUC {val['roc_auc']:.4f}  gap {block['overfit_gap_pr_auc']:+.4f}  {trees}"
        )

    shared = {
        **trigger_params(alert),
        "retrain.window_label": window.label,
        "retrain.window_days": window_days,
        "retrain.window_start_dt": window.start_dt,
        "retrain.window_end_dt": window.end_dt,
        "retrain.sources": ",".join(window.sources),
        "retrain.fit_rows": len(X_fit),
        "retrain.stop_rows": len(X_stop),
        "retrain.eval_rows": len(X_eval),
        "retrain.fit_fraud_rate_pct": round(float(y_fit.mean()) * 100, 4),
        "retrain.eval_fraud_rate_pct": round(float(y_eval.mean()) * 100, 4),
        "retrain.n_features": X_fit.shape[1],
        "retrain.scale_pos_weight": round(weight, 6),
        "early_stopping.ceiling": N_ESTIMATORS_CEILING,
        "early_stopping.rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping.monitored": f"aucpr / {LGB_EVAL_METRIC}",
        **environment_params(N_JOBS),
    }

    print(f"\n{tracking.describe_store()}")
    run_ids = log_training_runs(
        blocks,
        {"xgboost": xgb_model, "lightgbm": lgb_model},
        shared,
        artifacts=[],
    )
    for name, run_id in run_ids.items():
        print(f"  logged {name:<9} run {run_id}")

    registered = None
    if register and "xgboost" in run_ids:
        registered = register_challenger(
            run_ids["xgboost"], alert["id"], blocks["xgboost"]["val"]
        )
        if registered:
            print(
                f"\nregistered {registered['name']} v{registered['version']} "
                f"-> {registered['stage']} (@{registered['alias']})"
            )

    resolved = store.mark_resolved(
        alert["id"],
        run_id=run_ids.get("xgboost"),
        model_version=(registered or {}).get("version"),
        path=db_path,
    )
    print(f"alert #{alert['id']} {'marked resolved' if resolved else 'was already resolved'}")

    record = {
        "created_at": store.utc_now(),
        "trigger": trigger_params(alert),
        "window": {
            "label": window.label, "days": window_days, "sources": window.sources,
            "start_transaction_dt": window.start_dt, "end_transaction_dt": window.end_dt,
            "fit_rows": len(X_fit), "stop_rows": len(X_stop), "eval_rows": len(X_eval),
        },
        "n_jobs": N_JOBS,
        "models": blocks,
        "mlflow_run_ids": run_ids,
        "registered": registered,
        "alert_resolved": resolved,
    }
    RETRAIN_REPORT_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote    {RETRAIN_REPORT_PATH}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drift-triggered retraining.")
    parser.add_argument("--window-days", type=float, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--alert-id", type=int, default=None,
                        help="answer a specific alert instead of the newest open one")
    parser.add_argument("--check-only", action="store_true",
                        help="report whether a retrain is due and exit")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    alert = None
    if args.alert_id is not None:
        matches = [a for a in store.open_retrain_alerts(args.db) if a["id"] == args.alert_id]
        if not matches:
            print(f"alert #{args.alert_id} is not an open retrain trigger")
            return 1
        alert = matches[0]
    else:
        alert = store.latest_retrain_trigger(args.db)

    if args.check_only:
        if alert is None:
            print("no unresolved retrain trigger; nothing to do")
            return 1
        print(
            f"retrain due: alert #{alert['id']} ({alert['verdict']}), "
            f"value-drift PSI {alert['value_drift_psi']:.4f} over "
            f"{alert['value_drift_features']} features"
        )
        return 0

    if alert is None:
        print("no unresolved retrain trigger; nothing to do")
        return 0

    try:
        retrain(window_days=args.window_days, alert=alert, db_path=args.db,
                register=not args.no_register)
    except ThreadPinningError as exc:
        print(f"REFUSING TO REGISTER: {exc}")
        return 2
    except Exception as exc:
        print(f"retrain failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
