"""Read-only access to the stores the dashboard displays.

Every number shown comes from somewhere it was already written: PSI from
``drift_alerts``, promotion verdicts from ``model_promotions`` and the
registry, model metrics from ``baseline_metrics.json``. Nothing is
recalculated.

That is a correctness requirement, not tidiness. A dashboard that recomputed
PSI would eventually disagree with the monitor that fired the alert -- a
different window, a different bin edge, a different epsilon -- and the
operator would be looking at a number nobody can reconcile with the decision
that was actually taken. The only aggregation done here is over raw audit
rows (counting flags, counting unseen), which summarises stored facts rather
than re-deriving a published metric.

Separate from ``app.py`` so it can be tested without a browser.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.config import REPORTS_DIR
from src.drift import store
from src.training import tracking

BASELINE_METRICS_PATH: Path = REPORTS_DIR / "baseline_metrics.json"
RETRAIN_METRICS_PATH: Path = REPORTS_DIR / "retrain_metrics.json"
SHADOW_PATH: Path = REPORTS_DIR / "shadow_comparison.json"


def _table_exists(connection, name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Champion
# --------------------------------------------------------------------------


def champion() -> dict:
    """The model currently deciding, from the registry plus its own run.

    Metrics come from the run that produced the model, not from a fresh
    evaluation -- the point is to show what was promoted and on what
    evidence, which is a historical fact rather than a live measurement.
    """
    mlflow = tracking.mlflow_module()
    if mlflow is None:
        return {"available": False, "reason": "MLflow is not installed"}

    try:
        mlflow.set_tracking_uri(tracking.tracking_uri())
        from mlflow import MlflowClient

        client = MlflowClient()
        name = tracking.REGISTERED_MODEL_NAME
        version = None
        for alias in ("champion", "staging"):
            try:
                version = client.get_model_version_by_alias(name, alias)
                break
            except Exception:
                continue
        if version is None:
            return {"available": False, "reason": "no champion alias is set"}

        run = client.get_run(version.run_id)
        params, metrics = run.data.params, run.data.metrics
        return {
            "available": True,
            "name": name,
            "version": version.version,
            "stage": version.current_stage,
            "run_id": version.run_id,
            "run_name": run.info.run_name,
            "created": datetime.fromtimestamp(
                version.creation_timestamp / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
            "pr_auc": metrics.get("val.pr_auc"),
            "roc_auc": metrics.get("val.roc_auc"),
            "threshold": metrics.get("val.at_best_f1_threshold.threshold"),
            "f1": metrics.get("val.at_best_f1_threshold.f1"),
            "precision": metrics.get("val.at_best_f1_threshold.precision"),
            "recall": metrics.get("val.at_best_f1_threshold.recall"),
            "n_trees": metrics.get("n_trees_used"),
            "overfit_gap": metrics.get("overfit_gap_pr_auc"),
            "n_jobs": params.get("env.n_jobs_effective"),
            "train_rows": params.get("retrain.fit_rows") or params.get("split.train_rows_reduced"),
            "trigger_alert": params.get("trigger.alert_id"),
            "drifting_features": params.get("trigger.drifting_features"),
            "tags": dict(version.tags),
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# Live scoring
# --------------------------------------------------------------------------


def scoring_activity(db_path: Path = store.DEFAULT_DB_PATH) -> dict:
    """Throughput, flag rate and unseen rate from the audit table.

    An idle consumer is a normal state, not an error: the table holds every
    row ever scored, so the panel shows history rather than failing. The
    distinction is surfaced as ``live`` so the display can say which it is.
    """
    empty = {"rows": 0, "live": False, "last_scored": None, "recent": pd.DataFrame()}
    with store.connect(db_path) as connection:
        if not _table_exists(connection, "stream_predictions"):
            return {**empty, "reason": "nothing has been scored yet"}

        summary = connection.execute(
            """
            SELECT COUNT(*) AS rows,
                   SUM(CASE WHEN decision='BLOCK' THEN 1 ELSE 0 END) AS blocked,
                   SUM(CASE WHEN unseen_categories NOT IN ('[]','') THEN 1 ELSE 0 END)
                       AS with_unseen,
                   SUM(CASE WHEN challenger_probability IS NOT NULL THEN 1 ELSE 0 END)
                       AS shadowed,
                   MIN(scored_at) AS first_scored, MAX(scored_at) AS last_scored,
                   AVG(latency_ms) AS avg_latency
            FROM stream_predictions
            """
        ).fetchone()
        if not summary or not summary["rows"]:
            return {**empty, "reason": "nothing has been scored yet"}

        recent = pd.read_sql_query(
            "SELECT scored_at, decision, fraud_probability, model_version, "
            "unseen_categories, true_label, latency_ms "
            "FROM stream_predictions ORDER BY id DESC LIMIT 500",
            connection,
        )
        by_version = pd.read_sql_query(
            "SELECT model_version, COUNT(*) AS rows, "
            "SUM(CASE WHEN decision='BLOCK' THEN 1 ELSE 0 END) AS blocked "
            "FROM stream_predictions GROUP BY model_version ORDER BY rows DESC",
            connection,
        )

    rows = int(summary["rows"])
    last = summary["last_scored"]
    live = False
    if last:
        try:
            age = (
                datetime.now(timezone.utc) - datetime.fromisoformat(last)
            ).total_seconds()
            live = age < 120
        except Exception:
            pass

    # Throughput over the observed span. Reported as an average rather than
    # an instantaneous rate: the audit table records when rows were scored,
    # not a sampled counter, so anything finer would be invented.
    throughput = None
    try:
        span = (
            datetime.fromisoformat(summary["last_scored"])
            - datetime.fromisoformat(summary["first_scored"])
        ).total_seconds()
        if span > 0:
            throughput = rows / span
    except Exception:
        pass

    return {
        "rows": rows,
        "blocked": int(summary["blocked"] or 0),
        "with_unseen": int(summary["with_unseen"] or 0),
        "shadowed": int(summary["shadowed"] or 0),
        "flag_rate": (summary["blocked"] or 0) / rows,
        "unseen_rate": (summary["with_unseen"] or 0) / rows,
        "avg_latency_ms": summary["avg_latency"],
        "throughput_per_s": throughput,
        "first_scored": summary["first_scored"],
        "last_scored": last,
        "live": live,
        "recent": recent,
        "by_version": by_version,
    }


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------


def psi_history(db_path: Path = store.DEFAULT_DB_PATH) -> pd.DataFrame:
    """Every recorded alert, with the two drift types kept apart.

    Read straight from ``drift_alerts``. The monitor already classified each
    feature and stored both maxima; a blended series would erase exactly the
    distinction the Phase 0 decomposition exists to make -- genuine value
    drift needs a retrain, missingness-driven drift needs someone to look at
    the upstream feed.
    """
    with store.connect(db_path) as connection:
        if not _table_exists(connection, "drift_alerts"):
            return pd.DataFrame()
        frame = pd.read_sql_query(
            "SELECT id, created_at, window_label, window_rows, overall_psi, "
            "overall_psi_feature, value_drift_psi, value_drift_features, "
            "missingness_psi, missingness_features, threshold, "
            "retrain_triggered, investigate_pipeline, verdict, "
            "value_drift_feature_list, missingness_feature_list, "
            "resolved_at, resolved_model_version "
            "FROM drift_alerts ORDER BY id",
            connection,
        )
    if not frame.empty:
        frame["created_at"] = pd.to_datetime(frame["created_at"], format="mixed", utc=True)
    return frame


def unseen_categories(db_path: Path = store.DEFAULT_DB_PATH) -> pd.DataFrame:
    """The fast drift signal, aggregated by feature.

    Uses ``store.aggregate_unseen`` rather than its own query, so the
    dashboard and the monitor report the same counts from the same code.
    """
    try:
        rows = store.aggregate_unseen(path=db_path)
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["examples"] = frame["top_values"].apply(
        lambda values: ", ".join(f"{v['value']} ({v['n']})" for v in values[:3])
    )
    return frame.drop(columns=["top_values"])


# --------------------------------------------------------------------------
# Promotion history -- the evidence the system declines bad models
# --------------------------------------------------------------------------


def promotion_history(db_path: Path = store.DEFAULT_DB_PATH) -> pd.DataFrame:
    """Every registry transition with the verdict that produced it.

    Assembled from two places because rejections and promotions are recorded
    differently: ``model_promotions`` holds attempts that reached the
    registry (including one later rolled back), while a challenger rejected
    outright never gets a row there -- its verdict lives on the model version
    as tags. Showing only the promotions table would display a history in
    which every challenger succeeded.
    """
    records: list[dict] = []

    with store.connect(db_path) as connection:
        if _table_exists(connection, "model_promotions"):
            for row in connection.execute(
                "SELECT * FROM model_promotions ORDER BY id"
            ):
                records.append(
                    {
                        "when": row["promoted_at"],
                        "version": f"v{row['to_version']}",
                        "from_version": f"v{row['from_version']}" if row["from_version"] else "",
                        "verdict": row["verdict"],
                        "pr_auc_delta": row["pr_auc_delta"],
                        "margin_ses": row["margin_in_ses"],
                        "rows": row["comparison_rows"],
                        "alert": row["trigger_alert_id"],
                        "source": "model_promotions",
                    }
                )

    mlflow = tracking.mlflow_module()
    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(tracking.tracking_uri())
            from mlflow import MlflowClient

            client = MlflowClient()
            name = tracking.REGISTERED_MODEL_NAME
            for version in client.search_model_versions(f"name='{name}'"):
                tags = dict(version.tags)
                if "shadow_verdict" not in tags:
                    continue
                records.append(
                    {
                        "when": datetime.fromtimestamp(
                            version.last_updated_timestamp / 1000, tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        "version": f"v{version.version}",
                        "from_version": "",
                        "verdict": tags["shadow_verdict"],
                        "pr_auc_delta": float(tags.get("shadow_pr_auc_delta", "nan")),
                        "margin_ses": float("nan"),
                        "rows": int(tags.get("shadow_rows", 0)),
                        "alert": tags.get("trigger_alert_id"),
                        "source": "registry tags",
                    }
                )
        except Exception:
            pass

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).sort_values("when").reset_index(drop=True)
    frame["outcome"] = frame["verdict"].map(classify_verdict)
    return frame


def classify_verdict(verdict: str) -> str:
    """One word for what happened, for colouring a table."""
    lowered = (verdict or "").lower()
    if "rolled back" in lowered:
        return "rolled back"
    if lowered.startswith("promote"):
        return "promoted"
    if "no verdict" in lowered:
        return "refused"
    return "rejected"


def registry_versions() -> pd.DataFrame:
    """Every registered version and its current stage."""
    mlflow = tracking.mlflow_module()
    if mlflow is None:
        return pd.DataFrame()
    try:
        mlflow.set_tracking_uri(tracking.tracking_uri())
        from mlflow import MlflowClient

        client = MlflowClient()
        name = tracking.REGISTERED_MODEL_NAME
        rows = []
        for version in client.search_model_versions(f"name='{name}'"):
            tags = dict(version.tags)
            rows.append(
                {
                    "version": int(version.version),
                    "stage": version.current_stage,
                    "aliases": ", ".join(version.aliases or []),
                    "origin": tags.get("origin", "baseline"),
                    "note": tags.get("archived_reason")
                    or tags.get("promotion_rolled_back")
                    or tags.get("shadow_verdict", ""),
                }
            )
        return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def latest_shadow() -> dict:
    """The most recent comparison, including its leakage check."""
    return read_json(SHADOW_PATH)
