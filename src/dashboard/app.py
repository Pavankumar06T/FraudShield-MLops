"""FraudShield operations dashboard.

    streamlit run src/dashboard/app.py

Reads the stores the rest of the system already writes -- ``reports/drift.db``
and the MLflow registry -- and recalculates nothing. A dashboard that
recomputed PSI would eventually disagree with the monitor that fired the
alert, and an operator would be looking at a number nobody can reconcile
with the decision actually taken.

**Streamlit rather than Prometheus and Grafana**, deliberately. This machine
has 2 physical cores and 8 GB, already running Redpanda, an MLflow store and
a 799-tree model, and training pins ``n_jobs=2`` to both cores. Adding a
scrape server, a time-series database and a rendering service would contend
for exactly the resources the pipeline needs, to display numbers that are
already durable in SQLite. The demonstrative value is the same.

The panel that matters most is the last one. Anyone can show a dashboard
where models get promoted; the evidence that this system works is that it
declined two challengers and rolled back a third.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import data

st.set_page_config(page_title="FraudShield", page_icon="🛡", layout="wide")

REFRESH_SECONDS = 30


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    for column, (label, value, help_text) in zip(st.columns(len(items)), items):
        column.metric(label, value, help=help_text)


def fmt(value, spec: str = ".4f", dash: str = "--") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return dash


# --------------------------------------------------------------------------

st.title("FraudShield")
st.caption(
    "Self-retraining fraud detection. Every figure below is read from the "
    "store that produced it -- nothing on this page is recalculated."
)

with st.sidebar:
    st.header("Stores")
    from src.drift import store as _store
    from src.training import tracking as _tracking

    st.code(
        f"alerts   {_store.DEFAULT_DB_PATH.name}\n"
        f"mlflow   {_tracking.tracking_uri().split('///')[-1]}\n"
        f"registry {_tracking.REGISTERED_MODEL_NAME}",
        language=None,
    )
    st.caption(
        "SQLite and a local MLflow store. No Postgres, no Prometheus, no "
        "Grafana -- see the note at the bottom of this page."
    )
    if st.button("Refresh", width="stretch"):
        st.rerun()

# --------------------------------------------------------------------------
# 1. Current champion
# --------------------------------------------------------------------------

st.header("Current champion")

champion = data.champion()
if not champion.get("available"):
    st.error(f"No champion resolved: {champion.get('reason')}")
else:
    shadow = data.shadow_evidence(champion["version"])
    metric_row(
        [
            (
                "Version",
                f"v{champion['version']}",
                f"stage {champion['stage']}, promoted {champion['created']}",
            ),
            (
                "PR-AUC on unseen rows",
                fmt(shadow.get("pr_auc")) if shadow.get("available") else "not tested",
                "Shadow A/B, on live rows the model had never been trained on. "
                "This is what it does in production.",
            ),
            (
                "PR-AUC on its own eval slice",
                fmt(champion["pr_auc"]),
                "Recorded by the training run, on the tail of its own training "
                "window. A different window from the shadow figure -- the two "
                "are not comparable and routinely disagree.",
            ),
            (
                "Threshold",
                fmt(champion["threshold"]),
                "swept best-F1 point, not 0.5 -- scale_pos_weight makes the "
                "probabilities deliberately uncalibrated",
            ),
            (
                "Trees",
                f"{int(champion['n_trees']):,}" if champion.get("n_trees") else "--",
                "chosen by early stopping, not the 2,000 ceiling",
            ),
        ]
    )

    if shadow.get("available"):
        beat = (
            f" beating v{shadow['beat_version']} at {fmt(shadow['beat_pr_auc'])}"
            if shadow.get("beat_version")
            else ""
        )
        st.info(
            f"**The two PR-AUC figures above measure different windows and are "
            f"not in conflict.** {fmt(shadow['pr_auc'])} is from the shadow test "
            f"on {int(shadow['rows']):,} genuinely unseen rows"
            f"{beat} — a margin of {fmt(shadow['delta'], '+.4f')} at "
            f"{fmt(shadow['margin_ses'], '.2f')} standard errors. "
            f"{fmt(champion['pr_auc'])} is what the training run recorded on the "
            f"tail of its own window. Source: `{shadow['source']}`."
        )
    else:
        st.warning(
            "This model has no shadow-test result. The PR-AUC shown is its own "
            "evaluation slice, which is not evidence of production performance."
        )
    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            f"**Run** `{champion['run_id'][:12]}`  \n"
            f"**Trained on** {champion.get('train_rows') or '--'} rows, "
            f"`n_jobs={champion.get('n_jobs')}` pinned  \n"
            f"**Train/val gap** {fmt(champion['overfit_gap'], '+.4f')}"
        )
    with right:
        if champion.get("trigger_alert"):
            st.markdown(
                f"**Exists because of drift alert #{champion['trigger_alert']}**  \n"
                f"drifting features: `{champion.get('drifting_features', '')}`"
            )
        else:
            st.markdown("**Origin** baseline training run (no drift trigger)")

    versions = data.registry_versions()
    if not versions.empty:
        with st.expander(f"Registry — {len(versions)} versions"):
            st.dataframe(versions, width="stretch", hide_index=True)

st.divider()

# --------------------------------------------------------------------------
# 2. Live scoring
# --------------------------------------------------------------------------

st.header("Live scoring")

activity = data.scoring_activity()
if not activity["rows"]:
    st.info(
        f"{activity.get('reason', 'no rows')}. Start the consumer with "
        "`python -m src.streaming.consumer` once the producer has published."
    )
else:
    if activity["live"]:
        st.success(f"Consumer active — last row {activity['last_scored']}")
    else:
        st.warning(
            f"Consumer idle. Showing **{activity['rows']:,} historical rows**, "
            f"last scored {activity['last_scored']}."
        )

    metric_row(
        [
            ("Scored", f"{activity['rows']:,}", "every row ever scored"),
            (
                "Flag rate",
                f"{activity['flag_rate'] * 100:.2f}%",
                "BLOCK decisions at the promoted threshold",
            ),
            (
                "Unseen categories",
                f"{activity['unseen_rate'] * 100:.2f}%",
                "rows carrying a level absent from training -- the fast drift signal",
            ),
            (
                "Throughput",
                f"{activity['throughput_per_s']:.1f}/s"
                if activity["throughput_per_s"]
                else "--",
                "average over the observed span, not an instantaneous rate",
            ),
            ("Scoring latency", f"{activity['avg_latency_ms']:.1f} ms", "mean, per row"),
        ]
    )

    left, right = st.columns([3, 2])
    with left:
        recent = activity["recent"]
        if not recent.empty:
            recent = recent.copy()
            recent["scored_at"] = pd.to_datetime(
                recent["scored_at"], format="mixed", utc=True
            )
            bucket = (
                recent.set_index("scored_at")
                .assign(blocked=lambda f: (f["decision"] == "BLOCK").astype(int))
                .resample("10s")
                .agg({"decision": "count", "blocked": "sum"})
                .rename(columns={"decision": "scored"})
            )
            bucket = bucket[bucket["scored"] > 0]
            if not bucket.empty:
                st.caption("Most recent 500 rows, 10-second buckets")
                st.line_chart(bucket, height=220)
    with right:
        if not activity["by_version"].empty:
            st.caption("Rows scored per model version")
            st.dataframe(activity["by_version"], width="stretch", hide_index=True)
        st.caption(
            f"{activity['shadowed']:,} rows also scored by a challenger in shadow"
        )

st.divider()

# --------------------------------------------------------------------------
# 3. PSI trend, split by drift type
# --------------------------------------------------------------------------

st.header("Drift")
st.caption(
    "The two types are plotted separately on purpose. Genuine value drift "
    "means the populated values moved and a retrain fixes it; "
    "missingness-driven drift means coverage changed while the values held, "
    "and retraining on it would bake a transient upstream join into the "
    "model. A blended line would erase the distinction."
)

alerts = data.psi_history()
if alerts.empty:
    st.info("No alerts recorded. Run `python -m src.drift.monitor`.")
else:
    latest = alerts.iloc[-1]
    metric_row(
        [
            ("Alerts", f"{len(alerts)}", None),
            (
                "Latest verdict",
                latest["verdict"].split(" -- ")[0].split(" (")[0],
                latest["verdict"],
            ),
            (
                "Value drift PSI",
                fmt(latest["value_drift_psi"]),
                f"{int(latest['value_drift_features'])} features -- triggers retraining",
            ),
            (
                "Missingness PSI",
                fmt(latest["missingness_psi"]),
                f"{int(latest['missingness_features'])} features -- "
                "investigate the feed, do NOT retrain",
            ),
            ("Threshold", fmt(latest["threshold"], ".2f"), "retrain trigger"),
        ]
    )

    series = alerts.set_index("created_at")[["value_drift_psi", "missingness_psi"]]
    series.columns = ["genuine value drift", "missingness-driven"]
    st.line_chart(series, height=260)

    display = alerts[
        [
            "id", "created_at", "window_label", "window_rows", "overall_psi",
            "overall_psi_feature", "value_drift_psi", "value_drift_features",
            "missingness_psi", "missingness_features", "verdict",
            "resolved_model_version",
        ]
    ].rename(columns={"resolved_model_version": "answered_by"})
    with st.expander(f"All {len(alerts)} alerts"):
        st.dataframe(display, width="stretch", hide_index=True)

    worst = alerts.iloc[-1]
    if worst.get("value_drift_feature_list"):
        try:
            import json as _json

            names = _json.loads(worst["value_drift_feature_list"])
            if names:
                st.markdown(
                    "**Features driving the latest retrain trigger:** "
                    + ", ".join(f"`{n}`" for n in names)
                )
        except Exception:
            pass

st.divider()

# --------------------------------------------------------------------------
# 4. Unseen categories
# --------------------------------------------------------------------------

st.header("Unseen categories")
st.caption(
    "A level the encoder has never seen is visible on the first request "
    "carrying it, where windowed PSI needs a window of traffic to move. "
    "These are written by the serving and streaming layers as they meet them."
)

unseen = data.unseen_categories()
if unseen.empty:
    st.info("Nothing recorded yet — score some traffic through the API or consumer.")
else:
    metric_row(
        [
            ("Features affected", f"{len(unseen)}", None),
            ("Observations", f"{int(unseen['occurrences'].sum()):,}", None),
            (
                "Distinct unseen levels",
                f"{int(unseen['distinct_values'].sum()):,}",
                "each is a value absent from the training vocabulary",
            ),
            ("Worst feature", str(unseen.iloc[0]["feature"]), None),
        ]
    )
    left, right = st.columns([2, 3])
    with left:
        st.bar_chart(
            unseen.set_index("feature")["occurrences"], height=240, horizontal=True
        )
    with right:
        st.dataframe(
            unseen[
                ["feature", "occurrences", "distinct_values", "examples", "last_seen"]
            ],
            width="stretch",
            hide_index=True,
        )

st.divider()

# --------------------------------------------------------------------------
# 5. Promotion history -- the important one
# --------------------------------------------------------------------------

st.header("Promotion history")
st.caption(
    "The evidence that the system declines bad models rather than only "
    "promoting good ones. A pipeline that promotes everything it trains is "
    "not an A/B test."
)

history = data.promotion_history()
if history.empty:
    st.info("No promotion attempts recorded yet.")
else:
    counts = history["outcome"].value_counts().to_dict()
    metric_row(
        [
            ("Attempts", f"{len(history)}", None),
            ("Promoted", str(counts.get("promoted", 0)), "beat the champion beyond noise"),
            ("Rejected", str(counts.get("rejected", 0)), "not ahead, or inside noise"),
            (
                "Refused",
                str(counts.get("refused", 0)),
                "comparison could not be trusted -- no verdict issued",
            ),
            (
                "Rolled back",
                str(counts.get("rolled back", 0)),
                "promoted, then found invalid",
            ),
        ]
    )

    def colour(row: pd.Series):
        shades = {
            "promoted": "background-color: #1b5e20; color: #ffffff",
            "rejected": "background-color: #4e342e; color: #ffffff",
            "refused": "background-color: #4a148c; color: #ffffff",
            "rolled back": "background-color: #b71c1c; color: #ffffff",
        }
        return [shades.get(row["outcome"], "")] * len(row)

    table = history[
        ["when", "version", "outcome", "pr_auc_delta", "margin_ses", "rows",
         "alert", "verdict", "source"]
    ].rename(
        columns={
            "pr_auc_delta": "PR-AUC delta",
            "margin_ses": "margin (SE)",
            "rows": "judged rows",
        }
    )
    st.dataframe(
        table.style.apply(colour, axis=1).format(
            {"PR-AUC delta": "{:+.4f}", "margin (SE)": "{:.2f}", "judged rows": "{:,.0f}"},
            na_rep="--",
        ),
        width="stretch",
        hide_index=True,
    )

    shadow = data.latest_shadow()
    if shadow:
        window = shadow.get("window", {})
        leaked = window.get("leaked")
        st.subheader("Most recent comparison")
        cols = st.columns(4)
        cols[0].metric("Verdict", shadow.get("verdict", "--").split(" -- ")[0])
        cols[1].metric(
            "PR-AUC delta", fmt(shadow.get("decision", {}).get("pr_auc_delta"), "+.4f")
        )
        cols[2].metric(
            "Margin",
            f"{fmt(shadow.get('decision', {}).get('margin_in_ses'), '.2f')} SE",
            f"needs {fmt(shadow.get('decision', {}).get('required_ses'), '.2f')}",
        )
        cols[3].metric(
            "Leakage check",
            "clean" if leaked is False else "FAILED",
            f"{window.get('leakage_overlap', 0) * 100:.1f}% overlap with the "
            "challenger's training window",
        )
        if leaked:
            st.error(
                "The judged rows fell inside the challenger's own training "
                "window. No verdict was issued."
            )
        elif leaked is False:
            scored = window.get("scored_dt_range") or [0, 0]
            trained = window.get("trained_dt_range") or [0, 0]
            st.success(
                f"Zero overlap — judged TransactionDT {scored[0]:,.0f}..{scored[1]:,.0f} "
                f"begins after training ended at {trained[1]:,.0f}."
            )

st.divider()
with st.expander("Why Streamlit and SQLite rather than Grafana and Postgres"):
    st.markdown(
        """
This machine has **2 physical cores and 8 GB**, already running Redpanda, a
local MLflow store and a 799-tree model, and training pins `n_jobs=2` to
both cores. A Prometheus scraper, a time-series database and a Grafana
renderer would contend for exactly the resources the pipeline needs, in
order to display numbers already durable in SQLite.

**Postgres was considered and rejected for now.** An earlier note in
`docs/retraining.md` claimed the store layer avoided SQLite-only constructs
and would move with a connection string. That was checked and it is wrong —
the schema uses `AUTOINCREMENT` (4×), `PRAGMA` for WAL mode and for the
migration's column introspection, `executescript`, `cursor.lastrowid`, and
`?` placeholders throughout. Postgres needs `IDENTITY`/`SERIAL`,
`information_schema` for introspection, `RETURNING id`, and `%s`
placeholders. That is real rework, not a swap, and the note has been
corrected.
        """
    )
