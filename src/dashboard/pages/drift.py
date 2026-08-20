"""Drift — PSI over time, and the decomposition that decides the remedy."""

from __future__ import annotations

import json

import streamlit as st

from src.dashboard import data
from src.dashboard.common import fmt, metric_row, store_caption

st.title("Drift")
store_caption()

st.caption(
    "The two drift types are plotted separately on purpose. Genuine value "
    "drift means the populated values moved and a retrain fixes it. "
    "Missingness-driven drift means coverage changed while the values held, "
    "and retraining on it would bake a transient upstream join into the "
    "model. A blended line would erase the distinction."
)

alerts = data.psi_history()

if alerts.empty:
    st.info("No alerts recorded. Run `python -m src.drift.monitor`.")
    st.stop()

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
            f"{int(latest['missingness_features'])} features -- investigate the "
            "feed, do NOT retrain",
        ),
        ("Threshold", fmt(latest["threshold"], ".2f"), "retrain trigger"),
    ]
)

series = alerts.set_index("created_at")[["value_drift_psi", "missingness_psi"]]
series.columns = ["genuine value drift", "missingness-driven"]
st.line_chart(series, height=280)

st.divider()

# --------------------------------------------------------------------------

st.subheader("What is drifting, and how it was classified")

left, right = st.columns(2)


def feature_list(raw) -> list[str]:
    try:
        return json.loads(raw) if raw else []
    except (TypeError, ValueError):
        return []


with left:
    names = feature_list(latest.get("value_drift_feature_list"))
    st.markdown(f"**Genuine value drift — {len(names)} features**")
    st.caption("The populated values moved. This is what retraining fixes.")
    if names:
        st.code("\n".join(names), language=None)

with right:
    names = feature_list(latest.get("missingness_feature_list"))
    st.markdown(f"**Missingness-driven — {len(names)} features**")
    st.caption(
        "Coverage changed, values held. An upstream pipeline question, not a "
        "model one."
    )
    if names:
        st.code("\n".join(names), language=None)

st.info(
    "The split is made on populated-row PSI: a feature counts as genuine "
    "value drift when its PSI restricted to non-null rows is at least 0.05. "
    "The two clusters here sit at 0.000-0.015 and 0.07-8.40, so any cut "
    "between 0.01 and 0.07 gives the identical partition. See **Methodology**."
)

st.divider()

# --------------------------------------------------------------------------

st.subheader("Every alert")

display = alerts[
    [
        "id", "created_at", "window_label", "window_rows", "overall_psi",
        "overall_psi_feature", "value_drift_psi", "value_drift_features",
        "missingness_psi", "missingness_features", "verdict",
        "resolved_model_version",
    ]
].rename(columns={"resolved_model_version": "answered_by"})
st.dataframe(display, width="stretch", hide_index=True)

st.caption(
    "`overall_psi` is the worst single feature, not a mean. At real feature "
    "proportions the mean reads calm while the worst feature sits above 1.7 — "
    "see **Methodology** for the arithmetic."
)
