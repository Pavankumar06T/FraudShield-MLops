"""Overview — the champion, the current verdict, and the headline numbers."""

from __future__ import annotations

import streamlit as st

from src.dashboard import data
from src.dashboard.common import fmt, metric_row, store_caption

st.title("Overview")
store_caption()

champion = data.champion()
alerts = data.psi_history()
history = data.promotion_history()
activity = data.scoring_activity()

# --------------------------------------------------------------------------

if not champion.get("available"):
    st.error(f"No champion resolved: {champion.get('reason')}")
else:
    shadow = data.shadow_evidence(champion["version"])
    st.subheader(f"Champion — v{champion['version']}")
    metric_row(
        [
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
                "window. A different window -- the two are not comparable and "
                "routinely disagree.",
            ),
            (
                "Threshold",
                fmt(champion["threshold"]),
                "The swept best-F1 point from this version's own run, not 0.5 "
                "and not the previous champion's.",
            ),
            (
                "Trees",
                f"{int(champion['n_trees']):,}" if champion.get("n_trees") else "--",
                "chosen by early stopping, not the 2,000 ceiling",
            ),
        ]
    )
    if shadow.get("available") and shadow.get("beat_version"):
        st.success(
            f"Promoted over v{shadow['beat_version']} "
            f"({fmt(shadow['beat_pr_auc'])}) by {fmt(shadow['delta'], '+.4f')} "
            f"at {fmt(shadow['margin_ses'], '.2f')} standard errors, on "
            f"{int(shadow['rows']):,} genuinely unseen rows."
        )

st.divider()

# --------------------------------------------------------------------------

st.subheader("Current drift verdict")

if alerts.empty:
    st.info("No alerts recorded. Run `python -m src.drift.monitor`.")
else:
    latest = alerts.iloc[-1]
    open_trigger = latest["resolved_at"] is None and latest["retrain_triggered"] == 1
    (st.warning if open_trigger else st.success)(
        f"**{latest['verdict']}** — alert #{int(latest['id'])}, "
        + (
            "unresolved: a retrain is due."
            if open_trigger
            else f"answered by v{latest['resolved_model_version'] or '?'}."
        )
    )
    metric_row(
        [
            (
                "Genuine value drift",
                fmt(latest["value_drift_psi"]),
                f"{int(latest['value_drift_features'])} features -- this is what "
                "triggers a retrain",
            ),
            (
                "Missingness-driven",
                fmt(latest["missingness_psi"]),
                f"{int(latest['missingness_features'])} features -- investigate "
                "the upstream feed; retraining on it would bake in a transient join",
            ),
            ("Worst feature", str(latest["overall_psi_feature"]), "overall PSI is the max"),
            ("Threshold", fmt(latest["threshold"], ".2f"), "retrain trigger"),
        ]
    )

st.divider()

# --------------------------------------------------------------------------

st.subheader("At a glance")

counts = history["outcome"].value_counts().to_dict() if not history.empty else {}
metric_row(
    [
        ("Rows scored", f"{activity['rows']:,}", "every transaction ever scored"),
        (
            "Flag rate",
            f"{activity['flag_rate'] * 100:.2f}%" if activity["rows"] else "--",
            "BLOCK decisions at the promoted threshold",
        ),
        (
            "Unseen categories",
            f"{activity['unseen_rate'] * 100:.2f}%" if activity["rows"] else "--",
            "rows carrying a level absent from training -- the fast drift signal",
        ),
        ("Promotion attempts", str(len(history)), None),
        (
            "Promoted / declined",
            f"{counts.get('promoted', 0)} / "
            f"{counts.get('rejected', 0) + counts.get('refused', 0) + counts.get('rolled back', 0)}",
            "a pipeline that promotes everything it trains is not an A/B test",
        ),
    ]
)

st.caption(
    "Drift detail is on the **Drift** page, the registry and training "
    "configuration on **Model**, the full promotion record on **Promotions**, "
    "and how each of these numbers is computed on **Methodology**."
)
