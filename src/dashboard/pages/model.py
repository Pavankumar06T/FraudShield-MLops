"""Model — registry, lineage, and the configuration that produced it."""

from __future__ import annotations

import streamlit as st

from src.dashboard import data
from src.dashboard.common import fmt, metric_row, store_caption

st.title("Model")
store_caption()

champion = data.champion()
versions = data.registry_versions()

if not champion.get("available"):
    st.error(f"No champion resolved: {champion.get('reason')}")
    st.stop()

# --------------------------------------------------------------------------

st.subheader(f"v{champion['version']} — {champion['stage']}")

metric_row(
    [
        ("PR-AUC", fmt(champion["pr_auc"]), "on the run's own evaluation slice"),
        ("ROC-AUC", fmt(champion["roc_auc"]), None),
        ("F1", fmt(champion["f1"]), "at the swept threshold"),
        ("Precision", fmt(champion["precision"]), None),
        ("Recall", fmt(champion["recall"]), None),
    ]
)

left, right = st.columns(2)
with left:
    st.markdown(
        f"**Run** `{champion['run_id']}`  \n"
        f"**Registered** {champion['created']}  \n"
        f"**Trained on** {champion.get('train_rows') or '--'} rows  \n"
        f"**Trees** {int(champion['n_trees']):,} of a 2,000 ceiling  \n"
        f"**Train/val gap** {fmt(champion['overfit_gap'], '+.4f')}"
    )
with right:
    st.markdown(
        f"**Threshold** {fmt(champion['threshold'])}  \n"
        f"**n_jobs** `{champion.get('n_jobs')}` — pinned, not `-1`  \n"
        + (
            f"**Exists because of** drift alert #{champion['trigger_alert']}"
            if champion.get("trigger_alert")
            else "**Origin** baseline training run"
        )
    )

if champion.get("drifting_features"):
    st.markdown("**Drift that caused this model to exist**")
    st.code(champion["drifting_features"].replace(",", "\n"), language=None)

st.info(
    "`n_jobs` is pinned to 2 rather than `-1`. XGBoost's `hist` method is not "
    "thread-deterministic — its subsample and colsample RNG streams are "
    "per-thread, so the same seed and data produced 239, 300 and 140 trees at "
    "1, 2 and 4 threads. A CI runner differs in core count from any "
    "development machine, and an unpinned retrain would produce a model that "
    "differs for reasons unrelated to the drift."
)

st.divider()

# --------------------------------------------------------------------------

st.subheader("Registry")

if versions.empty:
    st.info("No registered versions found.")
else:
    st.dataframe(versions, width="stretch", hide_index=True)
    st.caption(
        "`@champion` is what serving resolves; `@challenger` is the candidate "
        "under shadow test. Archived versions keep the verdict that retired "
        "them — a model that lost a fair test is worth keeping on the record."
    )

st.divider()

# --------------------------------------------------------------------------

st.subheader("Lineage")

st.markdown(
    """
Each model resolves **its own** artifacts, by identity rather than by path:

| artifact | resolved from |
|---|---|
| model binary | the registered version |
| encoders | the MLflow run behind that version |
| threshold | `val.at_best_f1_threshold.threshold` on the same run |

Serving **raises** rather than falling back when a registered model's
encoders cannot be resolved. Falling back to `models/encoders.pkl` is not a
degraded mode — that file belongs to whichever run wrote it last, and a model
scored through another run's vocabulary produces predictions from levels it
never saw, with nothing downstream able to detect it.
"""
)
st.caption(
    "That failure happened here and survived a full promotion cycle. See "
    "**Methodology** for what it cost and what prevents it now."
)
