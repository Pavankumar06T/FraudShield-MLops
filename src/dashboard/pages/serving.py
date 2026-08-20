"""Serving — throughput, latency, and the fast drift signal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import data
from src.dashboard.common import metric_row, store_caption

st.title("Serving")
store_caption()

activity = data.scoring_activity()

if not activity["rows"]:
    st.info(
        f"{activity.get('reason', 'no rows')}. Start the consumer with "
        "`python -m src.streaming.consumer` once the producer has published."
    )
    st.stop()

if activity["live"]:
    st.success(f"Consumer active — last row {activity['last_scored']}")
else:
    st.warning(
        f"Consumer idle. Showing **{activity['rows']:,} historical rows**, "
        f"last scored {activity['last_scored']}. An idle consumer is a normal "
        "state, not an error."
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
            "rows carrying a level absent from training",
        ),
        (
            "Throughput",
            f"{activity['throughput_per_s']:.1f}/s"
            if activity["throughput_per_s"]
            else "--",
            "average over the observed span, not an instantaneous rate -- the "
            "audit table records when rows were scored, not a sampled counter",
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
            st.line_chart(bucket, height=240)
with right:
    if not activity["by_version"].empty:
        st.caption("Rows scored per model version")
        st.dataframe(activity["by_version"], width="stretch", hide_index=True)
    st.caption(
        f"{activity['shadowed']:,} rows also scored by a challenger in shadow. "
        "The challenger's output is recorded and never acted on."
    )

st.divider()

# --------------------------------------------------------------------------

st.subheader("Measured latency")

st.markdown(
    """
Single transaction, 431 features, **including SHAP**:

| stage | p50 | p95 |
|---|---|---|
| encode one transaction (dict → numpy) | 0.05 ms | 0.05 ms |
| `DMatrix` + `pred_contribs` (probability **and** SHAP) | 3.99 ms | 4.72 ms |
| **handler total** | **4.48 ms** | **5.67 ms** |
| **full HTTP round trip** | **6.74 ms** | **10.13 ms** |

Per-request SHAP is affordable, but only one way. The obvious
implementations are not:

| approach | p50 | p95 |
|---|---|---|
| `shap.TreeExplainer(model)(row)` per request | 24.35 ms | 116.57 ms |
| training's `build_features` on a single row | 22.10 ms | 54.46 ms |
| **booster `pred_contribs` + serving encoder** | **3.99 ms** | **4.72 ms** |

Assembled naively this endpoint would be **~58 ms p50 and over 150 ms at
p95**. Two changes close the gap: one booster call returning SHAP *and*
probability, and a serving encoder **429× faster** than the frame path and
asserted bit-identical to it on every column of every checked row.
"""
)

st.divider()

# --------------------------------------------------------------------------

st.subheader("Unseen categories — the fast drift signal")

st.caption(
    "A level the encoder has never seen is visible on the first request "
    "carrying it, where windowed PSI needs a window of traffic to move. The "
    "serving and streaming layers record them as they occur."
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
                "each a value absent from the training vocabulary",
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
            unseen[["feature", "occurrences", "distinct_values", "examples", "last_seen"]],
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "These are browser, OS and device strings — `id_31`, `id_30`, "
        "`DeviceInfo`, `id_33`. Software version turnover as users upgrade, "
        "not adversarial adaptation."
    )
