"""FraudShield operations dashboard.

    streamlit run src/dashboard/app.py

Six pages over ``reports/drift.db`` and the MLflow registry. **Nothing here
is recalculated** -- PSI comes from ``drift_alerts``, promotion verdicts from
``model_promotions`` and version tags, model metrics from the runs that
produced them.

That is a correctness requirement, not tidiness. A dashboard that recomputed
PSI would eventually disagree with the monitor that fired the alert -- a
different window, a different bin edge, a different epsilon -- and an
operator would be reading a number nobody can reconcile with the decision
actually taken. ``tests/test_dashboard.py`` asserts no module under this
package imports ``psi_numeric``, ``compute_psi_report``,
``average_precision_score`` or ``paired_bootstrap``.

The **Methodology** page explains every calculation with worked examples
drawn from real runs -- hardcoded values, never live computation, for the
same reason.

**Streamlit rather than Prometheus and Grafana**, deliberately. This machine
has 2 physical cores and 8 GB, already running Redpanda, a local MLflow store
and a 1,575-tree model, and training pins ``n_jobs=2`` to both cores. A
scrape server, a time-series database and a rendering service would contend
for exactly the resources the pipeline needs, to display numbers already
durable in SQLite.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FraudShield", page_icon="🛡", layout="wide")

PAGES = Path(__file__).parent / "pages"

navigation = st.navigation(
    {
        "Operations": [
            st.Page(PAGES / "overview.py", title="Overview", icon=":material/dashboard:",
                    default=True),
            st.Page(PAGES / "drift.py", title="Drift", icon=":material/trending_up:"),
            st.Page(PAGES / "serving.py", title="Serving", icon=":material/bolt:"),
        ],
        "Models": [
            st.Page(PAGES / "model.py", title="Model", icon=":material/inventory_2:"),
            st.Page(PAGES / "promotions.py", title="Promotions",
                    icon=":material/gavel:"),
        ],
        "Reference": [
            st.Page(PAGES / "methodology.py", title="Methodology",
                    icon=":material/functions:"),
        ],
    }
)

with st.sidebar:
    st.title("FraudShield")
    st.caption(
        "Self-retraining fraud detection. Every figure is read from the store "
        "that produced it."
    )

navigation.run()
