"""Promotions — every attempt, and the guards that stopped three of them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import data
from src.dashboard.common import fmt, metric_row, store_caption

st.title("Promotions")
store_caption()

st.caption(
    "The evidence that the system declines bad models rather than only "
    "promoting good ones. A pipeline that promotes everything it trains is "
    "not an A/B test."
)

history = data.promotion_history()

if history.empty:
    st.info("No promotion attempts recorded yet.")
    st.stop()

counts = history["outcome"].value_counts().to_dict()
metric_row(
    [
        ("Attempts", f"{len(history)}", None),
        ("Promoted", str(counts.get("promoted", 0)), "beat the champion beyond noise"),
        ("Rejected", str(counts.get("rejected", 0)), "not ahead, or inside noise"),
        (
            "Refused",
            str(counts.get("refused", 0)),
            "the comparison could not be trusted -- no verdict issued",
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

st.warning(
    "This table is assembled from **two** sources, and has to be. "
    "`model_promotions` holds only attempts that reached the registry — a "
    "challenger rejected outright never gets a row there, because the "
    "comparison exits before promoting. Its verdict survives as tags on the "
    "model version. A history built from the promotions table alone would "
    "show an unbroken run of successes."
)

st.divider()

# --------------------------------------------------------------------------

st.subheader("The three outcomes, in detail")

with st.expander("v2 — rejected on merit", expanded=False):
    st.markdown(
        "Trained on a 30-day window (61,723 fit rows) against the champion's "
        "271,938. It lost honestly:\n\n"
        "```\n"
        "  PR-AUC delta  -0.0100    bootstrap SE 0.0107    25,000 rows\n"
        "```\n\n"
        "Behind the champion, and the gap itself inside the noise. Archived "
        "with its verdict recorded on the version rather than deleted — a "
        "model that lost a fair test is worth keeping."
    )

with st.expander("v3 — promoted, then rolled back", expanded=True):
    st.markdown(
        "Scored **+0.2492 at 20.13 standard errors** and was promoted. The "
        "leakage guard then showed why that margin was impossible:\n\n"
        "```\n"
        "  judged   TransactionDT   10,569,737 .. 11,106,772\n"
        "  trained  TransactionDT    5,443,151 .. 15,811,131\n"
        "  overlap  100.0%\n"
        "```\n\n"
        "Every row it was judged on, it had been fitted on. It was recalling "
        "while the champion predicted. The promotion was reversed and v1 "
        "restored."
    )
    st.error(
        "A margin rule cannot catch this — a larger overlap produces a *more* "
        "decisive apparent win. Leakage is now checked first, before margin "
        "and before row count."
    )

with st.expander("v4 — promoted, on the second measurement", expanded=True):
    st.markdown(
        "Trained through the first half of the stream and judged on the 30 "
        "days after it, rows neither model had seen. Leakage check: "
        "**0.0% overlap**.\n\n"
        "```\n"
        "                    champion   challenger      delta\n"
        "  PR-AUC              0.4942       0.5270    +0.0327\n"
        "  precision           0.5300       0.5905    +0.0605\n"
        "  recall              0.4069       0.4300    +0.0231\n"
        "  false negatives        360          346        -14\n"
        "  flagged                466          442        -24\n\n"
        "  bootstrap SE  0.0068   95% CI [+0.0204, +0.0467]   4.81 SE on 20,000 rows\n"
        "```\n\n"
        "It catches **14 more frauds while raising 24 fewer flags** — better "
        "on both sides, not a threshold trade. That shape is what an honest "
        "comparison produces."
    )
    st.info(
        "This is v4's **second** comparison. The first reported +0.0361 at "
        "4.23 SE and was withdrawn: the challenger had been scored with the "
        "baseline's encoders. The conclusion survived re-measurement; the "
        "number moved, and the margin rose because the sample nearly doubled."
    )

st.divider()

# --------------------------------------------------------------------------

st.subheader("The gates a promotion must pass")

st.markdown(
    """
| gate | refuses when | fired on |
|---|---|---|
| **leakage** | judged rows overlap the challenger's training window by >1% | v3, and a stale-topic re-read |
| **sufficiency** | fewer than 5,000 jointly-scored rows or 100 positives | — |
| **noise** | the margin is inside one bootstrap standard error | v2 |

Leakage is checked first: once the rows are compromised, no other conclusion
is meaningful.
"""
)

shadow = data.latest_shadow()
if shadow:
    window = shadow.get("window", {})
    leaked = window.get("leaked")
    st.markdown("**Most recent comparison**")
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
        f"{window.get('leakage_overlap', 0) * 100:.1f}% overlap",
    )
    if leaked is False:
        scored = window.get("scored_dt_range") or [0, 0]
        trained = window.get("trained_dt_range") or [0, 0]
        st.success(
            f"Zero overlap — judged TransactionDT "
            f"{scored[0]:,.0f}..{scored[1]:,.0f} begins after training ended "
            f"at {trained[1]:,.0f}."
        )
    elif leaked:
        st.error(
            "The judged rows fell inside the challenger's own training "
            "window. No verdict was issued."
        )
