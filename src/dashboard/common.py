"""Shared rendering helpers for the dashboard pages.

Presentation only. Every figure still comes from ``src.dashboard.data``,
which reads the stores and recomputes nothing.
"""

from __future__ import annotations

import streamlit as st


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """A row of metrics, each with an optional hover explanation."""
    for column, (label, value, help_text) in zip(st.columns(len(items)), items):
        column.metric(label, value, help=help_text)


def fmt(value, spec: str = ".4f", dash: str = "--") -> str:
    """Format a number, or a dash when it is absent rather than zero."""
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return dash


def store_caption() -> None:
    """Where the page's numbers come from, in the sidebar."""
    from src.drift import store as _store
    from src.training import tracking as _tracking

    with st.sidebar:
        st.caption("Reading from")
        st.code(
            f"{_store.DEFAULT_DB_PATH.name}\n"
            f"{_tracking.tracking_uri().split('///')[-1].split('/')[-1]}\n"
            f"{_tracking.REGISTERED_MODEL_NAME}",
            language=None,
        )
