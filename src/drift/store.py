"""SQLite alert store shared by the drift monitor and the serving layer.

Two writers, one file. The scheduled monitor writes windowed PSI alerts;
the serving layer writes unseen-category observations as it meets them. They
share a store because they are two views of the same question, at different
latencies -- a category the model has never seen shows up in the next
request, while the PSI that eventually reflects it needs a window's worth of
traffic to accumulate.

SQLite rather than Postgres for now, deliberately: Phase 4 stays
self-contained and runnable with nothing installed. The schema is written to
survive the move -- explicit types, no SQLite-only constructs, timestamps as
ISO-8601 UTC text rather than the local-time default.

Concurrency: WAL mode, so the monitor reading a window does not block a
serving process appending observations. SQLite handles multiple readers and
one writer; if serving ever runs multiple processes that write at once, this
is the first thing to outgrow.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.common.config import REPORTS_DIR

#: Default store location. Gitignored with the rest of reports/.
DEFAULT_DB_PATH: Path = REPORTS_DIR / "drift.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_alerts (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at              TEXT    NOT NULL,
    window_label            TEXT    NOT NULL,
    window_start_dt         REAL,
    window_end_dt           REAL,
    window_rows             INTEGER NOT NULL,
    reference_rows          INTEGER NOT NULL,

    -- headline: the worst single feature, not a blend. There is no
    -- canonical "overall PSI"; averaging hides exactly the single collapsed
    -- feature the monitor exists to catch.
    overall_psi             REAL    NOT NULL,
    overall_psi_feature     TEXT,

    -- the split the Phase 0 analysis exists to make
    value_drift_psi         REAL    NOT NULL,
    value_drift_features    INTEGER NOT NULL,
    missingness_psi         REAL    NOT NULL,
    missingness_features    INTEGER NOT NULL,

    n_major                 INTEGER NOT NULL,
    n_moderate              INTEGER NOT NULL,
    n_stable                INTEGER NOT NULL,
    n_unmeasurable          INTEGER NOT NULL,

    threshold               REAL    NOT NULL,
    retrain_triggered       INTEGER NOT NULL,
    investigate_pipeline    INTEGER NOT NULL,
    verdict                 TEXT    NOT NULL,

    top_features            TEXT    NOT NULL,  -- JSON

    -- The two classified feature lists, stored in full rather than derived
    -- from top_features. top_features is ranked by raw PSI, and in this
    -- dataset the missingness cluster outranks most of the value-drift
    -- cluster -- so a top-15 slice names only 3 of the 10 features that
    -- actually triggered the retrain, and the run record would understate
    -- its own cause.
    value_drift_feature_list    TEXT,          -- JSON
    missingness_feature_list    TEXT,          -- JSON

    unseen_categories       TEXT,              -- JSON
    evidently_report_path   TEXT,

    -- Resolution. An alert stays open until a retrain consumes it, so the
    -- scheduled workflow cannot retrain twice on the same drift, and the
    -- run that answered an alert is recorded on the alert itself.
    resolved_at             TEXT,
    resolved_by_run_id      TEXT,
    resolved_model_version  TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_created ON drift_alerts (created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_retrain ON drift_alerts (retrain_triggered);

CREATE TABLE IF NOT EXISTS unseen_observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at   TEXT    NOT NULL,
    feature       TEXT    NOT NULL,
    value         TEXT    NOT NULL,
    occurrences   INTEGER NOT NULL DEFAULT 1,
    model_version TEXT,
    source        TEXT    NOT NULL DEFAULT 'serving'
);

CREATE INDEX IF NOT EXISTS idx_unseen_observed ON unseen_observations (observed_at);
CREATE INDEX IF NOT EXISTS idx_unseen_feature  ON unseen_observations (feature);
"""


def utc_now() -> str:
    """ISO-8601 UTC. Never local time -- alerts get read across timezones."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: Columns added after the table was first created. SQLite has no
#: ADD COLUMN IF NOT EXISTS, and CREATE TABLE IF NOT EXISTS silently leaves
#: an older table untouched -- so a store written before resolution tracking
#: existed would keep working and then fail on the first UPDATE.
MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("drift_alerts", "value_drift_feature_list TEXT"),
    ("drift_alerts", "missingness_feature_list TEXT"),
    ("drift_alerts", "resolved_at TEXT"),
    ("drift_alerts", "resolved_by_run_id TEXT"),
    ("drift_alerts", "resolved_model_version TEXT"),
    # Shadow scoring. stream_predictions predates the challenger columns, and
    # CREATE TABLE IF NOT EXISTS leaves an existing table untouched.
    ("stream_predictions", "challenger_version TEXT"),
    ("stream_predictions", "challenger_probability REAL"),
    ("stream_predictions", "challenger_decision TEXT"),
    ("stream_predictions", "challenger_threshold REAL"),
)


def _migrate(connection: sqlite3.Connection) -> None:
    for table, column in MIGRATIONS:
        name = column.split()[0]
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if existing and name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column}")


@contextmanager
def connect(path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open the store, creating it and its schema if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        # WAL so a reading monitor does not block a writing service.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)
        _migrate(connection)
        yield connection
        connection.commit()
    finally:
        connection.close()


@dataclass
class DriftAlert:
    """One monitor run, as recorded."""

    window_label: str
    window_rows: int
    reference_rows: int
    overall_psi: float
    overall_psi_feature: str | None
    value_drift_psi: float
    value_drift_features: int
    missingness_psi: float
    missingness_features: int
    n_major: int
    n_moderate: int
    n_stable: int
    n_unmeasurable: int
    threshold: float
    retrain_triggered: bool
    investigate_pipeline: bool
    verdict: str
    top_features: list[dict]
    value_drift_feature_list: list[str] = field(default_factory=list)
    missingness_feature_list: list[str] = field(default_factory=list)
    window_start_dt: float | None = None
    window_end_dt: float | None = None
    unseen_categories: list[dict] = field(default_factory=list)
    evidently_report_path: str | None = None
    created_at: str = field(default_factory=utc_now)

    def insert(self, path: Path = DEFAULT_DB_PATH) -> int:
        with connect(path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO drift_alerts (
                    created_at, window_label, window_start_dt, window_end_dt,
                    window_rows, reference_rows, overall_psi, overall_psi_feature,
                    value_drift_psi, value_drift_features,
                    missingness_psi, missingness_features,
                    n_major, n_moderate, n_stable, n_unmeasurable,
                    threshold, retrain_triggered, investigate_pipeline, verdict,
                    top_features, value_drift_feature_list,
                    missingness_feature_list, unseen_categories,
                    evidently_report_path
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.created_at, self.window_label, self.window_start_dt,
                    self.window_end_dt, self.window_rows, self.reference_rows,
                    self.overall_psi, self.overall_psi_feature,
                    self.value_drift_psi, self.value_drift_features,
                    self.missingness_psi, self.missingness_features,
                    self.n_major, self.n_moderate, self.n_stable, self.n_unmeasurable,
                    self.threshold, int(self.retrain_triggered),
                    int(self.investigate_pipeline), self.verdict,
                    json.dumps(self.top_features),
                    json.dumps(self.value_drift_feature_list),
                    json.dumps(self.missingness_feature_list),
                    json.dumps(self.unseen_categories),
                    self.evidently_report_path,
                ),
            )
            return int(cursor.lastrowid)


def recent_alerts(limit: int = 10, path: Path = DEFAULT_DB_PATH) -> list[dict]:
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM drift_alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def latest_retrain_trigger(
    path: Path = DEFAULT_DB_PATH, include_resolved: bool = False
) -> dict | None:
    """Most recent *unresolved* alert asking for a retrain, or None.

    The retrain reads this rather than re-deriving the decision, so the row
    that fires the retrain and the row that recorded why are the same one --
    a retrain can always be traced back to the drift measurement that caused
    it, months later, without recomputing anything.

    Resolved alerts are excluded so a scheduled workflow cannot retrain
    repeatedly on the same drift. ``retrain_triggered = 1`` is set only for
    genuine value drift; a missingness-driven alert never has it, so this
    query cannot return one.
    """
    query = "SELECT * FROM drift_alerts WHERE retrain_triggered = 1"
    if not include_resolved:
        query += " AND resolved_at IS NULL"
    query += " ORDER BY id DESC LIMIT 1"
    with connect(path) as connection:
        row = connection.execute(query).fetchone()
    return dict(row) if row else None


def open_retrain_alerts(path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Every unresolved retrain-triggering alert, oldest first."""
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM drift_alerts WHERE retrain_triggered = 1 "
            "AND resolved_at IS NULL ORDER BY id ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def mark_resolved(
    alert_id: int,
    run_id: str | None = None,
    model_version: str | None = None,
    path: Path = DEFAULT_DB_PATH,
) -> bool:
    """Close an alert, recording which run answered it.

    Returns False if the alert was already resolved, which is what makes
    this safe to call from a workflow that might run twice.
    """
    with connect(path) as connection:
        cursor = connection.execute(
            "UPDATE drift_alerts SET resolved_at = ?, resolved_by_run_id = ?, "
            "resolved_model_version = ? WHERE id = ? AND resolved_at IS NULL",
            (utc_now(), run_id, model_version, alert_id),
        )
        return cursor.rowcount > 0


# --------------------------------------------------------------------------
# Unseen categories, written by the serving layer
# --------------------------------------------------------------------------


def record_unseen(
    observations: Counter | dict[tuple[str, str], int],
    model_version: str | None = None,
    source: str = "serving",
    path: Path = DEFAULT_DB_PATH,
) -> int:
    """Persist a batch of (feature, value) counts.

    Batched rather than per request on purpose: a SQLite write inside the
    request path would add milliseconds to a 4 ms handler, and the signal
    does not need per-request durability to be useful.
    """
    if not observations:
        return 0
    stamp = utc_now()
    rows = [
        (stamp, feature, str(value), int(count), model_version, source)
        for (feature, value), count in observations.items()
    ]
    with connect(path) as connection:
        connection.executemany(
            "INSERT INTO unseen_observations "
            "(observed_at, feature, value, occurrences, model_version, source) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def aggregate_unseen(
    since: str | None = None, path: Path = DEFAULT_DB_PATH
) -> list[dict]:
    """Unseen levels per feature, worst first.

    This is the fast drift signal. Windowed PSI needs a window's worth of
    traffic before it moves; a category the encoder has never seen is
    visible on the first request that carries it.
    """
    query = (
        "SELECT feature, COUNT(DISTINCT value) AS distinct_values, "
        "SUM(occurrences) AS occurrences, MIN(observed_at) AS first_seen, "
        "MAX(observed_at) AS last_seen "
        "FROM unseen_observations "
    )
    params: tuple = ()
    if since:
        query += "WHERE observed_at >= ? "
        params = (since,)
    query += "GROUP BY feature ORDER BY occurrences DESC"

    with connect(path) as connection:
        rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            examples = connection.execute(
                "SELECT value, SUM(occurrences) AS n FROM unseen_observations "
                "WHERE feature = ? GROUP BY value ORDER BY n DESC LIMIT 5",
                (row["feature"],),
            ).fetchall()
            entry["top_values"] = [dict(e) for e in examples]
            result.append(entry)
    return result
