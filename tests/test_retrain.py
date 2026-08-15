"""Invariants for drift-triggered retraining.

Three carry the weight.

**Missingness-driven drift must never trigger a retrain.** Sixteen features
in this dataset exceed PSI 0.20 purely because their coverage improved.
Retraining on that bakes a transient upstream join into the model.

**Threads must be pinned and verified on the fitted estimators.** XGBoost's
hist method is not thread-deterministic, so a retrain on a differently-sized
runner produces a model that differs for reasons unrelated to the drift --
and the shadow comparison would then be measuring the core count.

**Registration is Staging, never Production.** A drift-triggered retrain is
a challenger. Whether it beats what is live is the A/B's decision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.drift import store
from src.training.retrain import (
    DEFAULT_WINDOW_DAYS,
    EVAL_FRACTION,
    TARGET_STAGE,
    ThreadPinningError,
    assert_threads_pinned,
    load_recent_window,
    trigger_params,
)
from src.training.train import N_JOBS, XGB_HYPERPARAMETERS


@pytest.fixture
def db(tmp_path):
    return tmp_path / "drift.db"


def insert_alert(db, *, retrain=True, value_psi=1.78, miss_psi=0.55, top=None):
    top = top if top is not None else [
        {"feature": "id_31", "psi": 1.7835, "type": "value"},
        {"feature": "id_13", "psi": 0.5086, "type": "value"},
        {"feature": "M9", "psi": 0.5490, "type": "missingness"},
    ]
    return store.DriftAlert(
        window_label="last 30 days", window_rows=85_431, reference_rows=319_927,
        overall_psi=max(value_psi, miss_psi), overall_psi_feature="id_31",
        value_drift_psi=value_psi, value_drift_features=10,
        missingness_psi=miss_psi, missingness_features=16,
        n_major=26, n_moderate=6, n_stable=277, n_unmeasurable=122,
        threshold=0.20, retrain_triggered=retrain,
        investigate_pipeline=miss_psi >= 0.20,
        verdict="RETRAIN + INVESTIGATE PIPELINE" if retrain else "INVESTIGATE PIPELINE (no retrain)",
        top_features=top, window_start_dt=1.3e7, window_end_dt=1.58e7,
    ).insert(db)


# --------------------------------------------------------------------------
# Only genuine value drift retrains
# --------------------------------------------------------------------------


def test_missingness_only_alert_never_becomes_a_trigger(db):
    """The coverage-change alert must be invisible to the retrain query."""
    insert_alert(db, retrain=False, value_psi=0.0, miss_psi=0.55)
    assert store.latest_retrain_trigger(db) is None
    assert store.open_retrain_alerts(db) == []


def test_value_drift_alert_is_picked_up(db):
    alert_id = insert_alert(db, retrain=True)
    trigger = store.latest_retrain_trigger(db)
    assert trigger is not None
    assert trigger["id"] == alert_id
    assert trigger["retrain_triggered"] == 1


def test_a_pipeline_alert_alongside_a_retrain_alert_does_not_confuse_it(db):
    insert_alert(db, retrain=False, value_psi=0.0, miss_psi=0.60)
    wanted = insert_alert(db, retrain=True)
    insert_alert(db, retrain=False, value_psi=0.0, miss_psi=0.61)
    assert store.latest_retrain_trigger(db)["id"] == wanted


# --------------------------------------------------------------------------
# An alert is consumed once
# --------------------------------------------------------------------------


def test_resolving_an_alert_stops_it_retriggering(db):
    alert_id = insert_alert(db)
    assert store.mark_resolved(alert_id, run_id="abc", model_version="2", path=db) is True
    assert store.latest_retrain_trigger(db) is None


def test_resolving_twice_is_refused(db):
    """A workflow that runs twice must not register two challengers for one
    drift measurement."""
    alert_id = insert_alert(db)
    assert store.mark_resolved(alert_id, path=db) is True
    assert store.mark_resolved(alert_id, path=db) is False


def test_resolution_records_which_run_answered_it(db):
    alert_id = insert_alert(db)
    store.mark_resolved(alert_id, run_id="run123", model_version="4", path=db)
    row = store.recent_alerts(1, path=db)[0]
    assert row["resolved_by_run_id"] == "run123"
    assert row["resolved_model_version"] == "4"
    assert row["resolved_at"]


def test_resolved_alerts_are_still_readable_for_audit(db):
    alert_id = insert_alert(db)
    store.mark_resolved(alert_id, run_id="r", path=db)
    assert store.latest_retrain_trigger(db, include_resolved=True)["id"] == alert_id


def test_migration_adds_resolution_columns_to_an_older_store(tmp_path):
    """CREATE TABLE IF NOT EXISTS silently leaves an older table alone, so a
    store written before resolution tracking would fail on the first UPDATE."""
    import sqlite3

    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE drift_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "created_at TEXT, window_label TEXT, window_rows INTEGER, "
        "reference_rows INTEGER, overall_psi REAL, value_drift_psi REAL, "
        "value_drift_features INTEGER, missingness_psi REAL, "
        "missingness_features INTEGER, n_major INTEGER, n_moderate INTEGER, "
        "n_stable INTEGER, n_unmeasurable INTEGER, threshold REAL, "
        "retrain_triggered INTEGER, investigate_pipeline INTEGER, "
        "verdict TEXT, top_features TEXT)"
    )
    legacy.commit()
    legacy.close()

    with store.connect(path) as connection:
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(drift_alerts)")}
    assert {"resolved_at", "resolved_by_run_id", "resolved_model_version"} <= columns


# --------------------------------------------------------------------------
# The decision is read, not re-derived
# --------------------------------------------------------------------------


def test_trigger_params_carry_the_cause_onto_the_run(db):
    alert = store.recent_alerts(1, path=db) if False else None
    insert_alert(db)
    alert = store.latest_retrain_trigger(db)
    params = trigger_params(alert)

    assert params["trigger.alert_id"] == alert["id"]
    assert params["trigger.value_drift_psi"] == pytest.approx(1.78)
    assert params["trigger.overall_psi_feature"] == "id_31"
    assert params["trigger.window_start_dt"] == alert["window_start_dt"]
    assert params["trigger.window_end_dt"] == alert["window_end_dt"]
    assert params["trigger.threshold"] == 0.20


def test_only_value_drift_features_are_named_as_the_cause(db):
    """A missingness feature must not be recorded as a reason for retraining
    -- the registry entry would then misattribute the model's existence."""
    insert_alert(db)
    params = trigger_params(store.latest_retrain_trigger(db))
    named = params["trigger.drifting_features"].split(",")
    assert "id_31" in named and "id_13" in named
    assert "M9" not in named


def test_trigger_params_survive_an_alert_with_no_top_features(db):
    insert_alert(db, top=[])
    params = trigger_params(store.latest_retrain_trigger(db))
    assert params["trigger.drifting_features"] == "(none recorded)"


# --------------------------------------------------------------------------
# Thread pinning, asserted on the fitted estimators
# --------------------------------------------------------------------------


class FakeModel:
    def __init__(self, n_jobs):
        self._n_jobs = n_jobs

    def get_params(self):
        return {"n_jobs": self._n_jobs}


def test_pinned_models_pass():
    assert assert_threads_pinned(FakeModel(N_JOBS), FakeModel(N_JOBS)) == N_JOBS


@pytest.mark.parametrize("bad", [-1, 1, 4, None])
def test_unpinned_model_is_refused(bad):
    """Checked on the fitted estimator, because that is the only thing that
    proves it -- a config value can be overridden silently."""
    if bad == N_JOBS:
        pytest.skip("that is the pinned value")
    with pytest.raises(ThreadPinningError, match="n_jobs"):
        assert_threads_pinned(FakeModel(bad))


def test_a_config_of_minus_one_is_refused_outright(monkeypatch):
    monkeypatch.setitem(XGB_HYPERPARAMETERS, "n_jobs", -1)
    with pytest.raises(ThreadPinningError, match="never reproducible"):
        assert_threads_pinned(FakeModel(-1))


def test_config_still_pins_to_two():
    assert N_JOBS == 2
    assert XGB_HYPERPARAMETERS["n_jobs"] == N_JOBS


# --------------------------------------------------------------------------
# Staging, never Production
# --------------------------------------------------------------------------


def test_target_stage_is_never_production():
    assert TARGET_STAGE == "Staging"
    assert TARGET_STAGE.lower() != "production"


def test_register_refuses_production(monkeypatch):
    import src.training.retrain as retrain_module

    monkeypatch.setattr(retrain_module, "TARGET_STAGE", "Production")
    with pytest.raises(RuntimeError, match="never register straight to Production"):
        retrain_module.register_challenger("run", 1, {"pr_auc": 0.5, "roc_auc": 0.9})


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_window_splits_are_disjoint_and_ordered_in_time(monkeypatch, tmp_path):
    """fit < stop < evaluate, strictly. Evaluating on the newest rows is the
    point: a retrain exists because recent behaviour changed."""
    import src.training.retrain as retrain_module

    n = 20_000
    frame = pd.DataFrame({
        "TransactionDT": np.arange(n) * 100.0,
        "isFraud": (np.arange(n) % 30 == 0).astype(int),
        "TransactionAmt": np.random.default_rng(0).lognormal(3, 1, n),
    })
    path = tmp_path / "part.parquet"
    frame.to_parquet(path, index=False)

    monkeypatch.setattr(retrain_module, "TRAIN_PARQUET", path)
    monkeypatch.setattr(retrain_module, "VAL_PARQUET", tmp_path / "absent.parquet")
    monkeypatch.setattr(retrain_module, "STREAM_PARQUET", tmp_path / "absent2.parquet")

    window = retrain_module.load_recent_window(window_days=10_000)
    assert window.fit["TransactionDT"].max() < window.stop["TransactionDT"].min()
    assert window.stop["TransactionDT"].max() < window.evaluation["TransactionDT"].min()
    assert window.rows == n
    assert len(window.evaluation) == pytest.approx(n * EVAL_FRACTION, rel=0.05)


def test_too_small_a_window_is_refused(monkeypatch, tmp_path):
    import src.training.retrain as retrain_module

    frame = pd.DataFrame({"TransactionDT": np.arange(50) * 100.0, "isFraud": 0})
    path = tmp_path / "tiny.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(retrain_module, "TRAIN_PARQUET", path)
    monkeypatch.setattr(retrain_module, "VAL_PARQUET", tmp_path / "a.parquet")
    monkeypatch.setattr(retrain_module, "STREAM_PARQUET", tmp_path / "b.parquet")

    with pytest.raises(ValueError, match="too few to retrain"):
        retrain_module.load_recent_window(window_days=10_000)


def test_retrain_without_an_alert_is_refused(db):
    import src.training.retrain as retrain_module

    with pytest.raises(RuntimeError, match="No unresolved retrain-triggering alert"):
        retrain_module.retrain(db_path=db)
