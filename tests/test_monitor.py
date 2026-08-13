"""Invariants for the drift monitor and its alert store.

The one that carries the most weight: missingness-driven drift must never
fire a retrain. Sixteen features in this dataset have a PSI above 0.20
purely because their coverage improved -- their populated values are
unchanged to three decimal places. Retraining on that would bake a transient
upstream join into the model, and a blended PSI number would make it
indistinguishable from the drift that genuinely needs a retrain.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.drift import store
from src.drift.monitor import (
    DEFAULT_THRESHOLD,
    PHASE0_BANDS,
    VALUE_DRIFT_RESIDUAL,
    DriftResult,
    load_window,
    split_drift_types,
    write_decision,
)
from src.drift.report import MAJOR_ABOVE


@pytest.fixture
def db(tmp_path):
    return tmp_path / "drift.db"


def make_result(**overrides) -> DriftResult:
    """A result shaped like the real one, with both drift types present."""
    psi = pd.Series(
        {"id_31": 1.5254, "id_13": 0.5661, "M8": 0.5303, "D11": 0.4496,
         "V3": 0.4131, "TransactionAmt": 0.02},
        name="psi",
    ).sort_values(ascending=False)
    decomposition = pd.DataFrame(
        {
            "psi_all": [1.5254, 0.5661, 0.5303, 0.4496, 0.4131],
            "psi_non_null": [7.5787, 2.4739, 0.0001, 0.0735, 0.0066],
            "na_rate_reference": [0.7184, 0.7529, 0.7459, 0.6166, 0.6166],
            "na_rate_current": [0.8135, 0.8199, 0.3941, 0.3033, 0.3033],
        },
        index=pd.Index(["id_31", "id_13", "M8", "D11", "V3"], name="feature"),
    )
    value, missingness = split_drift_types(decomposition, psi, MAJOR_ABOVE)
    defaults = dict(
        psi=psi, decomposition=decomposition, value_drift=value,
        missingness_drift=missingness,
        bands={"stable": 278, "moderate": 4, "major": 27, "unmeasurable": 122},
        window_rows=50_000, reference_rows=319_927, window_label="last 30 days",
        window_start_dt=1.0e7, window_end_dt=1.5e7, threshold=MAJOR_ABOVE, unseen=[],
    )
    defaults.update(overrides)
    return DriftResult(**defaults)


# --------------------------------------------------------------------------
# The separation the Phase 0 analysis exists to make
# --------------------------------------------------------------------------


def test_missingness_and_value_drift_are_separated():
    result = make_result()
    assert set(result.value_drift) == {"id_31", "id_13", "D11"}
    assert set(result.missingness_drift) == {"M8", "V3"}


def test_missingness_drift_alone_never_retrains():
    """M8 sits at PSI 0.53 with populated-row PSI of 0.0001. Its coverage
    went from 74.6% missing to 39.4% missing -- more data arrived. Retraining
    on that bakes an upstream join into the model."""
    psi = pd.Series({"M8": 0.5303, "V3": 0.4131})
    decomposition = pd.DataFrame(
        {"psi_non_null": [0.0001, 0.0066],
         "na_rate_reference": [0.7459, 0.6166], "na_rate_current": [0.3941, 0.3033]},
        index=pd.Index(["M8", "V3"], name="feature"),
    )
    value, missingness = split_drift_types(decomposition, psi, MAJOR_ABOVE)
    result = make_result(
        psi=psi, decomposition=decomposition,
        value_drift=value, missingness_drift=missingness,
    )
    assert result.value_drift == []
    assert result.missingness_psi > 0.5
    assert result.retrain_triggered is False
    assert result.investigate_pipeline is True
    assert result.verdict == "INVESTIGATE PIPELINE (no retrain)"


def test_value_drift_alone_retrains():
    psi = pd.Series({"id_31": 1.5254})
    decomposition = pd.DataFrame(
        {"psi_non_null": [7.5787], "na_rate_reference": [0.7184], "na_rate_current": [0.8135]},
        index=pd.Index(["id_31"], name="feature"),
    )
    value, missingness = split_drift_types(decomposition, psi, MAJOR_ABOVE)
    result = make_result(psi=psi, decomposition=decomposition,
                         value_drift=value, missingness_drift=missingness)
    assert result.retrain_triggered is True
    assert result.investigate_pipeline is False
    assert result.verdict == "RETRAIN"


def test_both_types_reported_independently():
    result = make_result()
    assert result.retrain_triggered and result.investigate_pipeline
    assert result.verdict == "RETRAIN + INVESTIGATE PIPELINE"
    # and the two maxima are genuinely different numbers, not one blend
    assert result.value_drift_psi != result.missingness_psi


def test_a_blended_average_would_have_hidden_the_signal():
    """The argument for max-not-mean, at the real proportions.

    431 features of which 278 are stable and 27 major. Averaging drowns the
    collapsed column: the mean lands below the 0.20 threshold while the
    worst feature sits at 1.53, so a blended monitor would report calm.
    """
    drifting = {"id_31": 1.5254, "id_13": 0.5661, "M8": 0.5303, "D11": 0.4496}
    stable = {f"V{i}": 0.01 for i in range(400)}
    psi = pd.Series({**drifting, **stable}).sort_values(ascending=False)

    result = make_result(psi=psi)
    assert psi.mean() < DEFAULT_THRESHOLD, "fixture must mirror the real shape"
    assert result.overall_psi > DEFAULT_THRESHOLD
    assert result.overall_psi_feature == "id_31"


def test_borderline_feature_lands_on_the_conservative_side():
    """D11's populated-row PSI is 0.0735 -- below the 0.10 stable band but
    well above the 0.000-0.007 of the true missingness cluster. Calling it
    genuine risks an unnecessary retrain; calling it missingness risks not
    retraining when the values really moved."""
    assert VALUE_DRIFT_RESIDUAL < 0.0735 < 0.10
    assert "D11" in make_result().value_drift


@pytest.mark.parametrize("cut", [0.01, 0.03, 0.05, 0.07])
def test_the_classification_is_insensitive_to_where_the_cut_falls(cut):
    """The two clusters separate at 0.000-0.007 and 0.07-7.58, so any cut in
    between gives the same answer. The threshold is a choice, not a knob
    that has to be tuned."""
    result = make_result()
    psi, decomposition = result.psi, result.decomposition
    value, _ = split_drift_types(decomposition, psi, MAJOR_ABOVE, residual=cut)
    assert set(value) == {"id_31", "id_13", "D11"}


def test_undecomposed_feature_defaults_to_genuine():
    """Not classifying is not the same as classifying as harmless -- the
    safer error is an unnecessary retrain, not a missed one."""
    psi = pd.Series({"mystery": 0.9})
    value, missingness = split_drift_types(pd.DataFrame(), psi, MAJOR_ABOVE)
    assert value == ["mystery"] and missingness == []


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


def test_default_threshold_matches_phase_0():
    assert DEFAULT_THRESHOLD == MAJOR_ABOVE == 0.20


def test_threshold_is_configurable():
    """id_31 drives value drift at 1.5254, so the threshold has to clear
    that to suppress the trigger."""
    assert make_result(threshold=2.0).retrain_triggered is False
    assert make_result(threshold=1.0).retrain_triggered is True
    assert make_result(threshold=0.10).retrain_triggered is True


def test_phase0_bands_are_pinned():
    assert PHASE0_BANDS == {"stable": 278, "moderate": 4, "major": 27}
    assert sum(PHASE0_BANDS.values()) == 309


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_window_is_selected_by_time_not_row_position(tmp_path):
    """The stream arrives ordered, but a monitor that assumed so would
    compare the wrong rows the first time it did not."""
    n = 1_000
    frame = pd.DataFrame(
        {"TransactionDT": np.arange(n) * 100.0, "isFraud": 0, "v": np.arange(n)}
    ).sample(frac=1.0, random_state=3)
    path = tmp_path / "stream.parquet"
    frame.to_parquet(path, index=False)

    window, label, start, end = load_window(window_rows=100, path=path)
    assert len(window) == 100
    assert window["TransactionDT"].min() == start
    # every selected row post-dates every excluded one
    excluded = frame[frame["TransactionDT"] < start]
    assert excluded["TransactionDT"].max() < start


def test_empty_window_is_refused(tmp_path):
    frame = pd.DataFrame({"TransactionDT": [1.0, 2.0], "v": [1, 2]})
    path = tmp_path / "s.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="selected no rows"):
        load_window(window_rows=0, path=path)


# --------------------------------------------------------------------------
# Alert store
# --------------------------------------------------------------------------


def test_alert_round_trips(db):
    result = make_result()
    alert = store.DriftAlert(
        window_label=result.window_label, window_rows=result.window_rows,
        reference_rows=result.reference_rows, overall_psi=result.overall_psi,
        overall_psi_feature=result.overall_psi_feature,
        value_drift_psi=result.value_drift_psi,
        value_drift_features=len(result.value_drift),
        missingness_psi=result.missingness_psi,
        missingness_features=len(result.missingness_drift),
        n_major=27, n_moderate=4, n_stable=278, n_unmeasurable=122,
        threshold=result.threshold, retrain_triggered=result.retrain_triggered,
        investigate_pipeline=result.investigate_pipeline, verdict=result.verdict,
        top_features=result.top_features(),
    )
    alert_id = alert.insert(db)
    assert alert_id > 0

    rows = store.recent_alerts(path=db)
    assert len(rows) == 1
    row = rows[0]
    assert row["verdict"] == "RETRAIN + INVESTIGATE PIPELINE"
    assert row["retrain_triggered"] == 1
    assert row["value_drift_features"] == 3
    assert row["missingness_features"] == 2
    assert json.loads(row["top_features"])[0]["feature"] == "id_31"


def test_per_type_breakdown_survives_the_round_trip(db):
    """The two numbers must stay distinguishable in storage -- collapsing
    them there would undo the separation the monitor computes."""
    result = make_result()
    store.DriftAlert(
        window_label="w", window_rows=1, reference_rows=1,
        overall_psi=result.overall_psi, overall_psi_feature="id_31",
        value_drift_psi=result.value_drift_psi, value_drift_features=3,
        missingness_psi=result.missingness_psi, missingness_features=2,
        n_major=27, n_moderate=4, n_stable=278, n_unmeasurable=122,
        threshold=0.20, retrain_triggered=True, investigate_pipeline=True,
        verdict="RETRAIN + INVESTIGATE PIPELINE", top_features=[],
    ).insert(db)
    row = store.recent_alerts(path=db)[0]
    assert row["value_drift_psi"] == pytest.approx(result.value_drift_psi)
    assert row["missingness_psi"] == pytest.approx(result.missingness_psi)
    assert row["value_drift_psi"] != row["missingness_psi"]


def test_latest_retrain_trigger_ignores_pipeline_only_alerts(db):
    common = dict(
        window_label="w", window_rows=1, reference_rows=1, overall_psi=0.5,
        overall_psi_feature="M8", n_major=1, n_moderate=0, n_stable=1,
        n_unmeasurable=0, threshold=0.20, top_features=[],
    )
    store.DriftAlert(
        value_drift_psi=0.0, value_drift_features=0, missingness_psi=0.53,
        missingness_features=1, retrain_triggered=False, investigate_pipeline=True,
        verdict="INVESTIGATE PIPELINE (no retrain)", **common
    ).insert(db)
    assert store.latest_retrain_trigger(db) is None

    store.DriftAlert(
        value_drift_psi=1.53, value_drift_features=1, missingness_psi=0.0,
        missingness_features=0, retrain_triggered=True, investigate_pipeline=False,
        verdict="RETRAIN", **common
    ).insert(db)
    assert store.latest_retrain_trigger(db)["verdict"] == "RETRAIN"


# --------------------------------------------------------------------------
# Unseen categories -- the fast signal
# --------------------------------------------------------------------------


def test_unseen_observations_aggregate_by_feature(db):
    store.record_unseen(
        Counter({("id_31", "chrome 65.0"): 12, ("id_31", "opera 51"): 3,
                 ("id_30", "Android 9"): 5}),
        model_version="1", path=db,
    )
    rows = store.aggregate_unseen(path=db)
    by_feature = {r["feature"]: r for r in rows}

    assert by_feature["id_31"]["occurrences"] == 15
    assert by_feature["id_31"]["distinct_values"] == 2
    assert by_feature["id_30"]["occurrences"] == 5
    assert rows[0]["feature"] == "id_31"  # worst first
    assert by_feature["id_31"]["top_values"][0]["value"] == "chrome 65.0"


def test_unseen_store_is_empty_not_broken_before_serving_runs(db):
    assert store.aggregate_unseen(path=db) == []


def test_recording_nothing_is_a_noop(db):
    assert store.record_unseen({}, path=db) == 0


# --------------------------------------------------------------------------
# The decision artifact later phases read
# --------------------------------------------------------------------------


def test_decision_file_records_both_types_and_their_remedies(tmp_path):
    path = write_decision(make_result(), tmp_path / "drift_decision.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["value_drift"]["count"] == 3
    assert payload["missingness_drift"]["count"] == 2
    assert "retrain" in payload["value_drift"]["remedy"]
    assert "upstream" in payload["missingness_drift"]["remedy"]
    assert "retrain" not in payload["missingness_drift"]["remedy"].split("retraining")[0]
    assert payload["retrain_triggered"] is True
    assert payload["threshold"] == 0.20
    assert payload["overall_psi_feature"] == "id_31"
