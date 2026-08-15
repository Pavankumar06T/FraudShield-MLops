"""Invariants for the dashboard's data layer.

The one that matters: the dashboard must read, never recompute. A panel that
recalculated PSI would eventually disagree with the monitor that fired the
alert -- a different window, a different bin edge, a different epsilon -- and
an operator would be looking at a number nobody can reconcile with the
decision actually taken.

Second: the promotion history must show rejections. A history assembled only
from the promotions table would display a run of unbroken successes, because
a challenger rejected outright never reaches that table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard import data
from src.drift import store


@pytest.fixture
def db(tmp_path):
    return tmp_path / "drift.db"


def insert_alert(db, *, value_psi=1.78, miss_psi=0.55, retrain=True,
                 value_list=None, miss_list=None):
    return store.DriftAlert(
        window_label="last 30 days", window_rows=85_431, reference_rows=319_927,
        overall_psi=max(value_psi, miss_psi), overall_psi_feature="id_31",
        value_drift_psi=value_psi, value_drift_features=11,
        missingness_psi=miss_psi, missingness_features=16,
        n_major=26, n_moderate=6, n_stable=277, n_unmeasurable=122,
        threshold=0.20, retrain_triggered=retrain, investigate_pipeline=True,
        verdict="RETRAIN + INVESTIGATE PIPELINE" if retrain else "STABLE",
        top_features=[{"feature": "id_31", "psi": value_psi, "type": "value"}],
        value_drift_feature_list=value_list or ["id_31", "id_13", "D11"],
        missingness_feature_list=miss_list or ["M9", "M8"],
    ).insert(db)


# --------------------------------------------------------------------------
# The two drift types stay separate
# --------------------------------------------------------------------------


def test_psi_history_keeps_the_two_drift_types_apart(db):
    """A blended series would erase the distinction the Phase 0
    decomposition exists to make."""
    insert_alert(db, value_psi=1.78, miss_psi=0.55)
    insert_alert(db, value_psi=2.27, miss_psi=0.60)

    frame = data.psi_history(db)
    assert len(frame) == 2
    assert "value_drift_psi" in frame.columns
    assert "missingness_psi" in frame.columns
    # they must be genuinely different series, not one column twice
    assert not frame["value_drift_psi"].equals(frame["missingness_psi"])
    assert frame["value_drift_psi"].tolist() == [1.78, 2.27]
    assert frame["missingness_psi"].tolist() == [0.55, 0.60]


def test_psi_values_are_read_verbatim_not_recomputed(db):
    """The exact figures the monitor stored must survive to the display."""
    insert_alert(db, value_psi=2.2655, miss_psi=0.6001)
    row = data.psi_history(db).iloc[0]
    assert row["value_drift_psi"] == pytest.approx(2.2655)
    assert row["missingness_psi"] == pytest.approx(0.6001)
    assert row["overall_psi_feature"] == "id_31"
    assert row["value_drift_features"] == 11
    assert row["missingness_features"] == 16


def test_psi_history_is_empty_not_broken_without_a_store(tmp_path):
    assert data.psi_history(tmp_path / "absent.db").empty


def test_feature_lists_survive_for_display(db):
    insert_alert(db, value_list=["id_31", "id_30", "V160"], miss_list=["M9"])
    row = data.psi_history(db).iloc[0]
    assert json.loads(row["value_drift_feature_list"]) == ["id_31", "id_30", "V160"]
    assert json.loads(row["missingness_feature_list"]) == ["M9"]


# --------------------------------------------------------------------------
# Live scoring degrades to history rather than erroring
# --------------------------------------------------------------------------


def make_rows(db, n=50, blocked_every=10, unseen_every=5):
    from src.streaming.consumer import ensure_schema, write_batch

    ensure_schema(db)
    rows = []
    for i in range(n):
        rows.append((
            store.utc_now(), str(i), 10_000_000.0 + i,
            0.9 if i % blocked_every == 0 else 0.1,
            "BLOCK" if i % blocked_every == 0 else "ALLOW",
            0.8018, "1", 799,
            json.dumps(["id_31"] if i % unseen_every == 0 else []),
            "[]", None, None, None, None,
            i % 30 == 0, 4.5,
        ))
    write_batch(rows, db)


def test_idle_consumer_shows_history_rather_than_an_error(db):
    """An idle consumer is a normal state. The panel must show the rows that
    exist, and say which state it is in."""
    make_rows(db, n=40)
    activity = data.scoring_activity(db)

    assert activity["rows"] == 40
    assert "reason" not in activity or activity["rows"] > 0
    assert isinstance(activity["live"], bool)
    assert not activity["recent"].empty


def test_no_rows_at_all_is_reported_not_raised(tmp_path):
    activity = data.scoring_activity(tmp_path / "empty.db")
    assert activity["rows"] == 0
    assert activity["live"] is False
    assert "reason" in activity


def test_flag_and_unseen_rates_summarise_stored_rows(db):
    make_rows(db, n=100, blocked_every=10, unseen_every=4)
    activity = data.scoring_activity(db)
    assert activity["blocked"] == 10
    assert activity["flag_rate"] == pytest.approx(0.10)
    assert activity["with_unseen"] == 25
    assert activity["unseen_rate"] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Unseen categories come from the shared aggregator
# --------------------------------------------------------------------------


def test_unseen_uses_the_same_aggregation_as_the_monitor(db):
    """Not its own query -- the dashboard and the monitor must report the
    same counts from the same code."""
    store.record_unseen(
        {("id_31", "chrome 66.0"): 253, ("id_31", "chrome 66.0 for android"): 154,
         ("id_30", "iOS 11.2.6"): 70, ("DeviceInfo", "SM-G950F"): 7,
         ("id_33", "2220x1081"): 10},
        model_version="1", path=db,
    )
    frame = data.unseen_categories(db)
    by_feature = frame.set_index("feature")

    assert by_feature.loc["id_31", "occurrences"] == 407
    assert by_feature.loc["id_31", "distinct_values"] == 2
    assert frame.iloc[0]["feature"] == "id_31"          # worst first
    assert set(by_feature.index) == {"id_31", "id_30", "DeviceInfo", "id_33"}


def test_unseen_examples_are_rendered_for_display(db):
    store.record_unseen({("id_31", "chrome 66.0"): 16}, path=db)
    frame = data.unseen_categories(db)
    assert "chrome 66.0 (16)" in frame.iloc[0]["examples"]
    assert "top_values" not in frame.columns


def test_unseen_is_empty_not_broken_before_anything_is_scored(tmp_path):
    assert data.unseen_categories(tmp_path / "none.db").empty


# --------------------------------------------------------------------------
# Promotion history must show refusals, not only successes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verdict,expected", [
    ("PROMOTE", "promoted"),
    ("ROLLED BACK -- evaluation rows were inside the training window", "rolled back"),
    ("NO VERDICT -- evaluation rows overlap the challenger's training window", "refused"),
    ("NO PROMOTION -- challenger is not ahead", "rejected"),
    ("no promotion, difference within noise", "rejected"),
])
def test_every_verdict_maps_to_an_outcome(verdict, expected):
    assert data.classify_verdict(verdict) == expected


def test_history_records_a_rollback_distinctly_from_a_rejection():
    """They are different failures: one was promoted then found invalid, the
    other never passed. Collapsing them would hide that a promotion was
    reversed."""
    assert data.classify_verdict("ROLLED BACK -- x") != data.classify_verdict(
        "NO PROMOTION -- challenger is not ahead"
    )


def test_promotion_history_reads_the_promotions_table(db):
    from src.serving.compare import PROMOTION_SCHEMA

    with store.connect(db) as connection:
        connection.executescript(PROMOTION_SCHEMA)
        connection.execute(
            "INSERT INTO model_promotions (promoted_at, from_version, to_version, "
            "trigger_alert_id, comparison_rows, comparison_positives, "
            "comparison_fraud_rate, champion_pr_auc, challenger_pr_auc, "
            "pr_auc_delta, bootstrap_std_error, margin_in_ses, verdict, metrics) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (store.utc_now(), "1", "4", 2, 11_429, 358, 0.03132,
             0.5299, 0.5661, 0.0361, 0.0085, 4.23, "PROMOTE", "{}"),
        )

    frame = data.promotion_history(db)
    promoted = frame[frame["version"] == "v4"]
    assert len(promoted) == 1
    assert promoted.iloc[0]["outcome"] == "promoted"
    assert promoted.iloc[0]["pr_auc_delta"] == pytest.approx(0.0361)
    assert promoted.iloc[0]["margin_ses"] == pytest.approx(4.23)
    assert promoted.iloc[0]["rows"] == 11_429


def test_promotion_history_is_empty_not_broken_without_the_table(tmp_path):
    frame = data.promotion_history(tmp_path / "bare.db")
    assert isinstance(frame, pd.DataFrame)


# --------------------------------------------------------------------------
# Nothing is recalculated
# --------------------------------------------------------------------------


def test_the_data_layer_does_not_import_the_psi_implementation():
    """A dashboard that could compute PSI would eventually compute it
    differently from the monitor."""
    source = Path("src/dashboard/data.py").read_text(encoding="utf-8")
    for forbidden in ("psi_numeric", "psi_categorical", "compute_psi_report",
                      "average_precision_score", "paired_bootstrap"):
        assert forbidden not in source, (
            f"{forbidden!r} in the dashboard means it can produce a number that "
            "disagrees with the tool that owns it"
        )


def test_the_app_does_not_import_the_psi_implementation():
    source = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    for forbidden in ("psi_numeric", "compute_psi_report", "average_precision_score"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# The two PR-AUC figures must both be available and distinguishable
# --------------------------------------------------------------------------


def seed_promotion(db, to_version="4", champion_pr=0.5299, challenger_pr=0.5661):
    from src.serving.compare import PROMOTION_SCHEMA

    with store.connect(db) as connection:
        connection.executescript(PROMOTION_SCHEMA)
        connection.execute(
            "INSERT INTO model_promotions (promoted_at, from_version, to_version, "
            "trigger_alert_id, comparison_rows, comparison_positives, "
            "comparison_fraud_rate, champion_pr_auc, challenger_pr_auc, "
            "pr_auc_delta, bootstrap_std_error, margin_in_ses, verdict, metrics) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (store.utc_now(), "1", to_version, 2, 11_429, 358, 0.03132,
             champion_pr, challenger_pr, challenger_pr - champion_pr,
             0.0085, 4.23, "PROMOTE", "{}"),
        )


def test_shadow_evidence_reports_the_unseen_row_figure(db):
    """The number that says what the model does in production, as opposed to
    on the tail of its own training window."""
    seed_promotion(db)
    evidence = data.shadow_evidence("4", db)

    assert evidence["available"] is True
    assert evidence["pr_auc"] == pytest.approx(0.5661)
    assert evidence["beat_version"] == "1"
    assert evidence["beat_pr_auc"] == pytest.approx(0.5299)
    assert evidence["delta"] == pytest.approx(0.0362, abs=1e-3)
    assert evidence["margin_ses"] == pytest.approx(4.23)
    assert evidence["rows"] == 11_429


def test_shadow_evidence_prefers_the_append_only_table(db):
    """shadow_comparison.json is overwritten by the next comparison; a
    promoted model's evidence must not vanish because a later candidate was
    tested."""
    seed_promotion(db)
    assert data.shadow_evidence("4", db)["source"] == "model_promotions"


def test_an_untested_model_says_so_rather_than_guessing(db):
    """Absent evidence must read as absent, not as a number."""
    seed_promotion(db, to_version="4")
    assert data.shadow_evidence("99", db).get("available") is not True


def test_the_two_figures_are_different_quantities(db):
    """v4 recorded 0.4834 on its own eval slice and 0.5661 on unseen rows.
    Showing one alone makes the other look like a contradiction."""
    seed_promotion(db)
    shadow = data.shadow_evidence("4", db)["pr_auc"]
    own_eval = 0.4834
    assert shadow != pytest.approx(own_eval, abs=0.01)


# --------------------------------------------------------------------------
# Rejections must survive alongside promotions
# --------------------------------------------------------------------------


def test_promotions_table_alone_omits_rejected_challengers(db):
    """The finding this dashboard exists to avoid repeating: the success
    path writes a row and the rejection path just returns, so a history
    built from one source shows an unbroken run of wins."""
    seed_promotion(db)
    with store.connect(db) as connection:
        rows = connection.execute(
            "SELECT to_version FROM model_promotions"
        ).fetchall()

    versions_in_table = {r["to_version"] for r in rows}
    assert "4" in versions_in_table
    assert "2" not in versions_in_table, (
        "a rejected challenger never reaches model_promotions -- which is "
        "exactly why promotion_history must read version tags too"
    )


def test_history_labels_which_source_each_row_came_from(db):
    """So the omission cannot recur silently."""
    seed_promotion(db)
    frame = data.promotion_history(db)
    assert "source" in frame.columns
    assert (frame["source"] == "model_promotions").any()
