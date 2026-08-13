"""Invariants for experiment tracking and data versioning.

The one that matters most is that none of it is load-bearing. Tracking a run
is valuable; it must never be the reason a twenty-minute fit dies. Every
entry point degrades to a no-op rather than raising.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import REPO_ROOT
from src.data import version
from src.training import tracking
from src.training.register_model import STAGE_ALIASES, promotion_description
from src.training.train import REFERENCE_BASELINE


# --------------------------------------------------------------------------
# Graceful degradation -- tracking must never be load-bearing
# --------------------------------------------------------------------------


@pytest.fixture
def no_mlflow(monkeypatch):
    """Simulate MLflow being absent or broken."""
    monkeypatch.setattr(tracking, "mlflow_module", lambda: None)
    monkeypatch.setattr(tracking, "_WARNED", True)  # keep the suite quiet


def test_start_run_yields_none_without_mlflow(no_mlflow):
    with tracking.start_run("anything") as run:
        assert run is None


def test_every_log_call_is_a_noop_without_mlflow(no_mlflow):
    """Callers guard on `run is None`, but the log functions must be safe
    even when handed a truthy run and no MLflow behind them."""
    tracking.log_params(None, {"a": 1})
    tracking.log_metrics(None, {"m": 1.0})
    tracking.log_artifact(None, REPO_ROOT / "requirements.txt")
    tracking.log_model(None, object(), "xgboost")
    tracking.log_params(object(), {"a": 1})
    tracking.log_metrics(object(), {"m": 1.0})


def test_describe_store_says_so_when_unavailable(no_mlflow):
    assert "not available" in tracking.describe_store()
    assert not tracking.is_available()


def test_log_artifact_skips_a_missing_file():
    tracking.log_artifact(object(), REPO_ROOT / "does-not-exist.bin")


# --------------------------------------------------------------------------
# Store resolution
# --------------------------------------------------------------------------


def test_default_store_is_sqlite_not_the_file_backend(monkeypatch):
    """MLflow 3 refuses the mlruns/ file store, and the Model Registry has
    never worked against it -- so the default has to be a database."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = tracking.tracking_uri()
    assert uri.startswith("sqlite:///")
    assert "mlflow.db" in uri
    # forward slashes: SQLAlchemy treats a backslash after sqlite:/// as an
    # escape rather than a separator
    assert "\\" not in uri


def test_explicit_override_is_honoured(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///somewhere/else.db")
    assert tracking.tracking_uri() == "sqlite:///somewhere/else.db"


def test_file_store_override_sets_the_opt_out(monkeypatch):
    """Someone insisting on the legacy store gets it, but MLflow 3 raises
    without this environment flag."""
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    tracking.tracking_uri()
    import os

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


# --------------------------------------------------------------------------
# Metric flattening
# --------------------------------------------------------------------------


def test_nested_threshold_blocks_flatten_to_queryable_keys():
    flat = tracking.flatten_metrics(
        {
            "pr_auc": 0.5255,
            "at_best_f1_threshold": {"precision": 0.4, "true_positives": 900},
        }
    )
    assert flat["pr_auc"] == pytest.approx(0.5255)
    assert flat["at_best_f1_threshold.precision"] == pytest.approx(0.4)
    assert flat["at_best_f1_threshold.true_positives"] == pytest.approx(900)


def test_flattening_drops_values_mlflow_cannot_store():
    flat = tracking.flatten_metrics(
        {
            "good": 1.0,
            "nan": float("nan"),
            "inf": float("inf"),
            "none": None,
            "flag": True,
            "text": "not a metric",
        }
    )
    assert set(flat) == {"good"}


def test_flattening_survives_the_real_metric_shape():
    """Guards against a nesting change silently dropping the confusion
    counts, which are the numbers an audit would ask for."""
    block = {
        "pr_auc": 0.5,
        "at_default_threshold": {"f1": 0.1, "true_positives": 5, "flagged": 9},
        "at_best_f1_threshold": {"f1": 0.2, "false_negatives": 3},
    }
    flat = tracking.flatten_metrics(block)
    assert len(flat) == 6
    assert all(isinstance(v, float) for v in flat.values())


# --------------------------------------------------------------------------
# Promotion description
# --------------------------------------------------------------------------


def test_description_reports_the_promoted_figures():
    """Sourced from the constant, so a reference change cannot leave the
    registry describing a model nobody trained."""
    text = promotion_description()
    assert f"{float(REFERENCE_BASELINE['pr_auc']):.4f}" in text
    assert f"{float(REFERENCE_BASELINE['roc_auc']):.4f}" in text
    assert f"+{float(REFERENCE_BASELINE['overfit_gap_pr_auc']):.4f}" in text
    assert str(int(REFERENCE_BASELINE["n_trees_used"])) in text
    assert "depth4_reg" in text
    assert "15% temporal carve" in text


def test_description_records_the_pinned_thread_count():
    """A tree count without the thread count that produced it is not
    reproducible information."""
    text = promotion_description()
    assert f"n_jobs         {REFERENCE_BASELINE['n_jobs']} (pinned)" in text
    assert "not\nthread-deterministic" in text or "thread-deterministic" in text
    assert "651, 799 or 969" in text


def test_description_warns_against_the_superseded_figure():
    """0.5477 must appear only as a thing not to compare against."""
    text = promotion_description()
    assert "0.5477" in text
    assert "NOT comparable" in text
    assert "48k rows" in text


def test_description_survives_a_pending_roc_auc(monkeypatch):
    monkeypatch.setitem(REFERENCE_BASELINE, "roc_auc", None)
    text = promotion_description()
    assert f"{float(REFERENCE_BASELINE['pr_auc']):.4f}" in text
    assert "ROC-AUC " not in text


def test_every_stage_has_an_alias():
    """Stages are deprecated in MLflow 3 and will be removed; the alias is
    what keeps the model loadable afterwards."""
    assert STAGE_ALIASES["Staging"] == "staging"
    assert set(STAGE_ALIASES) == {"Staging", "Production", "Archived"}


# --------------------------------------------------------------------------
# Data versioning
# --------------------------------------------------------------------------


def test_tracked_paths_are_repo_relative_and_cover_both_data_dirs():
    assert version.TRACKED == ("data/raw", "data/splits")
    for path in version.TRACKED:
        assert not path.startswith("/")


def test_link_path_is_the_repo_data_directory():
    assert version.LINK_PATH == REPO_ROOT / "data"


def test_link_status_reports_a_real_directory():
    assert version.link_status() in {"missing", "link", "directory"}


def test_ensure_link_is_a_noop_when_data_is_already_in_the_repo(monkeypatch):
    monkeypatch.setattr(version, "DATA_ROOT", version.LINK_PATH)
    assert "nothing to link" in version.ensure_link()


def test_ensure_link_refuses_to_destroy_a_populated_directory(monkeypatch, tmp_path):
    """The repo's data/ holds the real splits. Replacing it with a link to
    somewhere else would delete 81 MB to make a tool happy."""
    populated = tmp_path / "repo_data"
    (populated / "splits").mkdir(parents=True)
    (populated / "splits" / "train.parquet").write_bytes(b"not really parquet")

    monkeypatch.setattr(version, "LINK_PATH", populated)
    monkeypatch.setattr(version, "DATA_ROOT", tmp_path / "elsewhere")

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        version.ensure_link()
    assert (populated / "splits" / "train.parquet").exists()


def test_ensure_link_reports_a_missing_data_root(monkeypatch, tmp_path):
    monkeypatch.setattr(version, "LINK_PATH", tmp_path / "link")
    monkeypatch.setattr(version, "DATA_ROOT", tmp_path / "never-mounted")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        version.ensure_link()
