"""MLflow experiment tracking, optional by construction.

Every entry point here is a no-op when MLflow is missing or the store cannot
be opened. Training is the thing that matters; recording it is valuable but
must never be the reason a run fails. Import errors, a corrupt store, a
locked file -- all degrade to "not tracked" with one warning, never to a
traceback in the middle of a twenty-minute fit.

The store is a local SQLite file, so nothing depends on a server being up.
Point ``MLFLOW_TRACKING_URI`` elsewhere to override -- at a Drive path on
Colab, so runs survive the runtime being recycled.

SQLite rather than the older ``mlruns/`` directory store for two reasons,
both forced rather than chosen. MLflow 3 puts the filesystem backend in
maintenance mode and raises on it unless ``MLFLOW_ALLOW_FILE_STORE=true``;
and the Model Registry has never worked against the file store at all, so
promoting a model would be impossible. SQLite is still a single local file
with no server behind it.

One run per model. They are siblings rather than parent and children, and
each carries the full data description -- carve boundary, row counts, fraud
rates -- rather than inheriting it. A run has to be reproducible from its
own record; a record that only makes sense next to its siblings is not one.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.common.config import REPO_ROOT

#: Default experiment name. Runs land here unless MLFLOW_EXPERIMENT_NAME says
#: otherwise.
EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraudshield-baseline")

#: Local SQLite store. Gitignored -- it holds run records, not source.
DEFAULT_TRACKING_DB: Path = REPO_ROOT / "mlflow.db"

#: Where logged model binaries and artifacts land. Separate from the store
#: because a SQLite backend keeps only references, not blobs.
DEFAULT_ARTIFACT_ROOT: Path = REPO_ROOT / "mlartifacts"

#: Registered name for the production fraud model.
REGISTERED_MODEL_NAME: str = "fraudshield-xgboost"

_WARNED = False


def _warn_once(message: str) -> None:
    """Say it once. A per-run warning inside a loop is noise, not signal."""
    global _WARNED
    if not _WARNED:
        warnings.warn(f"MLflow tracking disabled: {message}", RuntimeWarning, stacklevel=2)
        _WARNED = True


def tracking_uri() -> str:
    """Resolve the store location, preferring an explicit override.

    Forward slashes even on Windows: SQLAlchemy parses the path after
    ``sqlite:///`` as a URL, and a backslash there is an escape rather than
    a separator.
    """
    override = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if override:
        # A caller who insists on the legacy directory store gets it, but
        # MLflow 3 raises on that backend without this opt-out -- and the
        # Model Registry will still be unavailable.
        if override.startswith("file:") or override.endswith("mlruns"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        return override
    return f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}"


def artifact_root() -> str:
    """Directory for logged models and files, as a URI."""
    override = os.environ.get("MLFLOW_ARTIFACT_ROOT", "").strip()
    if override:
        return override
    DEFAULT_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_ARTIFACT_ROOT.as_uri()


def mlflow_module():
    """Return the mlflow module, or None if it is unusable.

    Catches Exception rather than ImportError alone: MLflow pulls in a large
    dependency tree, and a version skew inside it surfaces as anything from
    AttributeError to a database error at import time.
    """
    try:
        import mlflow  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on the environment
        _warn_once(f"{type(exc).__name__}: {exc}")
        return None
    return mlflow


def is_available() -> bool:
    return mlflow_module() is not None


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    """Flatten nested metric dicts into dotted keys MLflow will accept."""
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), nested, out)
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        number = float(value)
        if number == number and abs(number) != float("inf"):  # drop NaN/inf
            out[prefix] = number


def flatten_metrics(metrics: dict) -> dict[str, float]:
    """Every numeric leaf of a metric block, as ``a.b.c`` keys.

    Turns the nested threshold blocks into ``at_best_f1_threshold.precision``
    and so on, so the confusion counts are queryable rather than buried in a
    JSON artifact.
    """
    flat: dict[str, float] = {}
    _flatten("", metrics, flat)
    return flat


def ensure_experiment(mlflow) -> str:
    """Point MLflow at the store and make sure the experiment exists.

    Created explicitly rather than via ``set_experiment`` alone, because the
    artifact location can only be set at creation time -- and with a SQLite
    backend the default lands relative to the working directory, so running
    from a notebook and from the repo root would scatter artifacts in two
    places.
    """
    mlflow.set_tracking_uri(tracking_uri())
    existing = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return EXPERIMENT_NAME


@contextmanager
def start_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Open an MLflow run, or yield None if tracking is unavailable.

    Callers write ``with start_run(...) as run:`` and guard on ``run`` being
    None, so the no-MLflow path costs one branch rather than a parallel
    code path.
    """
    mlflow = mlflow_module()
    if mlflow is None:
        yield None
        return

    try:
        ensure_experiment(mlflow)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as exc:  # pragma: no cover - depends on the environment
        _warn_once(f"could not open the store ({type(exc).__name__}: {exc})")
        yield None


def log_params(run: Any, params: dict[str, Any]) -> None:
    """Log parameters, stringified, skipping Nones."""
    if run is None:
        return
    mlflow = mlflow_module()
    if mlflow is None:
        return
    try:
        mlflow.log_params(
            {key: str(value) for key, value in params.items() if value is not None}
        )
    except Exception as exc:  # pragma: no cover
        _warn_once(f"log_params failed ({type(exc).__name__}: {exc})")


def log_metrics(run: Any, metrics: dict[str, float]) -> None:
    if run is None:
        return
    mlflow = mlflow_module()
    if mlflow is None:
        return
    try:
        mlflow.log_metrics(metrics)
    except Exception as exc:  # pragma: no cover
        _warn_once(f"log_metrics failed ({type(exc).__name__}: {exc})")


def log_artifact(run: Any, path: Path, artifact_path: str | None = None) -> None:
    """Attach a file to the run, if it exists."""
    if run is None or not Path(path).exists():
        return
    mlflow = mlflow_module()
    if mlflow is None:
        return
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover
        _warn_once(f"log_artifact failed ({type(exc).__name__}: {exc})")


def log_model(run: Any, model: Any, flavour: str, artifact_path: str = "model") -> None:
    """Log a fitted model under its native MLflow flavour.

    Flavour-specific logging rather than a generic pickle, so the artifact
    carries the library version and can be loaded back without this
    codebase present.
    """
    if run is None:
        return
    mlflow = mlflow_module()
    if mlflow is None:
        return
    try:
        module = getattr(mlflow, flavour)
        # `name` replaced `artifact_path` in MLflow 3; try the modern spelling
        # first and fall back so both lines work.
        try:
            module.log_model(model, name=artifact_path)
        except TypeError:
            module.log_model(model, artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover
        _warn_once(f"log_model failed ({type(exc).__name__}: {exc})")


def describe_store() -> str:
    """One line for the run log: where records are going, or why nowhere."""
    if not is_available():
        return "  MLflow not available -- training proceeds untracked."
    return (
        f"  MLflow store {tracking_uri()}\n"
        f"  artifacts    {artifact_root()}\n"
        f"  experiment   {EXPERIMENT_NAME!r}"
    )
