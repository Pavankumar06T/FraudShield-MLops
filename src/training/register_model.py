"""Register the promoted XGBoost model in the MLflow Model Registry.

The registry is the answer to "what was running on the 14th, and how did it
score?" -- the question an auditor asks and a metrics file cannot answer,
because a metrics file is overwritten by the next run.

Registration is deliberately a separate command rather than a step inside
training. Every training run produces a model; almost none should be
promoted. Coupling the two would make promotion a side effect of fitting,
which is exactly the silent-baseline-drift this project keeps guarding
against.

    python -m src.training.register_model                # newest xgboost run
    python -m src.training.register_model --run-id abc123
    python -m src.training.register_model --stage Production

A note on stages. MLflow 3 deprecates the Staging/Production stage strings
in favour of named aliases, and will remove them. Both are set: the stage
because it is what was asked for and what the tooling around this still
reads, and an equivalent alias so nothing breaks when stages go.
"""

from __future__ import annotations

import argparse
import sys
import warnings

from src.training import tracking
from src.training.train import REFERENCE_BASELINE

#: Stage the promoted model lands in.
DEFAULT_STAGE: str = "Staging"

#: Alias mirroring the stage, for the MLflow 3+ world where stages are gone.
STAGE_ALIASES: dict[str, str] = {
    "Staging": "staging",
    "Production": "champion",
    "Archived": "archived",
}


def promotion_description() -> str:
    """What this version is, in the terms the promotion was decided on.

    Sourced from REFERENCE_BASELINE rather than restated, so the registry
    entry and the constant the training run checks against cannot drift
    apart -- a description claiming a score the code no longer expects is
    worse than no description.
    """
    reference = REFERENCE_BASELINE
    lines = [
        "depth4_reg with early stopping on a 15% temporal carve of train.",
        "",
        f"  PR-AUC         {float(reference['pr_auc']):.4f}",
    ]
    roc = reference.get("roc_auc")
    if roc is not None:
        lines.append(f"  ROC-AUC        {float(roc):.4f}")
    lines += [
        f"  train/val gap  +{float(reference['overfit_gap_pr_auc']):.4f}",
        f"  trees used     {int(reference['n_trees_used'])} of 2000 (early stopped)",
        f"  n_jobs         {reference.get('n_jobs')} (pinned)",
        "",
        "Promoted over the depth-6 configuration on a comparison where both ran "
        "at the same thread count: 0.5291 at a +0.2734 gap against 0.5255 at "
        "+0.1932, trading 0.0036 of PR-AUC for 0.0802 of overfit gap.",
        "",
        "The figures above come from a later run at a pinned n_jobs, so they "
        "differ slightly from that comparison. XGBoost's hist method is not "
        "thread-deterministic -- its subsample and colsample RNG streams are "
        "per-thread -- so the same configuration yields 651, 799 or 969 trees "
        "depending on core count. All score within 0.0023 PR-AUC, a quarter of "
        "the 0.0080 bootstrap standard error, so none is better; the pin makes "
        "the choice repeatable. See docs/reproducibility.md.",
        "",
        "NOT comparable to the earlier 0.5477 figure, which was trained on the "
        "full 319,927-row split with no early-stopping carve. The entire "
        "difference between the two is the 48k rows the carve removes -- early "
        "stopping itself cost nothing.",
    ]
    return "\n".join(lines)


def latest_run_id(mlflow, model: str = "xgboost") -> str | None:
    """Most recent finished run for a given model in the experiment."""
    from mlflow import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(tracking.EXPERIMENT_NAME)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.model = '{model}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def register(
    run_id: str | None = None,
    stage: str = DEFAULT_STAGE,
    name: str = tracking.REGISTERED_MODEL_NAME,
    artifact_path: str = "model",
) -> dict | None:
    """Register a run's model and move it to ``stage``. None if unavailable."""
    mlflow = tracking.mlflow_module()
    if mlflow is None:
        print("MLflow is not available -- nothing to register.")
        return None

    mlflow.set_tracking_uri(tracking.tracking_uri())
    from mlflow import MlflowClient

    client = MlflowClient()

    if run_id is None:
        run_id = latest_run_id(mlflow)
        if run_id is None:
            print(
                f"No xgboost run found in experiment {tracking.EXPERIMENT_NAME!r}.\n"
                "Train first:  python -m src.training.train"
            )
            return None

    uri = f"runs:/{run_id}/{artifact_path}"
    print(f"registering {uri}\n  as {name!r}")

    with warnings.catch_warnings():
        # Stage transitions warn loudly on MLflow 3; the deprecation is
        # acknowledged in the module docstring and handled by also setting
        # an alias.
        warnings.simplefilter("ignore")
        version = mlflow.register_model(model_uri=uri, name=name)

        client.update_model_version(
            name=name, version=version.version, description=promotion_description()
        )
        client.set_model_version_tag(
            name, version.version, "regime", "depth4_reg + 15% temporal carve"
        )
        client.set_model_version_tag(
            name, version.version, "pr_auc", f"{float(REFERENCE_BASELINE['pr_auc']):.4f}"
        )

        staged = False
        try:
            client.transition_model_version_stage(
                name=name, version=version.version, stage=stage
            )
            staged = True
        except Exception as exc:
            print(f"  stage transition unavailable ({type(exc).__name__}); alias only")

        alias = STAGE_ALIASES.get(stage, stage.lower())
        client.set_registered_model_alias(name, alias, version.version)

    print(
        f"  version  {version.version}\n"
        f"  stage    {stage if staged else '(unset -- stages removed)'}\n"
        f"  alias    @{alias}\n"
        f"  load as  models:/{name}@{alias}"
    )
    if str(version.version) != "1":
        print(
            f"\n  Note: this is version {version.version}, not 1 -- the registry "
            "already\n  held earlier versions of this model name."
        )
    return {
        "name": name,
        "version": str(version.version),
        "stage": stage if staged else None,
        "alias": alias,
        "run_id": run_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register the promoted XGBoost model in the MLflow registry."
    )
    parser.add_argument("--run-id", default=None, help="defaults to the newest xgboost run")
    parser.add_argument("--stage", default=DEFAULT_STAGE, choices=sorted(STAGE_ALIASES))
    parser.add_argument("--name", default=tracking.REGISTERED_MODEL_NAME)
    args = parser.parse_args(argv)

    print(tracking.describe_store())
    result = register(run_id=args.run_id, stage=args.stage, name=args.name)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
