"""Shadow A/B: does the challenger actually beat the champion?

Compares the two on **the same rows** -- audit rows where both models
scored, judged against the true label that arrived afterwards. Same
transactions, same labels, so the only thing varying is the model.

**Promotion requires a margin larger than noise.** This is the whole point.
Phase 2 measured three XGBoost runs spanning 0.0023 PR-AUC against a
bootstrap standard error of 0.0080 and correctly called them
indistinguishable -- the same standard applies here. A challenger ahead by
less than one standard error has not been shown to be better; it has been
shown to be ahead on this particular sample, which is a different claim and
a coin flip dressed as a decision.

The bootstrap is **paired**: each resample draws row indices once and scores
both models on those same rows, so the correlation between two models that
mostly agree is preserved. An unpaired comparison of two independent
standard errors would be far too conservative here, because the models
share their inputs and make correlated errors -- the 0.9778 probability
correlation measured in Phase 2 is the same phenomenon.

**Each model is judged at its own threshold.** The champion's swept best-F1
point is 0.8018 and the challenger's is 0.6844; scoring the challenger at
the champion's cut would measure the threshold rather than the model. PR-AUC
is threshold-free and is the headline for exactly that reason.

    python -m src.serving.compare                     # report only
    python -m src.serving.compare --promote           # promote if it wins
    python -m src.serving.compare --min-rows 20000 --bootstrap 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.config import REPORTS_DIR, ensure_dirs
from src.drift import store
from src.serving.predictor import CHALLENGER_ALIAS, CHAMPION_ALIAS
from src.training import tracking

COMPARISON_PATH: Path = REPORTS_DIR / "shadow_comparison.json"

#: Bootstrap resamples. 1000 is enough for a stable standard error; the cost
#: is a few seconds and the estimate stops moving well before this.
DEFAULT_BOOTSTRAP: int = 1000

#: Below this many jointly-scored rows the comparison is not a decision.
#: PR-AUC on a few thousand rows at a 3.4% base rate rests on a hundred-odd
#: positives, and its standard error swamps any plausible margin.
MIN_ROWS: int = 5_000

#: And below this many positives, likewise -- rows alone are not enough when
#: the metric lives entirely on the minority class.
MIN_POSITIVES: int = 100

#: Fraction of the judged rows that may fall inside the challenger's
#: training window before the comparison is refused outright. Not zero: a
#: boundary row or a rounding difference in the recorded window should not
#: void an otherwise clean test. Anything above this is memory, not skill.
MAX_LEAKAGE_OVERLAP: float = 0.01

#: Margin required, in standard errors. 1.0 is the Phase 2 standard: a
#: difference smaller than one SE is noise. Note this is roughly 84%
#: one-sided confidence, not 95% -- raise it to 1.65 or 2.0 for a stricter
#: bar. It is deliberately not hidden inside the promotion logic.
DEFAULT_MARGIN_SES: float = 1.0

PROMOTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_promotions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    promoted_at             TEXT    NOT NULL,
    from_version            TEXT,
    to_version              TEXT    NOT NULL,
    trigger_alert_id        INTEGER,
    comparison_rows         INTEGER NOT NULL,
    comparison_positives    INTEGER NOT NULL,
    comparison_fraud_rate   REAL    NOT NULL,
    champion_pr_auc         REAL    NOT NULL,
    challenger_pr_auc       REAL    NOT NULL,
    pr_auc_delta            REAL    NOT NULL,
    bootstrap_std_error     REAL    NOT NULL,
    margin_in_ses           REAL    NOT NULL,
    verdict                 TEXT    NOT NULL,
    metrics                 TEXT    NOT NULL   -- full JSON
);
"""


@dataclass
class ModelScores:
    """One model's performance on the shared rows."""

    version: str
    threshold: float
    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def flagged(self) -> int:
        return self.true_positives + self.false_positives


@dataclass
class Comparison:
    """The shadow A/B result."""

    rows: int
    positives: int
    fraud_rate: float
    champion: ModelScores
    challenger: ModelScores
    pr_auc_delta: float
    bootstrap_std_error: float
    margin_in_ses: float
    required_ses: float
    sufficient_data: bool
    reason: str
    leakage_overlap: float = 0.0
    scored_dt_range: tuple[float, float] | None = None
    trained_dt_range: tuple[float, float] | None = None
    trigger_alert_id: int | None = None
    window_start: str | None = None
    window_end: str | None = None
    bootstrap_ci: tuple[float, float] = field(default=(0.0, 0.0))

    @property
    def leaked(self) -> bool:
        """Were the judged rows inside the challenger's training window?

        Any material overlap invalidates the comparison: the challenger is
        being asked to recall rows it was fitted on while the champion
        predicts them. That is a contest between memory and prediction, and
        no margin rule can rescue it -- the larger the overlap, the more
        decisively it will report a win.
        """
        return self.leakage_overlap > MAX_LEAKAGE_OVERLAP

    @property
    def trustworthy(self) -> bool:
        return self.sufficient_data and not self.leaked

    @property
    def promote(self) -> bool:
        """Only a margin beyond noise, on enough data, promotes."""
        return bool(
            self.trustworthy
            and self.pr_auc_delta > 0
            and self.margin_in_ses >= self.required_ses
        )

    @property
    def verdict(self) -> str:
        if self.leaked:
            return "NO VERDICT -- evaluation rows overlap the challenger's training window"
        if not self.sufficient_data:
            return "NO PROMOTION -- insufficient data"
        if self.promote:
            return "PROMOTE"
        if self.pr_auc_delta <= 0:
            return "NO PROMOTION -- challenger is not ahead"
        return "no promotion, difference within noise"


def threshold_scores(
    y: np.ndarray, proba: np.ndarray, threshold: float, version: str
) -> ModelScores:
    """Every reported metric for one model, at its own operating point."""
    predicted = (proba >= threshold).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denominator = precision + recall
    f1 = (2 * precision * recall / denominator) if denominator else 0.0

    return ModelScores(
        version=version,
        threshold=threshold,
        pr_auc=float(average_precision_score(y, proba)),
        roc_auc=float(roc_auc_score(y, proba)),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def paired_bootstrap(
    y: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Standard error of the PR-AUC *difference*, and its 95% interval.

    Paired: one index draw per resample, scoring both models on the same
    rows. The two models see identical inputs and make correlated errors, so
    treating their errors as independent would inflate the standard error
    and refuse promotions that are real.

    Resamples that lose one class entirely are skipped -- average precision
    is undefined there, and at a 3.4% base rate a small window can produce
    them.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(resamples):
        index = rng.integers(0, n, n)
        labels = y[index]
        if labels.min() == labels.max():
            continue
        deltas.append(
            average_precision_score(labels, challenger[index])
            - average_precision_score(labels, champion[index])
        )
    if len(deltas) < 2:
        return float("nan"), (float("nan"), float("nan"))
    array = np.asarray(deltas)
    return float(array.std(ddof=1)), (
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    )


def challenger_training_window(
    challenger_version: str, model_name: str | None = None
) -> tuple[float, float] | None:
    """The TransactionDT range the challenger was fitted on, from its run.

    Read from the MLflow run parameters the retrain logged, rather than
    recomputed -- the run record is what ties a model to the data behind it,
    and recomputing would let the two disagree.
    """
    mlflow = tracking.mlflow_module()
    if mlflow is None:
        return None
    try:
        mlflow.set_tracking_uri(tracking.tracking_uri())
        from mlflow import MlflowClient

        client = MlflowClient()
        name = model_name or tracking.REGISTERED_MODEL_NAME
        version = client.get_model_version(name, challenger_version)
        params = client.get_run(version.run_id).data.params
        return (
            float(params["retrain.window_start_dt"]),
            float(params["retrain.window_end_dt"]),
        )
    except Exception:
        return None


def overlap_fraction(
    scored: tuple[float, float], trained: tuple[float, float]
) -> float:
    """Fraction of the scored range that sits inside the trained range."""
    span = scored[1] - scored[0]
    if span <= 0:
        return 1.0 if trained[0] <= scored[0] <= trained[1] else 0.0
    covered = min(scored[1], trained[1]) - max(scored[0], trained[0])
    return max(0.0, covered) / span


def load_paired_rows(
    limit: int | None = None,
    db_path: Path = store.DEFAULT_DB_PATH,
    challenger_version: str | None = None,
    champion_version: str | None = None,
) -> list[dict]:
    """Audit rows where both models scored and the label is known.

    The join is implicit: a row carries both predictions because both models
    scored that same transaction in the same pass, so there is nothing to
    align and no chance of comparing different rows.

    Pinned to **one** champion/challenger pair. Once a second challenger has
    been registered the table holds rows from both, and an unfiltered query
    would average two different models together while reporting whichever
    version happened to be on the newest row -- a comparison of a model
    against itself-plus-its-predecessor, which would look entirely normal.
    Defaults to the most recently scored pair.
    """
    with store.connect(db_path) as connection:
        try:
            if challenger_version is None or champion_version is None:
                newest = connection.execute(
                    "SELECT model_version, challenger_version FROM stream_predictions "
                    "WHERE challenger_probability IS NOT NULL AND true_label IS NOT NULL "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if newest is None:
                    return []
                champion_version = champion_version or str(newest["model_version"])
                challenger_version = challenger_version or str(newest["challenger_version"])

            query = (
                "SELECT * FROM stream_predictions "
                "WHERE challenger_probability IS NOT NULL AND true_label IS NOT NULL "
                "AND model_version = ? AND challenger_version = ? "
                "ORDER BY id DESC"
            )
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = connection.execute(
                query, (champion_version, challenger_version)
            ).fetchall()
        except Exception:
            return []
    return [dict(row) for row in rows]


def compare(
    limit: int | None = None,
    resamples: int = DEFAULT_BOOTSTRAP,
    challenger_version: str | None = None,
    min_rows: int = MIN_ROWS,
    min_positives: int = MIN_POSITIVES,
    required_ses: float = DEFAULT_MARGIN_SES,
    db_path: Path = store.DEFAULT_DB_PATH,
) -> Comparison:
    """Score both models on the shared rows and decide."""
    rows = load_paired_rows(limit, db_path, challenger_version=challenger_version)
    if not rows:
        raise RuntimeError(
            "No audit rows carry both a champion and a challenger prediction. "
            "Run the consumer while a @challenger is registered."
        )

    y = np.array([r["true_label"] for r in rows], dtype=int)
    champion_proba = np.array([r["fraud_probability"] for r in rows], dtype=float)
    challenger_proba = np.array([r["challenger_probability"] for r in rows], dtype=float)

    champion_threshold = float(rows[0]["threshold"])
    challenger_threshold = float(rows[0]["challenger_threshold"])
    champion_version = str(rows[0]["model_version"])
    challenger_version = str(rows[0]["challenger_version"])

    positives = int(y.sum())
    fraud_rate = float(y.mean())

    # bool(...) rather than the raw comparison: numpy returns np.bool_,
    # which is not JSON-serialisable and would fail only at write time,
    # after the comparison had already been printed as if it succeeded.
    sufficient = bool(
        len(rows) >= min_rows and positives >= min_positives and y.min() != y.max()
    )
    reason = ""
    if len(rows) < min_rows:
        reason = f"{len(rows):,} jointly-scored rows, need {min_rows:,}"
    elif positives < min_positives:
        reason = f"{positives} positives, need {min_positives}"
    elif y.min() == y.max():
        reason = "the window contains only one class"

    champion = threshold_scores(y, champion_proba, champion_threshold, champion_version)
    challenger = threshold_scores(
        y, challenger_proba, challenger_threshold, challenger_version
    )
    delta = challenger.pr_auc - champion.pr_auc

    standard_error, interval = paired_bootstrap(
        y, champion_proba, challenger_proba, resamples
    )
    margin = abs(delta) / standard_error if standard_error and standard_error == standard_error else 0.0

    scored_range = (
        min(r["transaction_dt"] for r in rows if r["transaction_dt"] is not None),
        max(r["transaction_dt"] for r in rows if r["transaction_dt"] is not None),
    )
    trained_range = challenger_training_window(challenger.version)
    leakage = overlap_fraction(scored_range, trained_range) if trained_range else 0.0
    if leakage > MAX_LEAKAGE_OVERLAP:
        reason = (
            f"{leakage * 100:.1f}% of the judged rows fall inside the "
            "challenger's training window"
        )

    alert = store.latest_retrain_trigger(db_path, include_resolved=True)
    return Comparison(
        leakage_overlap=leakage,
        scored_dt_range=scored_range,
        trained_dt_range=trained_range,
        rows=len(rows),
        positives=positives,
        fraud_rate=fraud_rate,
        champion=champion,
        challenger=challenger,
        pr_auc_delta=float(delta),
        bootstrap_std_error=float(standard_error),
        margin_in_ses=float(margin),
        required_ses=required_ses,
        sufficient_data=sufficient,
        reason=reason,
        trigger_alert_id=(alert or {}).get("id"),
        window_start=rows[-1]["scored_at"],
        window_end=rows[0]["scored_at"],
        bootstrap_ci=interval,
    )


def format_comparison(result: Comparison) -> str:
    champion, challenger = result.champion, result.challenger
    lines = [
        "",
        "=" * 78,
        f"  {result.verdict}",
        "=" * 78,
        f"  window     {result.rows:,} jointly-scored rows, "
        f"{result.positives:,} fraud ({result.fraud_rate * 100:.3f}%)",
        f"             {result.window_start} .. {result.window_end}",
        "",
        f"  {'':<14}{'champion':>12}{'challenger':>13}{'delta':>11}",
        f"  {'version':<14}{'v' + champion.version:>12}{'v' + challenger.version:>13}",
        f"  {'threshold':<14}{champion.threshold:>12.4f}{challenger.threshold:>13.4f}",
        "  " + "-" * 50,
        f"  {'PR-AUC':<14}{champion.pr_auc:>12.4f}{challenger.pr_auc:>13.4f}"
        f"{result.pr_auc_delta:>+11.4f}",
        f"  {'ROC-AUC':<14}{champion.roc_auc:>12.4f}{challenger.roc_auc:>13.4f}"
        f"{challenger.roc_auc - champion.roc_auc:>+11.4f}",
        f"  {'F1':<14}{champion.f1:>12.4f}{challenger.f1:>13.4f}"
        f"{challenger.f1 - champion.f1:>+11.4f}",
        f"  {'precision':<14}{champion.precision:>12.4f}{challenger.precision:>13.4f}"
        f"{challenger.precision - champion.precision:>+11.4f}",
        f"  {'recall':<14}{champion.recall:>12.4f}{challenger.recall:>13.4f}"
        f"{challenger.recall - champion.recall:>+11.4f}",
        f"  {'false negs':<14}{champion.false_negatives:>12,}"
        f"{challenger.false_negatives:>13,}"
        f"{challenger.false_negatives - champion.false_negatives:>+11,}",
        f"  {'flagged':<14}{champion.flagged:>12,}{challenger.flagged:>13,}"
        f"{challenger.flagged - champion.flagged:>+11,}",
        "",
        f"  PR-AUC delta          {result.pr_auc_delta:+.4f}",
        f"  bootstrap std error    {result.bootstrap_std_error:.4f}  "
        f"(paired, {DEFAULT_BOOTSTRAP} resamples)",
        f"  95% CI on the delta   [{result.bootstrap_ci[0]:+.4f}, "
        f"{result.bootstrap_ci[1]:+.4f}]",
        f"  margin                 {result.margin_in_ses:.2f} SE  "
        f"(need {result.required_ses:.2f})",
    ]
    if result.leaked:
        scored = result.scored_dt_range or (0.0, 0.0)
        trained = result.trained_dt_range or (0.0, 0.0)
        lines += [
            "",
            f"  LEAKAGE: {result.leakage_overlap * 100:.1f}% of the judged rows sit "
            "inside the challenger's",
            "  own training window.",
            f"    judged  TransactionDT {scored[0]:,.0f} .. {scored[1]:,.0f}",
            f"    trained TransactionDT {trained[0]:,.0f} .. {trained[1]:,.0f}",
            "",
            "  The challenger is recalling rows it was fitted on while the champion",
            "  predicts them. No margin rule can rescue that -- a larger overlap",
            "  produces a more decisive apparent win. No verdict is issued.",
        ]
    elif not result.sufficient_data:
        lines.append(f"\n  Not a decision: {result.reason}.")
    elif result.promote:
        lines.append(
            "\n  The challenger is ahead by more than the noise in the estimate."
        )
    elif result.pr_auc_delta > 0:
        lines.append(
            f"\n  The challenger is ahead by {result.pr_auc_delta:+.4f}, but the "
            f"standard error is {result.bootstrap_std_error:.4f}. That is a "
            "difference\n  this sample cannot distinguish from chance -- the same "
            "standard that\n  ruled three XGBoost runs indistinguishable in Phase 2."
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def promote(result: Comparison, db_path: Path = store.DEFAULT_DB_PATH) -> dict | None:
    """Move the challenger to Production and archive the old champion."""
    if not result.promote:
        raise RuntimeError(f"refusing to promote: {result.verdict}")

    mlflow = tracking.mlflow_module()
    if mlflow is None:
        print("  MLflow unavailable; cannot promote")
        return None

    mlflow.set_tracking_uri(tracking.tracking_uri())
    from mlflow import MlflowClient

    client = MlflowClient()
    name = tracking.REGISTERED_MODEL_NAME
    old, new = result.champion.version, result.challenger.version

    client.transition_model_version_stage(name, new, "Production")
    client.transition_model_version_stage(name, old, "Archived")
    # The alias is what serving resolves; the stage is for humans reading the
    # registry. Moving the alias is the act that actually changes behaviour.
    client.set_registered_model_alias(name, CHAMPION_ALIAS, new)
    client.set_registered_model_alias(name, "staging", new)
    try:
        client.delete_registered_model_alias(name, CHALLENGER_ALIAS)
    except Exception:
        pass

    client.set_model_version_tag(name, new, "promoted_from", old)
    client.set_model_version_tag(
        name, new, "promotion_pr_auc_delta", f"{result.pr_auc_delta:+.4f}"
    )
    client.set_model_version_tag(name, new, "promotion_rows", str(result.rows))

    record = {
        "from_version": old,
        "to_version": new,
        "trigger_alert_id": result.trigger_alert_id,
        "rows": result.rows,
        "pr_auc_delta": result.pr_auc_delta,
    }
    with store.connect(db_path) as connection:
        connection.executescript(PROMOTION_SCHEMA)
        connection.execute(
            """
            INSERT INTO model_promotions (
                promoted_at, from_version, to_version, trigger_alert_id,
                comparison_rows, comparison_positives, comparison_fraud_rate,
                champion_pr_auc, challenger_pr_auc, pr_auc_delta,
                bootstrap_std_error, margin_in_ses, verdict, metrics
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                store.utc_now(), old, new, result.trigger_alert_id,
                result.rows, result.positives, result.fraud_rate,
                result.champion.pr_auc, result.challenger.pr_auc,
                result.pr_auc_delta, result.bootstrap_std_error,
                result.margin_in_ses, result.verdict,
                json.dumps(as_dict(result)),
            ),
        )
    return record


def as_dict(result: Comparison) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": result.verdict,
        "promote": result.promote,
        "window": {
            "rows": result.rows,
            "positives": result.positives,
            "fraud_rate": result.fraud_rate,
            "start": result.window_start,
            "end": result.window_end,
            "sufficient": result.sufficient_data,
            "reason": result.reason,
            "leakage_overlap": result.leakage_overlap,
            "leaked": result.leaked,
            "scored_dt_range": list(result.scored_dt_range or ()),
            "trained_dt_range": list(result.trained_dt_range or ()),
        },
        "champion": vars(result.champion),
        "challenger": vars(result.challenger),
        "decision": {
            "pr_auc_delta": result.pr_auc_delta,
            "bootstrap_std_error": result.bootstrap_std_error,
            "bootstrap_ci_95": list(result.bootstrap_ci),
            "margin_in_ses": result.margin_in_ses,
            "required_ses": result.required_ses,
        },
        "trigger_alert_id": result.trigger_alert_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shadow A/B champion vs challenger.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--challenger-version", default=None,
                        help="pin the comparison to one challenger (default: newest)")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--min-positives", type=int, default=MIN_POSITIVES)
    parser.add_argument("--margin-ses", type=float, default=DEFAULT_MARGIN_SES)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    ensure_dirs()
    try:
        result = compare(
            limit=args.limit, resamples=args.bootstrap, min_rows=args.min_rows,
            challenger_version=args.challenger_version,
            min_positives=args.min_positives, required_ses=args.margin_ses,
            db_path=args.db,
        )
    except Exception as exc:
        print(f"comparison failed: {type(exc).__name__}: {exc}")
        return 1

    print(format_comparison(result))
    COMPARISON_PATH.write_text(json.dumps(as_dict(result), indent=2), encoding="utf-8")
    print(f"\nwrote    {COMPARISON_PATH}")

    if args.promote:
        if result.promote:
            record = promote(result, args.db)
            if record:
                print(
                    f"\npromoted v{record['to_version']} to Production; "
                    f"v{record['from_version']} archived\n"
                    f"  answering alert #{record['trigger_alert_id']}, "
                    f"{record['rows']:,} rows, "
                    f"PR-AUC {record['pr_auc_delta']:+.4f}"
                )
        else:
            print(f"\nnot promoting: {result.verdict}")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
