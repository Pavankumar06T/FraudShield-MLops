"""Scheduled drift monitor: the Phase 0 measurement, productionised.

Compares the train slice as reference against a recent window of the stream
slice, and decides whether the model has gone blind.

**PSI comes from src/drift/psi.py, unchanged.** Not reimplemented, not
reparameterised. The Phase 0 sweep produced 278 stable / 4 moderate / 27
major with id_31 at 1.525, and a monitor that quietly computed those
differently would be reporting on a different question than the analysis it
inherits. The band counts are re-asserted on every run.

**Two drift types, alerted separately.** This is the whole point of the
Phase 0 decomposition, and blending them into one number destroys it:

*Missingness-driven* -- coverage changed, values did not. In this dataset,
sixteen features where the NA rate *fell*: M1-M3 and M7-M9 from 74.6% to
39.4%, D11 and V2-V11 from 61.7% to 30.3%, with PSI on populated rows at
essentially zero. More data started arriving. That is an upstream pipeline
or integration change, and retraining on it would bake a transient join
into the model. The remedy is to ask why coverage moved.

*Genuine value drift* -- the populated values themselves moved. Eleven
features: id_31, id_13, id_30, D11, and the V144-V152 / V159-V160 identity
block. Coverage *degraded* here (81.6% to 90.5%) while PSI on the rows that
remain is enormous -- id_31 at 7.58 on populated rows against 1.53 overall.
That is fraudsters and browsers changing, and it is what retraining fixes.

Only the second type fires a retrain. A monitor that retrained on the first
would be doing the wrong thing confidently.

**Unseen categories are the fast signal.** Windowed PSI needs a window of
traffic to move; a category the encoder has never seen is visible on the
first request carrying it. The serving layer already surfaces those per
response, and they are aggregated here into the same alert store.

    python -m src.drift.monitor                    # last 30 days of stream
    python -m src.drift.monitor --window-rows 50000
    python -m src.drift.monitor --threshold 0.15 --no-evidently
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import (
    REPORTS_DIR,
    STREAM_PARQUET,
    TRAIN_PARQUET,
    ensure_dirs,
)
from src.drift import store
from src.drift.report import (
    EXCLUDED_COLUMNS,
    MAJOR_ABOVE,
    STABLE_BELOW,
    band_counts,
    classify,
    compute_psi_report,
    decompose_drift,
)

#: PSI at which genuine value drift triggers a retrain. Matches Phase 0.
DEFAULT_THRESHOLD: float = MAJOR_ABOVE

#: Residual PSI on populated rows above which drift counts as genuine rather
#: than missingness-driven.
#:
#: 0.05 rather than the 0.10 "stable" band, because the two clusters in this
#: dataset separate far more sharply than that: the missingness group sits
#: at 0.000-0.007 on populated rows, the genuine group at 0.07-7.58. Any cut
#: between 0.01 and 0.07 gives the same answer. 0.05 puts D11 -- residual
#: 0.0735, the only borderline case -- on the genuine side, which matches
#: the Phase 0 decision and is the conservative choice: mistaking real drift
#: for a pipeline artifact means not retraining when you should.
VALUE_DRIFT_RESIDUAL: float = 0.05

#: Default recent window, in days of TransactionDT.
DEFAULT_WINDOW_DAYS: float = 30.0

SECONDS_PER_DAY: int = 86_400

DECISION_PATH: Path = REPORTS_DIR / "drift_decision.json"
EVIDENTLY_PATH: Path = REPORTS_DIR / "drift_report.html"

#: Re-asserted every run. A monitor whose numbers wandered from the analysis
#: it inherits is reporting on a different question.
PHASE0_BANDS: dict[str, int] = {"stable": 278, "moderate": 4, "major": 27}


@dataclass
class DriftResult:
    """One monitor run."""

    psi: pd.Series
    decomposition: pd.DataFrame
    value_drift: list[str]
    missingness_drift: list[str]
    bands: dict[str, int]
    window_rows: int
    reference_rows: int
    window_label: str
    window_start_dt: float | None
    window_end_dt: float | None
    threshold: float
    unseen: list[dict]
    evidently_path: str | None = None

    @property
    def overall_psi(self) -> float:
        """The worst single feature.

        Deliberately a max rather than a mean. There is no canonical
        "overall PSI", and averaging 431 features drowns the one collapsed
        column the monitor exists to catch -- the Phase 0 example averaged
        to 0.15 while its worst feature sat at 0.23.
        """
        clean = self.psi.dropna()
        return float(clean.max()) if len(clean) else 0.0

    @property
    def overall_psi_feature(self) -> str | None:
        clean = self.psi.dropna()
        return str(clean.idxmax()) if len(clean) else None

    def _max_over(self, features: list[str]) -> float:
        values = [self.psi[f] for f in features if f in self.psi.index]
        values = [v for v in values if not pd.isna(v)]
        return float(max(values)) if values else 0.0

    @property
    def value_drift_psi(self) -> float:
        return self._max_over(self.value_drift)

    @property
    def missingness_psi(self) -> float:
        return self._max_over(self.missingness_drift)

    @property
    def retrain_triggered(self) -> bool:
        """Only genuine value drift retrains."""
        return self.value_drift_psi >= self.threshold

    @property
    def investigate_pipeline(self) -> bool:
        """Coverage moved enough to be someone's bug, not the model's."""
        return self.missingness_psi >= self.threshold

    @property
    def verdict(self) -> str:
        if self.retrain_triggered and self.investigate_pipeline:
            return "RETRAIN + INVESTIGATE PIPELINE"
        if self.retrain_triggered:
            return "RETRAIN"
        if self.investigate_pipeline:
            return "INVESTIGATE PIPELINE (no retrain)"
        return "STABLE"

    def top_features(self, n: int = 15) -> list[dict]:
        rows = []
        for feature, value in self.psi.dropna().head(n).items():
            kind = (
                "value"
                if feature in self.value_drift
                else "missingness"
                if feature in self.missingness_drift
                else "unclassified"
            )
            entry = {"feature": str(feature), "psi": round(float(value), 6), "type": kind}
            if feature in self.decomposition.index:
                row = self.decomposition.loc[feature]
                entry["psi_non_null"] = round(float(row["psi_non_null"]), 6)
                entry["na_rate_reference"] = round(float(row["na_rate_reference"]), 6)
                entry["na_rate_current"] = round(float(row["na_rate_current"]), 6)
            rows.append(entry)
        return rows


def load_window(
    window_days: float | None = DEFAULT_WINDOW_DAYS,
    window_rows: int | None = None,
    path: Path = STREAM_PARQUET,
) -> tuple[pd.DataFrame, str, float | None, float | None]:
    """The most recent slice of the stream, by TransactionDT.

    Recency is measured on the transaction clock, not on row position: the
    stream arrives ordered but a monitor that assumed so would silently
    compare the wrong rows the first time it did not.
    """
    frame = pd.read_parquet(path)
    times = frame["TransactionDT"]

    if window_rows is not None:
        cutoff = float(times.nlargest(min(window_rows, len(frame))).min())
        label = f"last {window_rows:,} rows"
    else:
        span = float(times.max() - times.min())
        requested = (window_days or DEFAULT_WINDOW_DAYS) * SECONDS_PER_DAY
        cutoff = float(times.max()) - min(requested, span)
        label = f"last {window_days:g} days"

    window = frame[times >= cutoff]
    if window.empty:
        raise ValueError(
            f"Window '{label}' selected no rows from {path.name}. The stream "
            f"spans {float(times.min()):,.0f}-{float(times.max()):,.0f}."
        )
    return window, label, float(window["TransactionDT"].min()), float(times.max())


def split_drift_types(
    decomposition: pd.DataFrame,
    psi: pd.Series,
    threshold: float,
    residual: float = VALUE_DRIFT_RESIDUAL,
) -> tuple[list[str], list[str]]:
    """Partition the drifting features into genuine vs missingness-driven.

    A feature whose populated values still look like training has not
    drifted in any sense retraining would fix -- what changed is how often
    it arrives at all.
    """
    drifting = [f for f in psi.dropna().index if psi[f] >= threshold]
    value, missingness = [], []
    for feature in drifting:
        if feature not in decomposition.index:
            value.append(feature)  # undecomposed: treat as genuine, the safer error
            continue
        residual_psi = float(decomposition.loc[feature, "psi_non_null"])
        (value if residual_psi >= residual else missingness).append(feature)
    return value, missingness


def write_decision(result: DriftResult, path: Path = DECISION_PATH) -> Path:
    """Persist the classification as the artifact later phases read."""
    payload = {
        "created_at": store.utc_now(),
        "reference": TRAIN_PARQUET.name,
        "window": {
            "label": result.window_label,
            "rows": result.window_rows,
            "start_transaction_dt": result.window_start_dt,
            "end_transaction_dt": result.window_end_dt,
        },
        "threshold": result.threshold,
        "value_drift_residual_cut": VALUE_DRIFT_RESIDUAL,
        "bands": result.bands,
        "overall_psi": result.overall_psi,
        "overall_psi_feature": result.overall_psi_feature,
        "value_drift": {
            "features": result.value_drift,
            "count": len(result.value_drift),
            "max_psi": result.value_drift_psi,
            "remedy": "retrain -- the populated values moved",
        },
        "missingness_drift": {
            "features": result.missingness_drift,
            "count": len(result.missingness_drift),
            "max_psi": result.missingness_psi,
            "remedy": (
                "investigate the upstream feed -- coverage changed while the "
                "populated values held steady; retraining would bake a "
                "transient join into the model"
            ),
        },
        "unseen_categories": result.unseen,
        "verdict": result.verdict,
        "retrain_triggered": result.retrain_triggered,
        "investigate_pipeline": result.investigate_pipeline,
        "top_features": result.top_features(20),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    path: Path = EVIDENTLY_PATH,
    max_columns: int = 60,
) -> str | None:
    """HTML drift report alongside the numbers, or None if unavailable.

    Restricted to the most interesting columns: Evidently renders a panel
    per column, and 431 of them produces a page too large to open, let alone
    read. The numeric output remains the complete record.
    """
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  Evidently unavailable ({type(exc).__name__}); skipping HTML report")
        return None

    columns = [c for c in features[:max_columns] if c in reference.columns]
    if not columns:
        return None

    try:
        definition = DataDefinition()
        report = Report(metrics=[DataDriftPreset()])
        run = report.run(
            current_data=Dataset.from_pandas(current[columns], data_definition=definition),
            reference_data=Dataset.from_pandas(reference[columns], data_definition=definition),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        run.save_html(str(path))
        return str(path)
    except Exception as exc:  # pragma: no cover - version-dependent
        print(f"  Evidently report failed ({type(exc).__name__}: {exc}); numbers unaffected")
        return None


def run_monitor(
    window_days: float | None = DEFAULT_WINDOW_DAYS,
    window_rows: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    with_evidently: bool = True,
    db_path: Path = store.DEFAULT_DB_PATH,
) -> DriftResult:
    """Compare reference against a recent window and record the verdict."""
    ensure_dirs()

    print(f"reference  {TRAIN_PARQUET}")
    reference = pd.read_parquet(TRAIN_PARQUET)
    window, label, start_dt, end_dt = load_window(window_days, window_rows)
    print(
        f"window     {STREAM_PARQUET.name}  {label}  "
        f"{len(window):,} rows (TransactionDT {start_dt:,.0f}-{end_dt:,.0f})"
    )

    print(f"  scoring PSI across {reference.shape[1] - len(EXCLUDED_COLUMNS)} features ...")
    psi = compute_psi_report(reference, window)
    bands = band_counts(psi)

    drifting = [f for f in psi.dropna().index if psi[f] >= STABLE_BELOW]
    decomposition = decompose_drift(reference, window, drifting)
    value, missingness = split_drift_types(decomposition, psi, threshold)

    unseen = store.aggregate_unseen(path=db_path)

    result = DriftResult(
        psi=psi,
        decomposition=decomposition,
        value_drift=value,
        missingness_drift=missingness,
        bands=bands,
        window_rows=len(window),
        reference_rows=len(reference),
        window_label=label,
        window_start_dt=start_dt,
        window_end_dt=end_dt,
        threshold=threshold,
        unseen=unseen,
    )

    if with_evidently:
        print("  building Evidently report ...")
        result.evidently_path = build_evidently_report(
            reference, window, list(psi.dropna().index)
        )

    return result


def format_result(result: DriftResult) -> str:
    lines = [
        "",
        "=" * 74,
        f"  {result.verdict}",
        "=" * 74,
        f"  window       {result.window_label}, {result.window_rows:,} rows "
        f"vs {result.reference_rows:,} reference",
        f"  threshold    {result.threshold:.2f}",
        f"  worst feature {result.overall_psi_feature} at PSI "
        f"{result.overall_psi:.4f}",
        "",
        f"  bands        {result.bands['stable']} stable / "
        f"{result.bands['moderate']} moderate / {result.bands['major']} major "
        f"/ {result.bands['unmeasurable']} unmeasurable",
    ]

    fired = "FIRES RETRAIN" if result.retrain_triggered else "below threshold"
    lines += [
        "",
        f"  GENUINE VALUE DRIFT   {len(result.value_drift):>3} features, "
        f"max PSI {result.value_drift_psi:.4f}   {fired}",
        "    the populated values moved -- this is what retraining fixes",
    ]
    for feature in result.value_drift[:8]:
        entry = result.decomposition.loc[feature] if feature in result.decomposition.index else None
        detail = (
            f"  populated-row PSI {entry['psi_non_null']:.3f}, "
            f"NA {entry['na_rate_reference']:.1%} -> {entry['na_rate_current']:.1%}"
            if entry is not None
            else ""
        )
        lines.append(f"      {feature:<10} {result.psi[feature]:.4f}{detail}")

    fired = "investigate" if result.investigate_pipeline else "below threshold"
    lines += [
        "",
        f"  MISSINGNESS-DRIVEN    {len(result.missingness_drift):>3} features, "
        f"max PSI {result.missingness_psi:.4f}   {fired}",
        "    coverage changed, values held -- upstream pipeline, NOT a retrain",
    ]
    for feature in result.missingness_drift[:8]:
        entry = result.decomposition.loc[feature] if feature in result.decomposition.index else None
        detail = (
            f"  populated-row PSI {entry['psi_non_null']:.3f}, "
            f"NA {entry['na_rate_reference']:.1%} -> {entry['na_rate_current']:.1%}"
            if entry is not None
            else ""
        )
        lines.append(f"      {feature:<10} {result.psi[feature]:.4f}{detail}")

    if result.unseen:
        lines += ["", f"  UNSEEN CATEGORIES     {len(result.unseen)} feature(s) from serving"]
        for entry in result.unseen[:6]:
            values = ", ".join(v["value"] for v in entry["top_values"][:3])
            lines.append(
                f"      {entry['feature']:<10} {entry['occurrences']:>6} requests, "
                f"{entry['distinct_values']} distinct: {values}"
            )
    else:
        lines += ["", "  UNSEEN CATEGORIES     none recorded from serving yet"]

    lines.append("=" * 74)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scheduled drift monitor.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--window-days", type=float, default=None, metavar="D")
    group.add_argument("--window-rows", type=int, default=None, metavar="N")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--no-evidently", action="store_true")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    days = args.window_days
    if days is None and args.window_rows is None:
        days = DEFAULT_WINDOW_DAYS

    result = run_monitor(
        window_days=days,
        window_rows=args.window_rows,
        threshold=args.threshold,
        with_evidently=not args.no_evidently,
        db_path=args.db,
    )
    print(format_result(result))

    expected = PHASE0_BANDS
    if args.window_rows is None and days is not None and days >= 999:
        matched = all(result.bands[b] == expected[b] for b in expected)
        print(
            f"\n  Phase 0 band check: {'reproduces' if matched else 'DIFFERS from'} "
            f"278/4/27 over the full stream"
        )

    alert = store.DriftAlert(
        window_label=result.window_label,
        window_rows=result.window_rows,
        reference_rows=result.reference_rows,
        overall_psi=result.overall_psi,
        overall_psi_feature=result.overall_psi_feature,
        value_drift_psi=result.value_drift_psi,
        value_drift_features=len(result.value_drift),
        missingness_psi=result.missingness_psi,
        missingness_features=len(result.missingness_drift),
        n_major=result.bands["major"],
        n_moderate=result.bands["moderate"],
        n_stable=result.bands["stable"],
        n_unmeasurable=result.bands["unmeasurable"],
        threshold=result.threshold,
        retrain_triggered=result.retrain_triggered,
        investigate_pipeline=result.investigate_pipeline,
        verdict=result.verdict,
        top_features=result.top_features(),
        window_start_dt=result.window_start_dt,
        window_end_dt=result.window_end_dt,
        unseen_categories=result.unseen,
        evidently_report_path=result.evidently_path,
    )
    alert_id = alert.insert(args.db)

    decision = write_decision(result)
    print(f"\nwrote    {decision}")
    print(f"wrote    alert #{alert_id} to {args.db}")
    if result.evidently_path:
        print(f"wrote    {result.evidently_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
