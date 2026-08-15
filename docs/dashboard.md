# Dashboard

```bash
streamlit run src/dashboard/app.py
```

Five panels over `reports/drift.db` and the MLflow registry. **Nothing on the
page is recalculated** — PSI comes from `drift_alerts`, promotion verdicts
from `model_promotions` and version tags, model metrics from the runs that
produced them.

That is a correctness requirement. A dashboard that recomputed PSI would
eventually disagree with the monitor that fired the alert — a different
window, a different bin edge, a different epsilon — and an operator would be
reading a number nobody can reconcile with the decision actually taken.
`tests/test_dashboard.py` asserts neither dashboard module imports
`psi_numeric`, `compute_psi_report`, `average_precision_score` or
`paired_bootstrap`.

## Successes and failures are not stored together

**This is the finding worth carrying to any system with a promotion gate.**

`model_promotions` holds only attempts that *reached the registry*. A
challenger rejected by the shadow test never gets a row there — the
comparison exits before promoting, and nothing durable records that it was
tried. Its verdict survives only as tags on the model version:
`shadow_verdict`, `shadow_pr_auc_delta`, `shadow_rows`.

So a promotion history assembled from `model_promotions` alone would show:

```
  v3   rolled back  +0.2492
  v4   promoted     +0.0361
```

and a history assembled from both shows:

```
  v2   rejected     -0.0100      --     25,000 rows
  v3   rolled back  +0.2492  20.13 SE   25,000 rows
  v4   promoted     +0.0361   4.23 SE   11,429 rows
```

The first version is not wrong about any individual row. It is wrong about
the system, because it silently omits the case where the gate did its job.
**A system looks flawless when its failures are stored somewhere its
successes are not** — not through dishonesty, but because the success path
writes a record and the rejection path just returns.

The general shape: whenever an outcome is written by the code that *acts*,
the outcomes where nothing was done leave no trace. Rejections, refusals,
no-ops and early exits all vanish, and what remains reads as an unbroken run
of wins. The fix is to make the decision durable rather than the action —
record the verdict wherever the comparison happens, not only where the
promotion happens.

Until that refactor, `promotion_history()` reads both sources and labels
which one each row came from, so the omission cannot recur silently.

### Four outcomes, kept distinct

| outcome | meaning |
|---|---|
| `promoted` | beat the champion beyond one standard error |
| `rejected` | behind, or ahead by less than the noise |
| `refused` | the comparison could not be trusted — no verdict issued |
| `rolled back` | promoted, then found invalid |

`rolled back` and `rejected` are deliberately separate. Collapsing them
would hide that a promotion was *reversed*, which is a different and more
serious event than one that never happened.

## Two PR-AUC figures, and why they differ

Panel 1 shows the champion's PR-AUC twice:

```
PR-AUC on unseen rows          0.5661   shadow A/B, rows never trained on
PR-AUC on its own eval slice   0.4834   recorded by the training run
```

These are **not in conflict and not comparable.** The training run evaluates
on the tail of its own window; the shadow test evaluates on rows after that
window, which is where the model actually operates. For v4 the shadow figure
is *higher*, but the ordering is not fixed and either can lead.

Showing only one is the trap. Whichever appears reads as "the model's
accuracy", and the other then looks like a contradiction — particularly next
to the promotion panel's `+0.0361`, which is a delta against the champion on
the shadow rows and belongs to neither figure alone.

The shadow figure is read from `model_promotions` in preference to
`reports/shadow_comparison.json`, because the table is append-only while the
file is overwritten by the next comparison run. A promoted model's evidence
should not disappear because someone later tested a different candidate.

## An idle consumer is not an error

The live-scoring panel shows historical rows when nothing is currently
streaming, and says which state it is in. An idle consumer is a normal
state; failing the panel would make the dashboard unusable exactly when
someone opens it to check whether anything is running.

Throughput is reported as an average over the observed span rather than an
instantaneous rate, because the audit table records when rows were scored,
not a sampled counter. Anything finer would be invented.

## Why not Prometheus and Grafana

This machine has **2 physical cores and 8 GB**, already running Redpanda, a
local MLflow store and a 799-tree model, with training pinned to
`n_jobs=2` — both cores. A scrape server, a time-series database and a
rendering service would contend for exactly the resources the pipeline
needs, to display numbers already durable in SQLite.

Postgres was assessed separately and skipped; see
[retraining.md](retraining.md) for the audit. Short version: the store layer
uses `AUTOINCREMENT`, `PRAGMA`, `executescript`, `lastrowid` and `?`
placeholders, so it is a day of rework rather than a connection string — an
earlier note claimed otherwise and was wrong.
