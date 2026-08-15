# Drift-triggered retraining

```bash
python -m src.training.retrain --check-only      # exit 0 if a retrain is due
python -m src.training.retrain --window-days 120
```

## Can the workflow run on GitHub Actions?

**No — not on a GitHub-hosted runner.** Not with a workaround, not with a
secret. Nothing it needs is in the repository:

| what a retrain needs | size | in git? |
|---|---|---|
| `data/splits/*.parquet` | 78 MB | no — gitignored, DVC-tracked |
| `mlflow.db` (tracking store **and** registry) | 1.3 MB | no |
| `mlartifacts/` (model binaries) | 6.2 MB | no |
| `reports/drift.db` (the alert being answered) | 4.0 MB | no |
| `reports/baseline_metrics.json` (the threshold) | 17 KB | no |

And DVC has **no remote configured**, so `dvc pull` has nowhere to pull from:

```
$ dvc status --cloud
ERROR: config file error: no remote specified
```

A hosted runner would check out the code, find an empty `data/` directory and
no registry, and fail. The `preflight` job checks for exactly these files and
fails with that explanation rather than a traceback forty lines into
training.

**It runs correctly on a self-hosted runner** on the machine that holds the
data and the store — which is what `runs-on: [self-hosted]` means in the
workflow. Everything else there is real: the schedule, the alert query, the
resolution marking, the concurrency guard, the step summary.

### What a real deployment would need

Three things move, and none is hard. They were skipped because Phase 4 was
scoped to stay self-contained.

**1. Data behind a DVC remote.** S3, GCS, or Azure Blob.

```bash
dvc remote add -d storage s3://fraudshield/dvcstore
dvc push
```

The runner then does `dvc pull` with credentials from repository secrets. The
`.dvc` pointers are already committed, so the content hash is already
version-controlled — only the bytes need a home.

**2. MLflow on a tracking server, not a local file.** A server with a
Postgres backend and object-store artifacts:

```
MLFLOW_TRACKING_URI=https://mlflow.internal:5000
```

`src/training/tracking.py` already reads that variable and needs no change.
This also removes a limitation that is easy to miss: SQLite tolerates one
writer, so two runners retraining concurrently would collide. The workflow's
`concurrency` group prevents that today, but that is a lock, not a fix.

**3. Alerts in Postgres, not SQLite.** This is real work, not a connection
string — an earlier version of this note claimed otherwise and was wrong.
Audited, `src/drift/store.py` and its callers use:

| construct | count | Postgres needs |
|---|---|---|
| `INTEGER PRIMARY KEY AUTOINCREMENT` | 4 | `GENERATED ... AS IDENTITY` or `SERIAL` |
| `PRAGMA journal_mode` / `PRAGMA table_info` | 2 | dropped; `information_schema.columns` for the migration |
| `executescript` | 3 | split into separate statements |
| `cursor.lastrowid` | 1 | `INSERT ... RETURNING id` |
| `?` placeholders | throughout | `%s` |
| `sqlite3.Row` | 1 | `RealDictCursor` |

The timestamps being ISO-8601 UTC text does help, and the queries themselves
are ordinary SQL. But the schema DDL, the migration mechanism, the insert
path and every parameter marker change. Budget a day, not an afternoon.

With those three, `runs-on:` becomes `ubuntu-latest` and nothing else in the
workflow changes.

### One thing that would still be wrong

Even fully deployed, a nightly cron is the wrong trigger shape. The monitor
already knows the instant drift breaches — it writes the alert — so a
scheduled job that polls for it adds up to 24 hours of latency to a decision
that was already made. The push equivalent is a `repository_dispatch` from
the monitor:

```python
POST /repos/{owner}/{repo}/dispatches
{"event_type": "drift-detected", "client_payload": {"alert_id": 1}}
```

Keep the schedule as a backstop for a monitor run that failed to fire.

## What retraining does differently from `train.py`

Exactly one thing that matters: the training data is a **configurable recent
window of everything labelled**, rather than the fixed Phase 0 train slice.
Hyperparameters, the temporal carve, early stopping, the metric block and
MLflow logging are imported from `train.py` rather than restated, so a
retrained model and the baseline stay measured the same way.

The window is split three ways, strictly ordered in time:

```
|<---------------- last 120 days of labelled data ---------------->|
|<----------- fit ----------->|<-- stop -->|<---- evaluate ---->|
            ~72%                   ~13%            ~15%
```

Evaluating on the **newest** slice is the point. A retrain exists because
recent behaviour changed, so scoring it against older rows answers the wrong
question. The original val slice is deliberately not reused — it sits inside
this window and would be leakage.

## The decision is read, never re-derived

`latest_retrain_trigger()` returns the alert row the monitor wrote, and its
PSI, drifting features and window boundaries become run parameters:

```
trigger.alert_id              1
trigger.value_drift_psi       1.7835
trigger.overall_psi_feature   id_31
trigger.drifting_features     id_31,id_13,D11,id_30,V160,V145,...
trigger.window_start_dt       13219186.0
trigger.window_end_dt         15811131.0
```

Any model in the registry can therefore be traced back to the drift
measurement that caused it, months later, without recomputing anything.
Re-deriving the decision inside the retrain would let the model and the alert
disagree about why the model exists — and the alert is the auditable record.

**Only genuine value drift reaches this module.** `retrain_triggered` is set
solely for drift whose populated values moved. Missingness-driven drift gets
`investigate_pipeline` and is invisible to the trigger query. Retraining on a
coverage change would bake a transient upstream join into the model — and
sixteen features in this dataset breach 0.20 for exactly that reason.

An alert is consumed once. `mark_resolved` records which run answered it and
refuses a second attempt, so a workflow that runs twice cannot register two
challengers for one drift measurement.

## Staging, never Production

A drift-triggered retrain produces a **challenger**. Whether it beats what is
live is the shadow A/B's decision, and that decision does not belong to the
thing that produced the challenger. `register_challenger` raises if
`TARGET_STAGE` is ever set to Production, and a test asserts it.

## Threads are asserted, not assumed

```python
assert_threads_pinned(xgb_model, lgb_model)   # checks the FITTED estimators
```

Checked on the models rather than on the config, because that is the only
thing that proves it — a config value can be overridden by an environment
variable, a stale import, or a caller passing `n_jobs` explicitly, and every
one of those failures is silent.

XGBoost's `hist` method is not thread-deterministic: its subsample and
colsample RNG streams are per-thread, so thread count changes which rows and
columns each tree sees. Measured on the real split, identical seed and data,
only `n_jobs` varying — 799 trees at 2 threads against 969 at 4, with
predicted probabilities differing by up to 0.44.

A runner's core count differs from any development machine. Unpinned, a
drift-triggered retrain would produce a model differing for reasons having
nothing to do with the drift, and the Phase 7 shadow comparison would be
measuring the core count as much as the model. If the assertion fails the run
exits `2` and registers nothing — a model that cannot be reproduced must not
become a challenger. See [reproducibility.md](reproducibility.md).
