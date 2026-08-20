# FraudShield

A fraud detection system that monitors its own decay, retrains when the data
moves, and **refuses to promote the result unless it can prove it is better**.

The model is the smallest part. Gradient boosting on tabular fraud data is a
solved problem; the hard part is what happens in month four, when the inputs
have shifted and nothing in the pipeline notices. FraudShield is built around
that question.

Every figure on this page comes from a run that actually happened. They are
reproducible from `reports/`, `mlflow.db` and the audit tables — see
[Reproducing the numbers](#reproducing-the-numbers).

---

## What it does

```
 transaction ──▶ Kafka/Redpanda ──▶ consumer ──┬──▶ champion  ──▶ decision + SHAP ──▶ audit DB
                                               └──▶ challenger ──▶ recorded, never acted on
                                                                                        │
      ┌─────────────────────────────────────────────────────────────────────────────────┘
      ▼
 drift monitor ──▶ PSI per feature ──▶ decomposed ──┬── genuine value drift ──▶ RETRAIN
 (train vs recent window)                           └── missingness-driven  ──▶ INVESTIGATE PIPELINE
                                                                  │                    (no retrain)
      ┌───────────────────────────────────────────────────────────┘
      ▼
 retrain on a recent window ──▶ MLflow run ──▶ registered @challenger (Staging, never Production)
      │
      ▼
 shadow A/B on rows BOTH models scored ──▶ leakage guard ──▶ margin vs bootstrap SE
      │                                          │                     │
      │                                    refuse if the         reject if inside
      │                                    rows were trained on   noise
      ▼
 promote to Production, archive the old champion, record the verdict
```

Two properties matter more than any component:

**The challenger never decides anything.** It scores every transaction in
parallel and its output is written to the same audit row, but the returned
decision is computed from the champion before the challenger runs. A
challenger returning 1.0 for every row changes nothing but the shadow
columns.

**A promotion has to survive two independent gates.** The comparison is
refused outright if the judged rows overlap the challenger's training window,
and rejected if the margin is inside the bootstrap standard error. Both have
fired on real attempts.

---

## The data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) —
real anonymised card-not-present transactions from Vesta Corporation, in
**USD**. Transaction and identity tables joined on `TransactionID`.

| | rows | fraud rate |
|---|---|---|
| total | 590,540 × 434 cols | 3.50% |
| `train` | 319,927 | 3.401% |
| `val` | 98,305 | 3.933% |
| `stream` | 172,308 | 3.433% |

Split on `TransactionDT` sixths across 182 days — sixths 1–3 train, sixth 4
validate, sixths 5–6 replayed as the live stream. Boundaries are recorded in
`split_manifest.json` and re-derived from the sixths rule on every load, so
the repo cannot assert a boundary the data does not have.

`TransactionDT` is a seconds offset from an unknown epoch, so wall-clock time
is unrecoverable. Hour-of-day is derivable and internally consistent;
absolute dates are not.

---

## Results

### Baseline model — v1

XGBoost, 431 features, `depth4_reg` regularisation, early stopping on a
temporal carve of train.

| | PR-AUC | lift over floor | ROC-AUC | F1 | train/val gap | trees |
|---|---|---|---|---|---|---|
| XGBoost | **0.5248** | 13.3× | 0.8917 | 0.5143 | +0.2177 | 799 |
| LightGBM | 0.5239 | 13.3× | 0.8918 | 0.5120 | +0.1932 | 624 |
| ensemble | 0.5264 | 13.4× | 0.8932 | 0.5149 | +0.2064 | — |

PR-AUC is the headline because at a 3.93% positive rate the alternatives
mislead. Accuracy is worthless — "never fraud" scores 96%. ROC-AUC is
optimistic: its false-positive axis is normalised by the 94k negatives, so
thousands of false alarms barely move it. **PR-AUC's no-skill floor is the
base rate itself, 0.0393**, which is why the lift is quoted beside it.

At the swept operating point (threshold **0.8018**, not 0.5 — `scale_pos_weight`
of 29 deliberately decalibrates the probabilities):

```
precision 0.6305   recall 0.4343   2.71% of traffic flagged
```

The ensemble's +0.0016 over XGBoost is not worth the second model in serving;
member probability correlation is 0.9778.

### Drift is real, and it decomposes

Comparing `train` against the stream, **27 of 431 features breach PSI 0.20**
(full-stream sweep: 278 stable / 4 moderate / 27 major; band counts shift with
window size, the major count does not). Those 27 split into two groups with
**different remedies**:

| | features | max PSI | populated-row PSI | NA rate | remedy |
|---|---|---|---|---|---|
| **missingness-driven** | 16 | 0.6001 | 0.000–0.015 | 61.7% → 26.6% | investigate the feed |
| **genuine value drift** | 11 | 2.2655 | 1.0–8.4 | 81.6% → 89.4% | **retrain** |

The first group — `M1`–`M3`, `M7`–`M9`, `V2`–`V11` — has PSI above 0.20 purely
because *more data started arriving*. Their populated values are unchanged to
three decimal places. Retraining on that would bake a transient upstream join
into the model.

The second group — `id_31`, `id_13`, `id_30`, `D11`, `V144`, `V145`, `V150`–`V152`,
`V159`, `V160`, `DeviceInfo` — genuinely moved. **Only this group triggers a
retrain.** Missingness raises `INVESTIGATE PIPELINE (no retrain)`.

**Overall PSI is the max, not the mean.** There is no canonical blended PSI,
and at real feature proportions the mean reads calm while `id_31` sits at
1.78. A test asserts exactly that.

#### What actually drifted

`id_31` is a **browser string**. Its unseen levels are `chrome 65.0`,
`chrome 65.0 for android`, `samsung browser 6.4`. `id_30` is an OS string:
`iOS 11.2.6`, `Mac OS X 10_13_4`. `DeviceInfo` is a device model.

This is **software version turnover**, not adversarial adaptation. Browsers
update, users upgrade phones, and the vocabulary the encoder was fitted on
goes stale. The dataset carries no evidence that fraudsters changed tactics,
and none is claimed here.

### The fast drift signal

A category the encoder has never seen is visible on the **first request
carrying it**, where windowed PSI needs a window of traffic to move. The
serving and streaming layers record them as they occur:

| feature | observations | distinct unseen levels |
|---|---|---|
| `id_31` | 5,959 | 14 |
| `id_30` | 835 | 3 |
| `DeviceInfo` | 506 | 87 |
| `id_33` | 297 | 27 |

8.80% of 66,429 scored rows carried at least one.

### Three promotion outcomes

**This is the substance.** A pipeline that promotes everything it trains is
not an A/B test.

| version | outcome | PR-AUC Δ | margin | judged rows | why |
|---|---|---|---|---|---|
| **v2** | rejected | −0.0100 | — | 25,000 | behind the champion on merit |
| **v3** | rolled back | +0.2492 | 20.13 SE | 25,000 | **100% training-window overlap** |
| **v4** | **promoted** | **+0.0327** | **4.81 SE** | 20,000 | ahead beyond noise, zero overlap |

**v2** trained on a 30-day window (61,723 fit rows) and lost honestly. Its
verdict is recorded on the model version; it is Archived, not deleted.

**v3** scored +0.2492 at *twenty* standard errors and was promoted — then the
leakage guard showed its judged rows sat entirely inside its own training
window (`10,569,737..11,106,772` inside `5,443,151..15,811,131`). It was
recalling rows it had been fitted on while the champion predicted them. The
promotion was rolled back and v1 restored. **A margin rule cannot catch this**:
a larger overlap produces a *more* decisive apparent win.

**v4** trained through the first half of the stream (264,006 fit rows, bounded
at `13,219,129`) and was judged on the 30 days after it — rows neither model
had seen. Leakage check: **0.0%**.

```
                    champion   challenger      delta
  PR-AUC              0.4942       0.5270    +0.0327
  precision           0.5300       0.5905    +0.0605
  recall              0.4069       0.4300    +0.0231
  false negatives        360          346        -14
  flagged                466          442        -24

  bootstrap std error  0.0068    95% CI [+0.0204, +0.0467]
  margin               4.81 SE   (need 1.00)
```

It catches 14 more frauds while raising 24 *fewer* flags — better on both
sides, not a threshold trade. That shape is what an honest comparison
produces; a leaked one does not.

This is the *second* comparison of v4. The first reported +0.0361 at 4.23 SE
and was withdrawn: the challenger had been scored with the baseline's
encoders, because `retrain.py` logged no artifacts and serving fell back to
whatever `encoders.pkl` was on disk. Five of thirty-one categorical columns
assign different codes between the two runs, including 970 `DeviceInfo`
levels and 49 in `id_31` — the feature whose drift triggered the retrain.
The conclusion survived re-measurement; the number moved.

**The leakage guard has fired twice, both on mistakes actually made** — once on
the promotion above, and again minutes later when a fresh consumer group
re-read a stale topic from the beginning and fed v4 the old replay batch. It
refused both.

### Serving latency

Single transaction, 799-tree model, 431 features, **including SHAP**:

```
  full HTTP round trip     6.74 ms p50    10.13 ms p95
  handler only             4.48 ms p50     5.67 ms p95
```

Per-request SHAP is affordable, but only one way. The obvious implementations
are not:

| approach | p50 | p95 |
|---|---|---|
| `shap.TreeExplainer(model)(row)` per request | 24.35 ms | 116.57 ms |
| training's `build_features` on one row | 22.10 ms | 54.46 ms |
| **booster `pred_contribs` + serving encoder** | **3.99 ms** | **4.72 ms** |

Assembled naively this endpoint would be **~58 ms p50 and over 150 ms at p95**.
Two changes fix it: one booster call returning SHAP *and* probability
(`sigmoid(Σcontribs)` is the margin, verified equal to `predict_proba` within
3.3e-07), and a serving encoder that is **429× faster** than the frame path
and asserted bit-identical to it on every column of every checked row.

---

## Reproducibility: a finding, not a checkbox

**XGBoost's `hist` tree method is not thread-deterministic.** The `subsample`
and `colsample_bytree` RNG streams are per-thread, so the thread count changes
which rows and columns each tree sees. Same seed, same data, only `n_jobs`
varying:

| `n_jobs` | trees at early stop | max Δ predicted probability |
|---|---|---|
| 1 | 239 | — |
| 2 | 300 | 0.318 |
| 4 | 140 | 0.438 |

A 0.44 absolute swing is not floating-point summation order. Confirmed at full
scale: 799 trees at `n_jobs=2` against 969 at `n_jobs=-1`.

**LightGBM was the control** and reproduced to every digit across thread counts
— 624 trees, 0.5239 PR-AUC, 0.8918 ROC-AUC. That ruled out the data, the split,
the encoders and the pipeline, leaving XGBoost's sampling RNG.

This surfaced as two conflicting descriptions of one model: a Colab run
reporting 651 trees at 0.5255 and a local run reporting 969 at 0.5271. Same
code, same data, same seed, different core count.

**`N_JOBS = 2` is pinned, never `-1`.** CI runners differ in core count from any
development machine. Unpinned, a drift-triggered retrain would produce a model
that differs for reasons having nothing to do with the drift — and the shadow
comparison would be measuring the runner's core count as much as the model.
`retrain.py` asserts the effective value on the *fitted estimators* and exits
without registering if it does not match.

No thread count is better: the three runs span 0.0023 PR-AUC against a
bootstrap SE of 0.0080. Pinning buys repeatability, not quality.

---

## MLOps pillars

| pillar | implementation |
|---|---|
| **Data versioning** | DVC tracks `data/raw` and `data/splits`; git holds only the `.dvc` pointers with content hashes |
| **Experiment tracking** | MLflow, one run per model, 32–34 params and 83 metrics each, threshold blocks flattened to dotted keys so confusion counts are queryable |
| **Model registry** | 4 versions with stages and aliases; `@champion` is what serving resolves, `@challenger` is the candidate |
| **Reproducibility** | `n_jobs` pinned and asserted; seeds fixed; encoders fitted on the reduced train and logged beside the model; environment recorded on every run |
| **Serving** | FastAPI, model resolved from the registry at startup, per-decision SHAP at 6.74 ms p50 |
| **Streaming** | Redpanda replay preserving `TransactionDT` order; one shared `predict_one` for HTTP and stream so they cannot diverge |
| **Drift monitoring** | PSI per feature, decomposed into missingness vs value drift, alerts in SQLite, Evidently HTML alongside |
| **Automated retraining** | Reads the alert row rather than re-deriving; logs the triggering PSI and drifting features as run params |
| **Safe deployment** | Shadow A/B with a leakage guard and a bootstrap-noise floor; Staging never straight to Production |
| **Explainability** | Exact TreeSHAP per decision, additivity asserted every run, ordinal codes decoded to browser strings |
| **Audit trail** | Every scored transaction with probability, decision, threshold, model version, SHAP factors, unseen categories and true label |
| **Observability** | Streamlit dashboard reading the stores; a test asserts it cannot recompute what it displays |

---

## How to run this project

Start to finish, in the order that actually works. Steps depend on each other
— the consumer needs a producer to have published, the shadow comparison needs
both models to have scored the same rows, the dashboard needs the stores to
exist.

### 0. Prerequisites

| | requirement |
|---|---|
| **Python** | 3.12 (developed on 3.12.6; 3.11 should work, untested) |
| **Docker Desktop** | only for Redpanda in step 6. Everything else runs without it |
| **RAM** | **8 GB minimum, and that is genuinely the floor** |
| **Disk** | ~4 GB total |
| **Cores** | any; `n_jobs` is pinned to 2 regardless — see [Reproducibility](#reproducibility-a-finding-not-a-checkbox) |

On RAM, honestly: this was built on 8 GB / 2 physical cores and it fits, but
not comfortably. The 120-day retrain peaked at **1.2 GB resident** for the
Python process alone, with Redpanda holding 133 MB and Docker reserving ~3.9 GB
of the host. Close other applications before a full retrain. That constraint is
why the stack is SQLite and Streamlit rather than Postgres, Prometheus and
Grafana.

Disk, measured on this machine:

```
  raw Kaggle download   ~1.2 GB   (train_transaction.csv is 652 MB)
  data/splits             78 MB
  .dvc/cache              78 MB   (a second copy of the splits)
  virtualenv             ~2.5 GB  (xgboost, lightgbm, shap, mlflow,
                                   evidently, streamlit and their trees)
  mlartifacts             31 MB   grows ~6 MB per registered model
  reports/                65 MB   mostly drift.db and the Evidently HTML
```

### 1. Clone and enter

```bash
git clone https://github.com/Pavankumar06T/FraudShield-MLops.git
cd FraudShield-MLops/fraudshield
```

The Python package lives in the `fraudshield/` subdirectory. Every command
below is run from there.

### 2. Virtual environment

**PowerShell (Windows)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**bash (Linux/macOS/WSL/Git Bash)**

```bash
python -m venv .venv
source .venv/bin/activate
```

Confirm you are inside it before installing — the single most common failure
here is installing into the system Python and then wondering why imports fail:

```bash
python -c "import sys; print(sys.prefix)"    # must contain '.venv'
```

### 3. Install

```bash
pip install --upgrade pip
pip install -r requirements.lock.txt          # exact pins, reproducible
```

`requirements.txt` holds floors with the reasoning; `requirements.lock.txt`
holds the exact versions this was verified on. Use the lock file.

Installing takes several minutes — xgboost, lightgbm, shap, mlflow, evidently
and streamlit are all large.

### 4. Get the data — read this before starting

The dataset is [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
on Kaggle. Two things to know before you begin:

1. **Kaggle requires phone verification** to download competition data, even
   for a closed competition. Account → Settings → Phone Verification. There is
   no way around it and it is not obvious until the download fails.
2. **You must accept the competition rules** on the competition page, or the
   API returns 403.

```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/ (Windows: %USERPROFILE%\.kaggle\)
kaggle competitions download -c ieee-fraud-detection -p data/raw
cd data/raw && unzip ieee-fraud-detection.zip && cd ../..
```

You need `train_transaction.csv` (652 MB) and `train_identity.csv` (26 MB).
`test_*.csv` are unlabelled and unused here.

#### Producing the splits — **not automated in this repo**

Stated plainly: **there is no split script in this repository.** `src/data/`
contains loaders only. The splits were produced once in a Colab notebook
before the repo existed, and `src/common/config.py` *reads* the resulting
manifest rather than creating it.

DVC tracks `data/splits` (md5 `09d6c70a…`, 4 files, 81,374,856 bytes) so the
content is versioned — but **no DVC remote is configured**, so `dvc pull` has
nowhere to pull from. Cloning this repo does not get you the data.

To reproduce the splits yourself, join the two tables and cut on
`TransactionDT` sixths:

```python
import pandas as pd, json

tx = pd.read_csv("data/raw/train_transaction.csv")
idf = pd.read_csv("data/raw/train_identity.csv")
df = tx.merge(idf, on="TransactionID", how="left")      # 590,540 x 434

lo, hi = df.TransactionDT.min(), df.TransactionDT.max()
m3 = lo + (hi - lo) / 6 * 3        # train | val
m4 = lo + (hi - lo) / 6 * 4        # val   | stream

splits = {"train":  df[df.TransactionDT <  m3],
          "val":    df[(df.TransactionDT >= m3) & (df.TransactionDT < m4)],
          "stream": df[df.TransactionDT >= m4]}

meta = {}
for name, part in splits.items():
    part.to_parquet(f"data/splits/{name}.parquet", index=False)
    meta[name] = {"rows": len(part),
                  "fraud_rate_pct": round(part.isFraud.mean() * 100, 3)}

json.dump({"source": "IEEE-CIS train_transaction + train_identity "
                     "(left join on TransactionID)",
           "total_rows": len(df), "total_columns": df.shape[1],
           "dt_min": float(lo), "dt_max": float(hi),
           "boundaries": {"month3_end": float(m3), "month4_end": float(m4)},
           "splits": meta},
          open("data/splits/split_manifest.json", "w"), indent=2)
```

You should get 319,927 / 98,305 / 172,308 rows at 3.401% / 3.933% / 3.433%
fraud. `config.py` re-derives both boundaries from the sixths rule on load and
raises if the manifest disagrees, so a wrong cut fails loudly.

### 5. Point at the data

```bash
# bash
export FRAUDSHIELD_DATA=/absolute/path/to/data

# PowerShell
$env:FRAUDSHIELD_DATA = "E:\path\to\data"

# Colab, with Drive mounted
import os; os.environ["FRAUDSHIELD_DATA"] = "/content/drive/MyDrive/fraudshield/data"
```

Unset, it defaults to `<repo>/data`. Verify before going further:

```bash
python -m src.common.config
```

Expect an `x` beside every path that exists, and the split boundaries with row
counts and fraud rates. If `TRAIN_PARQUET` has no `x`, nothing downstream will
work.

Other variables, all optional: `MLFLOW_TRACKING_URI` (defaults to
`sqlite:///mlflow.db`), `MLFLOW_EXPERIMENT_NAME`, `KAFKA_BOOTSTRAP`
(`localhost:9092`; use `redpanda:29092` from inside a container).

### 6. Run it

Each step says what to expect. Times are from an 8 GB / 2-core machine.

#### 6a. Drift report — the Phase 0 measurement (~6 min)

```bash
python -m src.drift.report
```

> `278 stable / 4 moderate / 27 major`, `id_31` top at 1.525, and
> `17 of the major drifters are missingness-driven`. Writes
> `reports/psi_report.csv` and `reports/psi_decomposition.csv`.

Optional, but it is the evidence the whole project rests on.

#### 6b. Train the baseline (~25 min)

```bash
python -m src.training.train
```

> Temporal carve, then XGBoost and LightGBM, then the comparison table.
> XGBoost lands at PR-AUC 0.5248 / 799 trees. Writes `models/`,
> `reports/baseline_metrics.json`, and three MLflow runs.

`--sample 50000` gives a fast smoke test that deliberately **refuses to save**
— a model fitted on a slice is not a baseline.

#### 6c. Register it

```bash
python -m src.training.register_model
```

> `fraudshield-xgboost` v1 → Staging, `@champion` and `@staging` aliases.
> Everything downstream resolves the model from the registry, not from disk.

#### 6d. Drift monitor (~8 min)

```bash
python -m src.drift.monitor --window-days 30
```

> `RETRAIN + INVESTIGATE PIPELINE`, worst feature `id_31`, the two drift types
> reported separately. Writes alert #1 to `reports/drift.db`,
> `reports/drift_decision.json`, and a ~7 MB Evidently HTML.

Add `--no-evidently` to skip the HTML if you only want the numbers.

#### 6e. Start Redpanda

```bash
docker compose up -d redpanda
docker compose ps                    # wait for (healthy)
```

> ~133 MB, under 1% CPU idle. Kafka API on `localhost:9092`.

#### 6f. Produce, then consume — **order matters**

```bash
python -m src.streaming.producer --rate 200 --limit 5000
python -m src.streaming.consumer --from-beginning --limit 5000
```

> Producer: `5,000 published in 25.0s (200/s), 0 delivery failures`.
> Consumer: ~2.6% blocked, ~10% carrying unseen categories, 0 errors.

The consumer needs messages to already exist. Started first it waits and then
exits on its idle timeout.

**Use a fresh topic for each experiment.** A new consumer group defaults to
`auto.offset.reset=earliest`, so pointing one at a reused topic silently
re-reads the old messages. This caused a real invalid comparison during
development — see [Common failures](#common-failures).

#### 6g. Serve

```bash
uvicorn src.serving.app:app --port 8080
```

> `loaded fraudshield-xgboost v1 @champion`, 799 trees, threshold 0.8018 read
> from `baseline_metrics.json` — not 0.5.

In another terminal:

```bash
curl -s localhost:8080/health | python -m json.tool

curl -s -X POST localhost:8080/predict \
     -H 'Content-Type: application/json' \
     -d @examples/transaction_fraud.json | python -m json.tool
```

> `"fraud_probability": 0.999695, "decision": "BLOCK"`, top SHAP factors with
> categorical codes decoded, `"latency_ms"` around 6–8.

A transaction carrying an unseen browser scores rather than erroring:

```bash
curl -s -X POST localhost:8080/predict \
     -H 'Content-Type: application/json' \
     -d @examples/transaction_unseen_browser.json | python -m json.tool
```

> `"unseen_categories": ["id_31"]` — the fast drift signal, per request.

#### 6h. Retrain (~75 min at 120 days)

```bash
python -m src.training.retrain --check-only     # exit 0 if a retrain is due
python -m src.training.retrain --window-days 120 --end-dt 13219186
```

> Names the alert it answers, the window split, `thread pinning verified`,
> then registers the next version as `@challenger` in **Staging** and marks the
> alert resolved.

**`--end-dt` is not optional if you intend to run a shadow test.** Without it
the window runs to the newest data and swallows the rows the comparison will
judge; the guard then refuses the comparison. Set it below the `TransactionDT`
you will replay from.

#### 6i. Shadow scoring, then compare

With a challenger registered, the consumer scores every transaction twice
automatically. Replay rows the challenger has **not** been trained on:

```bash
python -m src.streaming.producer --rate 0 --start-dt 13219186 \
       --limit 20000 --topic judged_window
python -m src.streaming.consumer --topic judged_window --group judged \
       --from-beginning --limit 20000
python -m src.serving.compare                    # report only
python -m src.serving.compare --promote          # promote if it wins
```

> `PROMOTE` or `no promotion, difference within noise`, with the leakage check
> stated explicitly: `overlap 0.0%`. Promotion moves the challenger to
> Production, archives the old champion, and records the verdict.

Exit codes: `0` promoted or reported, `3` refused.

#### 6j. Dashboard

```bash
streamlit run src/dashboard/app.py
```

> Five panels at `localhost:8501`. Needs `reports/drift.db` and `mlflow.db` to
> exist; panels degrade to "nothing recorded yet" rather than erroring.

#### 6k. Tests — runnable without any of the above

```bash
pytest -q                                        # 321 tests, ~45s
pytest tests/test_dashboard.py -q                # one module
```

Tests needing the real splits or a trained model skip cleanly when absent, so
this works on a fresh clone.

### What you can run without the data

| step | works without data? |
|---|---|
| `pytest -q` | yes — data-dependent tests skip |
| `python -m src.common.config` | yes — reports what is missing |
| everything else | **no** |

There is no bundled sample dataset and no DVC remote. Without the IEEE-CIS
download you cannot train, monitor, serve, stream or compare.

### Common failures

**`ModuleNotFoundError: No module named 'src'`** — you are not in the
`fraudshield/` directory, or the venv is not active. Run from the directory
containing `src/`, and check `python -c "import sys; print(sys.prefix)"`.

**`Split manifest not found`** — `FRAUDSHIELD_DATA` is unset or wrong. The
error prints the path it resolved to. Run `python -m src.common.config`.

**`error during connect ... docker_engine`** — Docker Desktop is not running.
Start it and wait for the whale icon to settle before `docker compose up`.

**Consumer scores rows you did not just publish.** A new consumer group starts
at `auto.offset.reset=earliest` and re-reads the whole topic. During
development this fed a challenger the *previous* replay batch — rows inside its
own training window — and produced an invalid comparison that the leakage guard
caught at 100% overlap. Use `--topic` with a fresh name per experiment.

**`NO VERDICT -- evaluation rows overlap the challenger's training window`** —
working as intended. The retrain window included the rows being judged. Retrain
with `--end-dt` below your replay start.

**Colab: everything vanishes after a restart.** The runtime wipes local disk.
Keep `FRAUDSHIELD_DATA` on mounted Drive, and set
`MLFLOW_TRACKING_URI=sqlite:////content/drive/MyDrive/fraudshield/mlflow.db`
so the registry survives too. Otherwise a disconnect loses every registered
model.

**LightGBM warns about `eval_set` being deprecated.** Harmless, and specific to
one patched build. `eval_set` is the correct parameter; `eval_X`/`eval_y` raise
`TypeError` on stock LightGBM.

**Retrain seems to hang with no output.** PowerShell's `Out-File` buffers until
the process exits, so a redirected log stays empty mid-run. Check the process is
alive rather than the log.

---

## Limitations

Stated plainly, because a system that hides these is the thing this project
exists to argue against.

**The replay competes with the training window.** The stream is historical data
replayed, and a retrain window defined as "the last N days" will always swallow
the rows the shadow test is about to judge. That is exactly how v3's invalid
promotion happened. `--end-dt` and `--start-dt` bound the two apart, but the
operator has to choose the boundary — nothing enforces it automatically. On a
real feed with genuinely future traffic the problem does not arise.

**The promotion verdict rests on 11,429 rows, not 20,000.** The consumer hit its
idle timeout early. That is above the 5,000-row floor with 358 positives and the
verdict stands, but a larger window would tighten the 0.0085 standard error.

**Postgres is a day of work, not a connection string.** An earlier note claimed
otherwise and was wrong. The store uses `AUTOINCREMENT` (4×), `PRAGMA` for WAL
and column introspection, `executescript`, `cursor.lastrowid` and `?`
placeholders. Postgres needs `IDENTITY`, `information_schema`, `RETURNING id`
and `%s`. The schema DDL, migration mechanism and insert path all change.

**`shadow_comparisons` should be written unconditionally.** Today only
*promotions* get a durable row; a rejected challenger's verdict survives merely
as version tags. A promotion history built from one source would show an
unbroken run of successes. The general shape is worth naming: **when an outcome
is written by the code that acts, every outcome where nothing was done leaves no
trace.** The fix is to make the decision durable rather than the action.

**The GitHub Actions workflow needs a self-hosted runner.** Nothing it requires
is in git — 78 MB of parquet, `mlflow.db`, `mlartifacts/`, `reports/drift.db` —
and DVC has no remote configured. The preflight job detects exactly this and
fails with the reason.

**v3 is still in Staging, untested.** Its only honest verdict is "unknown"; it
has never had a fair trial.

**LightGBM stopped 7 trees short of the 2,000 ceiling** on the 120-day window.
The count was effectively capped rather than chosen, and the ceiling wants
raising for windows that size.

---

## Reproducing the numbers

| figure | source |
|---|---|
| baseline metrics, thresholds, band counts | `reports/baseline_metrics.json` |
| PSI per feature, decomposition | `reports/psi_report.csv`, `reports/psi_decomposition.csv`, `reports/drift_decision.json` |
| drift alerts and verdicts | `drift_alerts` table in `reports/drift.db` |
| unseen categories | `unseen_observations` table |
| scored transactions, shadow predictions | `stream_predictions` table |
| promotion history | `model_promotions` table + MLflow version tags |
| shadow comparison | `reports/shadow_comparison.json` |
| model versions and lineage | MLflow registry (`mlflow.db`) |

```bash
streamlit run src/dashboard/app.py    # every figure above, read not recomputed
```

---

## SDG alignment

**SDG 8 — Decent Work and Economic Growth.** Payment fraud is a direct drag on
commerce, and its cost falls hardest on small merchants who cannot absorb
chargebacks or afford enterprise fraud tooling. At the promoted operating point
this model flags **2.7% of traffic** to catch **47% of fraud** — a tractable
review queue rather than a blanket block. An open, self-maintaining pipeline
lowers the cost of that capability.

**SDG 16 — Peace, Justice and Strong Institutions**, specifically target **16.4**,
reducing illicit financial flows. Card fraud is a funding channel for organised
crime, and detection systems that silently decay are a channel left open. The
contribution here is not the detection rate; it is that the system *knows when
it has stopped working* and refuses to certify a replacement it cannot prove is
better.

A note on scope: SHAP attributions make bias auditing *possible* — a per-decision
factor breakdown is a prerequisite for asking whether a model treats groups
differently. No such audit has been run here, and IEEE-CIS carries no
demographic attributes to run one against. That is a capability this
architecture supports, not a claim about this model.

---

## Repository

```
src/
  common/config.py        paths, split boundaries read from the manifest
  data/                   split loaders
  features/               versioned feature engineering, fit/apply encoders
  training/               train, retrain, explain, register, compare configs
  drift/                  PSI, decomposition, monitor, SQLite alert store
  serving/                FastAPI app, shared predictor, serving encoder, shadow A/B
  streaming/              Redpanda producer and dual-scoring consumer
  dashboard/              Streamlit UI and its read-only data layer
docs/
  reproducibility.md      thread determinism, the pandas exoneration
  serving.md              latency measurements, curl examples
  retraining.md           what a real deployment needs
  dashboard.md            why rejections go missing from promotion histories
```

Run all these:

cd "E:\FraudShield MLOps\fraudshield"
.\.venv\Scripts\Activate.ps1
$env:FRAUDSHIELD_DATA = "E:\FraudShield MLOps\fraudshield\data"

python -m streamlit run src/dashboard/app.py

uvicorn src.serving.app:app --port 8080

pytest -q

FOR DATA: python -m src.common.config

For How does drift detection work / show me the numbers:
python -m src.drift.report

For an Instant answer : python -c "import pandas as pd; d=pd.read_csv('reports/psi_report.csv',index_col=0); print(d.head(15))"

How do you know the versions:
pytest -q
