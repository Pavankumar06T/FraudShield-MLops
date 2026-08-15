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
| **v4** | **promoted** | **+0.0361** | **4.23 SE** | 11,429 | ahead beyond noise, zero overlap |

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
  PR-AUC              0.5299       0.5661    +0.0361
  precision           0.5521       0.6022    +0.0501
  recall              0.4441       0.4693    +0.0251
  false negatives        199          190         -9
  flagged                288          279         -9

  bootstrap std error  0.0085    95% CI [+0.0192, +0.0532]
  margin               4.23 SE   (need 1.00)
```

It catches 9 more frauds while raising 9 *fewer* flags — better on both sides,
not a threshold trade. That shape is what an honest comparison produces; a
leaked one does not.

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

## Setup

Requires Python 3.12 and Docker.

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock.txt

# data lives outside the repo; point at it
export FRAUDSHIELD_DATA=/path/to/fraudshield/data
python -m src.common.config                        # prints resolved paths
```

`FRAUDSHIELD_DATA` exists so the same code runs locally and against
Drive-mounted storage on Colab. Nothing outside `src/common/config.py` reads
it.

### Run

```bash
# 1. baseline
python -m src.training.train
python -m src.training.register_model

# 2. serve
uvicorn src.serving.app:app --port 8080
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' \
     -d @examples/transaction_fraud.json | python -m json.tool

# 3. stream
docker compose up -d redpanda
python -m src.streaming.producer --rate 200 --limit 5000
python -m src.streaming.consumer --from-beginning --limit 5000

# 4. drift
python -m src.drift.monitor --window-days 30

# 5. retrain + shadow A/B
python -m src.training.retrain --check-only        # exit 0 if a retrain is due
python -m src.training.retrain --window-days 120 --end-dt 13219186
python -m src.serving.compare --promote

# 6. dashboard
streamlit run src/dashboard/app.py
```

`--end-dt` is not optional for an honest test — see the limitation below.

### Tests

```bash
pytest -q          # 321 tests
```

They encode the findings rather than the implementation: that missingness
drift never triggers a retrain, that flipping `isFraud` in a request changes
nothing, that a 20 SE margin cannot promote a leaked comparison, that the
dashboard cannot import a PSI function.

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
