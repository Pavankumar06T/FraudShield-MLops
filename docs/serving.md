# Serving

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080
```

The model is resolved from the MLflow registry at startup —
`models:/fraudshield-xgboost@staging` — not from `models/baseline_xgb.json`.
The registry alias *is* the definition of "promoted"; the local file is
whatever the last training run wrote, and the two diverge the moment anyone
experiments.

Encoders are downloaded from the **same run** that produced the registered
model. A model served against encoders from a different run fails silently:
the ordinal codes simply mean something else, and every prediction is
quietly wrong.

## A registered model is served with its own encoders, or not at all

`load_bundle` resolves encoders from the MLflow run behind the registered
version. If that run has none, it **raises** — it does not fall back to
`models/encoders.pkl`.

That file belongs to whichever run wrote it last. Measured between two runs
of this project, five of thirty-one categorical columns assigned different
codes to the same level — 970 `DeviceInfo` levels and 49 in `id_31`, the
feature whose drift triggered the retrain — and 57 of 400 predictions
changed, by up to 0.2502.

Every prediction still computes. That is precisely why it must raise: nothing
downstream can tell the codes mean something else. This failure survived a
full promotion cycle, and was caught only by someone reading `(FALLBACK)` in
a startup log.

```
EncoderResolutionError: Model version 2 (run 5894f0e9...) has no encoders.pkl
logged, so its own encoders cannot be resolved: ...
Refusing to serve it against models/encoders.pkl, which belongs to whichever
run wrote it last. ...
Fix by logging the encoders onto that run, or set
FRAUDSHIELD_ALLOW_ENCODER_FALLBACK=1 to accept the risk in development.
```

The escape hatch exists for development and is off by default. An absent
*challenger* is still not an error — `try_load_bundle` returns None, because
shadow scoring is optional and a real decision is not.

The same rule now applies to the threshold: `resolve_threshold` reads it from
the version's own run rather than from a file keyed by role. Resolving by role
meant the operating point followed the alias, so promoting a challenger
silently swapped its 0.8032 for the baseline's 0.8018.

## Latency, measured

On the promoted 799-tree model, 431 features, single transaction:

| stage | p50 | p95 |
|---|---|---|
| encode one transaction (dict → numpy) | 0.05 ms | 0.05 ms |
| `DMatrix` + `pred_contribs` (probability **and** SHAP) | 3.99 ms | 4.72 ms |
| **handler total** | **4.48 ms** | **5.67 ms** |
| **full HTTP round trip** | **6.74 ms** | **10.13 ms** |

**Per-request SHAP is affordable.** That was not obvious in advance, and the
obvious implementations are not:

| approach | p50 | p95 |
|---|---|---|
| `shap.TreeExplainer(model)(row)` per request | 24.35 ms | 116.57 ms |
| `build_features` on a single row | 22.10 ms | 54.46 ms |
| sklearn `predict_proba` + separate SHAP | ~35 ms | ~90 ms |
| **booster `pred_contribs` + `encode_row`** | **4.0 ms** | **4.7 ms** |

Assembled the natural way — training's `build_features`, then the `shap`
package — this endpoint would be **~58 ms p50 and over 150 ms at p95**, over
budget on the median and badly over on the tail. Two changes fix it:

1. **One booster call, not two.** `pred_contribs=True` returns per-feature
   SHAP values plus a bias term whose sum *is* the margin, so
   `sigmoid(sum(contributions))` gives the probability. Probability and
   explanation come from the same call and are consistent by construction
   rather than by coincidence. Verified equal to `predict_proba` within
   3.3e-07.

2. **A serving encoder.** `build_features` costs per *column* — 431 pandas
   operations at ~50 µs of overhead each. `encode_row` does the same work
   against dicts and numpy in 0.05 ms, **429× faster**, and
   `tests/test_serving.py` asserts the two produce bit-identical vectors
   across every column of every checked val row. It is safe because it is
   proven equivalent, not because it looks equivalent — if that test fails,
   the fast path is wrong and should be deleted rather than patched.

If SHAP ever does become the bottleneck (a deeper model, a larger ensemble),
the next move is to compute it only for flagged transactions. At the
promoted threshold that is 2.7% of traffic, so it would cut explanation cost
by ~97% while still explaining every decision anyone disputes. It is not
needed at 4 ms and has not been done — noting it so the option is on record
rather than rediscovered.

## `iteration_range` is load-bearing

Early stopping left **849 boosted rounds** in the booster, of which **799**
are the model. Predicting without bounding the range uses 50 trees that were
never validated and moves some probabilities by **8.2 percentage points**.
The sklearn wrapper applies `best_iteration` automatically; the raw booster
does not. `tests/test_serving.py::test_unbounded_prediction_would_differ_materially`
pins this.

## Endpoints

### `GET /health`

```bash
curl -s localhost:8080/health | python -m json.tool
```

```json
{
  "status": "ok",
  "model_name": "fraudshield-xgboost",
  "model_version": "1",
  "model_stage": "Staging",
  "n_trees": 799,
  "n_features": 431,
  "threshold": 0.8018161058425903,
  "threshold_source": "best-F1 from baseline_metrics.json",
  "run_id": "086411bc7958464aaa5a8a5ca8264e50"
}
```

### `POST /predict`

The body is an open JSON object. Absent fields are meaningful: ~76% of the
IEEE-CIS identity columns are empty on any given row, and XGBoost routes
missing values by a learned default, so requiring all 431 features would
reject most real traffic. Omitting a field and sending it as `null` mean the
same thing.

The threshold is the swept best-F1 point read from `baseline_metrics.json`,
**not** 0.5. With `scale_pos_weight` at 29 the probabilities are
deliberately uncalibrated, so 0.5 is an artifact of the class ratio rather
than a decision anyone made. At 0.8018 the model flags 2.7% of traffic.

```bash
curl -s -X POST localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d @examples/transaction_fraud.json | python -m json.tool
```

```json
{
  "transaction_id": "3222397",
  "fraud_probability": 0.999695,
  "decision": "BLOCK",
  "threshold": 0.801816,
  "top_factors": [
    {"feature": "V87",  "value": "5", "contribution": 1.0221, "direction": "toward fraud"},
    {"feature": "V258", "value": "7", "contribution": 0.8414, "direction": "toward fraud"},
    {"feature": "V45",  "value": "7", "contribution": 0.7242, "direction": "toward fraud"}
  ],
  "unseen_categories": [],
  "model_version": "1",
  "n_trees": 799,
  "latency_ms": 8.4
}
```

A minimal request scores fine:

```bash
curl -s -X POST localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"TransactionAmt": 500.0, "ProductCD": "W", "card4": "visa"}'
```

Ask for more or fewer factors with `?top_factors=12`.

## Unseen categories

A transaction carrying a category absent from training **scores; it does not
error.** This is not defensive programming — it is the drift already
measured: `id_31` has the highest PSI in the sweep (1.525) precisely because
the stream window carries browser strings train never saw.

```bash
curl -s -X POST localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d @examples/transaction_unseen_browser.json | python -m json.tool
```

```json
{
  "fraud_probability": 0.490573,
  "decision": "ALLOW",
  "unseen_categories": ["id_31"],
  "...": "..."
}
```

The unseen level maps to a dedicated code (`-1`), kept distinct from missing
(`NaN`) so the model can split on "a browser we have never seen" separately
from "no browser recorded". `unseen_categories` in the response makes
vocabulary drift visible per request, rather than waiting for a nightly PSI
job to notice.

Malformed input degrades rather than rejects: a non-numeric value in a
numeric field is treated as absent, because losing one field should not cost
the whole decision. An empty body is a `422` — that is a client bug, not a
transaction.

## Sample payloads

Drawn from real rows of the val split, with `isFraud` removed:

| file | fields populated | P(fraud) | actual |
|---|---|---|---|
| `examples/transaction_fraud.json` | 323 / 433 | 0.9997 | fraud |
| `examples/transaction_legitimate.json` | 225 / 433 | 0.0200 | legitimate |
| `examples/transaction_unseen_browser.json` | — | 0.4906 | carries `id_31="chrome 65.0"` |

## Docker

```bash
docker build -t fraudshield:latest .

docker run --rm -p 8080:8080 \
  -v "$PWD/mlflow.db:/app/mlflow.db:ro" \
  -v "$PWD/mlartifacts:/app/mlartifacts:ro" \
  -v "$PWD/reports:/app/reports:ro" \
  fraudshield:latest
```

The image carries the code, not the model. Baking a model in would let the
running container and the registry disagree about what is promoted, with
nothing to notice. `OMP_NUM_THREADS=2` is set inside the image for the same
reason `N_JOBS` is pinned — see [reproducibility.md](reproducibility.md).
