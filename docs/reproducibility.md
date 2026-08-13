# Reproducibility

What must be pinned for two training runs to produce the same model, and why
each pin exists. Every entry here came from an actual divergence, not from
caution.

## Thread count: `N_JOBS = 2`, never `-1`

**XGBoost's `hist` tree method is not thread-deterministic.** The `subsample`
and `colsample_bytree` RNG streams are per-thread, so the thread count
changes which rows and columns each tree sees. Not a rounding difference — a
different model.

Measured with an identical seed, identical data, and only `n_jobs` varying:

| `n_jobs` | trees at early stop | max Δ in predicted probability vs `n_jobs=1` |
|---|---|---|
| 1 | 239 | — |
| 2 | 300 (hit ceiling) | 0.318 |
| 4 (`-1` here) | 140 | 0.438 |

A 0.44 absolute swing in predicted probability is far beyond floating-point
summation order. It is a different sample of rows and columns per tree.

Confirmed at full scale on the real 271,938-row split, varying only
`n_jobs`:

| run | XGBoost trees | XGBoost PR-AUC | LightGBM trees | LightGBM PR-AUC |
|---|---|---|---|---|
| `n_jobs=2` (pinned) | 799 | 0.5248 | 624 | 0.5239 |
| `n_jobs=-1` → 4 | 969 | 0.5271 | 624 | 0.5239 |
| Colab, unpinned | 651 | 0.5255 | — | — |

**LightGBM is the control.** It reproduced to every digit — 624 trees,
0.5239 PR-AUC, 0.8918 ROC-AUC, +0.1932 gap — across both thread counts.
That rules out the data, the split, the encoders and the pipeline as sources
of the variation, and leaves XGBoost's sampling RNG.

This is how the promoted model came to have two conflicting descriptions: a
Colab run reported 651 trees at PR-AUC 0.5255, while the same configuration
on a 4-core machine reported 969 at 0.5271. Same code, same data, same seed.
Different core count.

**No thread count is better.** The three XGBoost runs span 0.0023 PR-AUC,
against a bootstrap standard error of 0.0080 on the 98,305-row val split
(95% CI width 0.0301). They are the same model quality reached by different
row and column samples. Pinning does not choose a better model — it makes
the choice repeatable, which is the only property that matters downstream.

`-1` is the one value that must never be used, because it resolves to the
core count of whatever machine happens to run it. `2` is chosen because it is
portable — Colab, GitHub Actions runners, and an 8 GB laptop all have at
least two cores — and roughly halves wall clock against single-threaded.

### Why this is not merely tidiness

Phase 6 retrains automatically on GitHub Actions runners, whose core count
differs from any development machine. Without a pinned `n_jobs`, a
drift-triggered retrain produces a model that differs for reasons having
nothing to do with the drift. The Phase 7 shadow A/B then compares the live
model against a challenger that differs in *two* respects — the new data and
the runner's core count — and there is no way to attribute the result to
either. The comparison would be uninterpretable, and it would look fine.

### LightGBM

Measured thread-deterministic: bit-identical predicted probabilities and an
identical tree count at 1 and 4 threads (`maxdiff = 0.000e+00`). It does not
need the pin.

It is pinned anyway, for symmetry. One knob in one place beats a comment
explaining which of two libraries needed it, and the cost is nil. If
LightGBM's determinism ever regresses, the pin is already there.

### If XGBoost fixes this

`tests/test_reproducibility.py::test_xgboost_thread_count_changes_the_model`
asserts that thread count *does* change the model. It fails if a future
XGBoost makes `hist` deterministic — deliberately, so the constraint gets
revisited rather than carried forever.

## pandas: `>=2.2,<3`

The cap is not a preference. `mlflow>=3` requires `pandas<3`, so
`pip install -r requirements.txt` silently downgrades a pandas 3.x
environment — as it did here, 3.0.5 → 2.3.3. Declaring `>=2.2` while the
resolver installs 2.3.3 would describe an environment nobody has.

**pandas was the first suspect for the tree-count divergence, and was
wrong.** Ruled out by inspection rather than argument:

- `FeatureEncoders.extension_columns` is empty — no nullable `Int64` or
  `boolean` columns exist, which was the only path by which pandas' dtype
  inference could reach the numbers.
- The parquet schema is explicit and version-independent: 399 `float32`,
  31 `string`, 4 `int64`. Arrow types do not change between pandas versions.
- The feature matrix is 430 `float32` + 1 `int64` under both lines, with no
  pandas extension dtypes surviving `build_features`.
- The 31 string columns arrive as `object` under pandas 2 and `str` under
  pandas 3, but both route to the same categorical path, and level ordering
  is `sorted(levels, key=str)` — deterministic either way.

The code supports both lines. `build_features` selects categoricals on
`not is_numeric_dtype` rather than `is_object_dtype` precisely because the
latter matches nothing under pandas 3. Lift the cap when MLflow supports
pandas 3.

## What every run records

`environment_params()` logs the following to MLflow on every run, so a future
divergence is attributable from the run record alone rather than
reconstructed months later:

```
env.python            env.cpu_count           env.pandas
env.platform          env.n_jobs_requested    env.numpy
env.xgboost           env.n_jobs_effective    env.sklearn
env.lightgbm                                  env.pyarrow
```

`n_jobs_effective` resolves `-1` to the actual core count. `-1` alone in a
record is useless: it says "all cores" without saying how many, which is
exactly the number needed to explain a divergence.

## What is already deterministic

- **Seeds.** `random_state=42` on both models.
- **The temporal carve.** Cut on the `TransactionDT` *value*, not row
  position, so it is identical regardless of row order in the parquet.
- **Encoder level ordering.** `sorted(levels, key=str)` — arbitrary but
  stable across runs and platforms.
- **Data content.** DVC records an md5 per tracked directory in
  `data/splits.dvc`, so a changed input is visible in a git diff rather than
  inferred from a moved metric.

## Running a reproducibility check

```bash
# fit and report, write nothing -- artifacts and registry stay consistent
python -m src.training.train --no-save --no-mlflow

# vary only the thread count
python -m src.training.train --n-jobs 1 --no-save --no-mlflow
```

`--no-save` exists because overwriting `models/` while the registry
describes a different model is precisely how the two silently disagree.
