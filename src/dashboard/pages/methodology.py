"""How every number on this dashboard is computed, and why that way.

**This page explains calculations; it never performs them.** Every worked
example is a hardcoded real value from a run that happened, not a live
computation. That is enforced by a test: no module under ``src/dashboard``
may import ``psi_numeric``, ``compute_psi_report``,
``average_precision_score`` or ``paired_bootstrap``.

The reason is the one that governs the whole dashboard. A page that
recomputed PSI would eventually disagree with the monitor that fired the
alert -- a different window, a different bin edge, a different epsilon -- and
a reader would be looking at a number nobody can reconcile with the decision
actually taken. An explanation that can drift from the thing it explains is
worse than no explanation.
"""

from __future__ import annotations

import streamlit as st

st.title("Methodology")
st.caption(
    "Every worked example is a real value from a real run, hardcoded. "
    "This page explains the calculations; it does not perform them."
)

# --------------------------------------------------------------------------

st.header("PSI — Population Stability Index")

st.markdown(
    r"""
PSI answers one deliberately narrow question: **has the shape of a feature's
distribution moved between a reference window and a current one?**

$$\mathrm{PSI} = \sum_{i} (c_i - r_i)\,\ln\!\frac{c_i}{r_i}$$

where $r_i$ and $c_i$ are the share of rows in bucket $i$ for the reference
and current windows.

It needs **no labels**, which is the whole reason it is the retraining
trigger. Fraud ground truth arrives weeks later via chargebacks, so accuracy
cannot be measured in real time. PSI can.
"""
)

st.subheader("Why the log ratio rather than a plain difference")

st.markdown(
    "Both factors share a sign, so every term is non-negative and drift in "
    "opposite directions across buckets can never cancel out. The log ratio "
    "also makes PSI sensitive to *relative* change, which is what catches a "
    "rare segment disappearing:"
)
st.table(
    {
        "bucket": ["a common bucket", "a rare bucket"],
        "reference": ["35%", "5%"],
        "current": ["30%", "0.1%"],
        "raw difference": ["-5 pts", "-4.9 pts"],
        "PSI contribution": ["0.0077", "0.1916"],
    }
)
st.markdown(
    "Both moved about five percentage points. The second is a **50x collapse** "
    "of a whole population segment. A raw difference cannot tell them apart; "
    "the log ratio can."
)

st.subheader("Worked example — `M9`, from the real monitor run")

st.markdown(
    "Numeric features are bucketed on **quantiles of the reference**, with the "
    "outer edges widened to plus/minus infinity so values beyond the training "
    "range still land somewhere instead of falling out as missing. "
    "Categorical features get one bucket per level. **Missing is always its "
    "own bucket** — a feature that stops arriving has drifted, and that must "
    "show up.\n\n"
    "`M9` is categorical, and its recorded figures are:"
)
st.code(
    "  M9   PSI (all rows)         0.5490\n"
    "       PSI (populated rows)   0.0010\n"
    "       NA rate  reference     74.6%\n"
    "       NA rate  current       38.8%",
    language=None,
)
st.markdown(
    "Almost the entire 0.5490 comes from **one bucket** — the missing bucket — "
    "whose share fell from 0.746 to 0.388. The populated values barely moved, "
    "which is what the 0.0010 says.\n\n"
    "Bin edges are computed once from the reference and **frozen with the "
    "model**. Recomputing them on live data would make every window uniform "
    "across its own quantiles, so PSI would read near zero forever and the "
    "monitor would never fire."
)

st.subheader("The bands")
st.table(
    {
        "PSI": ["< 0.10", "0.10 - 0.20", "> 0.20"],
        "reading": ["stable", "watch", "retrain"],
        "in this project": ["275-278 features", "4-7 features", "27 features"],
    }
)
st.info(
    "These thresholds come from 1990s credit-scorecard practice. They are a "
    "reasonable convention, not a derived optimum — worth saying plainly "
    "rather than presenting 0.20 as though it fell out of the mathematics."
)

st.subheader("Overall PSI is the max, not the mean")

st.markdown(
    "There is no canonical blended PSI. This project reports the worst single "
    "feature, because averaging destroys the signal it exists to catch.\n\n"
    "At the real proportions — 431 features of which roughly 278 are stable "
    "and 27 major — the arithmetic is decisive:"
)
left, right = st.columns(2)
left.metric("mean across all features", "~0.05", "reads calm", delta_color="off")
right.metric("max (id_31)", "1.78", "fires the retrain", delta_color="off")
st.markdown(
    "A blended monitor would have reported **stable** on the window where "
    "`id_31` sat at 1.78. `tests/test_monitor.py` builds a fixture at those "
    "real proportions and asserts the mean falls below 0.20 while the max "
    "clears it."
)

st.divider()

# --------------------------------------------------------------------------

st.header("The decomposition — two drift types, two remedies")

st.markdown(
    "A single PSI number cannot distinguish *the values changed* from *the "
    "feature started arriving more often*. Those need opposite responses, so "
    "the monitor computes both:\n\n"
    "| | what it scores |\n"
    "|---|---|\n"
    "| `psi_all` | every row, with missing as its own bucket |\n"
    "| `psi_non_null` | only rows where the feature is populated |\n\n"
    "Bin edges are identical between the two — they derive from non-null "
    "reference values either way — so the difference isolates the missing "
    "bucket's contribution."
)

st.subheader("Worked example — `M9` against `id_31`")
st.code(
    "  feature   psi_all   psi_non_null   NA reference -> current   classified as\n"
    "  M9         0.5490        0.0010          74.6% -> 38.8%      missingness-driven\n"
    "  id_31      2.2655        8.4030          71.8% -> 79.6%      genuine value drift",
    language=None,
)
st.markdown(
    "`M9` breaches 0.20 purely because **more data started arriving**. Its "
    "populated values are unchanged to three decimal places. Retraining on "
    "that would bake a transient upstream join into the model, so it raises "
    "`INVESTIGATE PIPELINE (no retrain)`.\n\n"
    "`id_31` moved the other way — coverage *degraded* while the populated "
    "values changed enormously. That is what retraining fixes.\n\n"
    "Note `psi_non_null` **exceeds** `psi_all` for `id_31`. Restricting to "
    "populated rows reveals *more* drift, not less, so a "
    "`psi_all - psi_non_null` framing is misleading whenever that happens — "
    "the classification uses the residual directly rather than the difference."
)

st.subheader("Where the cut falls, and why it barely matters")
st.markdown(
    "A feature is **genuine value drift** when `psi_non_null >= 0.05`. The two "
    "clusters separate far more sharply than that threshold implies:"
)
st.table(
    {
        "cluster": [
            "missingness-driven (16 features)",
            "genuine value drift (11 features)",
        ],
        "populated-row PSI range": ["0.000 - 0.015", "0.07 - 8.40"],
    }
)
st.markdown(
    "Any cut between **0.01 and 0.07** produces the identical partition, and a "
    "parametrised test asserts exactly that. `D11` at 0.0735 is the only "
    "borderline case; 0.05 places it on the genuine side, which is the "
    "conservative error — mistaking real drift for a pipeline artifact means "
    "not retraining when you should."
)

st.divider()

# --------------------------------------------------------------------------

st.header("PR-AUC, and why it leads")

st.markdown(
    "The positive rate on the evaluation window is **3.9%**. At that imbalance "
    "the familiar metrics mislead in specific, predictable ways."
)
st.table(
    {
        "metric": ["accuracy", "ROC-AUC", "PR-AUC"],
        "value here": [
            "~96% for a useless model",
            "0.8790 - 0.8941",
            "0.4942 - 0.5270",
        ],
        "why it does or does not work": [
            "Predicting 'never fraud' scores 96%. Not a measure of anything.",
            "Its false-positive axis is normalised by ~19,400 negatives, so "
            "thousands of false alarms barely move it.",
            "Lives on the positives. Its no-skill floor is the base rate.",
        ],
    }
)

st.subheader("The 0.0393 no-skill floor")
st.markdown(
    "PR-AUC's floor is **not 0.5**. A model assigning random scores achieves "
    "PR-AUC approximately equal to the positive base rate. On the baseline "
    "evaluation split that is **0.0393**, which is why every PR-AUC here is "
    "quoted with its lift beside it:"
)
st.code(
    "  baseline v1   PR-AUC 0.5248   floor 0.0393   lift 13.3x",
    language=None,
)
st.markdown(
    "0.5248 stated alone means nothing. Against a 0.0393 floor it is roughly "
    "thirteen times better than guessing. The floor claim is verified rather "
    "than asserted: a test scores random noise on 200,000 rows at a 3.4% rate "
    "and requires the result within plus/minus 0.004 of the base rate."
)

st.divider()

# --------------------------------------------------------------------------

st.header("The decision threshold")

st.markdown(
    "**0.5 is meaningless for this model.** `scale_pos_weight` is the "
    "negative/positive ratio — **29** on the training split — which multiplies "
    "the gradient contribution of positive rows so the model stops treating "
    "'call everything legitimate' as a good local minimum.\n\n"
    "The cost is calibration. Predicted probabilities come out inflated and "
    "can no longer be read as *0.8 means 80% of these are fraud*. That is fine "
    "for ranking, which is what PR-AUC measures and what a review queue "
    "consumes — but it makes 0.5 an artifact of the class ratio rather than a "
    "decision anyone made."
)

st.subheader("How the operating point is chosen")
st.markdown(
    "The full precision-recall curve is swept and the threshold maximising F1 "
    "is taken:\n\n"
    "```\n"
    "F1 = 2 * precision * recall / (precision + recall)\n"
    "```\n\n"
    "`precision_recall_curve` returns one more precision/recall point than it "
    "does thresholds — the trailing point is the degenerate recall=0, "
    "precision=1 corner, which has no threshold and is dropped."
)
st.code(
    "  v1   threshold 0.8018   precision 0.6305   recall 0.4343   2.71% flagged\n"
    "  v4   threshold 0.8032   precision 0.5905   recall 0.4300   2.21% flagged",
    language=None,
)
st.warning(
    "**Each model is judged at its own threshold.** v1's swept point is 0.8018 "
    "and v4's is 0.8032. Scoring a challenger at the champion's cut would "
    "measure the threshold rather than the model. The threshold is read from "
    "the version's own MLflow run — resolving it by *role* meant it followed "
    "the alias, so promoting v4 once silently swapped its 0.8032 for the "
    "baseline's 0.8018."
)

st.divider()

# --------------------------------------------------------------------------

st.header("The paired bootstrap, and the one-SE rule")

st.markdown(
    "The shadow test asks whether a challenger genuinely beats the champion or "
    "merely happens to be ahead on this sample. That needs a standard error on "
    "the **difference**."
)

st.subheader("Why paired")
st.markdown(
    "Each resample draws row indices **once** and scores *both* models on those "
    "same rows:\n\n"
    "```\n"
    "for b in 1..1000:\n"
    "    idx = resample row indices with replacement\n"
    "    d_b = PR-AUC(challenger[idx]) - PR-AUC(champion[idx])\n"
    "SE = standard deviation of d\n"
    "```\n\n"
    "The two models see identical inputs and make **correlated errors** — "
    "measured probability correlation between the XGBoost and LightGBM "
    "baselines was **0.9778**. Treating their errors as independent would "
    "inflate the standard error and refuse promotions that are real. Resamples "
    "that lose one class entirely are skipped, since average precision is "
    "undefined there and a small window at a 3% base rate can produce them."
)

st.subheader("Why one standard error")
st.markdown(
    "Phase 2 measured three XGBoost runs of the *same configuration on the "
    "same data*, differing only in thread count:"
)
st.code(
    "  n_jobs=1   239 trees      n_jobs=2   300 trees      n_jobs=4   140 trees\n"
    "  spread across the three runs:  0.0023 PR-AUC\n"
    "  bootstrap standard error:      0.0080",
    language=None,
)
st.markdown(
    "A spread of **a quarter of one standard error** across models provably "
    "identical in quality. That is the calibration for what noise means here: "
    "a challenger ahead by less than one SE has not been shown to be better, "
    "only to be ahead on this sample.\n\n"
    "A test feeds those exact numbers — 0.0023 against 0.0080 — into the "
    "promotion logic and requires the verdict "
    "`no promotion, difference within noise`."
)
st.info(
    "One SE is roughly **84% one-sided confidence**, not 95%. That is a "
    "deliberately modest bar and the multiplier is configurable — stated here "
    "rather than left implied by the phrase 'statistically significant'."
)

st.subheader("The promotion that passed")
st.code(
    "  champion v1   PR-AUC 0.4942        challenger v4   PR-AUC 0.5270\n"
    "  delta                 +0.0327\n"
    "  bootstrap SE           0.0068   (paired, 1000 resamples)\n"
    "  95% CI on the delta   [+0.0204, +0.0467]\n"
    "  margin                 4.81 SE  (need 1.00)     -> PROMOTE",
    language=None,
)

st.divider()

# --------------------------------------------------------------------------

st.header("The leakage guard")

st.markdown(
    "Before any margin is considered, the comparison checks that the judged "
    "rows sit **outside** the challenger's training window. That window is "
    "read from `retrain.window_start_dt` / `retrain.window_end_dt` on the "
    "challenger's own MLflow run — the record that already ties a model to its "
    "data, rather than a recomputation that could disagree.\n\n"
    "```\n"
    "overlap = |judged intersect trained| / |judged|\n"
    "```\n\n"
    "Above **1%** the comparison is refused outright and no verdict is issued. "
    "Not zero, because a boundary row or a rounding difference in the recorded "
    "window should not void an otherwise clean test."
)

st.subheader("Worked example — v3, the promotion that was rolled back")
st.code(
    "  judged   TransactionDT   10,569,737 .. 11,106,772\n"
    "  trained  TransactionDT    5,443,151 .. 15,811,131\n"
    "  overlap  100.0%\n\n"
    "  reported PR-AUC delta  +0.2492  at  20.13 SE     -> NO VERDICT",
    language=None,
)
st.error(
    "**A margin rule cannot catch this.** The larger the overlap, the more "
    "decisive the apparent win — v3 was recalling rows it had been fitted on "
    "while the champion predicted them. It scored twenty standard errors and "
    "was promoted before the guard existed. Leakage is therefore checked "
    "*first*, ahead of margin and row count."
)

st.subheader("A minimum that is not about leakage")
st.markdown(
    "Below **5,000 jointly-scored rows** or **100 positives** the comparison "
    "refuses regardless of margin. PR-AUC on a few thousand rows at a 3% base "
    "rate rests on a hundred-odd positives, and its standard error swamps any "
    "plausible difference. A comparison on too few rows is not a decision."
)

st.divider()

# --------------------------------------------------------------------------

st.header("Three times a comparison measured the wrong thing")

st.markdown(
    "Each produced a plausible number that answered the wrong question. Two "
    "were caught by a guard; the third was caught by a person reading a log, "
    "and is the one worth dwelling on."
)

st.subheader("1 — v3: the judged rows were inside the training window")
st.markdown(
    "**Measured:** how well v3 recalled rows it had been fitted on, against "
    "how well v1 predicted them.\n\n"
    "**Should have measured:** how both perform on unseen traffic.\n\n"
    "**Signal:** +0.2492 at 20.13 SE — a margin far too large for a retrain on "
    "drifted data, which is itself the tell.\n\n"
    "**Now prevented by:** the leakage guard, checking the judged range "
    "against the challenger's recorded training window before anything else. A "
    "test uses v3's actual ranges and asserts a 20 SE margin still cannot "
    "promote."
)

st.subheader("2 — the stale topic: a fresh consumer group re-read old rows")
st.markdown(
    "**Measured:** v4 on the *previous* replay batch at "
    "`10,569,737..10,768,871`, inside its own training window.\n\n"
    "**Should have measured:** v4 on the newly-produced judged window.\n\n"
    "**Cause:** `auto.offset.reset=earliest` means a new consumer group starts "
    "at the beginning of a topic, not at the newly-published tail. Replaying "
    "into a reused topic silently feeds old rows to a new group.\n\n"
    "**Now prevented by:** the same leakage guard, which refused at 100% "
    "overlap — and by a dedicated topic per experiment, so stale messages are "
    "unreachable."
)

st.subheader("3 — the encoder mismatch: the guard would not have caught it")
st.markdown(
    "**Measured:** v4 scored through the *baseline's* encoders.\n\n"
    "**Should have measured:** v4 through its own.\n\n"
    "**Cause:** the retrain logged no artifacts, so the model had no encoders "
    "on its run. Serving fell back to whatever `models/encoders.pkl` held — "
    "the baseline's — and labelled it `(FALLBACK)` in a startup log.\n\n"
    "**Why no guard fired:** the leakage check passed cleanly at 0.0% overlap. "
    "The rows were correct. The *vocabulary* was wrong, and every prediction "
    "still computed. Five of thirty-one categorical columns assigned different "
    "codes to the same level — 970 `DeviceInfo` levels, 49 in `id_31` — and 57 "
    "of 400 predictions differed by up to 0.2502.\n\n"
    "**Caught by:** a person reading `(FALLBACK)` in the startup output."
)
st.error(
    "**The correction moved the number but not the conclusion.** Re-measured "
    "with each model on its own encoders and its own threshold, v4 scored "
    "+0.0327 at 4.81 SE over 20,000 rows — a smaller delta on a larger sample, "
    "so a *larger* margin. v4 was promoted again, honestly."
)
st.markdown(
    "**Now prevented by:** three changes. The retrain logs its encoders under "
    "the exact filename serving resolves. Serving **raises** rather than "
    "falling back when a registered model's encoders cannot be resolved — an "
    "escape hatch exists for development, off by default and self-declaring in "
    "the source it reports. And the threshold is resolved from the version's "
    "own run rather than from a file keyed by role."
)

st.subheader("The pattern")
st.info(
    "All three shared a shape: **the pipeline resolved something by "
    "convenience rather than by identity** — rows by recency, messages by "
    "topic offset, artifacts by file path. Each produced output that looked "
    "entirely normal. The guards that exist now check identity: does this "
    "range belong to this model, does this encoder belong to this run."
)
