"""Invariants for shadow scoring and the promotion decision.

Two carry the weight.

**The challenger cannot influence the champion's decision.** It scores every
row in parallel and its output is recorded; if it could reach the returned
decision, the shadow deployment would be a live deployment nobody agreed to.
Asserted behaviourally: a challenger rigged to return 1.0 for everything
must change nothing but the shadow columns.

**A margin inside noise is not a promotion.** Phase 2 measured three XGBoost
runs spanning 0.0023 PR-AUC against a bootstrap SE of 0.0080 and called them
indistinguishable. The same standard here: ahead-on-this-sample is not
better, and promoting on it is a coin flip with extra steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drift import store
from src.serving.compare import (
    DEFAULT_MARGIN_SES,
    MAX_LEAKAGE_OVERLAP,
    format_comparison,
    MIN_POSITIVES,
    MIN_ROWS,
    Comparison,
    ModelScores,
    paired_bootstrap,
    threshold_scores,
)
from src.serving.predictor import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    ModelBundle,
    Prediction,
)
from src.streaming.consumer import (
    CHALLENGER_DECISION_IX,
    CHALLENGER_THRESHOLD_IX,
    CHALLENGER_VERSION_IX,
    DECISION_IX,
    TRUE_LABEL_IX,
    score_message,
)


# --------------------------------------------------------------------------
# The challenger must not be able to decide anything
# --------------------------------------------------------------------------


class FakeEncoder:
    feature_names = ("TransactionAmt",)

    def encode(self, record):
        from src.serving.encoding import EncodedRow

        return EncodedRow(np.array([1.0], dtype=np.float32), (), ())

    def decode(self, name, value):
        return str(value)


def bundle(version: str, threshold: float) -> ModelBundle:
    return ModelBundle(
        booster=None, encoder=FakeEncoder(), threshold=threshold,
        threshold_source="test", model_version=version, model_stage="Staging",
        n_trees=10, run_id=None, feature_names=("TransactionAmt",),
    )


@pytest.fixture
def rigged(monkeypatch):
    """predict_one stubbed: champion returns 0.10, challenger returns 1.0."""
    import src.streaming.consumer as consumer_module

    def fake_predict(b, message, top_factors=6):
        probability = 1.0 if b.model_version == "challenger" else 0.10
        return Prediction(
            fraud_probability=probability,
            decision="BLOCK" if probability >= b.threshold else "ALLOW",
            threshold=b.threshold, model_version=b.model_version, n_trees=10,
            top_factors=[], unseen_categories=[],
        )

    monkeypatch.setattr(consumer_module, "predict_one", fake_predict)
    return fake_predict


def test_challenger_cannot_change_the_returned_decision(rigged):
    """A challenger screaming BLOCK on every row must not block anything."""
    champion = bundle("champion", 0.80)
    challenger = bundle("challenger", 0.50)
    message = {"TransactionAmt": 1.0, "TransactionID": 1, "isFraud": 0}

    alone, _ = score_message(champion, message)
    shadowed, _ = score_message(champion, message, challenger)

    # index 4 is the decision, 3 the probability, 5 the threshold
    assert alone[4] == "ALLOW"
    assert shadowed[4] == alone[4]
    assert shadowed[3] == alone[3]
    assert shadowed[5] == alone[5]
    assert shadowed[6] == alone[6] == "champion"

    # the challenger's opinion is recorded, and only recorded
    assert shadowed[10] == "challenger"
    assert shadowed[11] == 1.0
    assert shadowed[12] == "BLOCK"


def test_champion_row_is_byte_identical_with_and_without_a_challenger(rigged):
    """Every champion-owned field, not just the decision."""
    champion = bundle("champion", 0.80)
    message = {"TransactionAmt": 1.0, "TransactionID": 7, "isFraud": 1}

    alone, _ = score_message(champion, message)
    shadowed, _ = score_message(champion, message, bundle("challenger", 0.01))

    champion_fields = slice(1, 10)  # id..top_factors, excluding scored_at
    assert alone[champion_fields] == shadowed[champion_fields]
    assert alone[TRUE_LABEL_IX] == shadowed[TRUE_LABEL_IX] == 1


def test_a_failing_challenger_does_not_cost_a_decision(monkeypatch):
    """Shadow scoring is best-effort; a real decision is not."""
    import src.streaming.consumer as consumer_module

    def fake_predict(b, message, top_factors=6):
        if b.model_version == "challenger":
            raise RuntimeError("challenger exploded")
        return Prediction(0.9, "BLOCK", 0.8, "champion", 10, [], [])

    monkeypatch.setattr(consumer_module, "predict_one", fake_predict)
    row, _ = score_message(bundle("champion", 0.8), {"TransactionAmt": 1.0},
                           bundle("challenger", 0.5))
    assert row[4] == "BLOCK"
    assert row[10] is None and row[11] is None


def test_each_model_is_judged_at_its_own_threshold(rigged):
    champion = bundle("champion", 0.80)
    challenger = bundle("challenger", 0.50)
    row, _ = score_message(champion, {"TransactionAmt": 1.0}, challenger)
    assert row[5] == pytest.approx(0.80)
    assert row[CHALLENGER_THRESHOLD_IX] == pytest.approx(0.50)


def test_champion_and_challenger_are_distinct_aliases():
    assert CHAMPION_ALIAS != CHALLENGER_ALIAS


# --------------------------------------------------------------------------
# Promotion requires a margin beyond noise
# --------------------------------------------------------------------------


def make_comparison(delta: float, se: float, rows: int = 50_000,
                    positives: int = 1_700, required: float = DEFAULT_MARGIN_SES):
    scores = lambda pr: ModelScores(
        version="1", threshold=0.8, pr_auc=pr, roc_auc=0.9, f1=0.5,
        precision=0.6, recall=0.44, true_positives=700, false_positives=460,
        false_negatives=1000, true_negatives=47840,
    )
    return Comparison(
        rows=rows, positives=positives, fraud_rate=positives / rows,
        champion=scores(0.53), challenger=scores(0.53 + delta),
        pr_auc_delta=delta, bootstrap_std_error=se,
        margin_in_ses=abs(delta) / se if se else 0.0, required_ses=required,
        sufficient_data=rows >= MIN_ROWS and positives >= MIN_POSITIVES,
        reason="",
    )


def test_the_phase_2_case_is_not_a_promotion():
    """0.0023 against an SE of 0.0080 -- the exact numbers Phase 2 judged
    indistinguishable. The A/B must reach the same conclusion."""
    result = make_comparison(delta=0.0023, se=0.0080)
    assert result.margin_in_ses < 1.0
    assert result.promote is False
    assert result.verdict == "no promotion, difference within noise"


def test_a_margin_beyond_one_se_promotes():
    result = make_comparison(delta=0.0250, se=0.0080)
    assert result.margin_in_ses > 3
    assert result.promote is True
    assert result.verdict == "PROMOTE"


def test_a_challenger_that_is_behind_never_promotes():
    """Even a large margin -- being decisively worse is still worse."""
    result = make_comparison(delta=-0.0400, se=0.0080)
    assert result.margin_in_ses > 1.0
    assert result.promote is False
    assert "not ahead" in result.verdict


@pytest.mark.parametrize("delta,se,expected", [
    (0.0079, 0.0080, False),   # just inside
    (0.0081, 0.0080, True),    # just outside
    (0.0000, 0.0080, False),
])
def test_the_boundary_is_exactly_one_standard_error(delta, se, expected):
    assert make_comparison(delta=delta, se=se).promote is expected


def test_a_stricter_margin_can_be_required():
    """1 SE is roughly 84% one-sided confidence; the bar is configurable and
    not buried in the promotion logic."""
    result = make_comparison(delta=0.0120, se=0.0080, required=2.0)
    assert result.margin_in_ses == pytest.approx(1.5)
    assert result.promote is False


# --------------------------------------------------------------------------
# Too few rows is not a decision
# --------------------------------------------------------------------------


def test_too_few_rows_refuses_even_a_huge_margin():
    result = make_comparison(delta=0.20, se=0.001, rows=500, positives=17)
    assert result.sufficient_data is False
    assert result.promote is False
    assert "insufficient data" in result.verdict


def test_too_few_positives_refuses():
    result = make_comparison(delta=0.20, se=0.001, rows=100_000, positives=40)
    assert result.sufficient_data is False
    assert result.promote is False


def test_minimums_are_stated_not_implied():
    assert MIN_ROWS >= 5_000
    assert MIN_POSITIVES >= 100


# --------------------------------------------------------------------------
# The bootstrap
# --------------------------------------------------------------------------


def make_labelled(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.034).astype(int)
    champion = np.clip(rng.normal(np.where(y == 1, 0.70, 0.30), 0.18), 0.001, 0.999)
    return y, champion


def test_bootstrap_of_a_model_against_itself_is_zero():
    """Paired: identical inputs must give a zero delta and no spread."""
    y, proba = make_labelled()
    se, (low, high) = paired_bootstrap(y, proba, proba, resamples=200)
    assert se == pytest.approx(0.0, abs=1e-12)
    assert low == pytest.approx(0.0, abs=1e-12) and high == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_se_shrinks_as_rows_grow():
    """More data, less noise -- the property the minimum-rows rule rests on."""
    small_y, small_c = make_labelled(2_000, seed=1)
    large_y, large_c = make_labelled(20_000, seed=1)
    rng = np.random.default_rng(3)
    small_x = np.clip(small_c + rng.normal(0, 0.05, len(small_c)), 0.001, 0.999)
    large_x = np.clip(large_c + rng.normal(0, 0.05, len(large_c)), 0.001, 0.999)

    small_se, _ = paired_bootstrap(small_y, small_c, small_x, resamples=200)
    large_se, _ = paired_bootstrap(large_y, large_c, large_x, resamples=200)
    assert large_se < small_se


def test_bootstrap_detects_a_genuinely_better_model():
    y, champion = make_labelled(20_000)
    # a challenger that separates the classes more sharply
    rng = np.random.default_rng(9)
    challenger = np.clip(rng.normal(np.where(y == 1, 0.85, 0.20), 0.15), 0.001, 0.999)

    se, (low, high) = paired_bootstrap(y, champion, challenger, resamples=200)
    from sklearn.metrics import average_precision_score

    delta = average_precision_score(y, challenger) - average_precision_score(y, champion)
    assert delta > 0
    assert delta / se > 1.0
    assert low > 0, "a real improvement should not straddle zero"


def test_bootstrap_survives_a_single_class_resample():
    """At a 3.4% base rate a small window can resample to one class, where
    average precision is undefined."""
    y = np.array([0] * 99 + [1])
    proba = np.linspace(0, 1, 100)
    se, _ = paired_bootstrap(y, proba, proba, resamples=50)
    assert se == se  # not NaN


# --------------------------------------------------------------------------
# Threshold-level metrics
# --------------------------------------------------------------------------


def test_threshold_scores_match_a_hand_counted_matrix():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.9, 0.4, 0.85, 0.95])
    scores = threshold_scores(y, proba, 0.8, "1")

    assert (scores.true_positives, scores.false_positives) == (2, 1)
    assert (scores.false_negatives, scores.true_negatives) == (1, 1)
    assert scores.precision == pytest.approx(2 / 3)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.flagged == 3


def test_false_negatives_are_reported_because_they_are_the_cost():
    y, proba = make_labelled()
    strict = threshold_scores(y, proba, 0.95, "1")
    lenient = threshold_scores(y, proba, 0.40, "1")
    assert strict.false_negatives > lenient.false_negatives
    assert strict.flagged < lenient.flagged


# --------------------------------------------------------------------------
# The promotion record
# --------------------------------------------------------------------------


def test_promotion_is_refused_when_the_verdict_says_no(tmp_path):
    from src.serving.compare import promote

    result = make_comparison(delta=0.0023, se=0.0080)
    with pytest.raises(RuntimeError, match="refusing to promote"):
        promote(result, tmp_path / "drift.db")


# --------------------------------------------------------------------------
# Leakage: the judged rows must be unseen by the challenger
# --------------------------------------------------------------------------


def leaky(delta=0.2492, se=0.0124, overlap=1.0,
          scored=(10_569_737.0, 11_106_772.0),
          trained=(5_443_151.0, 15_811_131.0)):
    result = make_comparison(delta=delta, se=se)
    result.leakage_overlap = overlap
    result.scored_dt_range = scored
    result.trained_dt_range = trained
    return result


def test_v3_actual_ranges_refuse_at_full_overlap():
    """The real numbers from the run that was wrongly promoted.

    v3 trained on TransactionDT 5,443,151..15,811,131 and was judged on
    10,569,737..11,106,772 -- entirely inside it. It scored +0.2492 at 20 SE
    and promoted. The margin rule cannot catch this: a larger overlap makes
    the apparent win more decisive, not less.
    """
    result = leaky()
    assert result.leakage_overlap == pytest.approx(1.0)
    assert result.leaked is True
    assert result.trustworthy is False
    assert result.promote is False, "a 20 SE margin must not promote a leaked comparison"
    assert result.verdict.startswith("NO VERDICT")
    assert "overlap" in result.verdict


def test_the_refusal_names_both_ranges_and_the_percentage():
    text = format_comparison(leaky())
    assert "100.0%" in text
    assert "10,569,737" in text and "11,106,772" in text
    assert "5,443,151" in text and "15,811,131" in text
    assert "No verdict is issued." in text


def test_a_clean_comparison_is_unaffected():
    """The guard must not refuse an honest test."""
    result = leaky(overlap=0.0, trained=(0.0, 9_000_000.0))
    assert result.leaked is False
    assert result.trustworthy is True
    assert result.promote is True


@pytest.mark.parametrize("overlap,expected_leak", [
    (0.000, False),
    (0.005, False),   # a boundary row should not void an honest test
    (0.011, True),
    (0.500, True),
    (1.000, True),
])
def test_the_tolerance_is_small_but_not_zero(overlap, expected_leak):
    assert leaky(overlap=overlap).leaked is expected_leak
    assert MAX_LEAKAGE_OVERLAP == pytest.approx(0.01)


def test_overlap_fraction_is_measured_against_the_judged_range():
    from src.serving.compare import overlap_fraction

    # fully inside
    assert overlap_fraction((10.0, 20.0), (0.0, 100.0)) == pytest.approx(1.0)
    # fully outside
    assert overlap_fraction((10.0, 20.0), (30.0, 40.0)) == pytest.approx(0.0)
    # half covered
    assert overlap_fraction((10.0, 20.0), (15.0, 40.0)) == pytest.approx(0.5)
    # training window strictly before the judged rows -- the clean case
    assert overlap_fraction((10_569_737.0, 11_106_772.0), (86_400.0, 10_569_554.0)) == 0.0


def test_leakage_outranks_every_other_verdict():
    """It is checked first because no other conclusion is meaningful once
    the rows are compromised."""
    result = leaky(delta=-0.5, se=0.001)   # decisively worse, and leaked
    assert result.verdict.startswith("NO VERDICT")


# --------------------------------------------------------------------------
# Artifacts must belong to the model that is serving them
# --------------------------------------------------------------------------


def test_retrain_logs_its_encoders_as_an_artifact():
    """A retrained model fits its own encoders on its own window. Logging
    none meant serving silently fell back to whatever encoders.pkl was on
    disk -- which belonged to a different run, and remapped 970 DeviceInfo
    levels and 49 in id_31."""
    from pathlib import Path

    source = Path("src/training/retrain.py").read_text(encoding="utf-8")
    assert "artifacts=[]," not in source, "the retrain logged no artifacts"
    assert "encoders.save(CHALLENGER_ENCODERS_PATH)" in source
    assert "artifacts=[staged, RETRAIN_REPORT_PATH]" in source


def test_encoders_are_staged_under_the_name_serving_looks_for():
    """Artifacts are logged by filename. Logging them as
    challenger_encoders.pkl would leave the download looking for
    encoders.pkl and falling back exactly as before."""
    from pathlib import Path

    source = Path("src/training/retrain.py").read_text(encoding="utf-8")
    assert "staged = Path(tempfile.mkdtemp()) / ENCODERS_PATH.name" in source


def test_threshold_is_resolved_per_version_not_per_role():
    """Resolving by role made the threshold follow the alias rather than the
    model: promoting v4 swapped its 0.8032 for the baseline's 0.8018, so a
    model measured at one operating point began deciding at another."""
    from pathlib import Path

    source = Path("src/serving/predictor.py").read_text(encoding="utf-8")
    assert "def resolve_threshold(" in source
    assert "resolve_threshold(client, version, alias)" in source
    assert "val.at_best_f1_threshold.threshold" in source


class _Version:
    def __init__(self, run_id="abc12345", version="4"):
        self.run_id, self.version = run_id, version


class _Client:
    def __init__(self, metrics):
        self._metrics = metrics

    def get_run(self, run_id):
        return type("R", (), {"data": type("D", (), {"metrics": self._metrics})})


def test_threshold_comes_from_the_versions_own_run():
    from src.serving.predictor import resolve_threshold

    client = _Client({"val.at_best_f1_threshold.threshold": 0.8032})
    threshold, source = resolve_threshold(client, _Version(), "champion")
    assert threshold == pytest.approx(0.8032)
    assert "abc12345" in source and "v4" in source


def test_threshold_falls_back_loudly_when_the_run_has_none():
    """Absent a run metric the role-keyed file is used, but the source says
    so rather than presenting it as the model's own."""
    from src.serving.predictor import resolve_threshold

    threshold, source = resolve_threshold(_Client({}), _Version(), "champion")
    assert "FALLBACK" in source
