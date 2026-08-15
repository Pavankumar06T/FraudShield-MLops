"""Scores the ``raw_transactions`` feed and writes an audit trail.

**Scoring is not implemented here.** Every transaction goes through
``src.serving.predictor.predict_one`` -- the same function the HTTP endpoint
calls, on the same bundle loaded from the same registry alias. There is no
streaming scoring path to drift away from the serving one, because there is
no second implementation.

That matters more than it looks. Phase 7 compares a shadow model against the
live one using rows scored here; if the stream scored transactions even
slightly differently from HTTP -- a threshold read from elsewhere, an
unbounded booster, encoders from another run -- the comparison would measure
the difference between two code paths rather than between two models, and
would look perfectly reasonable while doing it.

**The label never reaches the model.** ``isFraud`` arrives in the message
and is written to the audit table so decisions can be judged, but the
encoder reads exactly the fitted feature names, and ``build_features``
dropped the label before those names were recorded. It is unreachable by
construction rather than by convention -- ``tests/test_streaming.py``
asserts that flipping the label in a transaction changes nothing about the
prediction.

Offsets are committed only after a row is scored *and* written, so a crash
replays the transaction rather than losing it. A duplicated audit row is
harmless; a missing one is a hole in the record.

    python -m src.streaming.consumer
    python -m src.streaming.consumer --limit 5000 --from-beginning
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from src.drift import store
from src.serving.predictor import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    ModelBundle,
    load_bundle,
    predict_one,
    try_load_bundle,
)
from src.streaming.config import CONSUMER_GROUP, RAW_TOPIC, consumer_config

#: The label. Present in the message for scoring-quality monitoring, never
#: an input -- see the module docstring.
LABEL_FIELD: str = "isFraud"

#: Rows buffered before a write. SQLite commits are the slow part of this
#: loop; batching keeps the consumer ahead of a fast producer.
BATCH_SIZE: int = 200

PROGRESS_EVERY: int = 2_000

#: Positions in the audit row tuple. Named because they are read back by
#: index in the consume loop, and an off-by-two there reports every row as a
#: champion/challenger disagreement without anything looking wrong.
DECISION_IX: int = 4
CHALLENGER_VERSION_IX: int = 10
CHALLENGER_DECISION_IX: int = 12
CHALLENGER_THRESHOLD_IX: int = 13
TRUE_LABEL_IX: int = 14

_STOPPING = False


def _handle_signal(*_) -> None:
    global _STOPPING
    _STOPPING = True
    print("\n  draining and committing before exit ...")


SCHEMA = """
CREATE TABLE IF NOT EXISTS stream_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scored_at           TEXT    NOT NULL,
    transaction_id      TEXT,
    transaction_dt      REAL,
    fraud_probability   REAL    NOT NULL,
    decision            TEXT    NOT NULL,
    threshold           REAL    NOT NULL,
    model_version       TEXT    NOT NULL,
    n_trees             INTEGER NOT NULL,
    unseen_categories   TEXT,
    top_factors         TEXT,

    -- Shadow columns. The challenger scores every row in parallel and its
    -- output is recorded, never acted on: `decision` above is the champion's
    -- and nothing here can reach it.
    challenger_version      TEXT,
    challenger_probability  REAL,
    challenger_decision     TEXT,
    challenger_threshold    REAL,

    -- Monitoring only. Never an input; see the module docstring.
    true_label          INTEGER,
    latency_ms          REAL
);

CREATE INDEX IF NOT EXISTS idx_stream_scored   ON stream_predictions (scored_at);
CREATE INDEX IF NOT EXISTS idx_stream_decision ON stream_predictions (decision);
CREATE INDEX IF NOT EXISTS idx_stream_txn      ON stream_predictions (transaction_id);
"""


def ensure_schema(path: Path = store.DEFAULT_DB_PATH) -> None:
    with store.connect(path) as connection:
        connection.executescript(SCHEMA)


def write_batch(rows: list[tuple], path: Path = store.DEFAULT_DB_PATH) -> int:
    if not rows:
        return 0
    with store.connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO stream_predictions (
                scored_at, transaction_id, transaction_dt, fraud_probability,
                decision, threshold, model_version, n_trees,
                unseen_categories, top_factors,
                challenger_version, challenger_probability,
                challenger_decision, challenger_threshold,
                true_label, latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
    return len(rows)


def score_message(
    bundle: ModelBundle,
    message: dict,
    challenger: ModelBundle | None = None,
) -> tuple[tuple, list[str]]:
    """Score one transaction and shape it for the audit table.

    The label is pulled out for storage *before* scoring, purely for
    readability -- pulling it out is not what protects the model. The
    encoder only ever reads ``bundle.feature_names``, which does not contain
    it, so passing the whole message in would be equally safe.

    When a challenger is supplied it scores the same transaction through the
    same ``predict_one``, with its own bundle and its own swept threshold.
    Its output is recorded and nothing else: the returned decision is
    computed before the challenger runs and is never revisited. A challenger
    that returned 1.0 for every row would change nothing but the shadow
    columns.
    """
    label = message.get(LABEL_FIELD)
    started = time.perf_counter()
    prediction = predict_one(bundle, message)
    latency = (time.perf_counter() - started) * 1000

    # Computed here, from the champion alone, and used verbatim below.
    decision = prediction.decision

    shadow_version = shadow_probability = shadow_decision = shadow_threshold = None
    if challenger is not None:
        try:
            shadow = predict_one(challenger, message, top_factors=0)
            shadow_version = challenger.model_version
            shadow_probability = round(shadow.fraud_probability, 6)
            shadow_decision = shadow.decision
            shadow_threshold = round(challenger.threshold, 6)
        except Exception:
            # A failing challenger must not cost a real decision. The row is
            # still recorded, with the shadow columns left null.
            pass

    row = (
        store.utc_now(),
        str(message.get("TransactionID")) if message.get("TransactionID") is not None else None,
        float(message["TransactionDT"]) if message.get("TransactionDT") is not None else None,
        round(prediction.fraud_probability, 6),
        decision,
        round(prediction.threshold, 6),
        prediction.model_version,
        prediction.n_trees,
        json.dumps(prediction.unseen_categories),
        json.dumps(prediction.top_factors),
        shadow_version,
        shadow_probability,
        shadow_decision,
        shadow_threshold,
        int(label) if label is not None else None,
        round(latency, 3),
    )
    return row, prediction.unseen_categories


def consume(
    limit: int | None = None,
    topic: str = RAW_TOPIC,
    group: str = CONSUMER_GROUP,
    from_beginning: bool = False,
    db_path: Path = store.DEFAULT_DB_PATH,
    idle_timeout: float = 30.0,
) -> dict:
    """Consume, score, and record until the topic goes quiet or ``limit``."""
    from confluent_kafka import Consumer, KafkaError

    ensure_schema(db_path)
    bundle = load_bundle(alias=CHAMPION_ALIAS)
    challenger = try_load_bundle(CHALLENGER_ALIAS)
    if challenger is not None:
        print(
            f"  shadow scoring against challenger v{challenger.model_version} "
            f"at threshold {challenger.threshold:.4f} -- its output is recorded, "
            "never acted on"
        )

    extra = {"auto.offset.reset": "earliest"} if from_beginning else {}
    consumer = Consumer(consumer_config(group, extra))
    consumer.subscribe([topic])
    print(f"consuming {topic!r} as group {group!r}")

    pending: list[tuple] = []
    unseen_counter: Counter = Counter()
    stats = {"scored": 0, "blocked": 0, "allowed": 0, "unseen_rows": 0,
             "errors": 0, "shadow_scored": 0, "disagreements": 0}
    started = time.perf_counter()
    last_message = time.perf_counter()

    try:
        while not _STOPPING:
            if limit and stats["scored"] >= limit:
                break

            message = consumer.poll(1.0)
            if message is None:
                if time.perf_counter() - last_message > idle_timeout:
                    print(f"  no messages for {idle_timeout:.0f}s; stopping")
                    break
                continue
            if message.error():
                if message.error().code() != KafkaError._PARTITION_EOF:
                    stats["errors"] += 1
                continue

            last_message = time.perf_counter()
            try:
                payload = json.loads(message.value())
            except (ValueError, TypeError):
                stats["errors"] += 1
                continue

            row, unseen = score_message(bundle, payload, challenger)
            pending.append(row)
            stats["scored"] += 1
            stats["blocked" if row[4] == "BLOCK" else "allowed"] += 1
            # Index 12 is challenger_decision, 4 is the champion's. Comparing
            # against index 10 (challenger_version) would report every row as
            # a disagreement, which is how this was first written.
            if row[CHALLENGER_VERSION_IX] is not None:
                stats["shadow_scored"] += 1
                if row[CHALLENGER_DECISION_IX] != row[DECISION_IX]:
                    stats["disagreements"] += 1
            if unseen:
                stats["unseen_rows"] += 1
                for feature in unseen:
                    unseen_counter[(feature, str(payload.get(feature)))] += 1

            if len(pending) >= BATCH_SIZE:
                write_batch(pending, db_path)
                store.record_unseen(dict(unseen_counter), bundle.model_version,
                                    source="streaming", path=db_path)
                pending.clear()
                unseen_counter.clear()
                # Commit only once the rows are durable: a crash between
                # scoring and writing replays the batch rather than losing it.
                consumer.commit(asynchronous=False)

            if stats["scored"] % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {stats['scored']:>8,} scored  "
                    f"{stats['scored'] / elapsed:6.0f}/s  "
                    f"{stats['blocked']:,} blocked  "
                    f"{stats['unseen_rows']:,} with unseen categories"
                )
    finally:
        write_batch(pending, db_path)
        if unseen_counter:
            store.record_unseen(dict(unseen_counter), bundle.model_version,
                                source="streaming", path=db_path)
        try:
            consumer.commit(asynchronous=False)
        except Exception:
            pass
        consumer.close()

    elapsed = time.perf_counter() - started
    stats["elapsed_s"] = round(elapsed, 2)
    stats["rate"] = round(stats["scored"] / max(elapsed, 1e-9), 1)
    return stats


def format_stats(stats: dict) -> str:
    scored = max(stats["scored"], 1)
    return (
        f"\n  scored     {stats['scored']:,} in {stats['elapsed_s']}s "
        f"({stats['rate']}/s)\n"
        f"  blocked    {stats['blocked']:,} ({100 * stats['blocked'] / scored:.2f}%)\n"
        f"  allowed    {stats['allowed']:,}\n"
        f"  unseen     {stats['unseen_rows']:,} rows carried an unseen category\n"
        f"  errors     {stats['errors']:,}\n"
        f"  shadow     {stats['shadow_scored']:,} also scored by the challenger, "
        f"{stats['disagreements']:,} disagreements ("
        f"{100 * stats['disagreements'] / max(stats['shadow_scored'], 1):.2f}%)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the transaction stream.")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--topic", default=RAW_TOPIC)
    parser.add_argument("--group", default=CONSUMER_GROUP)
    parser.add_argument("--from-beginning", action="store_true")
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        stats = consume(
            limit=args.limit, topic=args.topic, group=args.group,
            from_beginning=args.from_beginning, idle_timeout=args.idle_timeout,
        )
    except Exception as exc:
        print(f"consumer failed: {type(exc).__name__}: {exc}")
        return 1
    print(format_stats(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
