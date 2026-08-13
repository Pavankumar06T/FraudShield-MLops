"""Replays the stream split onto ``raw_transactions``.

**Order is the point.** Rows are published sorted by ``TransactionDT``, so
the consumer meets transactions in the sequence they actually occurred. A
shuffled replay would still exercise the plumbing, but it would destroy the
one property this dataset was chosen for: the browser vocabulary in
``id_31`` and the identity-column coverage change arrive *gradually* across
the stream window, and drift that arrives gradually is what the monitor is
built to catch. Shuffling would smear that into a uniform background and
make every window look identical.

The label rides along in the message and is written to the audit log, but
never enters the feature vector -- see ``src/serving/predictor.py``, where
the encoder reads only the fitted feature names. It is here so the consumer
can record whether a decision was right, which is what Phase 7 needs to
compare a shadow model against the live one.

    python -m src.streaming.producer                 # 100/s, whole stream
    python -m src.streaming.producer --rate 500 --limit 20000
    python -m src.streaming.producer --rate 0        # as fast as possible
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import STREAM_PARQUET
from src.streaming.config import RAW_TOPIC, producer_config

#: Transactions per second. 0 means unthrottled.
DEFAULT_RATE: float = 100.0

#: How often to report progress, in messages.
PROGRESS_EVERY: int = 5_000

_STOPPING = False


def _handle_signal(*_) -> None:
    global _STOPPING
    _STOPPING = True
    print("\n  stopping after the current message ...")


def to_message(row: pd.Series) -> dict:
    """One transaction as JSON-safe primitives.

    NaN is dropped rather than serialised: JSON has no NaN literal, and an
    absent key is exactly how the encoder already represents a missing
    feature. Sending ``null`` would work too, but omitting keeps the payload
    small -- on this dataset most rows are majority-empty.

    Missingness is tested with ``pd.isna``, not ``isinstance(value, float)``.
    399 of the 434 columns are stored as float32, and ``np.float32`` is not
    a subclass of Python's ``float`` -- so an isinstance check silently
    passes float32 NaNs straight through, and ``json.dumps`` then emits a
    bare ``NaN`` token that is invalid JSON and rejected by strict parsers.
    """
    message = {}
    for key, value in row.items():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass  # arrays and unusual types are not missing
        message[key] = value.item() if hasattr(value, "item") else value
    return message


def load_ordered(limit: int | None = None, path: Path = STREAM_PARQUET) -> pd.DataFrame:
    """The stream split in TransactionDT order.

    Sorted explicitly rather than trusting the file's row order. The parquet
    happens to be ordered, but a replay whose temporal faithfulness depended
    on that would break silently the first time it was not.
    """
    frame = pd.read_parquet(path)
    frame = frame.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    return frame.head(limit) if limit else frame


def publish(
    rate: float = DEFAULT_RATE,
    limit: int | None = None,
    topic: str = RAW_TOPIC,
    path: Path = STREAM_PARQUET,
) -> int:
    """Publish the ordered stream, paced at ``rate`` messages per second."""
    from confluent_kafka import Producer

    frame = load_ordered(limit, path)
    span_days = (
        frame["TransactionDT"].max() - frame["TransactionDT"].min()
    ) / 86_400
    print(
        f"replaying {len(frame):,} transactions from {path.name}\n"
        f"  TransactionDT {frame['TransactionDT'].min():,.0f} -> "
        f"{frame['TransactionDT'].max():,.0f}  ({span_days:.1f} days)\n"
        f"  topic {topic!r} at "
        + (f"{rate:g}/s" if rate > 0 else "full speed")
    )

    producer = Producer(producer_config())
    interval = 1.0 / rate if rate > 0 else 0.0
    started = time.perf_counter()
    sent = 0
    failures = 0

    def on_delivery(err, _msg):
        nonlocal failures
        if err is not None:
            failures += 1

    for _, row in frame.iterrows():
        if _STOPPING:
            break
        message = to_message(row)
        key = str(message.get("TransactionID", sent))
        try:
            producer.produce(
                topic,
                key=key.encode(),
                value=json.dumps(message).encode(),
                on_delivery=on_delivery,
            )
        except BufferError:
            # The local queue is full: the broker is slower than the
            # requested rate. Drain and retry rather than dropping.
            producer.flush(5.0)
            producer.produce(
                topic,
                key=key.encode(),
                value=json.dumps(message).encode(),
                on_delivery=on_delivery,
            )
        producer.poll(0)
        sent += 1

        if sent % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {sent:>8,} sent  {sent / elapsed:7.0f}/s  "
                f"{failures} delivery failures"
            )

        if interval:
            # Paced against the wall clock from the start rather than by
            # sleeping a fixed interval each message: a per-message sleep
            # accumulates the send cost and drifts steadily slower.
            target = started + sent * interval
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

    producer.flush(30.0)
    elapsed = time.perf_counter() - started
    print(
        f"\ndone: {sent:,} published in {elapsed:.1f}s "
        f"({sent / max(elapsed, 1e-9):.0f}/s), {failures} delivery failures"
    )
    return sent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the stream split to Kafka.")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help="messages per second; 0 for unthrottled")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--topic", default=RAW_TOPIC)
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        publish(rate=args.rate, limit=args.limit, topic=args.topic)
    except Exception as exc:
        print(f"producer failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
