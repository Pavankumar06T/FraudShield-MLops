"""Shared Kafka/Redpanda settings for the producer and consumer."""

from __future__ import annotations

import os

#: Broker address. Containers use ``redpanda:29092``; the host uses
#: ``localhost:9092``. The broker advertises both, so this is the only knob
#: that has to change between the two.
BOOTSTRAP: str = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")

#: The replayed transaction feed.
RAW_TOPIC: str = os.environ.get("KAFKA_RAW_TOPIC", "raw_transactions")

#: Consumer group. Fixed so a restarted consumer resumes rather than
#: replaying the whole topic.
CONSUMER_GROUP: str = os.environ.get("KAFKA_CONSUMER_GROUP", "fraudshield-scorer")


def producer_config(extra: dict | None = None) -> dict:
    """librdkafka settings for the replay producer.

    ``linger.ms`` is deliberately small. Batching is what makes a Kafka
    producer fast, but this one is pacing a replay at a chosen rate, so
    holding messages to fill a batch would distort the very timing the
    replay exists to reproduce.
    """
    config = {
        "bootstrap.servers": BOOTSTRAP,
        "linger.ms": 5,
        "compression.type": "lz4",
        "acks": "1",
        "queue.buffering.max.messages": 100_000,
    }
    config.update(extra or {})
    return config


def consumer_config(group: str | None = None, extra: dict | None = None) -> dict:
    """librdkafka settings for the scoring consumer.

    ``enable.auto.commit`` is off: offsets are committed after a
    transaction has been scored *and* written, so a crash replays the row
    rather than silently dropping it. At-least-once is the right trade here
    -- a duplicated prediction row is harmless, a missing one is a hole in
    the audit trail.
    """
    config = {
        "bootstrap.servers": BOOTSTRAP,
        "group.id": group or CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "session.timeout.ms": 45_000,
    }
    config.update(extra or {})
    return config
