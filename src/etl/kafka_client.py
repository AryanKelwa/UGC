"""
src/etl/kafka_client.py
=======================
Thin wrappers around kafka-python producers and consumers.
All Kafka connection parameters are loaded from environment variables.

Environment variables (set in .env):
  KAFKA_BOOTSTRAP_SERVERS   e.g. localhost:9092
  KAFKA_RAW_TOPIC           e.g. ugc.raw.batches
  KAFKA_PROCESSED_TOPIC     e.g. ugc.processed.conversations
  KAFKA_DLQ_TOPIC           e.g. ugc.dead_letter
  KAFKA_CONSUMER_GROUP      e.g. ugc_etl_group
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterator

from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

load_dotenv()

log = logging.getLogger("pipeline.etl.kafka_client")

# ──────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────

def _bootstrap() -> list[str]:
    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    return [s.strip() for s in servers.split(",")]


def _raw_topic() -> str:
    return os.getenv("KAFKA_RAW_TOPIC", "ugc.raw.batches")


def _processed_topic() -> str:
    return os.getenv("KAFKA_PROCESSED_TOPIC", "ugc.processed.conversations")


def _dlq_topic() -> str:
    return os.getenv("KAFKA_DLQ_TOPIC", "ugc.dead_letter")


def _consumer_group() -> str:
    return os.getenv("KAFKA_CONSUMER_GROUP", "ugc_etl_group")


# ──────────────────────────────────────────────────────────
# Producer
# ──────────────────────────────────────────────────────────

def build_producer() -> KafkaProducer:
    """Build and return a JSON-serialising KafkaProducer."""
    producer = KafkaProducer(
        bootstrap_servers=_bootstrap(),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",            # wait for all in-sync replicas
        retries=5,
        max_block_ms=30_000,
    )
    log.info("KafkaProducer created → %s", _bootstrap())
    return producer


def publish_raw_batch(
    producer: KafkaProducer,
    batch_num: int,
    records: list[dict[str, Any]],
) -> int:
    """
    Publish every record from a raw batch to the raw Kafka topic.
    Returns the number of messages successfully enqueued.
    """
    topic = _raw_topic()
    sent = 0
    for idx, record in enumerate(records):
        message = {
            "batch_num":    batch_num,
            "record_index": idx,
            "user":         record.get("user", ""),
            "assistant":    record.get("assistant", ""),
        }
        key = f"{batch_num}:{idx}"
        try:
            future = producer.send(topic, key=key, value=message)
            future.get(timeout=10)   # block to confirm delivery
            sent += 1
        except KafkaError as exc:
            log.error("Failed to publish batch=%d record=%d: %s", batch_num, idx, exc)
    log.info("Published %d/%d records from batch %03d → topic '%s'",
             sent, len(records), batch_num, topic)
    return sent


def publish_dead_letter(
    producer: KafkaProducer,
    payload: dict[str, Any],
    reason: str,
) -> None:
    """Forward a failed record to the dead-letter queue."""
    topic = _dlq_topic()
    payload["_dlq_reason"] = reason
    try:
        producer.send(topic, value=payload).get(timeout=10)
        log.warning("Record sent to DLQ '%s': %s", topic, reason)
    except KafkaError as exc:
        log.error("DLQ publish failed: %s", exc)


# ──────────────────────────────────────────────────────────
# Consumer
# ──────────────────────────────────────────────────────────

def build_raw_consumer(
    *,
    auto_offset_reset: str = "earliest",
    enable_auto_commit: bool = False,
) -> KafkaConsumer:
    """
    Build a KafkaConsumer subscribed to the raw-batches topic.
    Manual commit is used by default for at-least-once ETL semantics.
    """
    consumer = KafkaConsumer(
        _raw_topic(),
        bootstrap_servers=_bootstrap(),
        group_id=_consumer_group(),
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        key_deserializer=lambda b: b.decode("utf-8") if b else None,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=enable_auto_commit,
        max_poll_records=50,
        session_timeout_ms=30_000,
        heartbeat_interval_ms=10_000,
    )
    log.info(
        "KafkaConsumer subscribed to '%s' (group=%s)",
        _raw_topic(), _consumer_group()
    )
    return consumer


def build_processed_producer() -> KafkaProducer:
    """Produce transformed records to the processed topic."""
    return build_producer()   # same config, different topic at call site


def iter_raw_messages(
    consumer: KafkaConsumer,
    *,
    poll_timeout_ms: int = 3_000,
    max_empty_polls: int = 5,
) -> Iterator[dict[str, Any]]:
    """
    Yield decoded message values from the raw consumer, committing offsets
    after each successful yield.  Stops after `max_empty_polls` consecutive
    empty poll rounds (useful for one-shot batch runs).
    """
    empty_rounds = 0
    while True:
        records_map = consumer.poll(timeout_ms=poll_timeout_ms)
        if not records_map:
            empty_rounds += 1
            if empty_rounds >= max_empty_polls:
                log.info("No new messages after %d polls — stopping consumer loop.", max_empty_polls)
                break
            continue
        empty_rounds = 0
        for tp, messages in records_map.items():
            for msg in messages:
                yield msg.value
                consumer.commit()   # manual commit after each record
