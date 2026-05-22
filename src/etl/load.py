"""
src/etl/load.py
===============
ETL — LOAD phase.

Consumes transformed conversation records from the processed Kafka topic
and persists them into the MongoDB `processed_records` collection.

Additionally marks the corresponding raw_batches document as "done".
"""

from __future__ import annotations

import logging
import os
from typing import Any

from kafka import KafkaConsumer
from kafka.errors import KafkaError
from dotenv import load_dotenv

from src.db.mongo_client import (
    processed_col,
    raw_col,
    insert_processed_record,
)
from src.etl.kafka_client import (
    _processed_topic,
    _consumer_group,
    _bootstrap,
    publish_dead_letter,
    build_producer,
)

load_dotenv()
log = logging.getLogger("pipeline.etl.load")


def _build_load_consumer() -> KafkaConsumer:
    """KafkaConsumer dedicated to the processed topic for the load phase."""
    import json

    consumer = KafkaConsumer(
        _processed_topic(),
        bootstrap_servers=_bootstrap(),
        group_id=f"{_consumer_group()}.loader",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        key_deserializer=lambda b: b.decode("utf-8") if b else None,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_records=50,
        session_timeout_ms=30_000,
        heartbeat_interval_ms=10_000,
    )
    log.info(
        "Load-consumer subscribed to '%s' (group=%s.loader)",
        _processed_topic(), _consumer_group()
    )
    return consumer


def run_load(
    *,
    poll_timeout_ms: int = 3_000,
    max_empty_polls: int = 5,
) -> dict[str, int]:
    """
    Load phase: consume transformed records from Kafka → persist to MongoDB.

    Returns stats: {"loaded": int, "skipped": int, "failed": int}
    """
    consumer = _build_load_consumer()
    dlq_producer = build_producer()   # for DLQ routing on load failure
    stats = {"loaded": 0, "skipped": 0, "failed": 0}

    log.info("=" * 60)
    log.info("ETL — LOAD Phase Starting")
    log.info("  Source topic : %s", _processed_topic())
    log.info("=" * 60)

    empty_rounds = 0
    try:
        while True:
            records_map = consumer.poll(timeout_ms=poll_timeout_ms)
            if not records_map:
                empty_rounds += 1
                if empty_rounds >= max_empty_polls:
                    log.info(
                        "No new messages after %d polls — stopping load loop.",
                        max_empty_polls
                    )
                    break
                continue
            empty_rounds = 0

            for tp, messages in records_map.items():
                for msg in messages:
                    record: dict[str, Any] = msg.value
                    source_id = record.get("source_mongo_id", "")
                    conversations = record.get("conversations", [])
                    metadata = record.get("metadata", {})

                    try:
                        # ── Persist to MongoDB processed collection ──────────
                        inserted_id = insert_processed_record(
                            source_id=source_id,
                            conversations=conversations,
                            metadata={
                                **metadata,
                                "batch_num":    record.get("batch_num", -1),
                                "record_index": record.get("record_index", -1),
                            },
                        )

                        # ── Mark corresponding raw doc as "done" ─────────────
                        if source_id:
                            from bson import ObjectId
                            try:
                                raw_col().update_one(
                                    {"_id": ObjectId(source_id)},
                                    {"$set": {"etl_status": "done"}},
                                )
                            except Exception as e:
                                log.warning(
                                    "Could not mark raw doc %s as done: %s",
                                    source_id, e
                                )

                        stats["loaded"] += 1
                        log.debug(
                            "Loaded source_id=%s → processed._id=%s",
                            source_id, inserted_id
                        )

                    except Exception as exc:
                        log.error(
                            "Load failed for source_id=%s: %s", source_id, exc
                        )
                        publish_dead_letter(dlq_producer, record, reason=str(exc))
                        stats["failed"] += 1

                    finally:
                        consumer.commit()

    finally:
        consumer.close()
        dlq_producer.flush()
        dlq_producer.close()

    log.info("=" * 60)
    log.info("ETL — LOAD Phase Complete")
    log.info("  Loaded  : %d", stats["loaded"])
    log.info("  Skipped : %d", stats["skipped"])
    log.info("  Failed  : %d", stats["failed"])
    log.info("=" * 60)
    return stats
