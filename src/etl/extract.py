"""
src/etl/extract.py
==================
ETL — EXTRACT phase.

Reads every pending raw_batch document from MongoDB and publishes each
conversation record as a Kafka message to the raw-batches topic.

This replaces/augments the file-based approach: data already stored in
MongoDB (via ingestion.py → mongo_client.upsert_raw_record) is streamed
into Kafka for downstream transform/load stages.
"""

from __future__ import annotations

import logging
from typing import Any

from src.db.mongo_client import raw_col
from src.etl.kafka_client import build_producer, publish_dead_letter

log = logging.getLogger("pipeline.etl.extract")


def run_extract(batch_filter: dict[str, Any] | None = None) -> int:
    """
    Extract phase: read raw records from MongoDB → publish to Kafka raw topic.

    Args:
        batch_filter: Optional MongoDB query filter (e.g. {"batch_num": {"$gte": 1}}).
                      Defaults to all documents with etl_status == "pending".

    Returns:
        Number of records successfully published to Kafka.
    """
    if batch_filter is None:
        batch_filter = {"etl_status": "pending"}

    col = raw_col()
    total_docs = col.count_documents(batch_filter)
    log.info("=" * 60)
    log.info("ETL — EXTRACT Phase Starting")
    log.info("  Filter       : %s", batch_filter)
    log.info("  Pending Docs : %d", total_docs)
    log.info("=" * 60)

    if total_docs == 0:
        log.warning("No pending documents found in MongoDB. Nothing to extract.")
        return 0

    producer = build_producer()
    published = 0
    failed = 0

    try:
        cursor = col.find(batch_filter)
        for doc in cursor:
            message: dict[str, Any] = {
                "mongo_id":     str(doc["_id"]),
                "batch_num":    doc.get("batch_num", -1),
                "record_index": doc.get("record_index", -1),
                "user":         doc.get("user", ""),
                "assistant":    doc.get("assistant", ""),
            }

            from src.etl.kafka_client import _raw_topic
            from kafka.errors import KafkaError

            topic = _raw_topic()
            key = f"{message['batch_num']}:{message['record_index']}"

            try:
                from src.etl.kafka_client import build_producer as _bp
                future = producer.send(topic, key=key, value=message)
                future.get(timeout=10)

                # Mark as in_progress so it isn't re-extracted on next run
                col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"etl_status": "in_progress"}},
                )
                published += 1
                log.debug(
                    "Extracted batch=%d record=%d → Kafka",
                    message["batch_num"], message["record_index"]
                )

            except Exception as exc:
                log.error(
                    "Failed to publish batch=%d record=%d: %s",
                    message["batch_num"], message["record_index"], exc
                )
                publish_dead_letter(producer, message, reason=str(exc))
                failed += 1

    finally:
        producer.flush()
        producer.close()

    log.info("=" * 60)
    log.info("ETL — EXTRACT Phase Complete")
    log.info("  Published : %d", published)
    log.info("  Failed    : %d", failed)
    log.info("=" * 60)
    return published
