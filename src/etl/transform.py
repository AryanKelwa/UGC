"""
src/etl/transform.py
====================
ETL — TRANSFORM phase.

Consumes raw conversation records from the Kafka raw topic, applies:
  1. PII anonymization  (reusing src.data.anonymization logic)
  2. Chat-ML formatting (reusing src.data.formatting logic)
  3. Quality validation (reusing src.data.quality_checks logic)

Publishes transformed records to the processed Kafka topic.
Failed records are routed to the dead-letter queue (DLQ).
"""

from __future__ import annotations

import logging
from typing import Any

from src.data.anonymization import clean_pii_text
from src.data.formatting import format_to_conversations, SYSTEM_PROMPT
from src.etl.kafka_client import (
    build_raw_consumer,
    build_processed_producer,
    iter_raw_messages,
    publish_dead_letter,
    _processed_topic,
)

log = logging.getLogger("pipeline.etl.transform")


# ──────────────────────────────────────────────────────────
# Transformation logic
# ──────────────────────────────────────────────────────────

def _apply_pii_clean(record: dict[str, Any]) -> dict[str, Any]:
    """Strip PII from user and assistant fields."""
    return {
        **record,
        "user":      clean_pii_text(record.get("user", "")),
        "assistant": clean_pii_text(record.get("assistant", "")),
    }


def _apply_format(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a cleaned record into Chat-ML conversation format."""
    conversations = format_to_conversations(
        record.get("user", ""),
        record.get("assistant", ""),
    )["conversations"]

    return {
        "source_mongo_id": record.get("mongo_id", ""),
        "batch_num":        record.get("batch_num", -1),
        "record_index":     record.get("record_index", -1),
        "conversations":    conversations,
        "metadata": {
            "system_prompt": SYSTEM_PROMPT,
        },
    }


def _validate(record: dict[str, Any]) -> tuple[bool, str]:
    """
    Basic quality gate:
    - user text must not be empty
    - assistant JSON must be a non-empty string
    - conversations list must have exactly 3 entries (system/user/assistant)
    """
    convs = record.get("conversations", [])
    if len(convs) != 3:
        return False, f"conversations length={len(convs)}, expected 3"

    user_content = next(
        (c["content"] for c in convs if c.get("role") == "user"), ""
    )
    asst_content = next(
        (c["content"] for c in convs if c.get("role") == "assistant"), ""
    )

    if not user_content.strip():
        return False, "Empty user content after PII clean"
    if not asst_content.strip():
        return False, "Empty assistant content after PII clean"

    return True, ""


# ──────────────────────────────────────────────────────────
# Transform runner
# ──────────────────────────────────────────────────────────

def run_transform(
    *,
    poll_timeout_ms: int = 3_000,
    max_empty_polls: int = 5,
) -> dict[str, int]:
    """
    Transform phase: consume raw Kafka messages → clean → format → validate → publish.

    Returns a stats dict: {"transformed": int, "failed": int, "dlq": int}
    """
    consumer = build_raw_consumer(
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    producer = build_processed_producer()
    topic_out = _processed_topic()

    stats = {"transformed": 0, "failed": 0, "dlq": 0}

    log.info("=" * 60)
    log.info("ETL — TRANSFORM Phase Starting")
    log.info("  Output topic : %s", topic_out)
    log.info("=" * 60)

    try:
        for raw_msg in iter_raw_messages(
            consumer,
            poll_timeout_ms=poll_timeout_ms,
            max_empty_polls=max_empty_polls,
        ):
            mongo_id = raw_msg.get("mongo_id", "?")
            try:
                # Step 1 — PII clean
                cleaned = _apply_pii_clean(raw_msg)

                # Step 2 — Format to Chat-ML
                transformed = _apply_format(cleaned)

                # Step 3 — Quality validation
                ok, reason = _validate(transformed)
                if not ok:
                    raise ValueError(f"Quality check failed: {reason}")

                # Step 4 — Publish to processed topic
                key = f"{transformed['batch_num']}:{transformed['record_index']}"
                future = producer.send(topic_out, key=key, value=transformed)
                future.get(timeout=10)

                stats["transformed"] += 1
                log.debug(
                    "Transformed mongo_id=%s batch=%d record=%d → '%s'",
                    mongo_id, transformed["batch_num"],
                    transformed["record_index"], topic_out,
                )

            except Exception as exc:
                log.error("Transform failed for mongo_id=%s: %s", mongo_id, exc)
                publish_dead_letter(producer, raw_msg, reason=str(exc))
                stats["failed"] += 1
                stats["dlq"] += 1

    finally:
        consumer.close()
        producer.flush()
        producer.close()

    log.info("=" * 60)
    log.info("ETL — TRANSFORM Phase Complete")
    log.info("  Transformed : %d", stats["transformed"])
    log.info("  Failed/DLQ  : %d", stats["failed"])
    log.info("=" * 60)
    return stats
