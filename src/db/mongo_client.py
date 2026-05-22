"""
src/db/mongo_client.py
======================
Singleton MongoDB client for the UGC / LLM Fine-Tuning pipeline.

Collections used:
  - raw_batches       : raw generated conversation records (one doc = one conversation)
  - processed_records : PII-cleaned, Chat-ML formatted conversations ready for training

Connection URL is read from the MONGODB_URL env variable (set in .env).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv()

log = logging.getLogger("pipeline.db.mongo_client")

# ──────────────────────────────────────────────────────────
# Collection names (single source of truth)
# ──────────────────────────────────────────────────────────
COL_RAW        = "raw_batches"
COL_PROCESSED  = "processed_records"

# ──────────────────────────────────────────────────────────
# Singleton state
# ──────────────────────────────────────────────────────────
_client:   MongoClient | None = None
_database: Database    | None = None


def get_client() -> MongoClient:
    """Return the shared MongoClient, creating it on first call."""
    global _client
    if _client is None:
        url = os.getenv("MONGODB_URL")
        if not url:
            raise EnvironmentError(
                "MONGODB_URL is not set. Add it to your .env file."
            )
        log.info("Connecting to MongoDB …")
        _client = MongoClient(url, serverSelectionTimeoutMS=10_000)
        # Eagerly verify connectivity
        try:
            _client.admin.command("ping")
            log.info("MongoDB connection established.")
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            _client = None
            raise ConnectionFailure(f"Cannot reach MongoDB at {url}: {exc}") from exc
    return _client


def get_db() -> Database:
    """Return the project database (MONGODB_DB_NAME env var)."""
    global _database
    if _database is None:
        db_name = os.getenv("MONGODB_DB_NAME", "ugc_pipeline")
        _database = get_client()[db_name]
        _ensure_indexes(_database)
        log.debug("Using database: %s", db_name)
    return _database


def _ensure_indexes(db: Database) -> None:
    """Create indexes if they don't already exist (idempotent)."""
    db[COL_RAW].create_index(
        [("batch_num", ASCENDING), ("record_index", ASCENDING)],
        unique=True,
        name="batch_record_unique",
    )
    db[COL_PROCESSED].create_index(
        [("source_id", ASCENDING)],
        unique=True,
        sparse=True,
        name="source_id_unique",
    )
    db[COL_PROCESSED].create_index(
        [("etl_status", ASCENDING)],
        name="etl_status_idx",
    )
    log.debug("MongoDB indexes ensured.")


# ──────────────────────────────────────────────────────────
# Collection helpers
# ──────────────────────────────────────────────────────────

def raw_col() -> Collection:
    return get_db()[COL_RAW]


def processed_col() -> Collection:
    return get_db()[COL_PROCESSED]


# ──────────────────────────────────────────────────────────
# CRUD helpers
# ──────────────────────────────────────────────────────────

def upsert_raw_record(
    batch_num: int,
    record_index: int,
    record: dict[str, Any],
) -> str:
    """
    Insert or replace a single raw conversation record.
    Returns the inserted/matched document _id as string.
    """
    col = raw_col()
    filter_q = {"batch_num": batch_num, "record_index": record_index}
    doc = {
        "batch_num":    batch_num,
        "record_index": record_index,
        "user":         record.get("user", ""),
        "assistant":    record.get("assistant", ""),
        "raw_record":   record,          # full original document
        "etl_status":   "pending",       # pending | in_progress | done | failed
    }
    result = col.replace_one(filter_q, doc, upsert=True)
    oid = result.upserted_id or col.find_one(filter_q, {"_id": 1})["_id"]
    return str(oid)


def insert_processed_record(
    source_id: str,
    conversations: list[dict],
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Insert a processed (anonymised + formatted) record.
    Idempotent — skips insert if source_id already exists.
    Returns the document _id as string.
    """
    col = processed_col()
    existing = col.find_one({"source_id": source_id}, {"_id": 1})
    if existing:
        return str(existing["_id"])

    doc: dict[str, Any] = {
        "source_id":     source_id,
        "conversations": conversations,
        "etl_status":    "done",
    }
    if metadata:
        doc["metadata"] = metadata

    result = col.insert_one(doc)
    return str(result.inserted_id)


def mark_raw_status(doc_id: str, status: str) -> None:
    """Update etl_status on a raw_batches document."""
    from bson import ObjectId
    raw_col().update_one({"_id": ObjectId(doc_id)}, {"$set": {"etl_status": status}})


def close() -> None:
    """Gracefully close the singleton connection."""
    global _client, _database
    if _client:
        _client.close()
        log.info("MongoDB connection closed.")
    _client = None
    _database = None
