"""
src/data/upload_to_mongo.py
============================
Merges all raw batch JSON files from data/raw/ into a single unified
temporary JSONL file, then bulk-upserts every record into the MongoDB
`raw_batches` collection as a unified dataset (not per-batch shards).

Usage (from project root):
    python -m src.data.upload_to_mongo
    python -m src.data.upload_to_mongo --raw-dir data/raw --dry-run
    python -m src.data.upload_to_mongo --overwrite-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.utils.logger import setup_logger
from src.utils.helpers import get_project_root

load_dotenv()

# ── Silence the root logger and all noisy third-party loggers ────────────────
# This prevents pymongo/urllib3 debug output (including raw document content)
# from leaking onto the terminal regardless of how the root logger is configured.
logging.root.setLevel(logging.WARNING)
for _noisy in ("pymongo", "pymongo.command", "pymongo.serverMonitor",
               "pymongo.topology", "motor", "urllib3", "bson"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = setup_logger("pipeline.data.upload_to_mongo")
log.propagate = False   # never bubble up to root logger


# ──────────────────────────────────────────────────────────────────
# Step 1: Merge all batch files → unified temp JSONL
# ──────────────────────────────────────────────────────────────────

def merge_batches_to_temp(raw_dir: Path) -> tuple[Path, int]:
    """
    Read every batch_*.json file in raw_dir, flatten all records
    into a single JSONL temp file.

    Returns:
        (temp_file_path, total_record_count)
    """
    batch_files = sorted(raw_dir.glob("batch_*.json"))
    if not batch_files:
        log.error("No batch_*.json files found in: %s", raw_dir)
        sys.exit(1)

    log.info("Found %d batch files to merge.", len(batch_files))

    # NamedTemporaryFile with delete=False so we control cleanup
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        prefix="ugc_unified_raw_",
        delete=False,
        encoding="utf-8",
    )
    tmp_path = Path(tmp.name)

    total = 0
    global_idx = 0  # unified sequential index across all batches

    try:
        with tmp:
            for batch_file in batch_files:
                batch_num = _extract_batch_num(batch_file)
                try:
                    records: list[dict] = json.loads(
                        batch_file.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    log.warning("Skipping %s — cannot read: %s", batch_file.name, exc)
                    continue

                if not isinstance(records, list):
                    log.warning("Skipping %s — expected a JSON array.", batch_file.name)
                    continue

                for local_idx, record in enumerate(records):
                    # Attach provenance metadata so we can trace back later.
                    # NOTE: batch_num + record_index MUST be present — the
                    # raw_batches collection has a unique compound index on
                    # (batch_num, record_index). Without them every document
                    # collides on (null, null) and only 1 record is ever stored.
                    enriched = {
                        "global_index":  global_idx,
                        "batch_num":     batch_num,   # satisfies batch_record_unique index
                        "record_index":  local_idx,   # satisfies batch_record_unique index
                        "source_batch":  batch_num,
                        "batch_index":   local_idx,
                        "user":          record.get("user", ""),
                        "assistant":     record.get("assistant", ""),
                        "raw_record":    record,
                        "etl_status":    "pending",
                    }
                    tmp.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    global_idx += 1
                    total += 1

                # no per-batch log — final summary is printed after the loop
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    log.info(
        "Merge complete: %d total records written to temp file → %s",
        total, tmp_path,
    )
    return tmp_path, total


def _extract_batch_num(path: Path) -> int:
    """Extract integer batch number from filenames like batch_007.json."""
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


# ──────────────────────────────────────────────────────────────────
# Step 2: Bulk-upsert unified records → MongoDB
# ──────────────────────────────────────────────────────────────────

def upload_unified_to_mongo(
    tmp_path: Path,
    total: int,
    overwrite_existing: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Stream records from the temp JSONL file and upsert them into MongoDB.

    Filter key: `global_index`  (unique across the unified dataset)

    Args:
        tmp_path:           Path to the temporary JSONL file.
        total:              Expected total number of records.
        overwrite_existing: If True, replace existing docs; else skip.
        dry_run:            Parse records but do not write to MongoDB.

    Returns:
        dict with keys: inserted, updated, skipped, failed
    """
    from pymongo import ReplaceOne
    from src.db.mongo_client import raw_col, _ensure_indexes, get_db

    if not dry_run:
        # Ensure the unified_index is also indexed
        db = get_db()
        db[raw_col().name].create_index(
            [("global_index", 1)],
            unique=True,
            name="global_index_unique",
        )
        _ensure_indexes(db)

    col = raw_col()

    stats = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
    batch_ops: list[ReplaceOne] = []
    BULK_SIZE = 500

    def _flush(ops: list[ReplaceOne]) -> None:
        """Execute a bulk write and update stats."""
        if not ops or dry_run:
            return
        try:
            result = col.bulk_write(ops, ordered=False)
            stats["inserted"] += result.upserted_count
            stats["updated"]  += result.modified_count
        except Exception as exc:
            log.error("Bulk write error: %s", exc)
            stats["failed"] += len(ops)

    log.info(
        "%s uploading %d unified records to MongoDB collection '%s' …",
        "[DRY-RUN]" if dry_run else "Now",
        total,
        col.name,
    )

    with tmp_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            try:
                doc: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Line %d — JSON error: %s", lineno, exc)
                stats["failed"] += 1
                continue

            filter_q = {"global_index": doc["global_index"]}

            # When overwrite_existing=False use update_one (not replace) so existing
            # docs are left untouched — no pre-check query needed, MongoDB handles it.
            if overwrite_existing or dry_run:
                batch_ops.append(ReplaceOne(filter_q, doc, upsert=True))
            else:
                from pymongo import UpdateOne
                # $setOnInsert only writes if the document does NOT exist yet
                batch_ops.append(
                    UpdateOne(filter_q, {"$setOnInsert": doc}, upsert=True)
                )

            if len(batch_ops) >= BULK_SIZE:
                _flush(batch_ops)
                batch_ops.clear()

    # Flush remainder
    _flush(batch_ops)

    log.info(
        "Upload complete — inserted: %d | updated: %d | skipped: %d | failed: %d",
        stats["inserted"], stats["updated"], stats["skipped"], stats["failed"],
    )
    return stats


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────

def run_upload(
    raw_dir: Path | None = None,
    overwrite_existing: bool = False,
    dry_run: bool = False,
    keep_temp: bool = False,
) -> bool:
    """
    Full pipeline: merge batches → temp file → upload to MongoDB.

    Returns True on success.
    """
    root = get_project_root()
    raw_dir = raw_dir or (root / "data" / "raw")

    if not raw_dir.exists():
        log.error("Raw data directory does not exist: %s", raw_dir)
        return False

    tmp_path: Path | None = None
    try:
        # ── Stage 1: Merge ─────────────────────────────────────────
        log.info("=" * 60)
        log.info("Stage 1/2 — Merging batch files into unified temp file")
        log.info("=" * 60)
        tmp_path, total = merge_batches_to_temp(raw_dir)

        if total == 0:
            log.error("No records were merged. Aborting upload.")
            return False

        # ── Stage 2: Upload ────────────────────────────────────────
        log.info("=" * 60)
        log.info("Stage 2/2 — Uploading unified dataset to MongoDB")
        log.info("=" * 60)
        stats = upload_unified_to_mongo(
            tmp_path=tmp_path,
            total=total,
            overwrite_existing=overwrite_existing,
            dry_run=dry_run,
        )

        success = stats["failed"] == 0
        return success

    finally:
        if tmp_path and tmp_path.exists() and not keep_temp:
            tmp_path.unlink()
            log.debug("Temporary file removed: %s", tmp_path)
        elif tmp_path and keep_temp:
            log.info("Temporary file kept at: %s", tmp_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge raw batch JSON files and upload unified dataset to MongoDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Path to the raw batches directory (default: data/raw).",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace existing MongoDB documents instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and merge records but do NOT write to MongoDB.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete the intermediate temp JSONL file after upload.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = run_upload(
        raw_dir=args.raw_dir,
        overwrite_existing=args.overwrite_existing,
        dry_run=args.dry_run,
        keep_temp=args.keep_temp,
    )
    sys.exit(0 if ok else 1)
