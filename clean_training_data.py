"""
clean_training_data.py
======================
Post-processing pipeline that reads all batch_*.json files from
OUTPUT_DIR (training-data/), applies PII anonymization to the `user`
field of every record, and writes cleaned files to CLEAN_OUTPUT_DIR
(training-data-clean/).

Concurrency model
-----------------
- **Batch-level parallelism**: each batch file is processed in its own
  worker thread (outer ThreadPoolExecutor, up to PII_WORKERS workers).
- **Record-level parallelism**: within each batch, each record's `user`
  field is cleaned concurrently (inner ThreadPoolExecutor).
- spaCy's nlp() call is thread-safe for inference; the model is loaded
  once at import time in pii_cleaner.py.

Run:
    python clean_training_data.py

Output:
    training-data-clean/batch_001.json
    training-data-clean/batch_002.json
    …
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── resolve project root ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.config import OUTPUT_DIR, CLEAN_OUTPUT_DIR, PII_WORKERS  # noqa: E402
from pii_cleaner import clean_pii_text                                # noqa: E402

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pii_clean")


# ─────────────────────────────────────────────────────────────────────────────
# Per-batch worker
# ─────────────────────────────────────────────────────────────────────────────

def _clean_record(record: dict) -> dict:
    """Return a copy of *record* with PII anonymized in user & assistant fields."""
    cleaned = dict(record)           # shallow copy
    if "user" in cleaned and isinstance(cleaned["user"], str):
        cleaned["user"] = clean_pii_text(cleaned["user"])
    if "assistant" in cleaned and isinstance(cleaned["assistant"], str):
        cleaned["assistant"] = clean_pii_text(cleaned["assistant"])
    return cleaned


def process_batch(src_file: Path, dst_file: Path) -> int:
    """
    Load *src_file*, clean every record in parallel, and save to *dst_file*.

    Returns
    -------
    int
        Number of records saved.  0 if the batch was skipped or failed.
    """
    if dst_file.exists():
        existing = json.loads(dst_file.read_text(encoding="utf-8"))
        log.info("Batch %s — skipping (cleaned copy exists, %d records)",
                 src_file.name, len(existing))
        return len(existing)

    try:
        records: list[dict] = json.loads(src_file.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Batch %s — failed to load: %s", src_file.name, exc)
        return 0

    if not records:
        log.warning("Batch %s — no records, skipping.", src_file.name)
        return 0

    log.info("Batch %s — cleaning %d records…", src_file.name, len(records))
    t0 = time.time()

    # ── Record-level concurrency within this batch ────────────────────────────
    cleaned_records: list[dict] = [None] * len(records)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(PII_WORKERS, len(records))) as ex:
        future_to_idx = {
            ex.submit(_clean_record, rec): idx
            for idx, rec in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                cleaned_records[idx] = future.result()
            except Exception as exc:
                log.error(
                    "Batch %s record %d — cleaning error: %s",
                    src_file.name, idx, exc,
                )
                cleaned_records[idx] = records[idx]  # keep original on error

    elapsed = time.time() - t0
    dst_file.write_text(
        json.dumps(cleaned_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Batch %s — %d records cleaned → %s  (%.1fs)",
        src_file.name, len(cleaned_records), dst_file.name, elapsed,
    )
    return len(cleaned_records)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all batch files in source directory
    batch_files = sorted(OUTPUT_DIR.glob("batch_*.json"))

    if not batch_files:
        log.warning("No batch_*.json files found in %s — nothing to do.", OUTPUT_DIR)
        sys.exit(0)

    log.info("=" * 60)
    log.info("PII Cleaning Pipeline")
    log.info("  Source      : %s", OUTPUT_DIR.resolve())
    log.info("  Destination : %s", CLEAN_OUTPUT_DIR.resolve())
    log.info("  Batches     : %d", len(batch_files))
    log.info("  Workers     : %d (batch-level parallel)", PII_WORKERS)
    log.info("=" * 60)

    total_records  = 0
    failed_batches: list[str] = []

    # ── Batch-level concurrency ───────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=PII_WORKERS) as executor:
        future_to_src: dict = {}
        for src_file in batch_files:
            dst_file = CLEAN_OUTPUT_DIR / src_file.name
            future = executor.submit(process_batch, src_file, dst_file)
            future_to_src[future] = src_file

        for future in as_completed(future_to_src):
            src = future_to_src[future]
            try:
                n = future.result()
                total_records += n
                if n == 0:
                    failed_batches.append(src.name)
            except Exception as exc:
                log.error("Batch %s — unhandled exception: %s", src.name, exc)
                failed_batches.append(src.name)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Batches processed : %d", len(batch_files))
    log.info("  Records cleaned   : %d", total_records)
    log.info("  Failed batches    : %s", sorted(failed_batches) or "none")
    log.info("=" * 60)

    if failed_batches:
        sys.exit(1)


if __name__ == "__main__":
    main()
