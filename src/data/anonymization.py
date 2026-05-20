"""
src/data/anonymization.py
=========================
Handles PII anonymization in synthetic dataset batches.
Replaces emails, phone numbers, and names using regexes and spaCy NER.
Runs dual concurrent workflows at the batch level and record level.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.utils.config_loader import load_yaml_config
from src.utils.helpers import get_project_root

log = logging.getLogger("pipeline.data.anonymization")

# --- Load NLP model once (thread-safe for concurrent inference) ---
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
    log.debug("spaCy model 'en_core_web_sm' loaded successfully.")
except (ImportError, OSError) as exc:
    _nlp = None
    _HAS_SPACY = False
    log.warning(
        "spaCy model 'en_core_web_sm' not available (%s). "
        "Person-name anonymization will be skipped. "
        "To enable name cleaning, run: python -m spacy download en_core_web_sm",
        exc,
    )

# --- Pre-compiled regex patterns ---
_RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
_RE_PHONE = re.compile(
    r'(?<!\d)'
    r'(?:\+?\d{1,3}[\s.-]?)?'
    r'(?:\?\(?\d{3}\)?[\s.-]?)?'
    r'\d{3}[\s.-]?\d{4}'
    r'(?!\d)'
)
_RE_PHONE_IN = re.compile(r'(?<!\d)(?:\+?91[\s.-]?)?[6-9]\d{9}(?!\d)')


def clean_pii_text(text: str) -> str:
    """Anonymize personal identifiers (emails, phone numbers, names) in text."""
    if not isinstance(text, str):
        return text

    # 1. Emails
    text = _RE_EMAIL.sub("[EMAIL]", text)

    # 2. Phone Numbers
    text = _RE_PHONE_IN.sub("[PHONE]", text)
    text = _RE_PHONE.sub("[PHONE]", text)

    # 3. Person Names via spaCy
    if _HAS_SPACY and _nlp is not None:
        try:
            doc = _nlp(text)
            replacements: list[tuple[str, str]] = []
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    replacements.append((ent.text, "[NAME]"))

            # Clean longest names first to prevent partial substrings bugs
            replacements.sort(key=lambda x: len(x[0]), reverse=True)
            for original, token in replacements:
                text = text.replace(original, token)
        except Exception as e:
            log.error("spaCy processing error: %s", e)

    return text


def _clean_record(record: dict) -> dict:
    """Mask PII in user and assistant dialogue fields."""
    cleaned = dict(record)
    if "user" in cleaned and isinstance(cleaned["user"], str):
        cleaned["user"] = clean_pii_text(cleaned["user"])
    if "assistant" in cleaned and isinstance(cleaned["assistant"], str):
        cleaned["assistant"] = clean_pii_text(cleaned["assistant"])
    return cleaned


def process_batch(src_file: Path, dst_file: Path, pii_workers: int) -> int:
    """Mask an entire batch in parallel."""
    if dst_file.exists():
        try:
            existing = json.loads(dst_file.read_text(encoding="utf-8"))
            log.info("Batch %s — skipping (cleaned exists, %d records)", src_file.name, len(existing))
            return len(existing)
        except Exception:
            log.warning("Batch %s — cleaned file corrupted, overwriting...", src_file.name)

    try:
        records: list[dict] = json.loads(src_file.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Batch %s — failed to load raw file: %s", src_file.name, exc)
        return 0

    if not records:
        log.warning("Batch %s — empty file, skipping.", src_file.name)
        return 0

    log.info("Batch %s — scrubbing PII in %d records...", src_file.name, len(records))
    t0 = time.time()

    cleaned_records: list[dict] = [None] * len(records)  # type: ignore[list-item]
    record_workers = min(pii_workers, len(records))

    with ThreadPoolExecutor(max_workers=record_workers) as record_ex:
        future_to_idx = {
            record_ex.submit(_clean_record, rec): idx
            for idx, rec in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                cleaned_records[idx] = future.result()
            except Exception as exc:
                log.error("Batch %s, record %d — cleaning error: %s. Keeping original.", src_file.name, idx, exc)
                cleaned_records[idx] = records[idx]

    elapsed = time.time() - t0

    try:
        dst_file.write_text(json.dumps(cleaned_records, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Batch %s — scrubbed %d records → %s (%.1fs)", src_file.name, len(cleaned_records), dst_file.name, elapsed)
        return len(cleaned_records)
    except Exception as exc:
        log.error("Batch %s — failed to write cleaned JSON: %s", src_file.name, exc)
        return 0


def run_cleaning(config_path: str | Path | None = None) -> bool:
    """Run data cleaning / anonymization pipeline."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "dataset" / "real_estate.yaml"

    cfg = load_yaml_config(config_path)

    raw_dir = root / cfg.paths.raw_dir
    interim_dir = root / cfg.paths.interim_dir
    interim_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(raw_dir.glob("batch_*.json"))
    if not batch_files:
        log.warning("No batch files found in %s. Run ingestion first.", raw_dir)
        return False

    log.info("=" * 60)
    log.info("Starting PII Anonymization Pipeline")
    log.info("  Raw Source   : %s", raw_dir.resolve())
    log.info("  Interim Dest : %s", interim_dir.resolve())
    log.info("  File Count   : %d files", len(batch_files))
    log.info("  Workers      : %d threads", cfg.cleaning.pii_workers)
    log.info("=" * 60)

    total_records = 0
    failed_batches: list[str] = []

    with ThreadPoolExecutor(max_workers=cfg.cleaning.pii_workers) as batch_ex:
        future_to_src = {}
        for src_file in batch_files:
            dst_file = interim_dir / src_file.name
            future = batch_ex.submit(process_batch, src_file, dst_file, cfg.cleaning.pii_workers)
            future_to_src[future] = src_file

        for future in as_completed(future_to_src):
            src = future_to_src[future]
            try:
                n = future.result()
                total_records += n
                if n == 0:
                    failed_batches.append(src.name)
            except Exception as exc:
                log.error("Batch %s — unhandled exception during cleaning: %s", src.name, exc)
                failed_batches.append(src.name)

    log.info("=" * 60)
    log.info("PII Scrubbing Complete.")
    log.info("  Total Scrubbed : %d records", total_records)
    log.info("  Failed Batches : %s", sorted(failed_batches) or "none")
    log.info("=" * 60)

    return len(failed_batches) == 0
