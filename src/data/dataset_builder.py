"""
src/data/dataset_builder.py
===========================
Processes anonymized interim datasets into final fine-tuning datasets.
Merges JSON batches, formats into Chat ML, shuffles, splits (70/15/15),
and saves train.jsonl, val.jsonl, and test.jsonl in processed/ directory.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from src.utils.config_loader import load_yaml_config
from src.utils.helpers import get_project_root
from src.data.formatting import format_to_conversations

log = logging.getLogger("pipeline.data.dataset_builder")


def run_preparation(config_path: str | Path | None = None) -> bool:
    """Merge interim anonymized batches, format, split and save fine-tuning datasets."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "dataset" / "real_estate.yaml"

    cfg = load_yaml_config(config_path)

    interim_dir = root / cfg.paths.interim_dir
    processed_dir = root / cfg.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(interim_dir.glob("batch_*.json"))
    if not batch_files:
        log.error("No anonymized batch files found in %s. Run cleaning first.", interim_dir)
        return False

    log.info("=" * 60)
    log.info("Starting Dataset Builder (Format & Split)")
    log.info("  Interim Source : %s", interim_dir.resolve())
    log.info("  Processed Dest : %s", processed_dir.resolve())
    log.info("  Files Found    : %d files", len(batch_files))
    log.info("=" * 60)

    all_data = []

    # Merge all batches
    for filepath in batch_files:
        try:
            raw = json.loads(filepath.read_text(encoding="utf-8"))
            all_data.extend(raw)
            log.debug("Merged %d records from %s", len(raw), filepath.name)
        except Exception as exc:
            log.error("Failed to merge batch %s: %s", filepath.name, exc)
            return False

    log.info("Total merged raw records: %d", len(all_data))
    if not all_data:
        log.error("No dataset records merged!")
        return False

    # Convert to standard Chat format
    converted = []
    for item in all_data:
        user_text = item.get("user", "")
        assistant_text = item.get("assistant", "")
        converted.append(format_to_conversations(user_text, assistant_text))

    # Shuffle dataset deterministically (seed=42)
    random.seed(42)
    random.shuffle(converted)

    # Split 70% Train / 15% Val / 15% Test
    total = len(converted)
    train_split = int(0.70 * total)
    eval_split = int(0.85 * total)

    train_data = converted[:train_split]
    val_data = converted[train_split:eval_split]
    test_data = converted[eval_split:]

    log.info("Split Outputs:")
    log.info("  Train Size : %d (70%%)", len(train_data))
    log.info("  Val Size   : %d (15%%)", len(val_data))
    log.info("  Test Size  : %d (15%%)", len(test_data))

    def save_jsonl(data: list[dict], filename: str) -> None:
        filepath = processed_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.info("Saved %d records -> %s", len(data), filepath.name)
        except Exception as exc:
            log.error("Failed to save split file %s: %s", filename, exc)
            raise

    try:
        save_jsonl(train_data, "train.jsonl")
        save_jsonl(val_data, "val.jsonl")
        save_jsonl(test_data, "test.jsonl")
    except Exception:
        return False

    log.info("=" * 60)
    log.info("Dataset Preparation Split Complete.")
    log.info("=" * 60)
    return True
