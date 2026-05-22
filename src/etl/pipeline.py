"""
src/etl/pipeline.py
===================
Orchestrates the full ETL pipeline in three sequential phases:

  1. EXTRACT  — MongoDB → Kafka raw topic
  2. TRANSFORM — Kafka raw topic → PII clean + format → Kafka processed topic
  3. LOAD      — Kafka processed topic → MongoDB processed_records collection

Usage (programmatic):
    from src.etl.pipeline import run_etl_pipeline
    run_etl_pipeline()

Usage (CLI):
    python -m src.etl.pipeline

The transform and load phases can optionally run concurrently using threads
(set parallel=True), which is useful for high-throughput scenarios.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.utils.logger import setup_logger
from src.etl.extract import run_extract
from src.etl.transform import run_transform
from src.etl.load import run_load

log = setup_logger("pipeline.etl.pipeline")


def run_etl_pipeline(
    *,
    batch_filter: dict[str, Any] | None = None,
    parallel_transform_load: bool = False,
    poll_timeout_ms: int = 3_000,
    max_empty_polls: int = 5,
) -> dict[str, Any]:
    """
    Execute the complete ETL pipeline.

    Args:
        batch_filter:              MongoDB filter for Extract phase.
                                   Defaults to {"etl_status": "pending"}.
        parallel_transform_load:   If True, Transform + Load run concurrently in
                                   separate threads (useful when Kafka topic has
                                   many messages already queued).
        poll_timeout_ms:           Kafka poll timeout per round (ms).
        max_empty_polls:           Max consecutive empty Kafka polls before stopping.

    Returns:
        Summary dict with stats from each phase.
    """
    log.info("╔" + "═" * 58 + "╗")
    log.info("║  UGC ETL Pipeline — Starting Full Run                  ║")
    log.info("╚" + "═" * 58 + "╝")

    summary: dict[str, Any] = {}

    # ── Phase 1: EXTRACT ────────────────────────────────────────
    log.info("▶ Phase 1/3 — EXTRACT")
    extracted = run_extract(batch_filter=batch_filter)
    summary["extract"] = {"published_to_kafka": extracted}

    if extracted == 0:
        log.warning("Nothing extracted. ETL pipeline aborted early.")
        return summary

    # ── Phase 2 + 3: TRANSFORM → LOAD ───────────────────────────
    transform_stats: dict[str, int] = {}
    load_stats: dict[str, int] = {}

    if parallel_transform_load:
        log.info("▶ Phase 2+3 — TRANSFORM & LOAD (parallel threads)")

        t_transform = threading.Thread(
            target=lambda: transform_stats.update(
                run_transform(
                    poll_timeout_ms=poll_timeout_ms,
                    max_empty_polls=max_empty_polls,
                )
            ),
            name="etl-transform",
            daemon=True,
        )
        t_load = threading.Thread(
            target=lambda: load_stats.update(
                run_load(
                    poll_timeout_ms=poll_timeout_ms,
                    max_empty_polls=max_empty_polls,
                )
            ),
            name="etl-load",
            daemon=True,
        )
        t_transform.start()
        t_load.start()
        t_transform.join()
        t_load.join()

    else:
        log.info("▶ Phase 2/3 — TRANSFORM")
        transform_stats = run_transform(
            poll_timeout_ms=poll_timeout_ms,
            max_empty_polls=max_empty_polls,
        )

        log.info("▶ Phase 3/3 — LOAD")
        load_stats = run_load(
            poll_timeout_ms=poll_timeout_ms,
            max_empty_polls=max_empty_polls,
        )

    summary["transform"] = transform_stats
    summary["load"] = load_stats

    log.info("╔" + "═" * 58 + "╗")
    log.info("║  UGC ETL Pipeline — Complete                           ║")
    log.info("╠" + "═" * 58 + "╣")
    log.info("║  Extract  → %d records published to Kafka              ", extracted)
    log.info("║  Transform→ %s", transform_stats)
    log.info("║  Load     → %s", load_stats)
    log.info("╚" + "═" * 58 + "╝")
    return summary


# ──────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the UGC ETL pipeline (Extract → Transform → Load)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run Transform and Load phases concurrently in separate threads.",
    )
    parser.add_argument(
        "--only-pending",
        action="store_true",
        default=True,
        help="Only extract records with etl_status='pending' (default: True).",
    )
    parser.add_argument(
        "--all",
        dest="extract_all",
        action="store_true",
        help="Re-extract ALL raw records regardless of etl_status.",
    )
    args = parser.parse_args()

    filt = {} if args.extract_all else {"etl_status": "pending"}
    run_etl_pipeline(
        batch_filter=filt,
        parallel_transform_load=args.parallel,
    )
